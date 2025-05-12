import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from functools import reduce
from operator import add
from pyspark.sql.functions import when
from pyspark.sql.functions import regexp_replace, when, abs, regexp_extract
from pyspark.sql.types import DoubleType

def process_silver_table_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_feature_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID":             StringType(),
        "Annual_Income":           FloatType(),
        "Monthly_Inhand_Salary":   FloatType(),
        "Num_Bank_Accounts":       FloatType(),
        "Num_Credit_Card":         FloatType(),
        "Interest_Rate":           FloatType(),
        "Num_of_Loan":             FloatType(),
        "Type_of_Loan":            StringType(),
        "Delay_from_due_date":     FloatType(),
        "Num_of_Delayed_Payment":  FloatType(),
        "Changed_Credit_Limit":    FloatType(),
        "Num_Credit_Inquiries":    FloatType(),
        "Credit_Mix":              StringType(),
        "Outstanding_Debt":        FloatType(),
        "Credit_Utilization_Ratio":FloatType(),
        "Credit_History_Age":      FloatType(),
        "Payment_of_Min_Amount":   StringType(),
        "Total_EMI_per_month":     FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour":       StringType(),
        "Monthly_Balance":         FloatType(),
        "snapshot_date":           DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # 1) Clean Annual_Income, Monthly_Inhand_Salary,
    #    Num_of_Delayed_Payment, Changed_Credit_Limit
    for c in [
        'Annual_Income',
        'Monthly_Inhand_Salary',
        'Num_of_Delayed_Payment',
        'Changed_Credit_Limit'
    ]:
        df = df.withColumn(
            c,
            F.regexp_replace(col(c).cast(StringType()), '_', '')
             .cast(FloatType())
        )
    median_ccl = df.approxQuantile('Changed_Credit_Limit', [0.5], 0.0)[0]
    df = df.withColumn(
        'Changed_Credit_Limit',
        when(col('Changed_Credit_Limit').isNull(), median_ccl)
        .otherwise(col('Changed_Credit_Limit'))
        .cast(FloatType())
    )

    # 2) Clean Outstanding_Debt
    df = df.withColumn(
        'Outstanding_Debt',
        F.regexp_replace(col('Outstanding_Debt').cast(StringType()), '_', '')
         .cast(FloatType())
    )

    # 3) Clean Num_of_Loan: strip “_”, replace -100→7, blank→7, cast
    df = df.withColumn(
        'Num_of_Loan',
        F.regexp_replace(col('Num_of_Loan').cast(StringType()), '_', '')
    )
    df = df.withColumn(
        'Num_of_Loan',
        when(col('Num_of_Loan') == '-100', '7')
        .when(col('Num_of_Loan') == '',      '7')
        .otherwise(col('Num_of_Loan'))
        .cast(FloatType())
    )

    # 4) Define caps and apply
    caps = {
        'Num_Bank_Accounts':  10,
        'Num_Credit_Card':    10,
        'Interest_Rate':      34,
        'Num_of_Loan':         9,
    }
    for c, cap in caps.items():
        df = df.withColumn(c, when(col(c) > cap, cap).otherwise(col(c)))

    # 5) Replace Monthly_Balance placeholder with mean
    df = df.withColumn(
        'Monthly_Balance',
        when(col('Monthly_Balance') == '__-333333333333333333333333333__', None)
        .otherwise(col('Monthly_Balance'))
        .cast(FloatType())
    )
    mean_bal = df.select(F.mean('Monthly_Balance')).first()[0]
    df = df.withColumn(
        'Monthly_Balance',
        when(col('Monthly_Balance').isNull(), mean_bal)
        .otherwise(col('Monthly_Balance'))
        .cast(FloatType())
    )

    # 6) Standardize Payment_Behaviour
    valid_pb = [
        'High_spent_Large_value_payments',
        'High_spent_Medium_value_payments',
        'High_spent_Small_value_payments',
        'Low_spent_Large_value_payments',
        'Low_spent_Medium_value_payments',
        'Low_spent_Small_value_payments'
    ]
    df = df.withColumn(
        'Payment_Behaviour',
        when(col('Payment_Behaviour').isin(valid_pb), col('Payment_Behaviour'))
        .otherwise('Unknown')
    )

    # 7) Standardize Credit_Mix
    valid_cm = ['Bad', 'Good', 'Standard']
    df = df.withColumn(
        'Credit_Mix',
        when(col('Credit_Mix').isin(valid_cm), col('Credit_Mix'))
        .otherwise('Unknown')
    )

    # 8) Clean Amount_invested_monthly
    df = df.withColumn(
        'Amount_invested_monthly',
        when(col('Amount_invested_monthly') == '__10000__', None)
        .otherwise(col('Amount_invested_monthly'))
        .cast(FloatType())
    )
    mean_inv = df.select(F.mean('Amount_invested_monthly')).first()[0]
    df = df.withColumn(
        'Amount_invested_monthly',
        when(col('Amount_invested_monthly').isNull(), mean_inv)
        .otherwise(col('Amount_invested_monthly'))
        .cast(FloatType())
    )

    # 9) Parse Credit_History_Age into numeric years
    df = df.withColumn(
        'years',
        F.regexp_extract(col('Credit_History_Age'), r'(\d+)\s+Years?', 1)
        .cast(IntegerType())
    )
    df = df.withColumn(
        'months',
        F.regexp_extract(col('Credit_History_Age'), r'(\d+)\s+Months?', 1)
        .cast(IntegerType())
    )
    df = df.withColumn(
        'Credit_History_Age_num',
        (col('years') + col('months')/F.lit(12)).cast(FloatType())
    ).drop('Credit_History_Age','years','months')

    # 11) Additional ratio features (all FloatType)
    df = df.withColumn(
        "debt_to_income_ratio",
        (col("Outstanding_Debt") / col("Annual_Income")).cast(FloatType())
    )
    df = df.withColumn(
        "monthly_repayment_to_income",
        (col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")).cast(FloatType())
    )
    df = df.withColumn(
        "days_overdue_per_late_payment",
        (col("Delay_from_due_date") / col("Num_of_Delayed_Payment")).cast(FloatType())
    )
    df = df.withColumn(
        "credit_inquiries_per_year",
        (col("Num_Credit_Inquiries") / col("Credit_History_Age_num")).cast(FloatType())
    )
 

    # save silver table - IRL connect to database to write
    partition_name = "silver_feature_financials_" + snapshot_date_str.replace('-','_') + '.parquet' 
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df



def process_silver_table_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_feature_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Occupation cleanup
    # 1) strip whitespace
    df = df.withColumn('Occupation', F.trim(col('Occupation')))
    # 2) turn "_______" into empty string
    df = df.withColumn('Occupation',
                       F.regexp_replace(col('Occupation'), r'^_+$', ''))
    # 3) replace '' or 'None' with null
    df = df.withColumn('Occupation',
                       when(col('Occupation').isin('', 'None'), None)
                       .otherwise(col('Occupation')))
    # 4) fill missing with 'Unknown'
    df = df.fillna({'Occupation': 'Unknown'})

    # Age cleanup
    # 1) strip underscores and cast
    df = df.withColumn('Age_num',
        F.regexp_replace(col('Age').cast(StringType()), '_', '')
          .cast(IntegerType())
    )
    # 2) flag out‐of‐range → null
    valid_min, valid_max = 0, 150
    df = df.withColumn('Age_outlier',
        (~col('Age_num').between(valid_min, valid_max)).cast('boolean')
    )
    df = df.withColumn('Age_num',
        when(col('Age_outlier'), None)
        .otherwise(col('Age_num'))
    )
    # 3) missing flag + fill
    df = df.withColumn('Age_missing',
        when(col('Age_num').isNull(), F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn('Age_num',
        when(col('Age_num').isNull(), F.lit(valid_max))
        .otherwise(col('Age_num'))
    )
    # 4) drop intermediate
    df = df.drop('Age', 'Age_outlier')
  
    # save silver table - IRL connect to database to write
    partition_name = "silver_features_attributes_" + snapshot_date_str.replace('-','_') + '.parquet' 
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_silver_table_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "fe_1": FloatType(),
        "fe_2": FloatType(),
        "fe_3": FloatType(),
        "fe_4": FloatType(),
        "fe_5": FloatType(),
        "fe_6": FloatType(),
        "fe_7": FloatType(),
        "fe_8": FloatType(),
        "fe_9": FloatType(),
        "fe_10": FloatType(),
        "fe_11": FloatType(),
        "fe_12": FloatType(),
        "fe_13": FloatType(),
        "fe_14": FloatType(),
        "fe_15": FloatType(),
        "fe_16": FloatType(),
        "fe_17": FloatType(),
        "fe_18": FloatType(),
        "fe_19": FloatType(),
        "fe_20": FloatType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: mean of fe_1 through fe_5
    first5 = [col(f"fe_{i}") for i in range(1, 6)]
    df = df.withColumn(
        "fe_1_5_mean",
        (reduce(add, first5) / F.lit(len(first5))).cast(FloatType())
    )
    
    # save silver table - IRL connect to database to write
    partition_name = "silver_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet' 
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

