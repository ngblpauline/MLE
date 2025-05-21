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
        "Customer_ID":           StringType(),
        "Num_Bank_Accounts":     IntegerType(),
        "Num_Credit_Card":       IntegerType(),
        "Interest_Rate":         IntegerType(),
        "Delay_from_due_date":   IntegerType(),
        "Num_Credit_Inquiries":  IntegerType(),
        "Payment_of_Min_Amount": StringType(),
        "Payment_Behaviour":     StringType(),
        "Type_of_Loan":          StringType(),
        "Credit_Mix":            StringType(),
        "snapshot_date":         DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # 3) keep these as strings for regex‐cleaning
    string_fields = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_of_Loan",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "Credit_History_Age",
    ]
    for c in string_fields:
        df = df.withColumn(c, col(c).cast(StringType()))

    # 4) strip underscores & cast to float
    for c in [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_of_Loan",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Outstanding_Debt",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "Credit_Utilization_Ratio",
    ]:
        df = df.withColumn(
            c,
            F.regexp_replace(col(c), "_", "")
             .cast(FloatType())
        )

    # 4b) impute Changed_Credit_Limit blanks → median
    median_ccl = df.approxQuantile("Changed_Credit_Limit", [0.5], 0.0)[0]
    df = df.withColumn(
        "Changed_Credit_Limit",
        when(col("Changed_Credit_Limit").isNull(), median_ccl)
         .otherwise(col("Changed_Credit_Limit"))
    )

    # 5) standardize Payment_Behaviour & Credit_Mix
    valid_pb = [
        'High_spent_Large_value_payments',
        'High_spent_Medium_value_payments',
        'High_spent_Small_value_payments',
        'Low_spent_Large_value_payments',
        'Low_spent_Medium_value_payments',
        'Low_spent_Small_value_payments'
    ]
    df = df.withColumn(
        "Payment_Behaviour",
        when(col("Payment_Behaviour").isin(valid_pb),
             col("Payment_Behaviour"))
        .otherwise("Unknown")
    )
    valid_cm = ['Bad', 'Good', 'Standard']
    df = df.withColumn(
        "Credit_Mix",
        when(col("Credit_Mix").isin(valid_cm), col("Credit_Mix"))
        .otherwise("Unknown")
    )

    df = df.withColumn(
    "days_overdue_per_late_payment",
    (
      # clean & cast the numerator, default to 0.0 if non-numeric or null
      coalesce(
        regexp_replace(col("Delay_from_due_date"), "[^0-9\\.]", "")
          .cast("double"),
        lit(0.0)
      )
      /
      # clean & cast the denominator, but if it’s <= 0 (or non-numeric → null), use 1.0
      when(
        regexp_replace(col("Num_of_Delayed_Payment"), "[^0-9\\.]", "")
          .cast("double") > 0,
        regexp_replace(col("Num_of_Delayed_Payment"), "[^0-9\\.]", "")
          .cast("double")
      )
      .otherwise(lit(1.0))
    ))

    # 6) parse Credit_History_Age while still a string
    df = (
        df.withColumn(
            "years",
            F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years?", 1)
             .cast(IntegerType())
        )
          .withColumn(
            "months",
            F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months?", 1)
             .cast(IntegerType())
        )
          .withColumn(
            "Credit_History_Age_num",
            (col("years") + col("months")/F.lit(12)).cast(FloatType())
        )
          .drop("Credit_History_Age","years","months")
    )

    # 7) compute ratio features exactly like pandas
    df = (
        df.withColumn(
            "debt_to_income_ratio",
            col("Outstanding_Debt") / col("Annual_Income")
        )
          .withColumn(
            "monthly_repayment_to_income",
            col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")
        )
          .withColumn(
            "days_overdue_per_late_payment",
            col("Delay_from_due_date") / col("Num_of_Delayed_Payment")
        )
          .withColumn(
            "credit_inquiries_per_year",
            col("Num_Credit_Inquiries") / col("Credit_History_Age_num")
        )
    )

    # 8) Loan type counts (exactly like pandas’ one‐hot counts)
    from pyspark.sql.functions import regexp_replace, split, size, expr, sum as spark_sum, lit

    loan_types = [
        'Auto Loan',
        'Credit-Builder Loan',
        'Debt Consolidation Loan',
        'Home Equity Loan',
        'Mortgage Loan',
        'Not Specified',
        'Payday Loan',
        'Personal Loan',
        'Student Loan'
    ]

    # normalize “ and ” → comma, then split into array
    df = df.withColumn(
        "_loan_array",
        split(
            regexp_replace(col("Type_of_Loan"), r"\s+and\s+", ","), 
            r",\s*"
        )
    )

    # for each loan type, count its occurrences
    for lt in loan_types:
        out_col = lt.replace('-', '_').replace(' ', '_') + "_count"
        df = df.withColumn(
            out_col,
            size(expr(f"filter(_loan_array, x -> x = '{lt}')"))
        )

    # if none matched, Unknown_count = 1
    known_cols = [lt.replace('-', '_').replace(' ', '_') + "_count" for lt in loan_types]
    df = df.withColumn(
        "_known_sum",
        sum(col(c) for c in known_cols)
    ).withColumn(
        "Unknown_count",
        when(col("_known_sum") == 0, lit(1)).otherwise(lit(0))
    ).drop("_loan_array", "_known_sum", "Type_of_Loan")

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

