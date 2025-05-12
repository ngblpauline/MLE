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
        "Customer_ID": StringType(),
        "Annual_Income": StringType(),  
        "Monthly_Inhand_Salary": StringType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": StringType(),  
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": StringType(),
        "Changed_Credit_Limit": StringType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": StringType(),
        "Credit_Utilization_Ratio": StringType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": StringType(),
        "Amount_invested_monthly": StringType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

     # clean numeric strings: strip non-numeric/minus/decimal and cast to float
    for c in [
        "Annual_Income", "Monthly_Inhand_Salary", "Changed_Credit_Limit",
        "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"
    ]:
        df = df.withColumn(
            c,
            regexp_replace(col(c), r"[^0-9.\-]", "").cast(FloatType())
        )
    # flag missing Changed_Credit_Limit
    df = df.withColumn(
        "changed_credit_limit_missing",
        when(col("Changed_Credit_Limit").isNull(), 1).otherwise(0).cast(IntegerType())
    )

    # handle missing sentinel for Num_Bank_Accounts (-1) and flag
    df = df.withColumn(
        "Num_Bank_Accounts",
        when(col("Num_Bank_Accounts") < 0, None)
          .otherwise(col("Num_Bank_Accounts")).cast(IntegerType())
    )
    df = df.withColumn(
        "bank_accounts_missing",
        when(col("Num_Bank_Accounts").isNull(), 1).otherwise(0).cast(IntegerType())
    )

    # clean and flag Num_of_Loan underscores and missing sentinel (-100)
    df = df.withColumn(
        "Num_of_Loan",
        regexp_replace(col("Num_of_Loan"), r"[^0-9]", "").cast(IntegerType())
    )
    df = df.withColumn(
        "num_of_loan_missing",
        when(col("Num_of_Loan") < 0, 1).otherwise(0).cast(IntegerType())
    )

    # clean Num_of_Delayed_Payment
    df = df.withColumn(
        "Num_of_Delayed_Payment",
        regexp_replace(col("Num_of_Delayed_Payment"), r"[^0-9]", "").cast(IntegerType())
    )

    # augment data: financial ratios
    df = df.withColumn(
        "debt_to_income_ratio",
        (col("Outstanding_Debt") / col("Annual_Income")).cast(FloatType())
    )
    df = df.withColumn(
        "loan_credit_ratio",
        (col("Num_of_Loan") / col("Num_Credit_Card")).cast(FloatType())
    )

    # flag cases where Num_Credit_Card was zero
    df = df.withColumn(
        "credit_card_zero_flag",
        when(col("Num_Credit_Card") == 0, 1).otherwise(0).cast(IntegerType())
    )
    
    df = df.withColumn(
        "dsi_ratio",
        (col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")).cast(FloatType())
    )

    # flag delayed payments
    df = df.withColumn(
        "delay_flag",
        when(col("Delay_from_due_date") > 0, 1).otherwise(0).cast(IntegerType())
    )

    # parse credit history age into months
    df = df.withColumn(
        "history_years",
        regexp_extract(col("Credit_History_Age"), r"(\d+) Years", 1).cast(IntegerType())
    ).withColumn(
        "history_months",
        regexp_extract(col("Credit_History_Age"), r"(\d+) Months", 1).cast(IntegerType())
    ).withColumn(
        "credit_history_months_total",
        (col("history_years") * 12 + col("history_months")).cast(IntegerType())
    )

    # payment_of_min cleaning, flag and missing
    df = df.withColumn(
        "Payment_of_Min_Amount",
        when(col("Payment_of_Min_Amount") == "NM", None).otherwise(col("Payment_of_Min_Amount"))
    )
    df = df.withColumn(
        "payment_of_min_flag",
        when(col("Payment_of_Min_Amount") == "Yes", 1)
        .when(col("Payment_of_Min_Amount") == "No", 0)
        .otherwise(None).cast(IntegerType())
    )
    df = df.withColumn(
        "payment_of_min_missing",
        when(col("Payment_of_Min_Amount").isNull(), 1).otherwise(0).cast(IntegerType())
    )

    # credit_mix cleaning, flag and missing
    df = df.withColumn(
        "Credit_Mix",
        when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix"))
    )
    df = df.withColumn(
        "credit_mix_missing",
        when(col("Credit_Mix").isNull(), 1).otherwise(0).cast(IntegerType())
    )
    df = df.withColumn(
        "credit_mix_flag",
        when(col("Credit_Mix") == "Good", 2)
        .when(col("Credit_Mix") == "Standard", 1)
        .when(col("Credit_Mix") == "Bad", 0)
        .otherwise(None).cast(IntegerType())
    )

    # clean and split Payment_Behaviour, flag missing
    valid_pb = [
        "High_spent_Large_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments"
    ]
    df = df.withColumn(
        "Payment_Behaviour",
        when(col("Payment_Behaviour").isin(valid_pb), col("Payment_Behaviour")).otherwise(None)
    )
    df = df.withColumn(
        "payment_behaviour_missing",
        when(col("Payment_Behaviour").isNull(), 1).otherwise(0).cast(IntegerType())
    )
    df = df.withColumn(
        "spend_level",
        regexp_extract(col("Payment_Behaviour"), r'^(High|Low)_spent_', 1).cast(StringType())
    )
    df = df.withColumn(
        "payment_size",
        regexp_extract(
            col("Payment_Behaviour"),
            r'^(?:High|Low)_spent_(Large_value_payments|Medium_value_payments|Small_value_payments)$',
            1
        ).cast(StringType())
    )

    # drop extreme Monthly_Balance outliers
    df = df.filter(abs(col("Monthly_Balance")) < 1e9)

    # drop raw and helper columns
    df = df.drop(
        "Credit_History_Age", "history_years", "history_months",
        "Payment_of_Min_Amount", "Credit_Mix", "Payment_Behaviour"
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
        "Customer_ID": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # clean age: strip non-digits (if any) and recast
    df = df.withColumn(
        "Age_clean",
        regexp_replace(col("Age"), "[^0-9]", "").cast(IntegerType()))
    
    # flag invalid ages (<0 or >120)
    df = df.withColumn(
        "age_invalid_flag",
        when((col("Age_clean") < 0) | (col("Age_clean") > 120), 1).otherwise(0))
    
    # finalize age: null invalid entries
    df = df.withColumn("Age_final",when(col("age_invalid_flag") == 1, None).otherwise(col("Age_clean")).cast(IntegerType())
                      )
    
    # drop intermediates
    df = df.drop("Age").drop("Age_clean")

    # augment data: is_minor flag on age
    df = df.withColumn(
        "is_minor",
        when(col("Age_final") < 18, 1).otherwise(0))

    # augment data: age buckets
    df = df.withColumn(
        "age_group",
        when(col("Age_final") < 25, "18-24")
        .when(col("Age_final") <= 35, "25-35")
        .when(col("Age_final") <= 50, "36-50")
        .otherwise("50+"))

    # augment data: SSN validity flag
    df = df.withColumn(
        "ssn_valid",
        when(col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), 1).otherwise(0).cast(IntegerType()))

     # convert placeholder '_______' to null
    df = df.withColumn("Occupation", when(col("Occupation") == "_______", None).otherwise(col("Occupation")))

    # augment data: missing occupation flag
    df = df.withColumn("occupation_missing", when(col("Occupation").isNull() | (col("Occupation") == ""), 1).otherwise(0).cast(IntegerType()))

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

    # augment data: total and average click features
    numeric_cols = [col(f"fe_{i}") for i in range(1, 21)]
    df = df.withColumn("click_sum", reduce(add, numeric_cols).cast(DoubleType()))
    df = df.withColumn("click_avg",(col("click_sum") / len(numeric_cols)).cast(DoubleType()))

    # augment data: active feature count (# of non-zero features)
    numeric_names = [f"fe_{i}" for i in range(1, 21)]
    indicators = [when(col(c) != 0, 1).otherwise(0) for c in numeric_names]
    df = df.withColumn("active_feat_count",reduce(add, indicators).cast(IntegerType()))
    
    # augment data: volatility (std deviation across features)
    sq_diffs = [ (col(c) - col("click_avg")) * (col(c) - col("click_avg")) for c in numeric_names ]
    df = df.withColumn(
        "click_volatility",
        F.sqrt(reduce(lambda a, b: a + b, sq_diffs) / len(numeric_names)).cast(DoubleType()))

    # outlier detection: flag values outside 1st and 99th percentiles per feature
    quantiles = df.stat.approxQuantile(numeric_names, [0.01, 0.99], 0.0)
    # attach outlier flags
    for feature, (q01, q99) in zip(numeric_names, quantiles):
        df = df.withColumn(
            f"{feature}_outlier_flag",
            when((col(feature) < q01) | (col(feature) > q99), 1).otherwise(0).cast(IntegerType()))
        
    # save silver table - IRL connect to database to write
    partition_name = "silver_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet' 
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


