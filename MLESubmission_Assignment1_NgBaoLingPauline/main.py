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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table_feature
import utils.data_processing_silver_table_feature
import utils.data_processing_gold_table_feature

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2025-01-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
dates_str_lst


# create bronze datalake (clickstream)
bronze_clickstream_directory = "datamart/bronze/feature/clickstream/"

if not os.path.exists(bronze_clickstream_directory):
    os.makedirs(bronze_clickstream_directory)

# run bronze backfill (clickstream)
for date_str in dates_str_lst:
    utils.data_processing_bronze_table_feature.process_bronze_table_clickstream(date_str, bronze_clickstream_directory, spark)

# create bronze datalake (attributes)
bronze_attributes_directory = "datamart/bronze/feature/attributes/"

if not os.path.exists(bronze_attributes_directory):
    os.makedirs(bronze_attributes_directory)

# run bronze backfill (attributes)
for date_str in dates_str_lst:
    utils.data_processing_bronze_table_feature.process_bronze_table_attributes(date_str, bronze_attributes_directory, spark)

# create bronze datalake (financials)
bronze_financials_directory = "datamart/bronze/feature/financials/"

if not os.path.exists(bronze_financials_directory):
    os.makedirs(bronze_financials_directory)

# run bronze backfill (financials)
for date_str in dates_str_lst:
    utils.data_processing_bronze_table_feature.process_bronze_table_financials(date_str, bronze_financials_directory, spark)

# create silver datalake (clickstream)
silver_clickstream_directory = "datamart/silver/feature/clickstream/"

if not os.path.exists(silver_clickstream_directory):
    os.makedirs(silver_clickstream_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table_feature.process_silver_table_clickstream(date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)

# create silver datalake (attributes)
silver_attributes_directory = "datamart/silver/feature/attributes/"

if not os.path.exists(silver_attributes_directory):
    os.makedirs(silver_attributes_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table_feature.process_silver_table_attributes(date_str, bronze_attributes_directory, silver_attributes_directory, spark)

# create silver datalake (financials)
silver_financials_directory = "datamart/silver/feature/financials/"

if not os.path.exists(silver_financials_directory):
    os.makedirs(silver_financials_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table_feature.process_silver_table_financials(date_str, bronze_financials_directory, silver_financials_directory, spark)

# create gold datalake (feature store)
gold_feature_store_directory = "datamart/gold/feature/"

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table_feature.process_features_gold_table(date_str, silver_clickstream_directory, silver_attributes_directory, silver_financials_directory, gold_feature_store_directory, spark)

folder_path = gold_feature_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

# Filter out any rows with nulls, take one row
complete_row = df.na.drop().limit(1).collect()[0]

# Convert to a Python dict and print
row_dict = complete_row.asDict()
for field, val in row_dict.items():
    print(f"{field}: {val}")



    