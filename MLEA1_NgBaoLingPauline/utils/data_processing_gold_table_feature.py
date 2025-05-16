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


def process_features_gold_table(snapshot_date_str, silver_clickstream_directory, silver_attributes_directory, silver_financials_directory, gold_feature_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to silver clickstream 
    part = snapshot_date_str.replace('-', '_')
    partition_name = f"silver_feature_clickstream_{part}.parquet"
    filepath = silver_clickstream_directory + partition_name
    df_click = spark.read.parquet(filepath)
    print(f"loaded clickstream from: {filepath}, rows: {df_click.count():,}")

    # connect to silver attributes 
    partition_name = f"silver_features_attributes_{part}.parquet"
    filepath = silver_attributes_directory + partition_name
    df_attr = spark.read.parquet(filepath)
    print(f"loaded attributes   from: {filepath}, rows: {df_attr.count():,}")

    # connect to silver financials 
    partition_name = f"silver_feature_financials_{part}.parquet"
    filepath = silver_financials_directory + partition_name
    df_fin = spark.read.parquet(filepath)
    print(f"loaded financials   from: {filepath}, rows: {df_fin.count():,}")

    # now full‐outer‐join
    df = (
        df_click
          .join(df_fin, on=["Customer_ID","snapshot_date"], how="outer")
          .join(df_attr, on=["Customer_ID","snapshot_date"], how="outer")
    )

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df