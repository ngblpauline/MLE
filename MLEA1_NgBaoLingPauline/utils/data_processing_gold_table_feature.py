import os
from datetime import datetime
from pyspark.sql import SparkSession

def process_features_gold_table(
    snapshot_date_str: str,
    silver_clickstream_directory: str,
    silver_attributes_directory: str,
    silver_financials_directory: str,
    gold_feature_store_directory: str
):
    spark = SparkSession.builder.getOrCreate()
    part = snapshot_date_str.replace('-', '_')

    # 1) read all clickstream events (24 per customer)
    click_pattern = os.path.join(
        silver_clickstream_directory,
        "silver_feature_clickstream_*.parquet"
    )
    df_click = spark.read.parquet(click_pattern)

    # 2) read the attributes slice for the given date
    attr_path = os.path.join(
        silver_attributes_directory,
        f"silver_features_attributes_{part}.parquet"
    )
    df_attr = spark.read.parquet(attr_path)

    # 3) read the financials slice for the given date
    fin_path = os.path.join(
        silver_financials_directory,
        f"silver_feature_financials_{part}.parquet"
    )
    df_fin = spark.read.parquet(fin_path)

    # 4) one-to-one join financials and attributes on (Customer_ID, snapshot_date)
    fin_attr = df_fin.join(
        df_attr,
        on=["Customer_ID", "snapshot_date"],
        how="left"
    )

    # 5) drop snapshot_date so each Customer_ID has exactly one fin/attr row
    fin_attr = fin_attr.drop("snapshot_date")

    # 6) left‐join clickstream to fin+attr on Customer_ID
    df_full = df_click.join(
        fin_attr,
        on="Customer_ID",
        how="left"
    )

    # 7) write out the gold feature store
    out_path = os.path.join(
        gold_feature_store_directory,
        f"gold_feature_store_{part}.parquet"
    )
    df_full.write.mode("overwrite").parquet(out_path)

    return df_full