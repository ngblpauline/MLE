import os
import glob
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import broadcast


def process_features_gold_table(
    snapshot_date_str,
    silver_clickstream_directory,
    silver_attributes_directory,
    silver_financials_directory,
    gold_feature_store_directory,
    spark,
    debug=False
):
    
    # Read every attributes partition
    df_attr_all = spark.read.parquet(
        os.path.join(silver_attributes_directory,
                     "silver_features_attributes_*.parquet")
    )
    print(f"Loaded attributes: {df_attr_all.count():,} rows ")
  
    
    # Read every financials partition
    df_fin_all = spark.read.parquet(
        os.path.join(silver_financials_directory,
                     "silver_feature_financials_*.parquet")
    )

    print(f"Loaded financials:  {df_fin_all.count():,} rows ")

      
    df_combined = (
        df_attr_all
          .join(df_fin_all,
                on=["customer_id","snapshot_date"],
                how="left")
    )

    part     = snapshot_date_str.replace("-", "_")
    click_fp = os.path.join(
        silver_clickstream_directory,
        f"silver_feature_clickstream_{part}.parquet"
    )
    df_click = spark.read.parquet(click_fp)
    print(f"Loaded clickstream for {snapshot_date_str}: {df_click.count():,} rows")
    
    
    df_static = (
        df_combined
          .drop("snapshot_date"))
          
    df_static.cache()
    
    
    df_gold_click = (
        df_click
          .join(
             broadcast(df_static),
             on="customer_id",
             how="left"
          )
    )

    df2 = (
        df_attr_all
          .join(df_fin_all,
                on=["customer_id","snapshot_date"],
                how="left")
    )
    
    df2_date = df_combined.filter(F.col("snapshot_date") == snapshot_date_str)

   
    missing_static_only = df2_date.join(
        df_click.select("customer_id").distinct(),
        on="customer_id",
        how="left_anti"
    )
    

    df_gold = df_gold_click.unionByName(
        missing_static_only,
        allowMissingColumns=True
    )
    
    out_name = f"gold_feature_store_{part}.parquet"
    out_path = os.path.join(gold_feature_store_directory, out_name)
    df_gold.write.mode("overwrite").parquet(out_path)
    print(f"Saved gold features to: {out_path} ({df_gold.count():,} rows)")

    return df_gold

