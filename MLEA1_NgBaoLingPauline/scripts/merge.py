#!/usr/bin/env python3
import argparse
from pyspark.sql import SparkSession, functions as F
from functools import reduce

def main():
    p = argparse.ArgumentParser(description="Merge label + feature tables and output CSV")
    p.add_argument("--snapshotdate", required=True,
                   help="YYYY-MM-DD string for the month to score")
    p.add_argument("--label_path",    required=True,
                   help="Directory containing your gold label Parquet files")
    p.add_argument("--feature_path",  required=True,
                   help="Directory containing your gold feature Parquet files")
    p.add_argument("--output_path",   required=True,
                   help="Full path to write merged CSV (e.g. /opt/airflow/scripts/data/final_merged.csv)")
    args = p.parse_args()

    spark = (
        SparkSession.builder
        .appName("JoinFeaturesLabels")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1) read and filter labels
    labels_sdf = (
        spark.read
             .option("mergeSchema", "true")
             .parquet(f"{args.label_path}/*.parquet")
             .withColumn("snapshot_date", F.to_date("snapshot_date"))
             .filter(F.col("snapshot_date") == F.lit(args.snapshotdate))
    )

    # 2) read and filter features
    features_sdf = (
        spark.read
             .option("mergeSchema", "true")
             .parquet(f"{args.feature_path}/*.parquet")
             .withColumn("snapshot_date", F.to_date("snapshot_date"))
             .filter(F.col("snapshot_date") == F.lit(args.snapshotdate))
    )

    # 3) pick out just the feature columns (here assuming they start with "fe_")
    feature_cols = [c for c in features_sdf.columns if c.startswith("fe_")]

    features_trimmed = features_sdf.select(
        "Customer_ID", "snapshot_date", *feature_cols
    )

    # 4) left join features onto labels
    joined = labels_sdf.join(
        features_trimmed,
        on=["Customer_ID", "snapshot_date"],
        how="left"
    )

    # 5) drop rows where all feature columns are null
    nonnull_condition = reduce(
        lambda a, b: a | b,
        (F.col(c).isNotNull() for c in feature_cols)
    )
    final_sdf = joined.filter(nonnull_condition)

    # 6) convert to Pandas and write CSV
    final_df = final_sdf.toPandas()
    final_df.to_csv(args.output_path, index=False)
    print(f"[✔] Wrote merged CSV to {args.output_path} ({len(final_df)} rows)")

    spark.stop()

if __name__ == "__main__":
    main()