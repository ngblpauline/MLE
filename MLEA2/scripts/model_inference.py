import argparse
import os
import glob
import pandas as pd
import pickle
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import glob
from datetime import datetime

import pandas as pd
import xgboost as xgb

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics      import (
    fbeta_score,
    roc_auc_score,
    classification_report,
)


# to call this script: python model_inference.py --snapshotdate "2024-09-01" --modelname "credit_model_2024_09_01.pkl"

def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    script_dir       = os.path.dirname(os.path.abspath(__file__))
    model_bank_dir   = os.path.join(script_dir, "model_bank")
    config["model_bank_directory"]   = model_bank_dir
    config["model_artefact_filepath"] = os.path.join(model_bank_dir, config["model_name"])

    print("Loading model from:", config["model_artefact_filepath"])

    # load the JSON booster directly
    bst = xgb.Booster()
    bst.load_model(config["model_artefact_filepath"])
    
    # config["model_bank_directory"] = "model_bank/"
    # config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    
    pprint.pprint(config)
    

    # --- load model artefact from model bank ---
    # Load the model
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = json.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])


    # --- load feature store ---
    
    
    feature_location = "datamart/gold/feature/"
    
    files_list = glob.glob(os.path.join(feature_location, "*.parquet"))
    if not files_list:
        raise FileNotFoundError(f"No .parquet files in {feature_location}")
    print("Reading these files:", files_list)
    
    features_store_sdf = spark.read.parquet(*files_list)
    print("TOTAL ROWS IN FEATURE STORE:", features_store_sdf.count())
    
    features_store_sdf.printSchema()
    features_store_sdf.show(5, truncate=False)
    
    features_sdf = features_store_sdf.filter(
        col("snapshot_date") == config["snapshot_date"]
    )
    print(f"Rows for {config['snapshot_date'].date()}:", features_sdf.count())
    
    features_pdf = features_sdf.toPandas()
    features_pdf

    # --- preprocess data for modeling ---
    # prepare X_inference
    X_raw = features_pdf.drop(columns=["Name", "SSN"])

    cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_raw.select_dtypes(include=["number"]).columns.tolist()
    
    print("Categorical columns to encode:", cat_cols)
    print("Numeric columns to scale:  ", num_cols)
    
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_cat = pd.DataFrame(
        ohe.fit_transform(X_raw[cat_cols]),
        columns=ohe.get_feature_names(cat_cols),
        index=X_raw.index
    )
    
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw[num_cols]),
        columns=num_cols,
        index=X_raw.index
    )
    
    X_processed = pd.concat([X_num_scaled, X_cat], axis=1)
    print("Final preprocessed shape:", X_processed.shape)
    
    print(X_processed.head())


    # --- model prediction inference ---
    # load model
    
    
    label_map = {
        "Annual_Income":                 "Annual Income ($)",
        "Num_Bank_Accounts":             "No. of Bank Accounts",
        "Num_Credit_Card":               "No. of Credit Cards Owned",
        "Interest_Rate":                 "Interest Rate (%)",
        "Num_of_Loan":                   "Total No. of Loans",
        "Delay_from_due_date":           "Payment Delay (days)",
        "Num_of_Delayed_Payment":        "No. of Delayed Payments",
        "Num_Credit_Inquiries":          "Total Credit Inquiries",
        "Outstanding_Debt":              "Outstanding Debt",
        "days_overdue_per_late_payment": "Avg Days Overdue per Late Payment",
        "Credit_History_Age_num":        "Credit History Age (yrs)",
        "debt_to_income_ratio":          "Debt-to-Income Ratio",
        "monthly_repayment_to_income":   "Repayment-to-Income Ratio",
        "credit_inquiries_per_year":     "Credit Inquiries / Year",
        "Mortgage_Loan_count":           "No. of Mortgage Loans",
        "Student_Loan_count":            "No. of Student Loans",
        "Credit_Mix_Bad":                "Poor Credit Profile",
        "Payment_of_Min_Amount_No":      "No Minimum Payment Made",
        "Payment_of_Min_Amount_Yes":     "Minimum Payment Made",
    }
    
    selected_features = list(label_map.keys())
    X_sel = X_processed[selected_features]
    print("Shape after selecting 19 features:", X_sel.shape)  # should be (n_rows, 19)
    
    model_filepath = os.path.join("model_bank", "xgboostv1.json")
    bst = xgb.Booster()
    bst.load_model(model_filepath)
    print("Loaded XGBoost JSON model from", model_filepath)
    
    dmat = xgb.DMatrix(X_sel.values)
    y_pred_proba = bst.predict(dmat)
    results_df = features_pdf[["Customer_ID", "snapshot_date"]].copy()
    results_df["model_name"]        = "xgboostv1.json"
    results_df["model_predictions"] = y_pred_proba
    
    print(results_df.head())


    # --- save model inference to datamart gold table ---
    model_base     = config["model_name"][:-4]
    gold_directory = f"datamart/gold/model_predictions/{model_base}/"
    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save gold table - IRL connect to database to write
    partition_name = (
    f"{config['model_name'][:-4]}"
    f"_predictions_{config['snapshot_date_str'].replace('-', '_')}.parquet")
    filepath = gold_directory + partition_name
    spark.createDataFrame(results_df).write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
