#!/usr/bin/env python3
"""
model_inference.py

Loads a saved XGBoost model (JSON format only) and runs inference on a CSV,
locking in exactly the 19 features expected by the model,
with support for specifying a custom input file path.
Appends probability and label columns back onto the original CSV.
"""
import argparse
import os
import sys
import pandas as pd
import xgboost as xgb

# 1) Define the canonical feature list (exactly 19 features)
FEATURE_COLS = [
    'num__Annual_Income',
    'num__Num_Bank_Accounts',
    'num__Num_Credit_Card',
    'num__Interest_Rate',
    'num__Num_of_Loan',
    'num__Delay_from_due_date',
    'num__Num_of_Delayed_Payment',
    'num__Num_Credit_Inquiries',
    'num__Outstanding_Debt',
    'num__Monthly_Balance',
    'num__days_overdue_per_late_payment',
    'num__Credit_History_Age_num',
    'num__debt_to_income_ratio',
    'num__monthly_repayment_to_income',
    'num__credit_inquiries_per_year',
    'num__Student_Loan_count',
    'cat__Credit_Mix_Bad',
    'cat__Payment_of_Min_Amount_No',
    'cat__Payment_of_Min_Amount_Yes',
]

# 1.5) Fixed threshold for binary label
THRESHOLD = 0.32

# 2) Determine base directories relative to this script
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IN_DIR     = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'in'))
OUT_DIR    = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))
MODEL_DIR  = os.path.normpath(os.path.join(BASE_DIR, 'model_bank'))


def load_and_prepare_data(input_path: str) -> pd.DataFrame:
    """
    Reads a CSV from the given path and selects only the locked-in features.
    Raises an error if any expected column is missing or the file is not found.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {sorted(missing)}")
    return df[FEATURE_COLS].copy()


def load_model(model_name: str) -> xgb.Booster:
    """
    Loads an XGBoost model from JSON (.json) format only.
    """
    if not model_name.endswith('.json'):
        raise ValueError("Model file must be in .json format")

    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def main(snapshot_date: str, model_name: str, input_filepath: str = None):
    # Determine input path: custom or based on snapshot_date
    if input_filepath:
        input_path = input_filepath.strip()
    else:
        input_path = os.path.join(IN_DIR, f"{snapshot_date}.csv")

    # Load full original CSV
    raw_df = pd.read_csv(input_path)

    # Prepare just the 19 features
    df_feat = load_and_prepare_data(input_path)
    dmat    = xgb.DMatrix(df_feat, feature_names=FEATURE_COLS)

    # Load model from JSON
    booster = load_model(model_name)

    # Sanity check feature count
    if dmat.num_col() != len(FEATURE_COLS):
        raise RuntimeError(
            f"Feature mismatch: model expects {len(FEATURE_COLS)}, "
            f"but DMatrix has {dmat.num_col()}"
        )

    # Predict probabilities and binary labels
    y_proba = booster.predict(dmat)
    y_pred  = (y_proba > THRESHOLD).astype(int)

    # Append results back onto the original data
    raw_df['prediction_proba'] = y_proba
    raw_df['prediction']       = y_pred

    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build output filename
    base_name   = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(
        OUT_DIR,
        f"{base_name}_{model_name}_predictions.csv"
    )

    # Save the enriched DataFrame
    raw_df.to_csv(output_path, index=False)
    print(f"Saved {len(raw_df)} rows with probability and label to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with XGBoost JSON model on locked features"
    )
    parser.add_argument(
        "--snapshotdate", required=False,
        help="Snapshot date string matching input CSV (e.g., 2023-07-01)"
    )
    parser.add_argument(
        "--modelname", required=True,
        help="Filename of the JSON model (e.g., xgboostv1.json)"
    )
    parser.add_argument(
        "--inputpath", required=False,
        help="Full path to input CSV (overrides --snapshotdate)"
    )
    args = parser.parse_args()

    try:
        main(args.snapshotdate, args.modelname, args.inputpath)
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        sys.exit(1)