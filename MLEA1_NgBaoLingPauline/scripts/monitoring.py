# dags/model_monitoring.py

import sys
import pandas as pd
from sklearn.metrics import fbeta_score

# Configuration
INPUT_CSV = '/opt/airflow/data/final_merged_2.csv'     # Path to your merged dataset
TRUE_COL = 'label'                        # Ground-truth label column
PRED_COL = 'prediction'                  # Model prediction column (adjust if different)
BETA = 2.0                               # Beta value for F-beta score
THRESHOLD = 0.6                          # Threshold below which to flag


def load_data(path):
    df = pd.read_csv(path)
    if TRUE_COL not in df.columns:
        raise KeyError(f"True-label column '{TRUE_COL}' not found in {path}")
    if PRED_COL not in df.columns:
        raise KeyError(f"Prediction column '{PRED_COL}' not found in {path}")
    return df


def compute_fbeta(y_true, y_pred, beta=BETA):
    return fbeta_score(y_true, y_pred, beta=beta)


def main():
    # Load data
    df = load_data(INPUT_CSV)
    y_true = df[TRUE_COL]
    y_pred = df[PRED_COL]

    # Compute F-beta
    fbeta_val = compute_fbeta(y_true, y_pred)
    print(f"F{BETA}-score: {fbeta_val:.4f}")

    # Flag if below threshold
    if fbeta_val < THRESHOLD:
        print(f"⚠️ ALERT: F{BETA}-score ({fbeta_val:.4f}) is below threshold {THRESHOLD}")
        sys.exit(1)
    else:
        print("✅ F-beta score is above threshold")
        sys.exit(0)


if __name__ == '__main__':
    main()