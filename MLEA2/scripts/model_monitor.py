import argparse
import os
import glob
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import (
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score
)

# Helper: compute Population Stability Index (PSI)
def compute_psi(expected, actual, bins):
    exp_counts = np.histogram(expected, bins=bins)[0] / len(expected)
    act_counts = np.histogram(actual,   bins=bins)[0] / len(actual)
    eps = 1e-8
    exp_counts = np.where(exp_counts == 0, eps, exp_counts)
    act_counts = np.where(act_counts == 0, eps, act_counts)
    return np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts))

# Helper: compute Characteristic Stability Index (CSI) per feature
def compute_csi_per_feature(train_df, current_df, features, n_bins=10):
    csi = {}
    eps = 1e-8
    for feat in features:
        tr = train_df[feat].dropna().values
        cu = current_df[feat].dropna().values
        if np.issubdtype(train_df[feat].dtype, np.number):
            edges = np.unique(np.percentile(tr, np.linspace(0, 100, n_bins+1)))
            if len(edges) < 2:
                csi[feat] = 0.0
                continue
            tr_counts = np.histogram(tr, bins=edges)[0] / len(tr)
            cu_counts = np.histogram(cu, bins=edges)[0] / len(cu)
        else:
            cats = np.unique(np.concatenate([tr, cu]))
            tr_counts = pd.Series(tr).value_counts(normalize=True).reindex(cats, fill_value=0).values
            cu_counts = pd.Series(cu).value_counts(normalize=True).reindex(cats, fill_value=0).values
        tr_p = np.where(tr_counts == 0, eps, tr_counts)
        cu_p = np.where(cu_counts == 0, eps, cu_counts)
        csi[feat] = np.sum((cu_p - tr_p) * np.log(cu_p / tr_p))
    return csi

# Monitoring thresholds
targets = {
    'F2': 0.60,
    'Recall': 0.70,
    'Precision': 0.40,
    'ROC_AUC': 0.70,
    'Gini': 0.50,
    'Avg_Savings_per_Cust': 1495,
    'PSI': 0.10,
    'CSI_max': 0.10
}

def main(snapshotdate, modelname, threshold):
    snap_dt = datetime.strptime(snapshotdate, "%Y-%m-%d")
    print(f"Using classification threshold = {threshold:.2f}")

    pred_dir  = os.path.join("datamart", "gold", "model_predictions", modelname[:-4])
    label_dir = os.path.join("datamart", "gold", "label_store")

    # 1) Load predictions
    pred_files = glob.glob(os.path.join(pred_dir, f"*{snapshotdate.replace('-','_')}*.parquet"))
    if not pred_files:
        print(f"No prediction file for {snapshotdate}. Skipping monitoring.")
        return
    preds = pd.read_parquet(pred_files[0])

    # 2) Load labels
    lbl_files = glob.glob(os.path.join(label_dir, f"*{snapshotdate.replace('-','_')}*.parquet"))
    if not lbl_files:
        print(f"No label file for {snapshotdate}. Skipping monitoring.")
        return
    labels = pd.read_parquet(lbl_files[0])

    # 3) Merge and guard empty
    df = preds.merge(
        labels[["Customer_ID","snapshot_date","label"]],
        on=["Customer_ID","snapshot_date"], how="inner"
    )
    if df.empty:
        print(f"No data to monitor for {snapshotdate}. Exiting without error.")
        return

    y_true  = df['label']
    y_proba = df['model_predictions']
    y_pred  = (y_proba >= threshold).astype(int)

    # Compute metrics
    metrics = {}
    metrics['F2']        = fbeta_score(y_true, y_pred, beta=2,   zero_division=0)
    metrics['Recall']    = recall_score(y_true, y_pred,            zero_division=0)
    metrics['Precision'] = precision_score(y_true, y_pred,         zero_division=0)
    metrics['ROC_AUC']   = roc_auc_score(y_true, y_proba)
    metrics['Gini']      = 2 * metrics['ROC_AUC'] - 1
    tn, fp, fn, tp       = confusion_matrix(y_true, y_pred).ravel()
    C_FP, C_FN           = 311, 7530
    baseline             = y_true.sum() * C_FN
    cost                 = fp * C_FP + fn * C_FN
    metrics['Avg_Savings_per_Cust'] = (baseline - cost) / len(df)

    # PSI
    train_probs_path = os.path.join(
        "datamart","gold","model_probabilities",
        f"train_proba_{modelname[:-4]}.parquet"
    )
    if os.path.exists(train_probs_path):
        train_probs = pd.read_parquet(train_probs_path)['model_proba']
        bins        = np.unique(np.percentile(train_probs, np.arange(0,101,10)))
        metrics['PSI'] = compute_psi(train_probs.values, y_proba.values, bins)
    else:
        metrics['PSI'] = np.nan

    # CSI
    features_path = os.path.join(
        "datamart","gold","model_features",
        f"features_{modelname[:-4]}.csv"
    )
    if os.path.exists(features_path):
        feature_list = pd.read_csv(features_path)['feature_name'].tolist()
        train_df     = pd.read_parquet(
            train_probs_path.replace('train_proba','train_data')
        )
        csi_dict     = compute_csi_per_feature(train_df, df, feature_list)
        metrics['CSI_max'] = max(csi_dict.values()) if csi_dict else np.nan
    else:
        metrics['CSI_max'] = np.nan

    # Check against targets
    alerts = []
    for m, val in metrics.items():
        tgt = targets.get(m)
        if tgt is not None:
            if (m in ['PSI','CSI_max'] and val >= tgt) or \
               (m not in ['PSI','CSI_max'] and val < tgt):
                alerts.append(f"{m}={val:.4f} fails target {tgt}")

    # Print summary and alert
    print(f"\nMonitoring for {snapshotdate} ({modelname}):")
    for m, v in metrics.items():
        print(f"  {m:25s}: {v:.4f}  target={targets.get(m)}")
    if alerts:
        print("\n*** ALERTS: ***")
        for a in alerts:
            print(" - ", a)
    else:
        print("All metrics within target thresholds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshotdate', required=True)
    parser.add_argument('--modelname',    required=True)
    parser.add_argument('--threshold',    type=float, default=0.32,
                        help="classification cutoff probability")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname, args.threshold)