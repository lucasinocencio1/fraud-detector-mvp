import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

from src.utils.metrics import auprc as calc_auprc
from src.utils.metrics import precision_at_k
from src.utils.psi import psi

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
TRAIN_FEAT_PATH = ARTIFACTS_DIR / "train_feat.parquet"
VAL_FEAT_PATH = ARTIFACTS_DIR / "val_feat.parquet"

BASE_FEATURES = [
    "Amount",
    "transaction_hour",
    "region",
    "device_type",
    "merchant_category",
    "is_weekend",
    "avg_amount_user",
    "amount_to_avg_ratio",
    "tx_last_24h",
    "tx_last_7d",
]

CATEGORICAL_FEATURES = ["region", "device_type", "merchant_category"]

PSI_NUMERIC_FEATURES = [
    "Amount",
    "transaction_hour",
    "avg_amount_user",
    "amount_to_avg_ratio",
    "tx_last_24h",
    "tx_last_7d",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "amount" in df.columns and "Amount" not in df.columns:
        df = df.rename(columns={"amount": "Amount"})
    if "class" in df.columns and "Class" not in df.columns:
        df = df.rename(columns={"class": "Class"})
    return df


def ensure_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BASE_FEATURES:
        if col not in df.columns:
            if col in CATEGORICAL_FEATURES:
                df[col] = "unknown"
            else:
                df[col] = 0.0
    return df[BASE_FEATURES]


# 1. Load artifacts and validation data
try:
    pipeline = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    train_df = pd.read_parquet(TRAIN_FEAT_PATH)
    val_df = pd.read_parquet(VAL_FEAT_PATH)
    print("Artifacts loaded")
except FileNotFoundError as e:
    raise SystemExit(f"File not found: {e}. Run training before evaluation.")

# 2. Prepare data
train_df = normalize_columns(train_df)
val_df = normalize_columns(val_df)

if "Class" not in train_df.columns or "Class" not in val_df.columns:
    raise SystemExit("Missing target column 'Class' in train/validation data")

y_train = train_df["Class"].astype(int)
y_val = val_df["Class"].astype(int)

train_features = ensure_base_features(train_df.drop(columns=["Class"], errors="ignore"))
val_features = ensure_base_features(val_df.drop(columns=["Class"], errors="ignore"))

# 3. Predict
scores = pipeline.predict_proba(val_features)[:, 1]

prec, rec, _ = precision_recall_curve(y_val, scores)
pr_auc = auc(rec, prec)
avg_prec = average_precision_score(y_val, scores)
roc_auc = roc_auc_score(y_val, scores)
auprc_score = calc_auprc(y_val, scores)
precision_at_1pct = precision_at_k(y_val, scores, k=0.01)

# 4. Save metrics
metrics = {
    "AUPRC": float(pr_auc),
    "Average_Precision": float(avg_prec),
    "AUPRC_Sklearn": float(auprc_score),
    "ROC_AUC": float(roc_auc),
    "Precision_at_1pct": float(precision_at_1pct),
    "Optimal_Threshold": float(threshold),
    "Samples_Train": int(len(y_train)),
    "Samples_Val": int(len(y_val)),
}

psi_values = {
    feature: float(psi(train_features[feature], val_features[feature]))
    for feature in PSI_NUMERIC_FEATURES
    if feature in train_features.columns and feature in val_features.columns
}

if psi_values:
    metrics["PSI"] = psi_values

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print("Evaluation metrics")
print(f"PR AUC: {pr_auc:.4f}")
print(f"AUPRC (average precision): {auprc_score:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision: {avg_prec:.4f}")
print(f"Precision@1%: {precision_at_1pct:.4f}")
print(f"Optimal Threshold: {float(threshold):.3f}")
if psi_values:
    print("PSI by feature:")
    for feature, value in psi_values.items():
        print(f"  {feature}: {value:.4f}")
print("Metrics saved to artifacts/metrics.json")
