import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

from src.utils.metrics import precision_at_k
from src.utils.psi import psi

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

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


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Amount" in df.columns:
        df["amount_log"] = np.log1p(df["Amount"])
        df["amount_sqrt"] = np.sqrt(df["Amount"])
        df["high_amount_flag"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)

    if "merchant_category" in df.columns:
        risky_categories = ["electronics", "travel", "gaming"]
        df["risky_category"] = df["merchant_category"].isin(risky_categories).astype(int)

    if "transaction_hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)
        df["hour_is_night"] = (
            (df["transaction_hour"] >= 0) & (df["transaction_hour"] <= 5)
        ).astype(int)

    if "region" in df.columns:
        risk_map = {"US": 3, "ASIA": 3, "EU": 2, "BR": 1}
        df["region_risk"] = df["region"].map(risk_map).fillna(1)

    if "Amount" in df.columns:
        for i in [1, 2, 3]:
            v_col = f"V{i}"
            if v_col in df.columns:
                df[f"amount_v{i}"] = df["Amount"] * df[v_col]

    v_cols = [c for c in df.columns if c.startswith("V") and len(c) <= 3]
    if len(v_cols) > 0:
        df["v_sum"] = df[v_cols].sum(axis=1)
        df["v_mean"] = df[v_cols].mean(axis=1)
        df["v_std"] = df[v_cols].std(axis=1)
        df["v_max"] = df[v_cols].max(axis=1)
        df["v_min"] = df[v_cols].min(axis=1)

    if "tx_last_24h" in df.columns:
        df["high_frequency"] = (df["tx_last_24h"] > 10).astype(int)
        if "tx_last_7d" in df.columns:
            df["tx_rate"] = df["tx_last_24h"] / (df["tx_last_7d"] + 1)

    if "amount_to_avg_ratio" in df.columns:
        df["unusual_amount"] = (df["amount_to_avg_ratio"] > 2.5).astype(int)

    return df


train = pd.read_parquet("artifacts/train_feat.parquet")
val = pd.read_parquet("artifacts/val_feat.parquet")

train = normalize_columns(train)
val = normalize_columns(val)

if "Class" not in train.columns or "Class" not in val.columns:
    raise SystemExit("Missing target column 'Class' in train/val data")

y_tr = train["Class"].astype(int)
y_val = val["Class"].astype(int)

X_tr = ensure_base_features(train)
X_val = ensure_base_features(val)

feature_transformer = FunctionTransformer(enrich_features, validate=False)

sample_enriched = enrich_features(X_tr.copy())
all_features = list(sample_enriched.columns)

numeric_features = [f for f in all_features if f not in CATEGORICAL_FEATURES]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", numeric_features),
    ],
    remainder="drop",
)

pos = max(1, int(y_tr.sum()))
neg = max(1, int(len(y_tr) - y_tr.sum()))
scale_pos_weight = float(neg / pos)

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
)

pipeline = Pipeline(
    steps=[
        ("features", feature_transformer),
        ("preprocess", preprocess),
        ("model", model),
    ]
)

pipeline.fit(X_tr, y_tr)


def find_optimal_threshold(y_true, y_pred_proba, cost_fp=1, cost_fn=10):
    if len(np.unique(y_true)) < 2:
        return 0.5, float("inf")
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = (fp * cost_fp) + (fn * cost_fn)
        costs.append(cost)
    optimal_idx = int(np.argmin(costs))
    return float(thresholds[optimal_idx]), float(costs[optimal_idx])


y_pred = pipeline.predict_proba(X_val)[:, 1]
prec, rec, _ = precision_recall_curve(y_val, y_pred)
auprc = auc(rec, prec)
avg_prec = average_precision_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred)
precision_at_1pct = precision_at_k(y_val, y_pred, k=0.01)

optimal_threshold, min_cost = find_optimal_threshold(y_val, y_pred)
y_pred_bin = (y_pred >= optimal_threshold).astype(int)

precision = precision_score(y_val, y_pred_bin)
recall = recall_score(y_val, y_pred_bin)
f1 = f1_score(y_val, y_pred_bin)

psi_values = {
    feature: float(psi(X_tr[feature], X_val[feature]))
    for feature in PSI_NUMERIC_FEATURES
    if feature in X_tr.columns and feature in X_val.columns
}

print("Performance metrics")
print(f"PR AUC: {auprc:.4f} | ROC-AUC: {roc_auc:.4f}")
print(f"AUPRC (average precision): {avg_prec:.4f}")
print(f"Precision@1%: {precision_at_1pct:.4f}")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
print(f"Optimal threshold: {optimal_threshold:.3f} | Min cost: {min_cost:.2f}")
print("PSI by feature:")
for feature, value in psi_values.items():
    print(f"  {feature}: {value:.4f}")
print("Classification report")
print(classification_report(y_val, y_pred_bin))

joblib.dump(pipeline, MODEL_PATH)
joblib.dump(optimal_threshold, THRESHOLD_PATH)

model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

metrics = {
    "AUPRC": float(auprc),
    "AUPRC_Sklearn": float(avg_prec),
    "ROC_AUC": float(roc_auc),
    "Average_Precision": float(avg_prec),
    "Precision_at_1pct": float(precision_at_1pct),
    "Precision": float(precision),
    "Recall": float(recall),
    "F1": float(f1),
    "Optimal_Threshold": float(optimal_threshold),
    "Min_Cost": float(min_cost),
    "Samples_Val": int(len(y_val)),
    "PSI": psi_values,
    "Model_Version": model_version,
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

metadata = {
    "model_version": model_version,
    "base_features": BASE_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "numeric_features": numeric_features,
    "artifacts": {
        "model": str(MODEL_PATH),
        "threshold": str(THRESHOLD_PATH),
        "metrics": str(METRICS_PATH),
    },
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print("Artifacts saved in artifacts/")
