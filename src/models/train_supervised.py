import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    auc,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

# ===============================
# 1. Caminhos e setup
# ===============================
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

train = pd.read_parquet("artifacts/train_feat.parquet")
val = pd.read_parquet("artifacts/val_feat.parquet")

print(f"Treino: {train.shape}, Validação: {val.shape}")

# ===============================
# 2. Preparação dos dados
# ===============================
y_tr = train["Class"].astype(int)
y_val = val["Class"].astype(int)
X_tr = train.drop(columns=["Class"])
X_val = val.drop(columns=["Class"])

# Codificação de variáveis categóricas
for col in ["region", "device_type", "merchant_category"]:
    if col in X_tr.columns:
        X_tr[col] = X_tr[col].astype("category").cat.codes
        X_val[col] = X_val[col].astype("category").cat.codes

# ===============================
# 3. Features derivadas
# ===============================
def enrich_features(df):
    df = df.copy()

    # Amount transformations
    if "Amount" in df.columns:
        df["amount_log"] = np.log1p(df["Amount"])
        df["high_amount_flag"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)

    # Temporal encoding
    if "transaction_hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)

    # Risk region
    if "region" in df.columns:
        df["region_risk"] = df["region"].map({
            "US": 3, "ASIA": 3, "EU": 2, "BR": 1
        }).fillna(1)

    # Statistical transformations (if V columns exist)
    v_cols = [c for c in df.columns if c.startswith("V")]
    if len(v_cols) > 0:
        df["v_mean"] = df[v_cols].mean(axis=1)
        df["v_std"] = df[v_cols].std(axis=1)

    return df


X_tr = enrich_features(X_tr)
X_val = enrich_features(X_val)

# ===============================
# 4. Modelo otimizado
# ===============================
scale_pos_weight = len(y_tr[y_tr == 0]) / len(y_tr[y_tr == 1])
print(f"Scale_pos_weight = {scale_pos_weight:.1f}")

model = XGBClassifier(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.2,
    reg_alpha=0.2,
    reg_lambda=1.0,
    min_child_weight=4,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_tr, y_tr)

# ===============================
# 5. Avaliação
# ===============================
y_pred = model.predict_proba(X_val)[:, 1]
prec, rec, _ = precision_recall_curve(y_val, y_pred)
auprc = auc(rec, prec)
avg_prec = average_precision_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred)

# Top-K precisions
def precision_at_k(y_true, y_score, k):
    k = int(k * len(y_true))
    top_k_idx = np.argsort(y_score)[-k:]
    return y_true.iloc[top_k_idx].mean()

metrics_at_k = {
    "precision@1%": precision_at_k(y_val, y_pred, 0.01),
    "precision@5%": precision_at_k(y_val, y_pred, 0.05),
    "precision@10%": precision_at_k(y_val, y_pred, 0.10),
}

# Threshold tuning
optimal_threshold = 0.5  # fixo para estabilidade
y_pred_bin = (y_pred > optimal_threshold).astype(int)

precision = precision_score(y_val, y_pred_bin)
recall = recall_score(y_val, y_pred_bin)
f1 = f1_score(y_val, y_pred_bin)

print("\n📊 Métricas de Performance:")
print(f"AUPRC: {auprc:.4f} | ROC-AUC: {roc_auc:.4f}")
for k, v in metrics_at_k.items():
    print(f"{k}: {v:.4f}")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
print("\n📋 Relatório de classificação:")
print(classification_report(y_val, y_pred_bin))

# ===============================
# 6. Salvar artefatos
# ===============================
joblib.dump(model, ARTIFACTS_DIR / "supervised_xgb.joblib")
joblib.dump(list(X_tr.columns), ARTIFACTS_DIR / "feature_names.joblib")

metrics = {
    "AUPRC": auprc,
    "ROC_AUC": roc_auc,
    "Precision@1%": metrics_at_k["precision@1%"],
    "Precision@5%": metrics_at_k["precision@5%"],
    "Precision@10%": metrics_at_k["precision@10%"],
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "N_features": len(X_tr.columns),
}
with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n💾 Modelo e métricas salvos em artifacts/")

# ===============================
# 7. Importância das features
# ===============================
import matplotlib.pyplot as plt

feat_imp = pd.DataFrame({
    "feature": X_tr.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp["feature"].head(15), feat_imp["importance"].head(15))
plt.gca().invert_yaxis()
plt.title("Top 15 Features Importantes (XGBoost)")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "feature_importance.png")
plt.close()
print("Gráfico salvo em artifacts/feature_importance.png")
