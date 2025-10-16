import joblib
import pandas as pd
import numpy as np
import json
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_auc_score
from pathlib import Path

# ===============================
# 1. Funções auxiliares
# ===============================
def enrich_features(df):
    """Feature engineering completo para detecção de fraude"""
    df = df.copy()

    # 1. Transformações de Amount
    if "Amount" in df.columns:
        df["amount_log"] = np.log1p(df["Amount"])
        df["amount_sqrt"] = np.sqrt(df["Amount"])
        df["high_amount_flag"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)

    # 2. Features de Risco por Categoria
    if "merchant_category" in df.columns:
        risky_categories = ["electronics", "travel", "gaming"]
        df["risky_category"] = df["merchant_category"].isin(risky_categories).astype(int)

    # 3. Features Temporais Cíclicas
    if "transaction_hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)
        df["hour_is_night"] = ((df["transaction_hour"] >= 0) & (df["transaction_hour"] <= 5)).astype(int)

    # 4. Features de Região de Risco
    if "region" in df.columns:
        risk_map = {"US": 3, "ASIA": 3, "EU": 2, "BR": 1}
        df["region_risk"] = df["region"].map(risk_map).fillna(1)

    # 5. Interações Amount x V-features (mais importantes)
    if "Amount" in df.columns:
        for i in [1, 2, 3]:
            v_col = f"V{i}"
            if v_col in df.columns:
                df[f"amount_v{i}"] = df["Amount"] * df[v_col]

    # 6. Estatísticas das V-features
    v_cols = [c for c in df.columns if c.startswith("V") and len(c) <= 3]
    if len(v_cols) > 0:
        df["v_sum"] = df[v_cols].sum(axis=1)
        df["v_mean"] = df[v_cols].mean(axis=1)
        df["v_std"] = df[v_cols].std(axis=1)
        df["v_max"] = df[v_cols].max(axis=1)
        df["v_min"] = df[v_cols].min(axis=1)

    # 7. Features de Comportamento
    if "tx_last_24h" in df.columns:
        df["high_frequency"] = (df["tx_last_24h"] > 10).astype(int)
        if "tx_last_7d" in df.columns:
            df["tx_rate"] = df["tx_last_24h"] / (df["tx_last_7d"] + 1)

    if "amount_to_avg_ratio" in df.columns:
        df["unusual_amount"] = (df["amount_to_avg_ratio"] > 2.5).astype(int)

    return df


# ===============================
# 2. Carregar modelos e dados
# ===============================
try:
    xgb_model = joblib.load("artifacts/supervised_xgb.joblib")
    lgbm_model = joblib.load("artifacts/supervised_lgbm.joblib")
    optimal_threshold = joblib.load("artifacts/optimal_threshold.joblib")
    feature_names = joblib.load("artifacts/feature_names.joblib")
    X_val = pd.read_parquet("artifacts/val_feat.parquet")
    print(f"✅ Modelos carregados (XGBoost + LightGBM) com {len(feature_names)} features")
    print(f"✅ Threshold ótimo: {optimal_threshold:.3f}")
    print(f"✅ Dados de validação: {X_val.shape[0]} amostras, {X_val.shape[1]} colunas")
except FileNotFoundError as e:
    raise SystemExit(f"❌ Arquivo não encontrado: {e}. Rode 'make train_sup' antes de avaliar.")

# ===============================
# 3. Preparar dados
# ===============================
y_val = X_val["Class"].astype(int)
X_val = X_val.drop(columns=["Class"])

for col in ["region", "device_type", "merchant_category"]:
    if col in X_val.columns:
        X_val[col] = X_val[col].astype("category").cat.codes

X_val = enrich_features(X_val)

# Garantir consistência das features
missing_features = set(feature_names) - set(X_val.columns)
if missing_features:
    print(f"⚠️ Features faltando: {missing_features}")
    for feat in missing_features:
        X_val[feat] = 0

extra_features = set(X_val.columns) - set(feature_names)
if extra_features:
    print(f"⚠️ Features extras ignoradas: {extra_features}")
    X_val = X_val.drop(columns=list(extra_features))

X_val = X_val[feature_names]
print(f"✅ Features preparadas: {X_val.shape[1]} colunas")

# ===============================
# 4. Predição com Ensemble
# ===============================
y_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
y_pred_lgbm = lgbm_model.predict_proba(X_val)[:, 1]
y_pred = 0.6 * y_pred_xgb + 0.4 * y_pred_lgbm  # Ensemble

prec, rec, _ = precision_recall_curve(y_val, y_pred)
auprc = auc(rec, prec)
avg_prec = average_precision_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred)

def precision_at_k(y_true, y_score, k):
    k = int(k * len(y_true))
    top_k_idx = np.argsort(y_score)[-k:]
    return y_true.iloc[top_k_idx].mean()

precision_at_1 = precision_at_k(y_val, y_pred, 0.01)
precision_at_5 = precision_at_k(y_val, y_pred, 0.05)
precision_at_10 = precision_at_k(y_val, y_pred, 0.10)

print("\n📊 Métricas de Avaliação (Ensemble):")
print(f"AUPRC: {auprc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision: {avg_prec:.4f}")
print(f"Precision@1%: {precision_at_1:.4f}")
print(f"Precision@5%: {precision_at_5:.4f}")
print(f"Precision@10%: {precision_at_10:.4f}")

# ===============================
# 5. Salvar métricas
# ===============================
metrics = {
    "AUPRC": float(auprc),
    "ROC_AUC": float(roc_auc),
    "Average_Precision": float(avg_prec),
    "Precision@1%": float(precision_at_1),
    "Precision@5%": float(precision_at_5),
    "Precision@10%": float(precision_at_10),
    "Optimal_Threshold": float(optimal_threshold),
    "Samples_Val": int(len(y_val)),
}

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n💾 Métricas salvas em artifacts/metrics.json")
