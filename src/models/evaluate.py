import joblib
import pandas as pd
import numpy as np
import json
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from pathlib import Path

# ===============================
# 1. Funções auxiliares
# ===============================
def enrich_features(df):
    """Aplica as mesmas features do treinamento"""
    df = df.copy()
    
    if "Amount" in df.columns:
        df["amount_log"] = np.log1p(df["Amount"])
        df["high_amount_flag"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
    
    if "transaction_hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)
    
    if "region" in df.columns:
        df["region_risk"] = df["region"].map({
            "US": 3, "ASIA": 3, "EU": 2, "BR": 1
        }).fillna(1)
    
    v_cols = [c for c in df.columns if c.startswith("V")]
    if len(v_cols) > 0:
        df["v_mean"] = df[v_cols].mean(axis=1)
        df["v_std"] = df[v_cols].std(axis=1)
    
    return df


# ===============================
# 2. Carregar modelo e dados
# ===============================
try:
    model = joblib.load("artifacts/supervised_xgb.joblib")
    feature_names = joblib.load("artifacts/feature_names.joblib")
    X_val = pd.read_parquet("artifacts/val_feat.parquet")
    print(f"✅ Modelo carregado com {len(feature_names)} features")
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
# 4. Predição e métricas
# ===============================
y_pred = model.predict_proba(X_val)[:, 1]

prec, rec, _ = precision_recall_curve(y_val, y_pred)
auprc = auc(rec, prec)
avg_prec = average_precision_score(y_val, y_pred)

def precision_at_k(y_true, y_score, k):
    k = int(k * len(y_true))
    top_k_idx = np.argsort(y_score)[-k:]
    return y_true.iloc[top_k_idx].mean()

precision_at_1 = precision_at_k(y_val, y_pred, 0.01)
precision_at_5 = precision_at_k(y_val, y_pred, 0.05)
precision_at_10 = precision_at_k(y_val, y_pred, 0.10)

print("\n📊 Métricas de Avaliação:")
print(f"AUPRC: {auprc:.4f}")
print(f"Average Precision: {avg_prec:.4f}")
print(f"Precision@1%: {precision_at_1:.4f}")
print(f"Precision@5%: {precision_at_5:.4f}")
print(f"Precision@10%: {precision_at_10:.4f}")

# ===============================
# 5. Salvar métricas
# ===============================
metrics = {
    "AUPRC": float(auprc),
    "Average_Precision": float(avg_prec),
    "Precision@1%": float(precision_at_1),
    "Precision@5%": float(precision_at_5),
    "Precision@10%": float(precision_at_10),
    "Samples_Val": int(len(y_val)),
}

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n💾 Métricas salvas em artifacts/metrics.json")
