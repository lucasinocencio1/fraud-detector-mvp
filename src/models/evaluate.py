import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from pathlib import Path

def enrich_features(df):
    """Aplica as mesmas features usadas no treinamento"""
    df = df.copy()
    
    # Features baseadas no Amount (com A maiúsculo)
    df["amount_log"] = np.log1p(df["Amount"])
    df["high_amount_flag"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
    
    # Features de interação entre Amount e V features
    df["amount_v1_interaction"] = df["Amount"] * df["V1"]
    df["amount_v2_interaction"] = df["Amount"] * df["V2"]
    
    # Features estatísticas das V features
    v_cols = [f"V{i}" for i in range(1, 29)]
    df["v_sum"] = df[v_cols].sum(axis=1)
    df["v_mean"] = df[v_cols].mean(axis=1)
    df["v_std"] = df[v_cols].std(axis=1)
    df["v_max"] = df[v_cols].max(axis=1)
    df["v_min"] = df[v_cols].min(axis=1)
    
    # Features temporais
    df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)
    
    return df

try:
    # Carregar modelo e dados
    model = joblib.load("artifacts/supervised_xgb.joblib")
    feature_names = joblib.load("artifacts/feature_names.joblib")
    X_val = pd.read_parquet("artifacts/val_feat.parquet")
    
    print(f"✅ Modelo carregado com {len(feature_names)} features")
    print(f"✅ Dados de validação: {X_val.shape[0]} amostras, {X_val.shape[1]} colunas")
    
except FileNotFoundError as e:
    raise SystemExit(f"❌ Arquivo não encontrado: {e}. Rode 'make train_sup' antes de avaliar.")

# Preparar dados
y_val = X_val["Class"]
X_val = X_val.drop(columns=["Class"])

# Converter colunas categóricas para códigos numéricos
for col in ["region", "device_type", "merchant_category"]:
    if col in X_val.columns:
        X_val[col] = X_val[col].astype("category").cat.codes

# Aplicar as mesmas features do treinamento
X_val = enrich_features(X_val)

# Garantir que temos as mesmas features do modelo
missing_features = set(feature_names) - set(X_val.columns)
if missing_features:
    print(f"⚠️ Features faltando: {missing_features}")
    for feat in missing_features:
        X_val[feat] = 0  # Preencher com zeros

# Reordenar colunas para corresponder ao modelo
X_val = X_val[feature_names]

print(f"✅ Features preparadas: {X_val.shape[1]} colunas")

# Predição
y_pred = model.predict_proba(X_val)[:, 1]

# Métricas
prec, rec, _ = precision_recall_curve(y_val, y_pred)
auprc = auc(rec, prec)
avg_prec = average_precision_score(y_val, y_pred)

# Precision@K
k_1 = int(0.01 * len(y_val))
k_5 = int(0.05 * len(y_val))
k_10 = int(0.10 * len(y_val))

precision_at_1 = y_val.iloc[y_pred.argsort()[-k_1:]].mean()
precision_at_5 = y_val.iloc[y_pred.argsort()[-k_5:]].mean()
precision_at_10 = y_val.iloc[y_pred.argsort()[-k_10:]].mean()

print(f"\n📊 Métricas de Avaliação:")
print(f"AUPRC: {auprc:.4f}")
print(f"Average Precision: {avg_prec:.4f}")
print(f"Precision@1%: {precision_at_1:.4f}")
print(f"Precision@5%: {precision_at_5:.4f}")
print(f"Precision@10%: {precision_at_10:.4f}")

# Salvar métricas atualizadas
metrics = {
    "auprc": float(auprc),
    "avg_precision": float(avg_prec),
    "precision_at_1pct": float(precision_at_1),
    "precision_at_5pct": float(precision_at_5),
    "precision_at_10pct": float(precision_at_10)
}

import json
with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n💾 Métricas salvas em artifacts/metrics.json")
