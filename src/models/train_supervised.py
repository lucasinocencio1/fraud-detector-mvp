import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, classification_report
from imblearn.over_sampling import SMOTE

# ===============================
# 1. Caminhos e setup
# ===============================
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

train = pd.read_parquet("artifacts/train_feat.parquet")
val = pd.read_parquet("artifacts/val_feat.parquet")

# ===============================
# 2. Preparação dos dados
# ===============================
y_tr = train["Class"]
y_val = val["Class"]
X_tr = train.drop(columns=["Class"])
X_val = val.drop(columns=["Class"])

# Convertendo colunas categóricas
for col in ["region", "device_type", "merchant_category"]:
    if col in X_tr.columns:
        X_tr[col] = X_tr[col].astype("category").cat.codes
        X_val[col] = X_val[col].astype("category").cat.codes

# ===============================
# 3. Novas features derivadas
# ===============================
def enrich_features(df):
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

X_tr = enrich_features(X_tr)
X_val = enrich_features(X_val)

# ===============================
# 4. Balanceamento (SMOTE + Undersampling)
# ===============================
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Estratégia híbrida: SMOTE + Undersampling
# Isso cria um dataset mais equilibrado sem perder muita informação
over_sampler = SMOTE(sampling_strategy=0.3, random_state=42)  # 30% de fraudes
under_sampler = RandomUnderSampler(sampling_strategy=0.7, random_state=42)  # 70% de normais

# Pipeline de balanceamento
sampler = Pipeline([
    ('over', over_sampler),
    ('under', under_sampler)
])

X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)

print(f"Dataset balanceado: {y_tr.value_counts().to_dict()} → {y_tr_res.value_counts().to_dict()}")
print(f"Taxa de fraude: {y_tr_res.mean():.1%}")

# ===============================
# 5. Treinamento com hiperparâmetros otimizados
# ===============================
model = XGBClassifier(
    n_estimators=500,        # Mais árvores para melhor performance
    max_depth=8,             # Profundidade adequada
    learning_rate=0.03,      # Taxa de aprendizado menor
    subsample=0.8,           # Subamostragem para evitar overfitting
    colsample_bytree=0.8,    # Subamostragem de features
    eval_metric="logloss",
    gamma=0.1,               # Regularização
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    min_child_weight=3,      # Evita overfitting em dados pequenos
    random_state=42,
    n_jobs=-1,
)

model.fit(X_tr_res, y_tr_res)

# ===============================
# 6. Avaliação no conjunto de validação
# ===============================
y_pred = model.predict_proba(X_val)[:, 1]
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

# Métricas de classificação com threshold otimizado
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Threshold otimizado baseado na curva Precision-Recall
optimal_threshold = prec[np.argmax(prec + rec)]
y_pred_binary = (y_pred > optimal_threshold).astype(int)

precision = precision_score(y_val, y_pred_binary)
recall = recall_score(y_val, y_pred_binary)
f1 = f1_score(y_val, y_pred_binary)
roc_auc = roc_auc_score(y_val, y_pred)

print(f"\n📊 Métricas de Performance:")
print(f"AUPRC: {auprc:.4f}")
print(f"Average Precision: {avg_prec:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Precision@1%: {precision_at_1:.4f}")
print(f"Precision@5%: {precision_at_5:.4f}")
print(f"Precision@10%: {precision_at_10:.4f}")

print(f"\n🎯 Métricas com Threshold Otimizado ({optimal_threshold:.3f}):")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print(f"\n📋 Relatório de classificação:")
print(classification_report(y_val, y_pred_binary))

# ===============================
# 7. Salvar modelo e métricas
# ===============================
joblib.dump(model, ARTIFACTS_DIR / "supervised_xgb.joblib")
joblib.dump(list(X_tr.columns), ARTIFACTS_DIR / "feature_names.joblib")

metrics = {
    "auprc": float(auprc), 
    "avg_precision": float(avg_prec), 
    "roc_auc": float(roc_auc),
    "precision_at_1pct": float(precision_at_1),
    "precision_at_5pct": float(precision_at_5),
    "precision_at_10pct": float(precision_at_10),
    "optimal_threshold": float(optimal_threshold),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "n_features": len(X_tr.columns),
    "n_train_samples": len(X_tr_res),
    "n_val_samples": len(X_val),
    "fraud_rate": float(y_tr_res.mean())
}

with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n💾 Modelo e métricas salvos em artifacts/")
print(f"   - Modelo: supervised_xgb.joblib")
print(f"   - Features: feature_names.joblib") 
print(f"   - Métricas: metrics.json")

# ===============================
# 8. Importância das features
# ===============================
import matplotlib.pyplot as plt

feat_imp = pd.DataFrame({"feature": X_tr.columns, "importance": model.feature_importances_})
feat_imp = feat_imp.sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp["feature"][:15], feat_imp["importance"][:15])
plt.gca().invert_yaxis()
plt.title("Importância das 15 Principais Features (XGBoost)")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "feature_importance.png")
plt.close()
print("Gráfico salvo em artifacts/feature_importance.png")
