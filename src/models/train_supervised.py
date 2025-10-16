import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    auc,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

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


X_tr = enrich_features(X_tr)
X_val = enrich_features(X_val)

# ===============================
# 4. Balanceamento de Dados
# ===============================
print("\n🔄 Balanceando dados...")
over_sampler = SMOTE(sampling_strategy=0.5, random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy=0.7, random_state=42)
sampler = Pipeline([
    ('over', over_sampler),
    ('under', under_sampler)
])

X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)
print(f"✅ Dados balanceados: {len(y_tr_res)} amostras ({len(y_tr_res[y_tr_res==1])} fraudes, {len(y_tr_res[y_tr_res==0])} normais)")

# ===============================
# 5. Otimização de Hiperparâmetros
# ===============================
print("\n🔍 Otimizando hiperparâmetros...")
param_dist = {
    'n_estimators': randint(300, 1000),
    'max_depth': randint(5, 15),
    'learning_rate': uniform(0.01, 0.1),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 0.5),
    'reg_lambda': uniform(0.5, 2.0),
    'min_child_weight': randint(1, 10)
}

base_model = XGBClassifier(
    eval_metric='aucpr',
    random_state=42,
    n_jobs=-1
)

search = RandomizedSearchCV(
    base_model,
    param_dist,
    n_iter=30,  # 30 iterações para velocidade
    cv=3,
    scoring='average_precision',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_tr_res, y_tr_res)
print(f"✅ Melhores parâmetros encontrados:")
for param, value in search.best_params_.items():
    print(f"   {param}: {value}")

model = search.best_estimator_

# ===============================
# 6. Threshold Optimization
# ===============================
def find_optimal_threshold(y_true, y_pred_proba, cost_fp=1, cost_fn=10):
    """Encontra threshold ótimo baseado em custo de negócio"""
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = (fp * cost_fp) + (fn * cost_fn)
        costs.append(cost)
    
    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx], costs[optimal_idx]

# ===============================
# 7. Avaliação
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

# Threshold tuning com custo de negócio
optimal_threshold, min_cost = find_optimal_threshold(y_val, y_pred)
print(f"\n💰 Threshold ótimo: {optimal_threshold:.3f} (custo mínimo: {min_cost:.2f})")
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
# 8. Ensemble com LightGBM
# ===============================
print("\n🎯 Treinando ensemble XGBoost + LightGBM...")
lgbm_model = LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='binary',
    metric='auc',
    random_state=42,
    n_jobs=-1
)
lgbm_model.fit(X_tr_res, y_tr_res)

# Predições do ensemble (média ponderada)
y_pred_xgb = model.predict_proba(X_val)[:, 1]
y_pred_lgbm = lgbm_model.predict_proba(X_val)[:, 1]
y_pred_ensemble = 0.6 * y_pred_xgb + 0.4 * y_pred_lgbm

# Avaliar ensemble
prec_ens, rec_ens, _ = precision_recall_curve(y_val, y_pred_ensemble)
auprc_ensemble = auc(rec_ens, prec_ens)
roc_auc_ensemble = roc_auc_score(y_val, y_pred_ensemble)

metrics_at_k_ensemble = {
    "precision@1%": precision_at_k(y_val, y_pred_ensemble, 0.01),
    "precision@5%": precision_at_k(y_val, y_pred_ensemble, 0.05),
    "precision@10%": precision_at_k(y_val, y_pred_ensemble, 0.10),
}

optimal_threshold_ens, min_cost_ens = find_optimal_threshold(y_val, y_pred_ensemble)
y_pred_bin_ensemble = (y_pred_ensemble > optimal_threshold_ens).astype(int)

precision_ensemble = precision_score(y_val, y_pred_bin_ensemble)
recall_ensemble = recall_score(y_val, y_pred_bin_ensemble)
f1_ensemble = f1_score(y_val, y_pred_bin_ensemble)

print(f"\n📊 Ensemble Performance:")
print(f"AUPRC: {auprc_ensemble:.4f} | ROC-AUC: {roc_auc_ensemble:.4f}")
print(f"Precision: {precision_ensemble:.4f} | Recall: {recall_ensemble:.4f} | F1: {f1_ensemble:.4f}")

# ===============================
# 9. Salvar artefatos
# ===============================
# Salvar threshold ótimo
joblib.dump(optimal_threshold_ens, ARTIFACTS_DIR / "optimal_threshold.joblib")
joblib.dump(model, ARTIFACTS_DIR / "supervised_xgb.joblib")
joblib.dump(lgbm_model, ARTIFACTS_DIR / "supervised_lgbm.joblib")
joblib.dump(list(X_tr.columns), ARTIFACTS_DIR / "feature_names.joblib")

metrics = {
    "AUPRC": float(auprc_ensemble),
    "ROC_AUC": float(roc_auc_ensemble),
    "Precision@1%": float(metrics_at_k_ensemble["precision@1%"]),
    "Precision@5%": float(metrics_at_k_ensemble["precision@5%"]),
    "Precision@10%": float(metrics_at_k_ensemble["precision@10%"]),
    "Precision": float(precision_ensemble),
    "Recall": float(recall_ensemble),
    "F1": float(f1_ensemble),
    "Optimal_Threshold": float(optimal_threshold_ens),
    "N_features": int(len(X_tr.columns)),
    "XGB_AUPRC": float(auprc),
    "XGB_ROC_AUC": float(roc_auc),
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
