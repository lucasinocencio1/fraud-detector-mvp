# 🚀 Melhorias Implementadas no Modelo de Detecção de Fraude

## 📊 Problema Identificado

O modelo anterior apresentava **performance muito baixa**:
- ❌ AUPRC: 0.042 (praticamente aleatório)
- ❌ ROC-AUC: 0.510 (não melhor que 50%)
- ❌ Precision/Recall/F1: 0.0 (modelo não detectava nada)

### Causa Raiz
1. **Falta de balanceamento de dados**: Modelo sobre-ajustado à classe majoritária
2. **Feature engineering incompleto**: Features agregadas não eram salvas
3. **Hiperparâmetros não otimizados**: Valores fixos e sub-ótimos
4. **Threshold fixo**: Sem otimização baseada em custo de negócio

---

## ✅ Melhorias Implementadas

### 1. 🔄 Balanceamento de Dados (SMOTE + Undersampling)

**Implementação:**
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

over_sampler = SMOTE(sampling_strategy=0.5, random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy=0.7, random_state=42)
sampler = Pipeline([('over', over_sampler), ('under', under_sampler)])

X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)
```

**Impacto:** +300% no AUPRC esperado

---

### 2. 🎯 Feature Engineering Completo

**Novas Features Implementadas:**

#### Transformações de Amount
- `amount_log`: Logaritmo natural de Amount
- `amount_sqrt`: Raiz quadrada de Amount
- `high_amount_flag`: Flag para valores acima do 95º percentil

#### Features de Risco
- `risky_category`: Flag para categorias de alto risco (electronics, travel, gaming)
- `region_risk`: Score de risco por região (US/ASIA: 3, EU: 2, BR: 1)

#### Features Temporais Cíclicas
- `hour_sin`: Seno da hora (0-23) para capturar padrões cíclicos
- `hour_cos`: Cosseno da hora
- `hour_is_night`: Flag para horário noturno (0-5h)

#### Interações Amount × V-features
- `amount_v1`, `amount_v2`, `amount_v3`: Interações com as 3 principais V-features

#### Estatísticas das V-features
- `v_sum`, `v_mean`, `v_std`, `v_max`, `v_min`: Estatísticas agregadas

#### Features de Comportamento
- `high_frequency`: Flag para alta frequência de transações (>10 em 24h)
- `tx_rate`: Taxa de transações (24h / 7d)
- `unusual_amount`: Flag para valores anormais (>2.5x média do usuário)

**Impacto:** +50-100% no AUPRC esperado

---

### 3. 🔍 Otimização de Hiperparâmetros (RandomizedSearchCV)

**Implementação:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

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

search = RandomizedSearchCV(
    base_model,
    param_dist,
    n_iter=30,
    cv=3,
    scoring='average_precision',
    n_jobs=-1
)
```

**Impacto:** +50-100% no AUPRC esperado

---

### 4. 💰 Threshold Optimization com Custo de Negócio

**Implementação:**
```python
def find_optimal_threshold(y_true, y_pred_proba, cost_fp=1, cost_fn=10):
    """Encontra threshold ótimo baseado em custo de negócio"""
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = (fp * cost_fp) + (fn * cost_fn)  # FP=1, FN=10
        costs.append(cost)
    
    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx], costs[optimal_idx]
```

**Impacto:** Melhora significativa em Precision e Recall balanceados

---

### 5. 🎯 Ensemble XGBoost + LightGBM

**Implementação:**
```python
from lightgbm import LGBMClassifier

# Treinar LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='binary',
    metric='auc',
    random_state=42
)

# Ensemble (média ponderada)
y_pred_ensemble = 0.6 * y_pred_xgb + 0.4 * y_pred_lgbm
```

**Impacto:** +20-30% no AUPRC esperado

---

### 6. 🐛 Correção de Bugs Críticos

#### Bug no `feature_build.py`
**Antes:**
```python
for df in [train, val, test]:
    df.merge(region_stats, on="region", how="left")  # ❌ Não salva!
    df.merge(cat_stats, on="merchant_category", how="left")
```

**Depois:**
```python
train = train.merge(region_stats, on="region", how="left")  # ✅ Salva!
train = train.merge(cat_stats, on="merchant_category", how="left")

val = val.merge(region_stats, on="region", how="left")
val = val.merge(cat_stats, on="merchant_category", how="left")

test = test.merge(region_stats, on="region", how="left")
test = test.merge(cat_stats, on="merchant_category", how="left")
```

---

## 📈 Resultados Esperados

| Métrica | Antes | Meta Fase 1 | Meta Fase 2 | Meta Fase 3 |
|---------|-------|-------------|-------------|-------------|
| **AUPRC** | 0.042 | > 0.15 | > 0.40 | > 0.65 |
| **ROC-AUC** | 0.510 | > 0.65 | > 0.75 | > 0.85 |
| **Precision@1%** | 0.05 | > 0.15 | > 0.30 | > 0.50 |
| **Precision** | 0.0 | > 0.20 | > 0.40 | > 0.60 |
| **Recall** | 0.0 | > 0.30 | > 0.50 | > 0.70 |

---

## 🚀 Como Usar

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Treinar Modelo com Melhorias
```bash
make data        # Gerar dados sintéticos
make features    # Construir features
make train_sup   # Treinar com balanceamento, GridSearch e ensemble
```

### 3. Avaliar Modelo
```bash
make evaluate    # Avaliar ensemble no conjunto de validação
```

### 4. Servir API
```bash
make serve       # Iniciar API FastAPI com ensemble
```

---

## 📦 Arquivos Modificados

1. **`src/models/train_supervised.py`**
   - ✅ Balanceamento SMOTE + Undersampling
   - ✅ Feature engineering completo
   - ✅ RandomizedSearchCV
   - ✅ Threshold optimization
   - ✅ Ensemble XGBoost + LightGBM

2. **`src/models/evaluate.py`**
   - ✅ Enrich_features completo
   - ✅ Avaliação com ensemble
   - ✅ Métricas expandidas

3. **`src/data/feature_build.py`**
   - ✅ Correção do bug de merge
   - ✅ Salvamento correto de features agregadas

4. **`src/serve/api.py`**
   - ✅ Suporte a ensemble predictions

5. **`src/serve/loader.py`**
   - ✅ Carregamento de modelos ensemble
   - ✅ Carregamento de threshold ótimo

6. **`requirements.txt`**
   - ✅ Adicionado `lightgbm`
   - ✅ Adicionado `imbalanced-learn`

---

## 🎯 Próximos Passos

### Fase 3: Features Avançadas (Futuro)
- [ ] Autoencoders para detecção de anomalias
- [ ] SHAP para interpretabilidade do modelo
- [ ] Métricas de drift (PSI) para monitoramento
- [ ] A/B testing framework
- [ ] Deploy com Docker + Kubernetes

---

## 📚 Referências

- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)

---

**Data:** 2025-01-27  
**Autor:** Fraud Detector Team  
**Versão:** 2.0

