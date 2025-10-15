
# Fraud Detector — MVP End-to-End (Português)

Primeiro projeto de **detecção de fraudes/anomalias** para você clonar e rodar localmente.  
Inclui: geração de dados **sintéticos**, split temporal, features básicas, modelos (XGBoost e IsolationForest), API FastAPI, **testes** e **CI**.

## Como rodar (local)
```bash
make setup
make data         # gera CSV sintético e faz split temporal
make features     # cria features e salva em artifacts/
make train_sup    # treina modelo supervisionado (XGBoost)
make train_unsup  # treina IsolationForest
make test         # roda testes
make serve        # inicia API em http://127.0.0.1:8000
```

### Exemplo de request
```bash
curl -X POST http://127.0.0.1:8000/predict       -H "Content-Type: application/json"       -d '{"Amount": 123.45, "V1":0,"V2":0,"V3":0,"V4":0,"V5":0,"V6":0,"V7":0,
       "V8":0,"V9":0,"V10":0,"V11":0,"V12":0,"V13":0,"V14":0,
       "V15":0,"V16":0,"V17":0,"V18":0,"V19":0,"V20":0,"V21":0,
       "V22":0,"V23":0,"V24":0,"V25":0,"V26":0,"V27":0,"V28":0}'
```

## Estrutura
```text
fraud-detector/
├─ src/
│  ├─ data/
│  ├─ models/
│  ├─ serve/
│  └─ utils/
├─ tests/
├─ .github/workflows/ci.yml
├─ artifacts/   (gerado)
├─ models/      (gerado)
└─ data/        (gerado)
```

## Métricas e validação
- Dados desbalanceados → foque em **AUPRC** e **Precision@K**.
- Split por **tempo** (anti-leakage).
- Testes: métricas, split, API e contrato de dados.

## Roadmap
- Autoencoder (PyTorch)
- Threshold por **custo esperado**
- Monitoramento de **drift** (PSI)
- Dashboard (Streamlit/Metabase)

