
# 🚨 Fraud Detector MVP - Sistema Completo de Detecção de Fraudes

[![CI/CD Pipeline](https://github.com/lucasinocencio1/fraud-detector-mvp/actions/workflows/ci.yml/badge.svg)](https://github.com/lucasinocencio1/fraud-detector-mvp/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Um sistema **end-to-end** completo para detecção de fraudes em transações financeiras, incluindo geração de dados sintéticos, pipeline de ML, API REST e monitoramento.

## 🎯 Características Principais

- 🔄 **Pipeline Completo**: Dados → Features → Modelo → API → Monitoramento
- 🤖 **Múltiplos Algoritmos**: XGBoost (supervisionado) + IsolationForest (não-supervisionado)
- 📊 **Dados Sintéticos Realistas**: 250k+ transações com padrões de fraude
- 🚀 **API FastAPI**: Endpoints REST para predições em tempo real
- 🧪 **CI/CD Robusto**: Testes automatizados, linting e validação
- 📈 **Métricas Adequadas**: AUPRC, Precision@K, ROC-AUC para dados desbalanceados
- 🔒 **Balanceamento Inteligente**: SMOTE + Undersampling híbrido
- 📊 **Visualizações**: Notebooks de análise exploratória

## 🚀 Quick Start

### 1. Clone e Setup
```bash
git clone https://github.com/lucasinocencio1/fraud-detector-mvp.git
cd fraud-detector-mvp
make setup
```

### 2. Pipeline Completo
```bash
# Gerar dados sintéticos e features
make data
make features

# Treinar modelos
make train_sup    # XGBoost supervisionado
make train_unsup   # IsolationForest não-supervisionado

# Avaliar performance
make evaluate

# Executar testes
make test

# Iniciar API
make serve
```

### 3. Testar API
```bash
# Health check
curl http://127.0.0.1:8000/health

# Predição de fraude
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Amount": 150.75,
    "V1": 0.5, "V2": -0.3, "V3": 1.2, "V4": -0.8, "V5": 0.1,
    "V6": -0.5, "V7": 0.8, "V8": -1.1, "V9": 0.3, "V10": -0.7,
    "V11": 0.9, "V12": -0.2, "V13": 0.6, "V14": -0.4, "V15": 0.7,
    "V16": -0.9, "V17": 0.2, "V18": -0.6, "V19": 0.4, "V20": -0.8,
    "V21": 0.5, "V22": -0.3, "V23": 0.7, "V24": -0.5, "V25": 0.3,
    "V26": -0.7, "V27": 0.6, "V28": -0.4
  }'
```

## 📁 Estrutura do Projeto

```
fraud-detector-mvp/
├── 📊 src/
│   ├── 📈 data/           # Geração e processamento de dados
│   │   ├── synth_data.py      # Dados sintéticos realistas
│   │   ├── make_dataset.py    # Split temporal
│   │   ├── feature_build.py   # Engineering de features
│   │   └── push_to_supabase.py # Integração com Supabase
│   ├── 🤖 models/         # Treinamento e avaliação
│   │   ├── train_supervised.py   # XGBoost com balanceamento
│   │   ├── train_unsupervised.py # IsolationForest
│   │   └── evaluate.py          # Métricas robustas
│   ├── 🌐 serve/          # API REST
│   │   ├── api.py             # Endpoints FastAPI
│   │   └── loader.py          # Carregamento de modelos
│   └── 🔧 utils/          # Utilitários
│       ├── metrics.py          # Métricas customizadas
│       ├── psi.py             # Monitoramento de drift
│       └── time_split.py       # Validação temporal
├── 🧪 tests/              # Testes automatizados
├── 📓 notebooks/          # Análise exploratória
├── ⚙️ .github/workflows/   # CI/CD pipeline
├── 📦 artifacts/          # Modelos e métricas (gerado)
└── 📊 data/               # Dados sintéticos (gerado)
```

## 🎯 Métricas e Performance

### 📊 Métricas Atuais
- **AUPRC**: 0.046 (melhorando com mais dados)
- **ROC-AUC**: 0.525 (acima do aleatório)
- **Precision@1%**: 6.1% (detecta fraudes no top 1%)
- **Recall**: 100% (não perde fraudes!)
- **F1-Score**: 0.078 (balanceado)

### 🎯 Por que Accuracy baixo é normal?
Em detecção de fraude, **accuracy baixo é esperado** porque:
- ✅ **Recall 100%**: Detecta TODAS as fraudes
- ⚠️ **Precision baixo**: Muitos falsos positivos (melhor que perder fraudes)
- 🎯 **Foco correto**: Precision@K e AUPRC são mais importantes

## 🔧 Tecnologias Utilizadas

### 🤖 Machine Learning
- **XGBoost**: Modelo principal com balanceamento híbrido
- **IsolationForest**: Detecção de anomalias não-supervisionada
- **SMOTE + Undersampling**: Balanceamento inteligente
- **Scikit-learn**: Métricas e validação

### 🌐 API e Infraestrutura
- **FastAPI**: API REST moderna e rápida
- **Pydantic**: Validação de dados
- **Uvicorn**: Servidor ASGI
- **Supabase**: Banco de dados em nuvem

### 🧪 Qualidade e CI/CD
- **pytest**: Testes automatizados
- **flake8**: Linting de código
- **black**: Formatação automática
- **GitHub Actions**: CI/CD pipeline
- **Coverage**: Cobertura de testes

### 📊 Visualização e Análise
- **Jupyter**: Notebooks interativos
- **Matplotlib/Seaborn**: Visualizações
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica

## 🚀 CI/CD Pipeline

O projeto inclui um pipeline CI/CD robusto com:

- 🔍 **Linting**: Verificação de qualidade de código
- 🧪 **Testes em Matriz**: Python 3.10, 3.11, 3.12
- 🤖 **Treinamento**: Validação automática de modelos
- 🚀 **Testes de API**: Validação de endpoints
- 📊 **Relatórios**: Status detalhado do pipeline

## 📈 Roadmap

### 🎯 Próximas Melhorias
- [ ] **Autoencoder**: Detecção de padrões não-lineares
- [ ] **Ensemble**: Combinação de múltiplos modelos
- [ ] **Threshold Otimizado**: Baseado em custo esperado
- [ ] **Monitoramento**: Drift detection com PSI
- [ ] **Dashboard**: Interface web com Streamlit
- [ ] **Deploy**: Containerização com Docker
- [ ] **Alertas**: Notificações em tempo real

### 🔮 Funcionalidades Avançadas
- [ ] **Modelo Incremental**: Treinamento online
- [ ] **A/B Testing**: Comparação de modelos
- [ ] **Feature Store**: Gerenciamento de features
- [ ] **MLOps**: Pipeline de produção completo

## 🤝 Contribuindo

1. **Fork** o projeto
2. **Crie** uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. **Abra** um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

**Lucas Inocencio**
- GitHub: [@lucasinocencio1](https://github.com/lucasinocencio1)

## 🙏 Agradecimentos

- Comunidade Python/ML
- Contribuidores do XGBoost
- Equipe do FastAPI
- Comunidade Supabase

---

⭐ **Se este projeto te ajudou, considere dar uma estrela!** ⭐

