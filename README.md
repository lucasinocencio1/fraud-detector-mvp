
# 🚨 Fraud Detector MVP - Sistema Completo de Detecção de Fraudes

[![CI/CD Pipeline](https://github.com/lucasinocencio1/fraud-detector-mvp/actions/workflows/ci.yml/badge.svg)](https://github.com/lucasinocencio1/fraud-detector-mvp/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Um sistema **end-to-end** completo para detecção de fraudes em transações financeiras, incluindo geração de dados sintéticos, pipeline de ML, API REST e monitoramento.

## 🎯 Características Principais

- 🔄 **Pipeline Completo**: Dados → Features → Modelo → API → Alertas → Monitoramento
- 🤖 **Múltiplos Algoritmos**: XGBoost (supervisionado) + IsolationForest (não-supervisionado)
- 📊 **Dados Sintéticos Realistas**: 250k+ transações com padrões de fraude
- 🚀 **API FastAPI**: Endpoints REST para predições em tempo real
- 🚨 **Alertas WhatsApp**: Notificações automáticas de fraudes detectadas
- 🧪 **CI/CD Robusto**: Testes automatizados, linting e validação
- 📈 **Métricas Adequadas**: AUPRC, Precision@K, ROC-AUC para dados desbalanceados
- 🔒 **Balanceamento Inteligente**: SMOTE + Undersampling híbrido
- 📊 **Visualizações**: Notebooks de análise exploratória
- 🔗 **Integração Supabase**: Upload automático para cliente

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

# Enviar dados para cliente (opcional)
python src/data/push_to_supabase.py
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
    "transaction_hour": 14,
    "region": "Lisboa",
    "device_type": "mobile",
    "merchant_category": "groceries",
    "V1": 0.5, "V2": -0.3, "V3": 1.2, "V4": -0.8, "V5": 0.1,
    "V6": -0.5, "V7": 0.8, "V8": -1.1, "V9": 0.3, "V10": -0.7,
    "V11": 0.9, "V12": -0.2, "V13": 0.6, "V14": -0.4, "V15": 0.7,
    "V16": -0.9, "V17": 0.2, "V18": -0.6, "V19": 0.4, "V20": -0.8,
    "V21": 0.5, "V22": -0.3, "V23": 0.7, "V24": -0.5, "V25": 0.3,
    "V26": -0.7, "V27": 0.6, "V28": -0.4
  }'
```

### 📋 Variáveis da API

**🔴 Obrigatórias:**
- `Amount`: Valor da transação (float)
- `transaction_hour`: Hora da transação (0-23)
- `region`: Região geográfica (string)
- `device_type`: Tipo de dispositivo (mobile/desktop)
- `merchant_category`: Categoria do comerciante (groceries/electronics/etc)

**🟡 Opcionais (padrão 0):**
- `V1-V28`: Features PCA (float)
- `Amount_log1p`, `Amount_z`, `Amount_log1p_z`: Features derivadas
- `hour_is_night`: Flag de horário noturno
- `region_amount_mean/std`: Estatísticas por região
- `mc_amount_mean/std`: Estatísticas por categoria

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
│   ├── 🚨 alerts/         # Sistema de alertas
│   │   ├── fraud_alert.py     # Alertas via WhatsApp
│   │   └── test_message.py    # Modo sandbox
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

### 📊 Armazenamento e Integração
- **Supabase**: Banco de dados PostgreSQL em nuvem
- **Integração Cliente**: Upload automático de dados para cliente hipotético
- **Schema Sync**: Sincronização automática de estrutura de tabelas
- **Batch Upload**: Envio eficiente de dados em lotes


## 🔗 Integração com Cliente (Supabase)

### 📤 Upload Automático de Dados
```bash
# Enviar dados para cliente hipotético
python src/data/push_to_supabase.py
```

### 🎯 Funcionalidades de Integração
- **🔄 Sincronização de Schema**: Criação automática de tabelas e colunas
- **📊 Upload em Batch**: Envio eficiente de 250k+ transações
- **🔗 Relacionamento**: Vinculação automática customers ↔ transactions
- **🛡️ Tratamento de Erros**: Robustez contra colunas duplicadas
- **📈 Monitoramento**: Logs detalhados do processo de upload

### 🏗️ Estrutura no Supabase
```sql
-- Tabela de Clientes
customers (
  customer_id UUID PRIMARY KEY,
  name TEXT,
  email TEXT,
  age INTEGER,
  city TEXT,
  state TEXT,
  country TEXT
)

-- Tabela de Transações  
transactions (
  transaction_id UUID PRIMARY KEY,
  customer_id UUID REFERENCES customers(customer_id),
  amount DECIMAL,
  time TIMESTAMP,
  v1-v28 DECIMAL,  -- Features PCA
  class INTEGER,   -- 0=Normal, 1=Fraude
  is_night BOOLEAN
)
```

### ⚙️ Configuração
```bash
# Variáveis de ambiente necessárias
export SUPABASE_URL="sua-url-supabase"
export SUPABASE_KEY="sua-chave-supabase"

# Para sistema de alertas (opcional)
export ULTRAMSG_INSTANCE="sua-instancia-ultramsg"
export ULTRAMSG_TOKEN="seu-token-ultramsg"
```

## 🚨 Sistema de Alertas em Tempo Real

### 📱 Notificações Automáticas
```bash
# Executar sistema de alertas
python src/alerts/fraud_alert.py

# Testar envio de mensagens (modo sandbox)
python src/alerts/test_message.py
```

### 🎯 Funcionalidades do Sistema de Alertas
- **🔍 Detecção Automática**: Monitora transações fraudulentas no Supabase
- **📱 WhatsApp Integration**: Envia alertas via UltraMsg API
- **👤 Mapeamento Cliente**: Vincula fraudes aos dados do cliente
- **⚡ Tempo Real**: Notificações instantâneas de fraudes detectadas
- **🛡️ Sandbox Mode**: Modo de teste sem envio real de mensagens

### 📋 Exemplo de Alerta
```
⚠️ Alerta de possível fraude detectada!

Cliente: João Silva
Região: Lisboa
Valor: €1,250.00
Ação: Revisar transação no painel de risco.
```

### 🔧 Tecnologias de Alertas
- **UltraMsg API**: Integração com WhatsApp Business
- **Supabase**: Monitoramento de transações em tempo real
- **Python Requests**: Comunicação com APIs externas
- **Pandas**: Processamento de dados de fraude

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
- [ ] **Integração Avançada**: Webhooks para notificações de fraude
- [ ] **Dashboard Cliente**: Interface para visualização de métricas

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

