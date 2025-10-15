"""
src/data/feature_build.py

Gera features agregadas (estatísticas por região e categoria),
mantendo compatibilidade com o novo dataset sintético.
"""

import pandas as pd
from pathlib import Path

# Caminhos
DATA_DIR = Path("artifacts")

# Carregar splits
train = pd.read_parquet(DATA_DIR / "train.parquet")
val = pd.read_parquet(DATA_DIR / "val.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

# Garantir colunas numéricas corretas
if "Amount" in train.columns:
    train.rename(columns={"Amount": "amount"}, inplace=True)
    val.rename(columns={"Amount": "amount"}, inplace=True)
    test.rename(columns={"Amount": "amount"}, inplace=True)

# Features agregadas por região e categoria
region_stats = train.groupby("region")["amount"].agg(["mean", "std"]).reset_index()
region_stats.columns = ["region", "region_amount_mean", "region_amount_std"]

cat_stats = train.groupby("merchant_category")["amount"].agg(["mean", "std"]).reset_index()
cat_stats.columns = ["merchant_category", "cat_amount_mean", "cat_amount_std"]

# Merge nos datasets
for df in [train, val, test]:
    df.merge(region_stats, on="region", how="left")
    df.merge(cat_stats, on="merchant_category", how="left")

# Salvar resultados
train.to_parquet("data/train_feat.parquet", index=False)
val.to_parquet("data/val_feat.parquet", index=False)
test.to_parquet("data/test_feat.parquet", index=False)

print("✅ Features geradas e salvas: train_feat, val_feat, test_feat")
