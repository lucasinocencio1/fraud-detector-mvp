import os

import numpy as np
import pandas as pd

os.makedirs("artifacts", exist_ok=True)

splits = ["train", "val", "test"]

for split in splits:
    df = pd.read_parquet(f"artifacts/{split}.parquet")

    # Feature: horário noturno
    df["hour_is_night"] = df["transaction_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    # Estatísticas por região
    region_stats = (
        df.groupby("region")["Amount"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "region_amount_mean", "std": "region_amount_std"})
        .reset_index()
    )
    df = df.merge(region_stats, on="region", how="left")

    # Estatísticas por categoria de comerciante
    mc_stats = (
        df.groupby("merchant_category")["Amount"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mc_amount_mean", "std": "mc_amount_std"})
        .reset_index()
    )
    df = df.merge(mc_stats, on="merchant_category", how="left")

    # Features derivadas de Amount
    df["Amount_log1p"] = np.log1p(df["Amount"])
    df["Amount_z"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-9)
    df["Amount_log1p_z"] = (df["Amount_log1p"] - df["Amount_log1p"].mean()) / (
        df["Amount_log1p"].std() + 1e-9
    )

    # Preencher NaN e salvar
    df.fillna(0, inplace=True)
    df.to_parquet(f"artifacts/{split}_feat.parquet", index=False)

print("Features geradas e salvas: train_feat, val_feat, test_feat")
