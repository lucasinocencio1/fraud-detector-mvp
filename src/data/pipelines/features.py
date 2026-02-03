from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.data.config import get_settings
from src.data.io import read_parquet, write_parquet
from src.data.logging import configure_logging
from src.data.validators import ENRICHED_TRANSACTIONS_SCHEMA, validate

logger = configure_logging(name="fraud_data.features")

BASE_FEATURE_COLUMNS = [
    "Amount",
    "transaction_hour",
    "region",
    "device_type",
    "merchant_category",
    "is_weekend",
    "avg_amount_user",
    "amount_to_avg_ratio",
    "tx_last_24h",
    "tx_last_7d",
]
BASE_CATEGORICAL = {"region", "device_type", "merchant_category"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "amount": "Amount",
        "class": "Class",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def ensure_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in BASE_FEATURE_COLUMNS:
        if column not in df.columns:
            if column in BASE_CATEGORICAL:
                df[column] = "unknown"
            else:
                df[column] = 0.0
    return df


def add_feature_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    if "region" in df.columns and "Amount" in df.columns:
        region_stats = df.groupby("region")["Amount"].agg(region_amount_mean="mean", region_amount_std="std").reset_index()
        df = df.merge(region_stats, on="region", how="left")
    if "merchant_category" in df.columns and "Amount" in df.columns:
        mc_stats = df.groupby("merchant_category")["Amount"].agg(mc_amount_mean="mean", mc_amount_std="std").reset_index()
        df = df.merge(mc_stats, on="merchant_category", how="left")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_base_features(df)
    df = add_feature_aggregates(df)
    df = validate(ENRICHED_TRANSACTIONS_SCHEMA, df)
    return df


def run_feature_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = get_settings()
    artifacts = Path(settings.artifacts_dir)

    train = read_parquet(artifacts / "train.parquet")
    val = read_parquet(artifacts / "val.parquet")
    test = read_parquet(artifacts / "test.parquet")

    train_feat = build_features(train)
    val_feat = build_features(val)
    test_feat = build_features(test)

    write_parquet(train_feat, artifacts / "train_feat.parquet")
    write_parquet(val_feat, artifacts / "val_feat.parquet")
    write_parquet(test_feat, artifacts / "test_feat.parquet")

    logger.info(
        "Feature artifacts saved: train=%s, val=%s, test=%s",
        len(train_feat),
        len(val_feat),
        len(test_feat),
    )

    return train_feat, val_feat, test_feat
