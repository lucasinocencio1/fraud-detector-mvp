from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.data.config import get_settings
from src.data.io import read_csv, write_parquet
from src.data.logging import configure_logging
from src.data.pipelines.ingest import _ensure_transaction_id
from src.data.validators import RAW_TRANSACTIONS_SCHEMA, validate

logger = configure_logging(name="fraud_data.split")


def time_split(
    df: pd.DataFrame, *, train_ratio: float, val_ratio: float, timestamp_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    df = df.sort_values(timestamp_column).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def run_split() -> None:
    settings = get_settings()
    raw_path = Path(settings.raw_data_dir) / "transactions.csv"
    df = read_csv(raw_path)
    df = _ensure_transaction_id(df, timestamp_column=settings.timestamp_column)
    df = validate(RAW_TRANSACTIONS_SCHEMA, df)

    train, val, test = time_split(
        df,
        train_ratio=settings.train_ratio,
        val_ratio=settings.val_ratio,
        timestamp_column=settings.timestamp_column,
    )

    write_parquet(train, Path(settings.artifacts_dir) / "train.parquet")
    write_parquet(val, Path(settings.artifacts_dir) / "val.parquet")
    write_parquet(test, Path(settings.artifacts_dir) / "test.parquet")

    logger.info("Split data into train=%s, val=%s, test=%s", len(train), len(val), len(test))
