from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.config import get_settings
from src.data.io import read_csv, write_csv
from src.data.logging import configure_logging
from src.data.validators import RAW_TRANSACTIONS_SCHEMA, validate

logger = configure_logging(name="fraud_data.ingest")


def _ensure_transaction_id(df: pd.DataFrame, *, timestamp_column: str) -> pd.DataFrame:
    df = df.copy()
    if "transaction_id" not in df.columns:
        df["transaction_id"] = (
            df[timestamp_column].astype(str)
            + "-"
            + df.reset_index().index.astype(str)
        )
    df["transaction_id"] = df["transaction_id"].astype(str)
    return df


def load_existing_transactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.info("No existing raw transactions at %s. Starting fresh.", path)
        return pd.DataFrame(columns=RAW_TRANSACTIONS_SCHEMA.columns)
    settings = get_settings()
    existing = read_csv(path)
    existing = _ensure_transaction_id(existing, timestamp_column=settings.timestamp_column)
    return validate(RAW_TRANSACTIONS_SCHEMA, existing)


def append_batch(batch: pd.DataFrame, *, destination: Optional[Path] = None) -> pd.DataFrame:
    settings = get_settings()
    dest = destination or Path(settings.raw_data_dir) / "transactions.csv"

    prepared_batch = _ensure_transaction_id(batch, timestamp_column=settings.timestamp_column)
    validated_batch = validate(RAW_TRANSACTIONS_SCHEMA, prepared_batch)
    existing = load_existing_transactions(dest)

    combined = pd.concat([existing, validated_batch], ignore_index=True)
    combined = combined.drop_duplicates(subset=["transaction_id"]).sort_values(settings.timestamp_column)

    write_csv(combined, dest)
    logger.info("Ingested batch with %s records. Total records: %s", len(validated_batch), len(combined))
    return combined
