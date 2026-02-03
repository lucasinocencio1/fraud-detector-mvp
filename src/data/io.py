from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from src.data.logging import configure_logging

logger = configure_logging(name="fraud_data.io")


def read_csv(path: Path, *, validator: Optional[Callable[[pd.DataFrame], Any]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded CSV %s with %s rows", path, len(df))
    if validator:
        df = validator(df)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote CSV %s with %s rows", path, len(df))


def read_parquet(path: Path, *, columns: Optional[list[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path, columns=columns)
    logger.info("Loaded Parquet %s with %s rows", path, len(df))
    return df


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote Parquet %s with %s rows", path, len(df))
