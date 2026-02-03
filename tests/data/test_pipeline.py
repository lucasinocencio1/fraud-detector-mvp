import os
from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

from src.data.config import get_settings
from src.data.pipelines.features import run_feature_pipeline
from src.data.pipelines.ingest import append_batch
from src.data.pipelines.split import run_split


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    artifacts_dir = tmp_path / "artifacts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DATA_RAW_DATA_DIR", str(raw_dir))
    monkeypatch.setenv("DATA_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("DATA_TRAIN_RATIO", "0.6")
    monkeypatch.setenv("DATA_VAL_RATIO", "0.2")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def sample_batch() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"time": 0, "amount": 120.5, "region": "US", "device_type": "mobile", "merchant_category": "electronics", "transaction_hour": 10, "is_weekend": 0, "avg_amount_user": 80.0, "amount_to_avg_ratio": 1.5, "tx_last_24h": 2, "tx_last_7d": 5, "class": 1},
            {"time": 1, "amount": 45.0, "region": "EU", "device_type": "desktop", "merchant_category": "groceries", "transaction_hour": 14, "is_weekend": 0, "avg_amount_user": 55.0, "amount_to_avg_ratio": 0.82, "tx_last_24h": 1, "tx_last_7d": 4, "class": 0},
            {"time": 2, "amount": 310.75, "region": "US", "device_type": "mobile", "merchant_category": "travel", "transaction_hour": 1, "is_weekend": 0, "avg_amount_user": 95.0, "amount_to_avg_ratio": 3.27, "tx_last_24h": 5, "tx_last_7d": 9, "class": 1},
            {"time": 3, "amount": 25.1, "region": "BR", "device_type": "mobile", "merchant_category": "restaurants", "transaction_hour": 20, "is_weekend": 1, "avg_amount_user": 30.0, "amount_to_avg_ratio": 0.84, "tx_last_24h": 0, "tx_last_7d": 2, "class": 0},
        ]
    )


def test_append_batch_generates_transaction_ids(tmp_path):
    df = sample_batch()
    combined = append_batch(df)

    transactions_path = Path(get_settings().raw_data_dir) / "transactions.csv"
    assert transactions_path.exists()
    assert "transaction_id" in combined.columns
    assert combined["transaction_id"].is_unique


def test_split_and_feature_pipeline(tmp_path):
    append_batch(sample_batch())
    run_split()
    train_feat, val_feat, test_feat = run_feature_pipeline()

    artifacts_dir = Path(get_settings().artifacts_dir)
    assert (artifacts_dir / "train.parquet").exists()
    assert (artifacts_dir / "train_feat.parquet").exists()

    for df in (train_feat, val_feat, test_feat):
        assert {"region_amount_mean", "region_amount_std", "mc_amount_mean", "mc_amount_std"}.issubset(df.columns)


def test_append_batch_requires_schema(tmp_path):
    invalid = sample_batch().drop(columns=["amount"])
    with pytest.raises(pa.errors.SchemaErrors):
        append_batch(invalid)
