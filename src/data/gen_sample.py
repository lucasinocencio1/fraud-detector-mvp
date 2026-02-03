"""
Generate synthetic transaction data for model training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import get_settings
from src.data.io import write_csv

RNG = np.random.default_rng(42)
REGIONS = ["EU", "US", "BR", "ASIA"]
DEVICE_TYPES = ["mobile", "desktop"]
MERCHANT_CATS = [
    "electronics",
    "groceries",
    "fashion",
    "gaming",
    "travel",
    "restaurants",
]


def gen_transactions(
    n: int = 250_000,
    fraud_rate: float = 0.07,
    start_time: int = 0,
) -> pd.DataFrame:
    time = np.arange(start_time, start_time + n)
    region = RNG.choice(REGIONS, size=n, p=[0.45, 0.25, 0.15, 0.15])
    device_type = RNG.choice(DEVICE_TYPES, size=n, p=[0.65, 0.35])
    merchant_category = RNG.choice(MERCHANT_CATS, size=n)
    transaction_hour = RNG.integers(0, 24, size=n)
    is_night = ((transaction_hour >= 0) & (transaction_hour <= 5)).astype(int)
    is_weekend = RNG.choice([0, 1], size=n, p=[0.7, 0.3])

    amount = np.clip(RNG.gamma(shape=2.2, scale=70.0, size=n), 0.01, 10_000)
    avg_amount_user = np.clip(RNG.gamma(shape=2.0, scale=80.0, size=n), 1, 5_000)
    amount_to_avg_ratio = amount / (avg_amount_user + 1e-6)
    tx_last_24h = np.clip(RNG.poisson(2, n), 0, 50)
    tx_last_7d = np.clip(tx_last_24h + RNG.poisson(4, n), 0, 100)

    base = np.full(n, fraud_rate, dtype=float)
    base += 0.00008 * np.clip(amount - 200, 0, None)
    base += 0.015 * is_night
    base += 0.01 * (device_type == "mobile")
    base += 0.02 * np.isin(merchant_category, ["electronics", "travel", "gaming"])
    base += 0.01 * np.isin(region, ["US", "ASIA"])
    base += 0.01 * (tx_last_24h > 10)
    base += 0.02 * (amount_to_avg_ratio > 2.5)
    base = np.clip(base, 0, 0.8)

    is_fraud = (RNG.random(n) < base).astype(int)
    transaction_id = [f"tx-{i}" for i in range(start_time, start_time + n)]

    df = pd.DataFrame(
        {
            "transaction_id": transaction_id,
            "time": time,
            "amount": amount,
            "region": region,
            "device_type": device_type,
            "merchant_category": merchant_category,
            "transaction_hour": transaction_hour,
            "is_weekend": is_weekend,
            "avg_amount_user": avg_amount_user,
            "amount_to_avg_ratio": amount_to_avg_ratio,
            "tx_last_24h": tx_last_24h,
            "tx_last_7d": tx_last_7d,
            "class": is_fraud,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic transaction dataset")
    parser.add_argument(
        "-n",
        "--rows",
        type=int,
        default=250_000,
        help="Number of rows to generate (default: 250000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/transactions.csv)",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.07,
        help="Base fraud rate (default: 0.07)",
    )
    args = parser.parse_args()

    settings = get_settings()
    out_path = Path(args.output) if args.output else Path(settings.raw_data_dir) / "transactions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = gen_transactions(n=args.rows, fraud_rate=args.fraud_rate)
    write_csv(df, out_path)
    print(f"Generated {len(df)} rows, fraud rate {df['class'].mean():.2%}")


if __name__ == "__main__":
    main()
