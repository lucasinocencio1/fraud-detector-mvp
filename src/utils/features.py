import numpy as np
import pandas as pd

BASE_FEATURES = [
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

CATEGORICAL_FEATURES = ["region", "device_type", "merchant_category"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "amount" in df.columns and "Amount" not in df.columns:
        df = df.rename(columns={"amount": "Amount"})
    if "class" in df.columns and "Class" not in df.columns:
        df = df.rename(columns={"class": "Class"})
    return df


def ensure_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BASE_FEATURES:
        if col not in df.columns:
            if col in CATEGORICAL_FEATURES:
                df[col] = "unknown"
            else:
                df[col] = 0.0
    return df[BASE_FEATURES]


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Amount" in df.columns:
        df["amount_log"] = np.log1p(df["Amount"])
        df["amount_sqrt"] = np.sqrt(df["Amount"])
        df["high_amount_flag"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)

    if "merchant_category" in df.columns:
        risky_categories = ["electronics", "travel", "gaming"]
        df["risky_category"] = df["merchant_category"].isin(risky_categories).astype(int)

    if "transaction_hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)
        df["hour_is_night"] = (
            (df["transaction_hour"] >= 0) & (df["transaction_hour"] <= 5)
        ).astype(int)

    if "region" in df.columns:
        risk_map = {"US": 3, "ASIA": 3, "EU": 2, "BR": 1}
        df["region_risk"] = df["region"].map(risk_map).fillna(1)

    if "Amount" in df.columns:
        for i in [1, 2, 3]:
            v_col = f"V{i}"
            if v_col in df.columns:
                df[f"amount_v{i}"] = df["Amount"] * df[v_col]

    v_cols = [c for c in df.columns if c.startswith("V") and len(c) <= 3]
    if len(v_cols) > 0:
        df["v_sum"] = df[v_cols].sum(axis=1)
        df["v_mean"] = df[v_cols].mean(axis=1)
        df["v_std"] = df[v_cols].std(axis=1)
        df["v_max"] = df[v_cols].max(axis=1)
        df["v_min"] = df[v_cols].min(axis=1)

    if "tx_last_24h" in df.columns:
        df["high_frequency"] = (df["tx_last_24h"] > 10).astype(int)
        if "tx_last_7d" in df.columns:
            df["tx_rate"] = df["tx_last_24h"] / (df["tx_last_7d"] + 1)

    if "amount_to_avg_ratio" in df.columns:
        df["unusual_amount"] = (df["amount_to_avg_ratio"] > 2.5).astype(int)

    return df
