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


def ensure_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BASE_FEATURES:
        if col not in df.columns:
            if col in CATEGORICAL_FEATURES:
                df[col] = "unknown"
            else:
                df[col] = 0.0
    return df[BASE_FEATURES]
