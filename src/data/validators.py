from typing import Any

from pandera.api.pandas.components import Column
from pandera.api.pandas.container import DataFrameSchema

RAW_TRANSACTIONS_SCHEMA = DataFrameSchema(
    {
        "transaction_id": Column(str, nullable=False),
        "time": Column(int, nullable=False),
        "amount": Column(float, nullable=False),
        "region": Column(str, nullable=False),
        "device_type": Column(str, nullable=False),
        "merchant_category": Column(str, nullable=False),
        "transaction_hour": Column(int, nullable=False),
        "is_weekend": Column(int, nullable=False),
        "avg_amount_user": Column(float, nullable=False),
        "amount_to_avg_ratio": Column(float, nullable=False),
        "tx_last_24h": Column(int, nullable=False),
        "tx_last_7d": Column(int, nullable=False),
        "class": Column(int, nullable=False),
    },
    coerce=True,
    strict=False,
)


ENRICHED_TRANSACTIONS_SCHEMA = DataFrameSchema(
    {
        "Amount": Column(float, nullable=False),
        "transaction_hour": Column(int, nullable=False),
        "region": Column(str, nullable=False),
        "device_type": Column(str, nullable=False),
        "merchant_category": Column(str, nullable=False),
        "is_weekend": Column(int, nullable=False),
        "avg_amount_user": Column(float, nullable=False),
        "amount_to_avg_ratio": Column(float, nullable=False),
        "tx_last_24h": Column(int, nullable=False),
        "tx_last_7d": Column(int, nullable=False),
        "Class": Column(int, nullable=True),
    },
    coerce=True,
)


def validate(schema: DataFrameSchema, data: Any) -> Any:
    return schema.validate(data, lazy=True)
