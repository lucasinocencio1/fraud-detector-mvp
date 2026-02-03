from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    Amount: float = Field(ge=0)
    transaction_hour: int = Field(ge=0, le=23)
    region: str
    device_type: str
    merchant_category: str
    is_weekend: int = 0
    avg_amount_user: float = 0.0
    amount_to_avg_ratio: float = 0.0
    tx_last_24h: int = 0
    tx_last_7d: int = 0


class PredictionOut(BaseModel):
    fraud_score: float
    decision: str
    threshold: float
    model_version: Optional[str] = None
