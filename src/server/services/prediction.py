from typing import Any, Dict

import pandas as pd

from src.models.features import ensure_base_features
from src.server.api.schemas import PredictionOut, TransactionIn
from src.server.loader import get_artifacts


class PredictionServiceError(RuntimeError):
    """Raised when the prediction service cannot fulfill a request."""


def _prepare_payload(tx: TransactionIn) -> pd.DataFrame:
    payload = tx.model_dump()
    df = pd.DataFrame([payload])
    return ensure_base_features(df)


def _build_response(score: float, threshold: float, metadata: Dict[str, Any]) -> PredictionOut:
    decision = "fraud" if score >= threshold else "not_fraud"
    model_version = metadata.get("model_version") if isinstance(metadata, dict) else None
    return PredictionOut(
        fraud_score=score,
        decision=decision,
        threshold=threshold,
        model_version=model_version,
    )


def predict_transaction(tx: TransactionIn) -> PredictionOut:
    try:
        model, threshold, metadata = get_artifacts()
    except Exception as exc:  # noqa: BLE001
        raise PredictionServiceError(str(exc)) from exc

    df = _prepare_payload(tx)
    score = float(model.predict_proba(df)[0, 1])
    return _build_response(score, float(threshold), metadata or {})
