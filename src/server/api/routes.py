import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.server.api.dependencies import require_api_key
from src.server.api.schemas import PredictionOut, TransactionIn
from src.server.loader import artifacts_ready
from src.server.services.prediction import PredictionServiceError, predict_transaction

router = APIRouter()

logger = logging.getLogger("fraud_api")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def readiness():
    ok, msg = artifacts_ready()
    if not ok:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=msg)
    return {"status": "ready"}


def _predict(tx: TransactionIn) -> PredictionOut:
    try:
        return predict_transaction(tx)
    except PredictionServiceError as exc:
        logger.exception("Prediction service error: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        ) from exc


@router.post(
    "/v1/predict",
    response_model=PredictionOut,
    dependencies=[Depends(require_api_key)],
)
def predict_v1(tx: TransactionIn):
    return _predict(tx)


@router.post(
    "/predict",
    response_model=PredictionOut,
    dependencies=[Depends(require_api_key)],
)
def predict_legacy(tx: TransactionIn):
    return _predict(tx)
