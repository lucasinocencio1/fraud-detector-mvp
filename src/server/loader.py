import json
from pathlib import Path

import joblib

ART_MODEL = Path("artifacts/model.joblib")
ART_THRESHOLD = Path("artifacts/threshold.joblib")
ART_METADATA = Path("artifacts/metadata.json")

_MODEL_CACHE = None
_THRESHOLD_CACHE = None
_METADATA_CACHE = None


def load_artifacts():
    if not ART_MODEL.exists():
        raise FileNotFoundError(f"Missing model artifact: {ART_MODEL}")
    if not ART_THRESHOLD.exists():
        raise FileNotFoundError(f"Missing threshold artifact: {ART_THRESHOLD}")

    model = joblib.load(ART_MODEL)
    threshold = joblib.load(ART_THRESHOLD)

    metadata = {}
    if ART_METADATA.exists():
        with open(ART_METADATA, "r") as f:
            metadata = json.load(f)

    return model, threshold, metadata


def get_artifacts():
    global _MODEL_CACHE, _THRESHOLD_CACHE, _METADATA_CACHE
    if _MODEL_CACHE is None or _THRESHOLD_CACHE is None:
        _MODEL_CACHE, _THRESHOLD_CACHE, _METADATA_CACHE = load_artifacts()
    return _MODEL_CACHE, _THRESHOLD_CACHE, _METADATA_CACHE


def artifacts_ready():
    try:
        load_artifacts()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)
