
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from src.serve.loader import load_supervised

app = FastAPI(title="Fraud Detector MVP")
model, FEATURES = load_supervised()

class Tx(BaseModel):
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount_log1p: float = 0.0
    Amount_z: float = 0.0
    Amount_log1p_z: float = 0.0

@app.get("/health")
def health():
    return {"status": "ok", "features": FEATURES[:5] + ["..."]}

@app.post("/predict")
def predict(tx: Tx):
    x = np.array([[getattr(tx, f) for f in FEATURES]], dtype=float)
    score = float(model.predict_proba(x)[0,1])
    return {"fraud_score": score}
