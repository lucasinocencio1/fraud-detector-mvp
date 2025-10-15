from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import subprocess
from src.serve.loader import load_supervised

app = FastAPI(title="Fraud Detector Pro")
model, FEATURES = load_supervised()
 
class Tx(BaseModel):
    # numéricas básicas
    Amount: float
    transaction_hour: int = Field(ge=0, le=23)

    # latentes (padrão 0)
    V1: float=0; V2: float=0; V3: float=0; V4: float=0; V5: float=0; V6: float=0; V7: float=0
    V8: float=0; V9: float=0; V10: float=0; V11: float=0; V12: float=0; V13: float=0; V14: float=0
    V15: float=0; V16: float=0; V17: float=0; V18: float=0; V19: float=0; V20: float=0
    V21: float=0; V22: float=0; V23: float=0; V24: float=0; V25: float=0; V26: float=0; V27: float=0; V28: float=0

    # categóricas
    region: str
    device_type: str
    merchant_category: str

    # features derivadas opcionais (se vierem 0, o pipeline cuida internamente)
    Amount_log1p: float = 0.0
    Amount_z: float = 0.0
    Amount_log1p_z: float = 0.0
    hour_is_night: int = 0
    region_amount_mean: float = 0.0
    region_amount_std: float = 0.0
    mc_amount_mean: float = 0.0
    mc_amount_std: float = 0.0

@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURES)}

@app.post("/predict")
def predict(tx: Tx):
    # Constrói o vetor na ordem de FEATURES
    vals = [getattr(tx, f) for f in FEATURES]
    X = np.array([vals], dtype=float) if isinstance(vals[0], (int, float)) else None
    # Para o Pipeline com OneHot + num, basta passar DataFrame dict-like
    # Construiremos input como dicionário:
    row = {f: getattr(tx, f) for f in FEATURES}
    import pandas as pd
    Xdf = pd.DataFrame([row])
    score = float(model.predict_proba(Xdf)[0, 1])
    return {"fraud_score": score}

@app.post("/retrain")
def retrain():
    """
    Dispara atualização e retreino:
    - update_data -> make_dataset -> feature_build -> train_supervised
    """
    cmds = [
        ["python3", "src/data/update_data.py"],
        ["python3", "src/data/make_dataset.py"],
        ["python3", "src/data/feature_build.py"],
        ["python3", "src/models/train_supervised.py"],
    ]
    out = []
    for c in cmds:
        proc = subprocess.run(c, capture_output=True, text=True)
        out.append({"cmd": " ".join(c), "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
        if proc.returncode != 0:
            return {"status": "failed", "log": out}
    # recarrega modelo em memória
    global model, FEATURES
    model, FEATURES = load_supervised()
    return {"status": "ok", "log": out}
