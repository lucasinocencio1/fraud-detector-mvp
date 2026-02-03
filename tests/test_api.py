import os

from fastapi.testclient import TestClient

from src.server.app import app


def test_health_and_readiness():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    r = client.get("/ready")
    assert r.status_code in (200, 503)


def test_predict_contract_when_ready():
    os.environ["API_KEY"] = "test"
    client = TestClient(app)

    payload = {
        "Amount": 100.0,
        "transaction_hour": 10,
        "region": "US",
        "device_type": "mobile",
        "merchant_category": "electronics",
        "tx_last_24h": 1,
        "tx_last_7d": 3,
        "amount_to_avg_ratio": 1.2,
        "is_weekend": 0,
        "avg_amount_user": 80.0,
    }

    readiness = client.get("/ready")
    if readiness.status_code == 503:
        r = client.post("/v1/predict", json=payload, headers={"X-API-Key": "test"})
        assert r.status_code == 503
        return

    r = client.post("/v1/predict", json=payload, headers={"X-API-Key": "test"})
    assert r.status_code == 200
    body = r.json()
    assert "fraud_score" in body and 0.0 <= body["fraud_score"] <= 1.0
    assert "decision" in body
    assert "threshold" in body
