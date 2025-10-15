from fastapi.testclient import TestClient

from src.serve.api import app


def test_predict_contract():
    client = TestClient(app)
    payload = {"Amount": 100.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "fraud_score" in body and 0.0 <= body["fraud_score"] <= 1.0
