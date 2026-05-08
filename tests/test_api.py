from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_predict():
    sample_input = {
        "trip_distance": 5.0,
        "passenger_count": 2,
        "fare_amount": 10.0,
        "tip_amount": 2.0,
        "tolls_amount": 0.0,
        "VendorID": 1,
        "payment_type": 1
    }

    response = client.post(
        "/predict",
        json=sample_input
    )

    assert response.status_code == 200
    assert "fare_amount" in response.json()