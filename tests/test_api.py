from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_query():
    response = client.post(
        "/query",
        json={
            "query": "I have fever and headache"
        }
    )

    assert response.status_code == 200