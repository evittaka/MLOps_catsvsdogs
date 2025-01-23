from fastapi.testclient import TestClient

from catsvsdogs.api import app

client = TestClient(app)


def test_predict_endpoint():
    with client as c:
        with open("tests/integration_tests/cat.jpg", "rb") as f:
            files = {"data": ("cat.jpg", f, "image/jpeg")}
            response = c.post("/predict", files=files)
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert isinstance(data["probability"], list)
            assert len(data["probability"]) == 2
            assert data["prediction"] in ["cat", "dog"]


def test_invalid_file():
    with client as c:
        files = {"data": ("test.txt", b"invalid data", "text/plain")}
        response = c.post("/predict", files=files)
        assert "error" in response.json()


def test_missing_file():
    with client as c:
        response = c.post("/predict")
        assert response.status_code == 422
