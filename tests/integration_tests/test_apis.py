import io
import os

import requests

# Use the fixed deployed API URL
DEPLOYED_MODEL_URL = os.getenv("DEPLOYED_MODEL_URL")
if DEPLOYED_MODEL_URL is None:
    DEPLOYED_MODEL_URL = "https://mlops-catsvsdogs-122709719634.us-central1.run.app"
    print(f"DEPLOYED_MODEL_URL not set, using default: {DEPLOYED_MODEL_URL}")
else:
    print(f"DEPLOYED_MODEL_URL: {DEPLOYED_MODEL_URL}")


def test_predict_valid_image():
    url = f"{DEPLOYED_MODEL_URL}/predict"
    with open("tests/integration_tests/cat.jpg", "rb") as img_file:
        response = requests.post(
            url,
            headers={"accept": "application/json"},
            files={"data": ("cat.jpg", img_file, "image/jpeg")},
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print("Valid image test passed.")


def test_predict_invalid_file():
    url = f"{DEPLOYED_MODEL_URL}/predict"
    invalid_file = io.BytesIO(b"This is not an image")
    response = requests.post(
        url,
        headers={"accept": "application/json"},
        files={"data": ("invalid.txt", invalid_file, "text/plain")},
    )
    assert response.status_code in [400, 422, 500], f"Expected 400, 422, or 500, got {response.status_code}"
    print("Invalid file test passed.")


def test_predict_no_file():
    url = f"{DEPLOYED_MODEL_URL}/predict"
    response = requests.post(url, headers={"accept": "application/json"})
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("No file test passed.")
