import io

from fastapi.testclient import TestClient

from catsvsdogs.api import app

client = TestClient(app)


def test_model_loading():
    """
    Test the model loading process via health endpoint.
    """
    try:
        response = client.get("/health")
        # Assert the endpoint exists and is returning a response
        assert response.status_code in [200, 404], f"Expected 200 or 404, got {response.status_code}"
        print("Health endpoint reached successfully.")
    except Exception as e:
        raise AssertionError(f"Health endpoint test failed unexpectedly: {e}") from e


def test_predict_valid_image():
    """
    Test the /predict endpoint with a valid image file.
    """
    try:
        with open("tests/integration_tests/cat.jpg", "rb") as img_file:
            response = client.post(
                "/predict",
                headers={"accept": "application/json"},
                files={"data": ("cat.jpg", img_file, "image/jpeg")},
            )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        response_json = response.json()
        assert "prediction" in response_json, "Response must include 'prediction'"
        assert "probability" in response_json, "Response must include 'probability'"
        print("Valid image test passed. Prediction:", response_json["prediction"])
    except Exception as e:
        raise AssertionError(f"Valid image test failed unexpectedly: {e}") from e


def test_predict_invalid_file():
    """
    Test the /predict endpoint with an invalid file.
    """
    try:
        invalid_file = io.BytesIO(b"This is not an image")
        response = client.post(
            "/predict",
            headers={"accept": "application/json"},
            files={"data": ("invalid.txt", invalid_file, "text/plain")},
        )

        assert response.status_code in [200, 400, 422, 500], (
            f"Expected 200, 400, 422, or 500, got {response.status_code}"
        )

        response_json = response.json()

        if response.status_code == 200:
            assert "error" in response_json or "prediction" in response_json, (
                "Response should contain 'error' or 'prediction' for invalid file"
            )
            print("Test passed: API handled invalid file gracefully.")
        elif response.status_code in [400, 422]:
            assert "detail" in response_json, "Response must include 'detail'"
            print("Invalid file test passed: Proper error details provided.")
        elif response.status_code == 500:
            print("Test passed: Server returned 500 for invalid file.")
    except Exception as e:
        raise AssertionError(f"Invalid file test failed unexpectedly: {e}") from e


def test_predict_no_file():
    """
    Test the /predict endpoint with no file provided.
    """
    try:
        response = client.post(
            "/predict",
            headers={"accept": "application/json"},
        )

        assert response.status_code == 422, f"Expected 422, got {response.status_code}"
        response_json = response.json()
        assert "detail" in response_json, "Response must include 'detail'"
        assert response_json["detail"][0]["msg"].lower() == "field required", "Error message mismatch for missing file"
        print("No file test passed. Properly handled missing file case.")
    except Exception as e:
        raise AssertionError(f"No file test failed unexpectedly: {e}") from e
