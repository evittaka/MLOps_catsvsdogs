import io

from fastapi.testclient import TestClient

from catsvsdogs.api import app

client = TestClient(app)


def test_model_loading():
    """
    Test the model loading process.
    """
    try:
        from catsvsdogs.api import load_model

        model = load_model()
        assert model is not None, "Model loading failed"
        print("Model loaded successfully.")
    except RuntimeError as e:
        raise AssertionError(f"Model loading test failed: {e}") from e


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
    invalid_file = io.BytesIO(b"This is not an image")
    response = client.post(
        "/predict",
        headers={"accept": "application/json"},
        files={"data": ("invalid.txt", invalid_file, "text/plain")},
    )

    assert response.status_code in [200, 500], f"Expected 200 or 500, got {response.status_code}"

    response_json = response.json()

    if response.status_code == 200:
        assert "error" in response_json, "Response must include 'error'"
        assert "cannot identify the uploaded file as a valid image" in response_json["error"].lower(), (
            "Error message should indicate invalid image file"
        )
        print("Test passed: Invalid file handled with error message.")
    elif response.status_code == 500:
        print("Test passed: Server returned 500 for invalid file.")
    else:
        raise AssertionError("Unexpected response for invalid file")


def test_predict_no_file():
    """
    Test the /predict endpoint with no file provided.
    """
    response = client.post(
        "/predict",
        headers={"accept": "application/json"},
    )

    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    response_json = response.json()
    assert "detail" in response_json, "Response must include 'detail'"
    assert response_json["detail"][0]["msg"].lower() == "field required", "Error message mismatch for missing file"
    print("No file test passed. Properly handled missing file case.")
