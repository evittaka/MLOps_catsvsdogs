import io
import os

import timm
import torch
from fastapi import FastAPI, File, UploadFile
from google.cloud import storage
from PIL import Image
from torchvision import transforms

app = FastAPI()

# Configuration
MODEL_BUCKET_NAME = "mlops_catsvsdogs"
MODEL_FILE = "models/model_latest.pth"  # This will always be the latest model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize GCP storage client
client = storage.Client()

# Define the image transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Load the model from the cloud storage
def load_model():
    try:
        bucket = client.get_bucket(MODEL_BUCKET_NAME)
        blob = bucket.blob(MODEL_FILE)
        model_data = blob.download_as_string()

        # Load the model (MobileNetV3 with modified classifier)
        model = timm.create_model("mobilenetv3_large_100", pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)  # Modify for 2 classes (cat, dog)

        # Load the saved weights into the model
        state_dict = torch.load(io.BytesIO(model_data), map_location=DEVICE)
        # Remove the 'model.' prefix from the keys in the state_dict
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        # Now load the modified state_dict
        model.load_state_dict(new_state_dict, strict=True)

        model.to(DEVICE)
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}") from e


# Load the model once when the Cloud Function is initialized
try:
    model = load_model()
except RuntimeError as e:
    raise RuntimeError(f"Failed to load model: {str(e)}") from e


@app.post(
    "/predict",
    summary="Predict whether the image is of a cat or a dog",
    description="""
    Upload an image file, and the model will classify it as either a "cat" or a "dog".

    ### Request:
    - **data**: An image file (e.g., JPG, PNG).

    ### Response:
    - **prediction**: The class label ("cat" or "dog").
    - **probability**: List of probabilities for each class, rounded to 2 decimal places.
    """,
)
def predict(data: UploadFile = File(...)):  # noqa: B008
    """
    Predict whether the uploaded image is of a cat or a dog.

    Args:
        data (UploadFile): The uploaded image file.

    Returns:
        dict: A dictionary containing the predicted class label and class probabilities.
    """
    if model is None:
        return {"error": "Model not loaded. Please check server logs."}

    try:
        # Read the image file
        image = Image.open(io.BytesIO(data.file.read()))
        image = transform(image).unsqueeze(0).to(DEVICE)

        # Make the prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            prediction = "cat" if predicted.item() == 0 else "dog"
            probability = torch.nn.functional.softmax(output, dim=1).tolist()[0]
            # two decimal places
            probability = [round(p, 2) for p in probability]
            return {"prediction": prediction, "probability": probability}
    except Exception as e:
        raise RuntimeError(f"Failed to make prediction: {str(e)}") from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))  # Default to 8080 if not set
    uvicorn.run(app, host="0.0.0.0", port=port)
