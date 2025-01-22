import os

import pandas as pd
import requests
import streamlit as st


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict"
    response = requests.post(predict_url, files={"data": image}, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = "https://mlops-catsvsdogs-122709719634.us-central1.run.app"
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probability"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", prediction)

            # make a nice bar chart
            classes = ["Cat", "Dog"]
            data = {"Class": [f"{i}" for i in classes], "Probability": probabilities}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()