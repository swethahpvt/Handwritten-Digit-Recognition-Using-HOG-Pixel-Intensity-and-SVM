import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_model()
model = artifacts["model"]
pca = artifacts["pca"]

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a digit (0-9)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    img = Image.open(uploaded_file).convert('L')  # Grayscale
    img = img.resize((28, 28))                     # Resize to 28x28

    # MNIST has white digit on BLACK background.
    # Most real photos have black digit on WHITE background — invert them.
    img_array = np.array(img)
    if img_array.mean() > 127:
        img = ImageOps.invert(img)
        img_array = np.array(img)

    st.image(img, caption="Preprocessed Image (28x28)", width=150)

    # ✅ Flatten and normalize to 784 raw pixels (matches model.pkl training)
    img_flat = img_array.flatten() / 255.0         # shape: (784,)

    # Apply PCA (trained on raw pixels)
    img_pca = pca.transform(img_flat.reshape(1, -1))  # shape: (1, 75)

    # Predict
    prediction = model.predict(img_pca)

    st.subheader(f"Predicted Digit: {prediction[0]}")
