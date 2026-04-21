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

    # Load image and convert to grayscale
    img = Image.open(uploaded_file).convert('L')

    # Add padding to mimic MNIST centering
    img = ImageOps.expand(img, border=20, fill='white')

    # Resize to 28x28 using high quality resampling
    img = img.resize((28, 28), Image.LANCZOS)

    st.image(img, caption="Preprocessed Image", use_container_width=True)

    # Flatten and normalize to [0, 1]
    img_array = np.array(img).flatten() / 255.0

    # Auto-invert if image has white background
    # MNIST was trained on WHITE digit on BLACK background
    if img_array.mean() > 0.5:
        img_array = 1 - img_array

    # Apply PCA (75 components, same as training)
    img_pca = pca.transform(img_array.reshape(1, -1))

    # Predict
    prediction = model.predict(img_pca)

    st.subheader(f"✅ Predicted Digit: {prediction[0]}")
