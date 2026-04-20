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

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Flatten and normalize
    img_array = np.array(img).flatten() / 255.0   # shape: (784,)

    # ✅ Invert if image has dark digit on white background
    # MNIST was trained on WHITE digit on BLACK background
    if img_array.mean() > 0.5:      # Background is bright (white) → invert
        img_array = 1 - img_array

    # Apply PCA (75 components, same as training)
    img_pca = pca.transform(img_array.reshape(1, -1))  # shape: (1, 75)

    # Predict
    prediction = model.predict(img_pca)

    st.subheader(f"✅ Predicted Digit: {prediction[0]}")