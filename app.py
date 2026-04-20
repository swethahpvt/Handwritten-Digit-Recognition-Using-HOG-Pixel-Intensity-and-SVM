import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle
from skimage.feature import hog

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

    # ✅ MNIST has white digit on BLACK background.
    # Most real photos have black digit on WHITE background — so invert them.
    img_array = np.array(img)
    if img_array.mean() > 127:
        # Bright background detected → invert so digit is white on black
        img = ImageOps.invert(img)
        img_array = np.array(img)

    st.image(img, caption="Preprocessed Image (28x28)", width=150)

    # Normalize (same as training: X / 255.0)
    img_array = img_array / 255.0                  # shape: (28, 28)

    # ✅ Extract HOG features (must match training pipeline exactly)
    hog_features = hog(
        img_array,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        orientations=9
    )                                              # shape: (1764,)

    # Apply PCA (75 components, same as training)
    img_pca = pca.transform(hog_features.reshape(1, -1))  # shape: (1, 75)

    # Predict
    prediction = model.predict(img_pca)

    st.subheader(f"Predicted Digit: {prediction[0]}")
