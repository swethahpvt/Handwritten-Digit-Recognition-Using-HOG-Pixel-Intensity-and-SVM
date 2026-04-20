import streamlit as st
import numpy as np
from PIL import Image
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

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy array and normalize (same as training: X / 255.0)
    img_array = np.array(img) / 255.0             # shape: (28, 28)

    # ✅ Extract HOG features (MUST match training pipeline exactly)
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
