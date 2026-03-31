import streamlit as st
import numpy as np
from PIL import Image
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
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(img).flatten() / 255.0
    img_pca = pca.transform(img_array.reshape(1, -1))

    prediction = model.predict(img_pca)
    st.subheader(f"Predicted Digit: {prediction[0]}")
