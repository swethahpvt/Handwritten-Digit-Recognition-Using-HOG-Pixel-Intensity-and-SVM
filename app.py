import streamlit as st
import numpy as np
from PIL import Image
import pickle
from skimage.feature import hog

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a digit (0-9)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    img_array = np.array(img) / 255.0
    
    hog_features = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(14, 14),
        cells_per_block=(1, 1),
        visualize=False
    )
    
    hog_features = hog_features.reshape(1, -1)
    
    prediction = model.predict(hog_features)
    st.subheader(f"Predicted Digit: {prediction[0]}")
