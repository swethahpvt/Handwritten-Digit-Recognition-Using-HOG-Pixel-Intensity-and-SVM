import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Load model safely
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Title
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a digit (0-9)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    img_array = np.array(img) / 255.0
    
    # OPTIONAL (fix inverted images)
    # img_array = 1 - img_array
    
    img_flat = img_array.reshape(1, -1)
    
    prediction = model.predict(img_flat)
    
    st.subheader(f"Predicted Digit: {prediction[0]}")
