import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

# Title
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a digit (0-9)")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    
    # Load image
    img = Image.open(uploaded_file).convert('L')  # grayscale
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Show image
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize
    img_array = img_array / 255.0
    
    # Flatten (same as training)
    img_flat = img_array.reshape(1, -1)
    
    # Prediction
    prediction = model.predict(img_flat)
    
    st.subheader(f"Predicted Digit: {prediction[0]}")