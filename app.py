# import streamlit as st
# import numpy as np
# from PIL import Image
# import pickle

# @st.cache_resource
# def load_model():
#     with open("model.pkl", "rb") as f:
#         return pickle.load(f)

# artifacts = load_model()
# model = artifacts["model"]
# pca = artifacts["pca"]

# st.title("Handwritten Digit Recognition")
# st.write("Upload an image of a digit (0-9)")

# uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert('L')
#     img = img.resize((28, 28))

#     st.image(img, caption="Uploaded Image", use_container_width=True)

#     img_array = np.array(img).flatten() / 255.0
#     img_pca = pca.transform(img_array.reshape(1, -1))

#     prediction = model.predict(img_pca)
#     st.subheader(f"Predicted Digit: {prediction[0]}")




import streamlit as st
import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skimage.feature import hog

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Digit Recognition App",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# Custom Styling (Professional Look)
# ---------------------------
st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
}
.sub-text {
    font-size: 18px;
    color: #7f8c8d;
}
.result-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Title Section
# ---------------------------
st.markdown('<p class="big-title">🧠 Handwritten Digit Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">HOG + PCA + SVM Model</p>', unsafe_allow_html=True)

st.divider()

# ---------------------------
# Sidebar (Professional Control Panel)
# ---------------------------
st.sidebar.title("⚙️ Settings")

show_steps = st.sidebar.checkbox("Show Processing Steps", True)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    mnist = fetch_openml('mnist_784', version=1)

    X = mnist.data / 255.0
    y = mnist.target.astype(int)

    X_small = X[:3000]
    y_small = y[:3000]

    def extract_hog(images):
        features = []
        for img in images.values.reshape(-1, 28, 28):
            f = hog(img,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2),
                    orientations=9)
            features.append(f)
        return np.array(features)

    hog_feat = extract_hog(X_small)

    pca = PCA(n_components=75)
    X_pca = pca.fit_transform(hog_feat)

    model = SVC(kernel='linear')
    model.fit(X_pca, y_small)

    return model, pca

model, pca = load_model()

# ---------------------------
# Main Layout (2 Columns)
# ---------------------------
col1, col2 = st.columns([1, 1])

# ---------------------------
# Upload Section
# ---------------------------
with col1:
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, caption="Original Image", use_column_width=True)

        # Processing
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (28, 28))
        img_norm = img_resized / 255.0

        if show_steps:
            st.image(img_resized, caption="Processed (28x28)", width=150)

# ---------------------------
# Prediction Section
# ---------------------------
with col2:
    st.subheader("🎯 Prediction")

    if uploaded_file is not None:

        # HOG
        hog_feat = hog(img_norm,
                       pixels_per_cell=(4, 4),
                       cells_per_block=(2, 2),
                       orientations=9)

        hog_feat = hog_feat.reshape(1, -1)

        # PCA
        pca_feat = pca.transform(hog_feat)

        # Prediction
        prediction = model.predict(pca_feat)

        # Result Box
        st.markdown(f"""
        <div class="result-box">
            <h2>Predicted Digit</h2>
            <h1 style="color:#27ae60;">{prediction[0]}</h1>
        </div>
        """, unsafe_allow_html=True)

        # Confidence (optional)
        confidence = model.decision_function(pca_feat)
        st.write("Confidence Scores:", confidence)

    else:
        st.info("Upload an image to get prediction")

# ---------------------------
# Footer Info
# ---------------------------
st.divider()

st.markdown("""
### 📊 Model Details
- Feature Extraction: HOG
- Dimensionality Reduction: PCA (75 components)
- Classifier: SVM (Linear Kernel)

### 💡 How to Use
1. Upload a handwritten digit image  
2. View processed image  
3. Get prediction instantly  
""")