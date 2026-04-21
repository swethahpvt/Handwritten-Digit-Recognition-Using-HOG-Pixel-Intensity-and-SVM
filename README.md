Handwritten Digit Recognition Using HOG, Pixel Intensity, and SVM

A machine learning web application that recognizes handwritten digits (0–9) from uploaded images using Support Vector Machine (SVM) with Pixel Intensity features and PCA dimensionality reduction.

Live Demo: [Click Here](https://handwritten-digit-recognition-using-hog-vti3.onrender.com)

Overview

This project compares three feature extraction techniques — **Pixel Intensity**, **HOG (Histogram of Oriented Gradients)**, and **Zernike Moments** — combined with multiple classifiers to classify handwritten digits from the MNIST dataset.

Problem Statement & Motivation

Handwritten digit recognition is a foundational problem in computer vision with real-world applications in:

- Postal automation* — reading handwritten ZIP codes and addresses
- Bank cheque processing* — automatically reading handwritten amounts
- Digitizing historical records* — converting handwritten ledgers and documents
- Accessibility technology* — assistive tools for people with disabilities

Despite decades of research, building accurate, fast, and lightweight digit recognizers remains a meaningful challenge. Deep learning methods achieve high accuracy but require significant compute. This project explores whether *classical machine learning* (SVM, KNN, Logistic Regression, Random Forest) combined with smart feature engineering can achieve competitive performance — and it does, reaching *98.1% accuracy* with an SVM on raw pixel features
The best performing model — **Pixel + SVM with 98.1% accuracy** — is deployed as an interactive Streamlit web app.

Source
The *MNIST (Modified National Institute of Standards and Technology)* dataset is the benchmark dataset for handwritten digit recognition.

- *Source:* [Yann LeCun's MNIST Page](http://yann.lecun.com/exdb/mnist/) / sklearn.datasets.fetch_openml('mnist_784')
- *Total Size:* 70,000 grayscale images
  - Training set: 60,000 images
  - Test set: 10,000 images
- Image Dimensions:* 28 × 28 pixels (784 features per image after flattening)
- Pixel Range:* 0–255 (normalized to 0–1 during preprocessing)
- Classes:* 10 (digits 0 through 9)
  
Model Performance Comparison

| Feature   | SVM   | KNN   | Logistic Regression | Random Forest |
|-----------|-------|-------|----------------------|---------------|
| Pixel     | 98.1% | 97.4% | 91.0%               | 95.2%         |
| HOG       | 95.6% | 95.7% | 96.7%               | 93.8%         |
| Zernike   | 84.9% | 85.3% | 73.3%               | 82.6%         |

Feature Extraction Methods

Pixel Intensity
- Raw pixel values (28×28 = 784 features) normalized to [0, 1]
- PCA applied to reduce to 75 components
- Highest accuracy among all methods

HOG (Histogram of Oriented Gradients)
- Captures edge and gradient structure
- Parameters: pixels_per_cell=(4,4), cells_per_block=(2,2), orientations=9
- Moderate robustness to thickness variations

Zernike Moments
- Captures global shape information using radius=10, degree=8
- Rotation-invariant but misses fine pixel-level detail

Project Structure

├── app.py               # Streamlit web application
├── code.ipynb           # Full experimentation notebook
├── model.pkl            # Trained Pixel SVM model + PCA
├── requirements.txt     # Python dependencies
├── runtime.txt          # Python version for deployment
└── README.md            # Project documentation

How It Works

1. User uploads an image of a handwritten digit
2. Image is converted to grayscale and resized to 28×28
3. Pixel values are normalized and flattened
4. PCA reduces dimensions to 75 components
5. SVM predicts the digit (0–9)

Tech Stack

- **Python 3.11**
- **Streamlit** — Web interface
- **scikit-learn** — SVM, PCA, model training
- **NumPy / Pillow** — Image processing
- **MNIST Dataset** — 70,000 training images
Run Locally
bash
git clone https://github.com/swethahpvt/Handwritten-Digit-Recognition-Using-HOG-Pixel-Intensity-and-SVM
cd Handwritten-Digit-Recognition-Using-HOG-Pixel-Intensity-and-SVM
pip install -r requirements.txt
streamlit run app.py

Deployment

Deployed on Render as a free web service.

- Build Command: pip install -r requirements.txt
- Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
- Python Version: 3.11.9

Author
Swetha Babu — [GitHub Profile](https://github.com/swethahpvt)
