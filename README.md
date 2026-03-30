Image Classification using HOG, Zernike & Pixel Features
Overview

This project focuses on comparing different feature extraction techniques and machine learning models for image classification. The goal is to analyze how feature representation impacts classification performance.

Three feature extraction methods are used:
- Pixel Features (Raw intensity values)
- HOG (Histogram of Oriented Gradients)
- Zernike Moments (Shape descriptors and rotation invariant)

These features are evaluated using multiple machine learning classifiers.
 Objectives

- Compare different feature extraction techniques  
- Evaluate multiple machine learning models  
- Analyze robustness to rotation and thickness variations  
- Identify the best feature-model combination  
 Technologies Used

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Mahotas (for Zernike moments)  
 Machine Learning Models

- Support Vector Machine (SVM)  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Random Forest  
 Methodology
 Pixel Features
- Images are flattened into 784-dimensional vectors  
- Used directly for training models  
 HOG Features
- Extracts gradient and edge-based features  
- Captures structural information  
Zernike Features
- Extracts shape-based features  
- Rotation invariant  
- Captures global image structure  
Results

| Feature Type | SVM   | Logistic | KNN    | Random Forest | Best Model | Accuracy |
|--------------|-------|----------|--------|---------------|-----------|-----------|
| Pixel        | 0.981 | 0.91     | 0.974  | 0.952         | SVM       | 0.981     |
| HOG          | 0.956 | 0.9666   | 0.955  | 0.9384        | Logistic  | 0.9666    |
| Zernike      | 0.849 | 0.733    | 0.8527 | 0.82          | KNN       | 0.8527    |
 Key Observations

- Pixel features achieved the highest accuracy with SVM  
- HOG features performed best with Logistic Regression  
- Zernike features provide strong shape representation but lower accuracy  
- Different models perform better with different feature types  
Robustness Analysis

- Pixel Features: Sensitive to rotation and thickness variations  
- HOG Features: Moderately robust  
- Zernike Features: Highly robust to rotation and thickness variations  
Evaluation Metrics

- Accuracy  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  

1. Clone the repository  
2. Install dependencies:
bash
pip install numpy pandas scikit-learn matplotlib mahotas
