# Alzheimer
🧠 Alzheimer’s Disease Detection – Two-Stage AI System

This project implements a two-stage machine learning pipeline for Alzheimer’s disease detection:

Stage 1: Clinical & demographic data–based prediction using CatBoost

Stage 2: MRI image–based disease stage classification using CNN (TensorFlow/Keras)

The system combines tabular ML and deep learning (medical imaging) to improve diagnostic accuracy.

📌 Project Overview

Alzheimer’s disease diagnosis benefits from combining:

Patient clinical data (age, gender, cognitive scores, etc.)

MRI brain scans showing structural changes

This project reflects that idea using:

CatBoost (excellent for categorical medical data)

CNN-based deep learning model for MRI classification

🏗️ Architecture
Input Data
│
├── Stage 1: Clinical Dataset (CSV)
│   └── CatBoost Classifier
│       └── Binary Alzheimer’s Diagnosis
│
└── Stage 2: MRI Images
    └── CNN (TensorFlow)
        └── Multi-class Disease Stage Prediction

🧪 Datasets Used
1️⃣ Stage 1 – Clinical Dataset

File: alzheimers_stage1_cleaned_dataset.csv

Target column: Alzheimer’s Diagnosis

Contains:

Numerical features

Categorical features (handled natively by CatBoost)

2️⃣ Stage 2 – MRI Dataset

Source: Hugging Face

Dataset name: Falah/Alzheimer_MRI

Classes: 4 Alzheimer’s stages

Image preprocessing:

Resize to 224 × 224

Grayscale → RGB conversion

Normalization (0–1)

🛠️ Technologies & Libraries
Core Libraries

Python 3.8+

TensorFlow / Keras

NumPy

Pandas

Scikit-learn

Hugging Face datasets

Machine Learning

CatBoost

SHAP (for explainability)

Utilities

Joblib

📦 Installation
pip install tensorflow datasets catboost shap joblib pandas numpy scikit-learn

🚀 Stage 2 – MRI CNN Model
Model Summary

Input shape: 224 × 224 × 3

Architecture:

3 × Conv2D + MaxPooling layers

Dense + Dropout

Softmax output (4 classes)

Loss: categorical_crossentropy

Optimizer: Adam

Epochs: 15

Batch size: 32

Training
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15
)

Model Output

Saved as:

alzheimers_mri_stage2_model.keras

🚀 Stage 1 – CatBoost Clinical Model
Why CatBoost?

Handles categorical features automatically

Excellent performance on medical tabular data

Robust to missing values

Model Configuration

Iterations: 800

Learning rate: 0.03

Depth: 6

Evaluation metric: ROC-AUC

Class balancing enabled

Training
final_model.fit(
    X,
    y,
    cat_features=cat_feature_indices
)

Saved Model
alzheimers_catboost_model.cbm

📊 Model Evaluation
Cross-Validation

5-fold Stratified K-Fold

Metric: ROC-AUC

Fold 1 ROC-AUC: xxxx
Fold 2 ROC-AUC: xxxx
Fold 3 ROC-AUC: xxxx
Fold 4 ROC-AUC: xxxx
Fold 5 ROC-AUC: xxxx

Mean ROC-AUC: xxxx
Std ROC-AUC : xxxx

🔍 Explainability (Optional)

SHAP can be used to interpret CatBoost predictions:

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

📁 Project Structure
├── alzheimers_stage1_cleaned_dataset.csv
├── alzheimers_catboost_model.cbm
├── alzheimers_mri_stage2_model.keras
├── train_mri_model.py
├── train_catboost_model.py
├── cross_validation.py
├── README.md

✅ Key Features

Two-stage AI diagnosis pipeline

Combines tabular + image data

Balanced learning for medical datasets

Cross-validated performance

Production-ready model saving

🔮 Future Improvements

Transfer learning (ResNet / EfficientNet)

Ensemble Stage 1 + Stage 2 predictions

Web-based diagnostic dashboard

Clinical feature importance visualization

Grad-CAM visualization for MRI scans
