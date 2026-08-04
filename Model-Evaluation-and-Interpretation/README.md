# Model Evaluation and Interpretation using Explainable AI

## Overview

This project demonstrates how to evaluate a machine learning classification model using advanced performance metrics and explain its predictions using SHAP (SHapley Additive Explanations). The project focuses not only on predictive performance but also on model interpretability, making it suitable for real-world AI applications where transparency and trust are essential.

---

## Objectives

* Train a high-performing classification model.
* Evaluate the model using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, and AUC.
* Explain model predictions using SHAP.
* Interpret both global and local feature contributions.

---

## Dataset

**Dataset:** UCI Heart Disease Dataset

The dataset contains patient medical information such as age, cholesterol level, blood pressure, chest pain type, maximum heart rate, and other clinical features to predict the presence of heart disease.

Target Variable:

* **0** → No Heart Disease
* **1** → Heart Disease Present

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SHAP
* Jupyter Notebook

---

## Project Structure

```text
Model-Evaluation-and-Interpretation/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── model_interpretation.ipynb
│
├── docs/
│   └── evaluation_report.md
│
└── assets/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── shap_summary.png
    └── shap_bar.png
```

---

## Workflow

1. Load and explore the dataset.
2. Handle missing values and preprocess data.
3. Train a Random Forest Classifier.
4. Evaluate model performance.
5. Generate ROC Curve and calculate AUC.
6. Explain model predictions using SHAP.
7. Interpret feature importance globally and locally.

---

## Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* Area Under the Curve (AUC)

---

## Explainable AI

SHAP was used to understand model predictions.

Generated visualizations include:

* SHAP Summary Plot
* SHAP Feature Importance Bar Plot
* SHAP Waterfall/Force Plot

These visualizations explain both overall feature importance and individual prediction behavior.

---

## Key Learning Outcomes

* Model evaluation using multiple classification metrics.
* Importance of Precision, Recall, and F1-score beyond Accuracy.
* ROC-AUC analysis for threshold-independent evaluation.
* Explainable AI concepts using SHAP.
* Understanding global and local feature contributions.
* Building interpretable machine learning models.

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV.
* Comparison with XGBoost and LightGBM.
* Model deployment using Streamlit or Flask.
* Experiment with LIME for model interpretation.

---

The NorthCap University
