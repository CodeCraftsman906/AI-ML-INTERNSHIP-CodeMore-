# Advanced Model Evaluation Techniques on Imbalanced Data

## Project Overview

This project demonstrates advanced evaluation techniques for machine learning models trained on highly imbalanced datasets using the **Credit Card Fraud Detection** dataset. Instead of focusing only on model accuracy, the project explores evaluation metrics that better reflect model performance when the positive class is rare.

A baseline Logistic Regression model is trained without applying any class imbalance handling techniques. The model is then evaluated using custom implementations of ROC Curve and Precision-Recall Curve along with threshold analysis and different F1-score averaging methods.

---

## Objectives

* Load and analyze an imbalanced credit card fraud dataset.
* Train a baseline Logistic Regression classifier.
* Evaluate model performance without handling class imbalance.
* Implement custom ROC Curve and Precision-Recall Curve functions.
* Analyze the effect of different classification thresholds.
* Compare Macro, Micro, and Weighted F1-Scores.
* Recommend the most appropriate evaluation metric for fraud detection.

---

## Dataset

**Dataset:** Credit Card Fraud Detection

The dataset contains anonymized credit card transactions made by European cardholders.

* Features: 30 input features (`Time`, `Amount`, `V1`–`V28`)
* Target Variable:

  * `0` → Legitimate Transaction
  * `1` → Fraudulent Transaction

The dataset is highly imbalanced, making it suitable for studying advanced evaluation metrics.

---

## Project Structure

```text
Advanced-Model-Evaluation/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── imbalanced_evaluation.ipynb
│
├── src/
│   └── custom_metrics.py
│
├── docs/
│   └── metric_suitability_report.md
│
└── data/
    └── creditcard.csv
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## Workflow

1. Load the Credit Card Fraud dataset.
2. Perform Exploratory Data Analysis.
3. Preprocess the dataset.
4. Train a baseline Logistic Regression model.
5. Evaluate using Accuracy, Precision, Recall, and F1-Score.
6. Implement custom ROC and Precision-Recall Curves.
7. Perform threshold analysis.
8. Compare Macro, Micro, and Weighted F1-Scores.
9. Recommend the most suitable evaluation metric for fraud detection.

---

## Key Findings

* Accuracy alone is misleading for highly imbalanced datasets.
* Precision and Recall provide a more meaningful evaluation of fraud detection models.
* The Precision-Recall Curve is more informative than the ROC Curve for highly imbalanced problems.
* Threshold selection significantly affects Precision and Recall.
* Macro F1-Score provides a more balanced evaluation than Micro or Weighted F1-Scores when minority class performance is important.

---

## Future Improvements

* Apply SMOTE and compare results.
* Evaluate ensemble models such as Random Forest and XGBoost.
* Perform hyperparameter tuning.
* Compare additional anomaly detection techniques.

---
