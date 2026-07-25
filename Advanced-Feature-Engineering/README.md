# Advanced Feature Engineering Techniques

## Overview

This project demonstrates how feature engineering techniques can improve the performance of machine learning models on tabular data. The Wine Quality Dataset is used to perform binary classification using Logistic Regression. Different feature engineering methods are applied and their impact on model performance is analysed.

---

## Objectives

- Perform data cleaning and preprocessing.
- Train a baseline Logistic Regression model.
- Generate Polynomial Features.
- Create Interaction Features.
- Apply Log Transformation.
- Apply Box-Cox Transformation.
- Compare the performance of all models.
- Analyse the impact of each feature engineering technique.

---

## Dataset

**Dataset:** Wine Quality Dataset

**Source:** UCI Machine Learning Repository

The dataset contains physicochemical properties of red wine samples and their quality ratings.

Target Variable:

- Quality ≥ 6 → Good Wine (1)
- Quality < 6 → Poor Wine (0)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- SciPy
- Jupyter Notebook

---

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Baseline Model
4. Polynomial Feature Generation
5. Interaction Feature Generation
6. Log Transformation
7. Box-Cox Transformation
8. Model Evaluation
9. Performance Comparison

---

## Feature Engineering Techniques

### Baseline

Original dataset without additional feature engineering.

### Polynomial Features

Creates squared and higher-order terms to capture non-linear relationships.

### Interaction Features

Creates interaction terms between different variables without generating squared terms.

### Log Transformation

Reduces skewness in highly skewed numerical features.

### Box-Cox Transformation

Transforms skewed numerical features into a more Gaussian-like distribution.

---

## Model Used

- Logistic Regression

Evaluation Metric:

- Accuracy
- Classification Report

---

## Repository Structure

```
Advanced-Feature-Engineering/
│
├── README.md
├── requirements.txt
├── data/
│   └── raw_dataset_info.txt
├── docs/
│   └── impact_report.md
└── notebooks/
    └── feature_engineering.ipynb
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

Open

```
notebooks/feature_engineering.ipynb
```

Run all cells sequentially.

---

## Results

The project compares five different approaches:

- Baseline Model
- Polynomial Features
- Interaction Features
- Log Transformation
- Box-Cox Transformation

The final comparison table and visualization show how feature engineering influences model performance.

---
