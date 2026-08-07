# Introduction to Ensemble Methods

## Credit Card Fraud Detection using Ensemble Learning

### Project Overview

This project demonstrates the application of **Ensemble Learning** techniques for detecting fraudulent credit card transactions. The performance of a baseline **Decision Tree** model is compared with two popular ensemble algorithms: **Random Forest** and **Gradient Boosting**.

The objective is to understand how combining multiple weak learners improves prediction accuracy, reduces overfitting, and produces more reliable classification results.

---

## Objectives

* Select a standard tabular classification dataset.
* Perform data preprocessing.
* Train a baseline Decision Tree model.
* Implement Random Forest and Gradient Boosting classifiers.
* Compare the performance of all three models using multiple evaluation metrics.

---

## Dataset

The project uses the **Credit Card Fraud Detection Dataset**, which contains anonymized credit card transactions.

Target Variable:

* **Class = 0** → Legitimate Transaction
* **Class = 1** → Fraudulent Transaction

The dataset is highly imbalanced, making metrics such as Precision, Recall, F1-Score, and ROC-AUC more informative than Accuracy alone.

---

## Project Structure

```text
README.md

notebooks/
└── ensemble_exploration.ipynb

reports/
└── performance_comparison.md

data/
└── raw_dataset_info.txt

requirements.txt
```

---

## Machine Learning Models

* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix
* ROC Curve

---

## Results

The notebook compares all three models and demonstrates that ensemble learning techniques generally outperform a single Decision Tree by providing better generalization and reducing overfitting.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## Conclusion

Ensemble learning significantly improves classification performance for fraud detection. Random Forest and Gradient Boosting achieve better predictive accuracy and robustness compared to a standalone Decision Tree, making them more suitable for real-world classification problems involving complex and imbalanced datasets.
