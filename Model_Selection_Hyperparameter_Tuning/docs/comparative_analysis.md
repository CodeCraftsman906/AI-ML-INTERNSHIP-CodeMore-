# Comparative Analysis Report

## Project Objective

The objective of this project was to compare multiple machine learning classification algorithms and improve their performance using hyperparameter tuning techniques. The final model was selected after evaluating predictive performance, computational efficiency, and model complexity.

---

# Dataset Overview

Dataset: Adult Income Census Dataset

Problem Type: Binary Classification

Target Variable:

* Income <=50K
* Income >50K

The dataset contains demographic and employment-related information with both numerical and categorical attributes.

---

# Data Preprocessing

The following preprocessing steps were performed before model training:

* Removed duplicate records.
* Handled missing values.
* Cleaned the target labels.
* Applied One-Hot Encoding to categorical features.
* Encoded the target variable using Label Encoding.
* Standardized the feature values using StandardScaler.
* Split the data into training, validation, and testing sets.

---

# Models Evaluated

## 1. Random Forest

Random Forest is an ensemble learning algorithm that constructs multiple decision trees and combines their predictions to improve accuracy while reducing overfitting.

### Hyperparameter Optimization

Technique Used:

* GridSearchCV

Parameters Tuned:

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf

---

## 2. Gradient Boosting

Gradient Boosting builds decision trees sequentially, where each new tree attempts to correct the errors of the previous model.

### Hyperparameter Optimization

Technique Used:

* RandomizedSearchCV

Parameters Tuned:

* n_estimators
* learning_rate
* max_depth
* min_samples_split
* min_samples_leaf
* subsample

---

## 3. Support Vector Machine

Support Vector Machine constructs an optimal decision boundary that maximizes the margin between different classes.

### Hyperparameter Optimization

Technique Used:

* RandomizedSearchCV

Parameters Tuned:

* C
* kernel
* gamma

---

# Performance Comparison

| Model                  |             Accuracy | Precision |   Recall | F1 Score|
| ---------------------- | -------------------: | --------: | -----:   | -------:|
| Random Forest          |             0.859249	|  0.780021 |  0.595135| 0.675149|
| Gradient Boosting      |             0.867552 |  0.769937 |  0.657568| 0.709329|
| Support Vector Machine |             0.845699 |  0.735063 |  0.581892| 0.649570|

---

# Hyperparameter Optimization Summary

Hyperparameter tuning significantly improved model performance by identifying parameter combinations that generalized better on unseen data.

GridSearchCV was selected for Random Forest because its search space was relatively small, allowing an exhaustive search.

RandomizedSearchCV was selected for Gradient Boosting and Support Vector Machine because these models have much larger search spaces. Randomized search reduced computational cost while still identifying high-performing parameter combinations.

---

# Model Complexity Analysis

## Random Forest

### Advantages

* High predictive accuracy.
* Robust to noise.
* Handles high-dimensional data effectively.
* Less prone to overfitting than a single decision tree.

### Limitations

* Higher memory consumption.
* Larger model size.
* Less interpretable than a single decision tree.

---

## Gradient Boosting

### Advantages

* Excellent predictive performance.
* Learns complex relationships.
* Often achieves state-of-the-art accuracy on structured datasets.

### Limitations

* Longer training time.
* Sensitive to hyperparameter selection.
* Can overfit if not properly tuned.

---

## Support Vector Machine

### Advantages

* Effective in high-dimensional feature spaces.
* Strong theoretical foundation.
* Works well with an appropriate kernel.

### Limitations

* Computationally expensive on large datasets.
* Sensitive to feature scaling.
* Kernel selection greatly affects performance.

---

# Final Model Selection

The final model was selected based on:

* Highest test accuracy.
* Balanced precision, recall, and F1-score.
* Generalization capability.
* Computational efficiency.
* Model complexity.

Based on the experimental evaluation, the model with the best overall performance was selected as the final classifier and saved as `best_model.pkl` for future inference.

---

# Conclusion

This project demonstrated the complete machine learning workflow, beginning with data preprocessing and progressing through baseline model development, hyperparameter optimization, model comparison, and final model selection.

The comparative analysis showed that systematic hyperparameter tuning can substantially improve predictive performance while enabling informed model selection based on both accuracy and computational considerations.
