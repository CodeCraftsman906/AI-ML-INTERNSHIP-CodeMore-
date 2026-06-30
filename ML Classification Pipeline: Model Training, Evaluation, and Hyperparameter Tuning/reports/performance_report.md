# Performance Analysis Report

# Project Overview

The objective of this project was to develop and evaluate supervised machine learning models capable of classifying the target variable (`Feature13`) using a processed dataset. After data preprocessing, feature engineering, and train-test splitting, two baseline classification algorithms—Logistic Regression and Decision Tree Classifier—were trained and evaluated. The best-performing model was further optimized using GridSearchCV for hyperparameter tuning.

---

# Models Evaluated

The following supervised learning algorithms were implemented:

* Logistic Regression
* Decision Tree Classifier

Both models were trained using 80% of the dataset and evaluated on the remaining 20% of unseen test data.

---

# Performance Metrics Used

The following evaluation metrics were used to measure model performance:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics provide a comprehensive understanding of the classification performance beyond overall accuracy.

---

# Baseline Model Results

| Metric    | Logistic Regression | Decision Tree |
| --------- | ------------------: | ------------: |
| Accuracy  |           **61.1%** |         44.7% |
| Precision |               37.3% |     **47.1%** |
| Recall    |           **61.1%** |         44.7% |
| F1-Score  |           **46.3%** |         45.8% |

---

# Model Comparison

### Logistic Regression

Strengths:

* Achieved the highest overall accuracy.
* Produced better Recall and F1-Score.
* Demonstrated better generalization on unseen data.
* Served as the strongest baseline model.

Limitations:

* Lower Precision compared to the Decision Tree.
* Required increasing the maximum number of iterations to achieve convergence.

---

### Decision Tree Classifier

Strengths:

* Slightly higher Precision.
* Easy to interpret and visualize.

Limitations:

* Lower Accuracy.
* Lower Recall.
* Lower F1-Score.
* Showed weaker generalization compared to Logistic Regression.

---

# Best Performing Model

Based on the evaluation metrics, **Logistic Regression** was selected as the best-performing baseline model.

Reasons:

* Highest Accuracy (61.1%)
* Highest Recall
* Highest F1-Score
* Better overall balance across evaluation metrics

Although the Decision Tree achieved slightly higher Precision, its lower Accuracy and Recall made it less suitable for this dataset.

---

# Hyperparameter Tuning

To further improve the selected model, **GridSearchCV** was used to perform hyperparameter tuning.

The following parameters were optimized:

* Regularization parameter (`C`)
* Solver
* Maximum iterations (`max_iter`)
* Penalty type

GridSearchCV evaluated multiple parameter combinations using 5-fold cross-validation and selected the best-performing configuration.

---

# Performance Insights

The analysis highlighted several important observations:

* Logistic Regression generalized better than the Decision Tree on this dataset.
* Multiple evaluation metrics should always be considered instead of relying only on Accuracy.
* Hyperparameter tuning is an effective way to improve model performance without changing the dataset.
* Confusion matrices provided valuable insights into classification errors and class-wise performance.

---

# Recommendations for Future Improvements

The following techniques can further improve the model:

* Collect additional training data.
* Improve feature engineering.
* Address potential class imbalance.
* Perform feature selection.
* Apply advanced hyperparameter optimization techniques such as RandomizedSearchCV or Bayesian Optimization.
* Perform k-fold cross-validation for more reliable performance estimation.

---

# Conclusion

The project successfully implemented a complete supervised machine learning workflow, beginning with data preprocessing and feature engineering, followed by model training, evaluation, comparison, and optimization. Among the evaluated models, Logistic Regression demonstrated the best overall performance and was selected as the final model for optimization using GridSearchCV. The project reinforced the importance of proper data preprocessing, systematic model evaluation, and hyperparameter tuning in developing reliable and effective machine learning solutions.
