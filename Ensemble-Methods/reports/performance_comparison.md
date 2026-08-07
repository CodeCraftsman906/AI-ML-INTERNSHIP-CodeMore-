# Performance Comparison Report

## Objective

The objective of this experiment was to evaluate the effectiveness of ensemble learning techniques by comparing a Decision Tree with Random Forest and Gradient Boosting on the Credit Card Fraud Detection dataset.

---

## Models Evaluated

1. Decision Tree
2. Random Forest
3. Gradient Boosting

---

## Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

These metrics provide a comprehensive assessment of classification performance, particularly for imbalanced datasets.

---

## Performance Summary

### Decision Tree

* Simple and interpretable baseline model.
* Fast training time.
* More susceptible to overfitting.
* Lower overall performance compared to ensemble methods.

### Random Forest

* Uses bagging to combine multiple Decision Trees.
* Produces more stable predictions.
* Reduces variance and overfitting.
* Achieves improved Precision, Recall, and F1-Score.

### Gradient Boosting

* Builds trees sequentially by correcting previous errors.
* Learns complex decision boundaries effectively.
* Generally provides the strongest predictive performance.
* Requires more computational time than Random Forest.

---

## Comparative Analysis

The comparison indicates that ensemble methods consistently outperform a single Decision Tree. Random Forest improves model stability through averaging multiple trees, while Gradient Boosting enhances predictive accuracy by iteratively correcting previous mistakes.

For fraud detection, where identifying minority-class transactions is critical, ensemble learning provides better Recall and F1-Score, resulting in more reliable classification performance.

---

## Conclusion

This experiment demonstrates that ensemble learning is an effective strategy for improving classification performance. Both Random Forest and Gradient Boosting deliver better generalization and higher predictive accuracy than a standalone Decision Tree, making them more suitable for credit card fraud detection.
