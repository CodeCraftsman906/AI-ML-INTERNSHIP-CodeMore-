# Metric Suitability Report for Credit Card Fraud Detection

## Introduction

Credit card fraud detection is a binary classification problem characterized by severe class imbalance, where fraudulent transactions represent only a very small percentage of all transactions. In such situations, relying solely on accuracy can lead to misleading conclusions about model performance.

This report evaluates different performance metrics and recommends the most appropriate metric for this business problem.

---

## Dataset Description

The Credit Card Fraud Detection dataset contains anonymized financial transaction records.

* Majority Class: Legitimate Transactions
* Minority Class: Fraudulent Transactions

The dataset is highly imbalanced, making it an ideal candidate for studying advanced evaluation techniques.

---

## Baseline Model

A Logistic Regression classifier was trained without applying any imbalance handling techniques such as oversampling, undersampling, SMOTE, or class weighting.

This baseline model demonstrates how traditional classifiers behave when trained on highly imbalanced data.

---

## Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC Curve
* Precision-Recall Curve
* Macro F1-Score
* Micro F1-Score
* Weighted F1-Score

---

## ROC Curve Analysis

The ROC Curve measures the relationship between the True Positive Rate and the False Positive Rate across different classification thresholds.

Although the ROC Curve provides a good overall measure of classification performance, it can present an overly optimistic view when the dataset is highly imbalanced because the large number of negative samples dominates the False Positive Rate.

---

## Precision-Recall Curve Analysis

The Precision-Recall Curve focuses exclusively on the minority (fraud) class.

Since fraudulent transactions are extremely rare, this curve provides a clearer understanding of how well the model detects fraud while minimizing false alarms.

For highly imbalanced datasets, the Precision-Recall Curve is generally more informative than the ROC Curve.

---

## Threshold Analysis

Changing the classification threshold directly affects model performance.

* Lower thresholds increase Recall by identifying more fraudulent transactions but also increase false positives, reducing Precision.
* Higher thresholds improve Precision by making fewer fraud predictions but may miss genuine fraud cases, reducing Recall.

The threshold should therefore be selected according to the business objective rather than using the default value of 0.5.

---

## Comparison of F1 Averaging Methods

**Macro F1-Score**

Treats all classes equally regardless of their frequency, making it suitable when minority class performance is important.

**Micro F1-Score**

Aggregates all predictions before computing the metric. On imbalanced datasets, this score is dominated by the majority class.

**Weighted F1-Score**

Computes the F1-Score for each class while weighting them according to class frequency. Although it considers class imbalance, the majority class still has a greater influence on the final result.

---

## Recommended Evaluation Metric

For credit card fraud detection, the **Precision-Recall Curve** should be considered the primary evaluation metric because it focuses on the minority class and clearly illustrates the trade-off between detecting fraud and minimizing false alarms.

Recall is especially important because failing to detect fraudulent transactions may result in significant financial losses. However, Precision should also be monitored to avoid an excessive number of false alerts.

Macro F1-Score serves as an additional balanced metric because it gives equal importance to both classes.

---

## Conclusion

This study demonstrates that evaluation metrics must be selected according to the business problem rather than relying solely on accuracy.

For highly imbalanced fraud detection tasks, the combination of the Precision-Recall Curve, Recall, and Macro F1-Score provides a more reliable assessment of model performance than traditional accuracy-based evaluation.
