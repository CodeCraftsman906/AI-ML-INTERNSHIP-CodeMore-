# Model Evaluation Report

## Objective

The objective of this project was to evaluate the performance of a machine learning classification model using advanced evaluation metrics and interpret its predictions through Explainable AI techniques. The project demonstrates how model evaluation and interpretability complement each other in developing reliable AI systems.

---

## Dataset Description

The UCI Heart Disease Dataset was used for this project. It contains demographic and clinical information collected from patients, including age, sex, chest pain type, cholesterol level, resting blood pressure, fasting blood sugar, ECG results, maximum heart rate achieved, and several other medical attributes.

The original target variable contained five classes (0–4). It was converted into a binary classification problem:

* **0:** No Heart Disease
* **1:** Heart Disease Present

---

## Data Preprocessing

The following preprocessing steps were performed:

* Removed unnecessary ID column.
* Converted the target variable into binary classes.
* Handled missing values using Median and Mode imputation.
* Encoded categorical variables.
* Standardized numerical features.
* Split the dataset into training and testing sets.

---

## Model Selection

A **Random Forest Classifier** was selected because it:

* Handles nonlinear relationships effectively.
* Performs well on structured tabular datasets.
* Is less prone to overfitting compared to a single decision tree.
* Works efficiently with SHAP's tree-based explainability methods.

---

## Performance Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* Area Under the ROC Curve (AUC)

These metrics provide a comprehensive understanding of model performance beyond overall accuracy.

---

## Confusion Matrix Interpretation

The confusion matrix summarizes the model's classification performance by displaying correctly and incorrectly classified samples.

It helps identify:

* True Positives
* True Negatives
* False Positives
* False Negatives

A higher number of correct classifications indicates better predictive performance.

---

## ROC Curve and AUC

The ROC Curve illustrates the trade-off between the True Positive Rate and False Positive Rate across different classification thresholds.

The Area Under the Curve (AUC) provides a threshold-independent measure of performance.

A higher AUC value indicates stronger discrimination between positive and negative classes.

---

## SHAP Interpretation

SHAP was used to explain the model's predictions.

### Global Interpretation

The SHAP Summary Plot identifies the most influential features affecting model predictions across the entire dataset.

The SHAP Bar Plot ranks features according to their average contribution to predictions.

### Local Interpretation

The SHAP Waterfall/Force Plot explains how individual feature values contribute to a single prediction.

Positive SHAP values increase the probability of heart disease, while negative values decrease it.

---

## Challenges Faced

* Handling missing values in the dataset.
* Converting categorical variables into numerical form.
* Understanding changes introduced in newer SHAP versions.
* Resolving compatibility issues between older tutorials and SHAP 0.52.
* Correctly interpreting SHAP explanation objects for visualization.

---

## Conclusion

This project demonstrates that successful machine learning extends beyond building accurate predictive models. Proper evaluation using multiple performance metrics and model interpretability using SHAP provides greater transparency, reliability, and confidence in AI systems. Explainable AI plays a crucial role in understanding model behavior, validating predictions, and supporting informed decision-making in real-world applications.
