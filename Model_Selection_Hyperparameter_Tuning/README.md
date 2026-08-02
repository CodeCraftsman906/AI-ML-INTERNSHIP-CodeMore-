# Model Selection and Hyperparameter Tuning

## Project Overview

This project demonstrates the process of selecting the most suitable machine learning model and optimizing its performance through hyperparameter tuning. Three supervised classification algorithms—Random Forest, Gradient Boosting, and Support Vector Machine (SVM)—are trained, evaluated, and compared. Hyperparameter optimization is performed using GridSearchCV and RandomizedSearchCV to identify the best-performing model.

---

## Objectives

* Prepare and preprocess the Adult Income dataset.
* Split the dataset into training, validation, and testing sets.
* Train baseline machine learning models.
* Optimize model performance using hyperparameter tuning.
* Compare the performance of all tuned models.
* Select and save the best-performing model.

---

## Dataset

**Dataset:** Adult Income Census Dataset

**Prediction Task:** Binary Classification

**Target Variable:** `income`

Classes:

* `<=50K`
* `>50K`

The dataset contains both numerical and categorical features describing demographic and employment-related information.

---

## Machine Learning Algorithms

The following classification algorithms were implemented:

* Random Forest Classifier
* Gradient Boosting Classifier
* Support Vector Machine (SVM)

---

## Hyperparameter Tuning Techniques

Two optimization techniques were used:

### GridSearchCV

Used for Random Forest to perform an exhaustive search over a predefined parameter grid.

### RandomizedSearchCV

Used for Gradient Boosting and SVM to efficiently search the hyperparameter space while reducing computational time.

---

## Evaluation Metrics

The following metrics were used to compare model performance:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

---

## Repository Structure

```text
Model_Selection_Hyperparameter_Tuning/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── hyperparameter_tuning.ipynb
│
├── docs/
│   └── comparative_analysis.md
│
├── src/
│   └── train_tuned_model.py
│

```

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

Open Jupyter Notebook and execute:

```text
notebooks/hyperparameter_tuning.ipynb
```

Run all cells sequentially.

---

## Running the Training Script

Execute:

```bash
python src/train_tuned_model.py
```

The script will:

* Load and preprocess the dataset
* Train the selected tuned model
* Evaluate model performance
* Save the trained model as `best_model.pkl`

---

## Results

The tuned machine learning models were evaluated using multiple performance metrics. Hyperparameter optimization improved the overall predictive performance compared to the baseline models. The model achieving the highest accuracy and balanced evaluation metrics was selected as the final model and saved for future use.

---

## Future Improvements

* Explore additional ensemble methods such as XGBoost and LightGBM.
* Apply advanced feature engineering techniques.
* Perform feature selection to reduce dimensionality.
* Experiment with Bayesian Optimization for hyperparameter tuning.
* Deploy the final model as a web application using Flask or FastAPI.
