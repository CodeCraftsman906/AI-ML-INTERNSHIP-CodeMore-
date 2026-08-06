# Hyperparameter Optimization for Credit Card Fraud Detection

## Project Overview

This project explores advanced hyperparameter optimization techniques to improve the performance of an XGBoost classifier on a Credit Card Fraud Detection dataset. The objective is to compare Random Search and Bayesian Optimization in terms of model performance and optimization efficiency.

---

## Objectives

- Build a baseline XGBoost classifier.
- Apply Random Search for hyperparameter tuning.
- Apply Bayesian Optimization using Optuna.
- Compare both optimization techniques.
- Evaluate execution time and predictive performance.

---

## Dataset

The dataset contains credit card transactions labeled as fraudulent or non-fraudulent.

Target Variable:

- Class
  - 0 → Legitimate Transaction
  - 1 → Fraudulent Transaction

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Optuna

---

## Project Structure

```
Hyperparameter-Optimization/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── bayesian_opt.py
│   └── random_search.py
│
└── docs/
    └── optimization_report.md
```

---

## Hyperparameter Optimization Techniques

### Baseline Model

A default XGBoost classifier is trained without tuning.

### Random Search

Random Search samples random combinations of hyperparameters from a predefined search space and evaluates them using cross-validation.

### Bayesian Optimization

Bayesian Optimization (Optuna) intelligently selects the next hyperparameter configuration based on the results of previous trials, reducing unnecessary evaluations.

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Execution Time

---

## How to Run

Install dependencies

```bash
pip install -r requirements.txt
```

Run Random Search

```bash
python src/random_search.py
```

Run Bayesian Optimization

```bash
python src/bayesian_opt.py
```

---

## Results

After execution, compare:

- Best Hyperparameters
- Cross Validation ROC-AUC
- Test ROC-AUC
- Accuracy
- Precision
- Recall
- F1 Score
- Optimization Time

---

## Conclusion

Bayesian Optimization generally converges faster toward high-performing hyperparameter configurations, while Random Search provides a strong baseline with simpler implementation.
