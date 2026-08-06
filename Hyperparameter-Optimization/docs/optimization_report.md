# Hyperparameter Optimization Report

## Introduction

Hyperparameter optimization plays an important role in improving machine learning model performance. This project compares Random Search and Bayesian Optimization for tuning an XGBoost classifier on the Credit Card Fraud Detection dataset.

---

# Dataset

- Dataset: Credit Card Fraud Detection
- Target Variable: Class
- Train-Test Split: 80:20
- Cross Validation: 5-Fold

---

# Baseline Model

A default XGBoost model was trained before performing any optimization.

## Performance

| Metric | Value |
|---------|------:|
| Accuracy |0.9994|
| Precision|0.8667|
| Recall   |0.7959|
| F1 Score |0.8298|
| ROC-AUC  |0.9390|
| Training Time |1.78 seconds|

---

# Random Search

Random Search explores randomly selected hyperparameter combinations from a predefined search space.

## Search Space

- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- gamma
- min_child_weight

### Best Parameters

```
{'subsample': 0.8, 'n_estimators': 200, 'min_child_weight': 5, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0.2, 'colsample_bytree': 0.8}
```

### Best Cross Validation Score:

'''
0.9874736774056008
'''


### Performance

| Metric | Value |
|---------|------:|
| Accuracy |0.9996|
| Precision|0.9518|
| Recall   |0.8061|
| F1 Score |0.8729|
| ROC-AUC  |0.9732|
| Optimization Time |237.27 sec|

---

# Bayesian Optimization

Bayesian Optimization uses Optuna to intelligently explore promising hyperparameter combinations based on previous evaluations.

### Best Parameters

```
{'n_estimators': 465, 'max_depth': 7, 'learning_rate': 0.02652938162277112, 'subsample': 0.9214530022769365, 'colsample_bytree': 0.7331839377303674, 'gamma': 0.16452919857268639, 'min_child_weight': 7
```

### Performance

| Metric | Value |
|---------|------:|
| Accuracy |0.9996|
| Precision|0.9412|
| Recall   |0.8163|
| F1 Score |0.8743|
| ROC-AUC  |0.9787|
| Optimization Time |685.13 sec|

---

# Observations

- Random Search evaluates randomly selected hyperparameter combinations.
- Bayesian Optimization learns from previous trials to identify better configurations.
- Bayesian Optimization generally requires fewer evaluations to achieve competitive performance.
- Random Search is simple to implement and serves as an effective baseline optimization strategy.

---

# Conclusion

Both optimization techniques improved the baseline XGBoost model. Random Search efficiently explored the search space through random sampling, whereas Bayesian Optimization demonstrated a more intelligent search strategy by focusing on promising regions of the hyperparameter space. Overall, Bayesian Optimization achieved competitive performance with a more efficient exploration process, making it a preferred approach for complex hyperparameter tuning tasks.
