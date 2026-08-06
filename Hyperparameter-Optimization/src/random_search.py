import time

import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("creditcard.csv")

# ------------------------------------------------------------
# Features and Target
# ------------------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

# ------------------------------------------------------------
# Train Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# Baseline Model
# ------------------------------------------------------------

baseline_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

baseline_start = time.time()

baseline_model.fit(X_train, y_train)

baseline_end = time.time()

baseline_predictions = baseline_model.predict(X_test)
baseline_probabilities = baseline_model.predict_proba(X_test)[:, 1]

print("=" * 60)
print("Baseline Performance")
print("=" * 60)

print(f"Accuracy : {accuracy_score(y_test, baseline_predictions):.4f}")
print(f"Precision: {precision_score(y_test, baseline_predictions):.4f}")
print(f"Recall   : {recall_score(y_test, baseline_predictions):.4f}")
print(f"F1 Score : {f1_score(y_test, baseline_predictions):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, baseline_probabilities):.4f}")
print(f"Training Time: {baseline_end-baseline_start:.2f} seconds")

# ------------------------------------------------------------
# Hyperparameter Space
# ------------------------------------------------------------

param_dist = {

    "n_estimators": [100, 200, 300, 500],

    "max_depth": [3, 5, 7, 9],

    "learning_rate": [0.01, 0.05, 0.1, 0.2],

    "subsample": [0.6, 0.8, 1.0],

    "colsample_bytree": [0.6, 0.8, 1.0],

    "gamma": [0, 0.1, 0.2, 0.3],

    "min_child_weight": [1, 3, 5]
}

# ------------------------------------------------------------
# Random Search
# ------------------------------------------------------------

random_search = RandomizedSearchCV(

    estimator=XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ),

    param_distributions=param_dist,

    n_iter=20,

    scoring="roc_auc",

    cv=5,

    verbose=1,

    random_state=42,

    n_jobs=-1

)

print("\nRunning Random Search...\n")

search_start = time.time()

random_search.fit(X_train, y_train)

search_end = time.time()

# ------------------------------------------------------------
# Best Model
# ------------------------------------------------------------

best_model = random_search.best_estimator_

predictions = best_model.predict(X_test)

probabilities = best_model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("=" * 60)
print("Random Search Results")
print("=" * 60)

print("\nBest Parameters:\n")
print(random_search.best_params_)

print("\nBest Cross Validation Score:")
print(random_search.best_score_)

print("\nExecution Time:")
print(f"{search_end-search_start:.2f} seconds")

print("\nTest Performance")

print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions):.4f}")
print(f"Recall   : {recall_score(y_test, predictions):.4f}")
print(f"F1 Score : {f1_score(y_test, predictions):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, probabilities):.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, predictions))
