import time

import optuna
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
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
    test_size=0.20,
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
print(f"Training Time: {baseline_end - baseline_start:.2f} seconds")


# ------------------------------------------------------------
# Objective Function
# ------------------------------------------------------------

def objective(trial):

    params = {

        "n_estimators": trial.suggest_int("n_estimators", 100, 500),

        "max_depth": trial.suggest_int("max_depth", 3, 10),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            0.5
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10
        ),

        "random_state": 42,

        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params)

    score = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    return score.mean()


# ------------------------------------------------------------
# Bayesian Optimization
# ------------------------------------------------------------

print("\nRunning Bayesian Optimization...\n")

study = optuna.create_study(direction="maximize")

optimization_start = time.time()

study.optimize(
    objective,
    n_trials=30,
    show_progress_bar=True
)

optimization_end = time.time()

# ------------------------------------------------------------
# Train Best Model
# ------------------------------------------------------------

best_model = XGBClassifier(
    **study.best_params,
    random_state=42,
    eval_metric="logloss"
)

best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)

probabilities = best_model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("=" * 60)
print("Bayesian Optimization Results")
print("=" * 60)

print("\nBest Parameters:\n")
print(study.best_params)

print("\nBest Cross Validation ROC-AUC:")
print(f"{study.best_value:.4f}")

print("\nOptimization Time:")
print(f"{optimization_end - optimization_start:.2f} seconds")

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