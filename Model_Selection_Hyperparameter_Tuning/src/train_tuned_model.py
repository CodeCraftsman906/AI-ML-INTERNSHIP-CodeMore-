import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Load Dataset


column_names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

df = pd.read_csv(
    "data/adult_train.csv",
    header=None,
    names=column_names,
    na_values="?",
    skipinitialspace=True
)

# ==========================================================
# Data Preprocessing
# ==========================================================

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df["income"] = df["income"].str.replace(".", "", regex=False)

# ==========================================================
# Separate Features and Target
# ==========================================================

X = df.drop("income", axis=1)
y = df["income"]

# ==========================================================
# One-Hot Encoding
# ==========================================================

X = pd.get_dummies(
    X,
    drop_first=True
)

# ==========================================================
# Encode Target
# ==========================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================================
# Initialize Best Model
# Replace these hyperparameters with your tuned values
# ==========================================================

best_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# ==========================================================
# Train Model
# ==========================================================

best_model.fit(X_train, y_train)

# ==========================================================
# Predictions
# ==========================================================

predictions = best_model.predict(X_test)

# ==========================================================
# Model Evaluation
# ==========================================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("=" * 60)
print("Model Performance")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, predictions))


# ==========================================================
# Save Model
# ==========================================================

joblib.dump(best_model, "models/best_model.pkl")

print("\nBest model saved successfully!")