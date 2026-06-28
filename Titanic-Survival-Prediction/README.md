# Titanic Survival Prediction using Logistic Regression

## Project Overview

This project demonstrates the complete Machine Learning workflow using the Titanic Survival Dataset. The objective is to predict whether a passenger survived the Titanic disaster based on passenger attributes such as age, gender, passenger class, fare, and family information.

The project was developed as part of an AI/ML internship task to gain practical experience in data preprocessing, exploratory data analysis, feature selection, model training, testing, and evaluation.

## Dataset

Dataset: Titanic Survival Dataset

Target Variable:

* Survived

  * 0 = Did Not Survive
  * 1 = Survived

Features Used:

* Pclass
* Sex
* Age
* SibSp
* Parch
* Fare
* Embarked

Removed Features:

* PassengerId
* Name
* Ticket
* Cabin

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Project Workflow

### 1. Data Loading

The dataset was loaded into a Jupyter Notebook using Pandas.

### 2. Exploratory Data Analysis

Basic statistical analysis was performed to understand:

* Dataset structure
* Missing values
* Feature distributions
* Survival trends

### 3. Data Preprocessing

The following preprocessing steps were applied:

* Missing value handling
* Feature selection
* Categorical variable encoding
* Data cleaning

### 4. Train-Test Split

The dataset was split into:

* Training Data: 80%
* Testing Data: 20%

### 5. Model Training

A Logistic Regression model was trained using Scikit-learn.

### 6. Model Evaluation

The model was evaluated using:

* Accuracy Score
* F1 Score
* Classification Report
* Confusion Matrix

## Results

Model: Logistic Regression

Evaluation Metrics:

* Accuracy: 0.7988826815642458
* F1 Score: 0.75

The model successfully learned survival patterns from the dataset and achieved satisfactory classification performance.

## Key Findings

* Female passengers had higher survival rates than male passengers.
* Passenger class significantly influenced survival probability.
* Younger passengers generally showed higher survival rates.
* Logistic Regression provided a strong baseline classification model.

## Limitations

* Limited feature engineering.
* Logistic Regression assumes a linear relationship between features and target.
* Some potentially useful information was removed during preprocessing.
* Performance may be improved using ensemble learning techniques.

## Future Improvements

* Hyperparameter tuning using GridSearchCV.
* Cross-validation.
* Feature scaling and engineering.
* Random Forest Classifier.
* Gradient Boosting Models.
* XGBoost implementation.

## Installation

Clone the repository:

git clone <repository-link>

Move into the project directory:

cd Titanic-Survival-Prediction

Install dependencies:

pip install -r requirements.txt

Run the notebook:

jupyter notebook

Open:

notebooks/ml_model_implementation.ipynb
