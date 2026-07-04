# 🚢 Mini End-To-End AI Project using Machine Learning

## 📌 Project Overview

This project applies Machine Learning techniques to predict passenger survival on the Titanic based on demographic and travel-related information. The objective is to analyze historical passenger data, identify key factors influencing survival, and build predictive classification models.

The project demonstrates a complete Machine Learning workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, hyperparameter tuning, and business insight generation.

---

## 🎯 Objectives

* Understand and analyze the Titanic dataset.
* Perform data cleaning and preprocessing.
* Explore patterns and relationships through visualizations.
* Engineer meaningful features to improve model performance.
* Train and compare multiple classification models.
* Evaluate models using standard performance metrics.
* Generate actionable insights from the data.

---

## 📂 Project Structure

```text
MINI AI PROJECT/
│
|
├── data/
│   ├── raw_dataset.csv
│   │
│   └── final_cleaned_dataset.csv
|
│
|── Mini_AI_Project.ipynb
│
|
├── Project_Report.pdf
│
|
|
│── README.md

```

---

## 📊 Dataset Information

The dataset contains passenger information from the Titanic disaster.

### Features Used

| Feature    | Description                       |
| ---------- | --------------------------------- |
| Pclass     | Passenger Class                   |
| Sex        | Gender                            |
| Age        | Passenger Age                     |
| SibSp      | Number of Siblings/Spouses Aboard |
| Parch      | Number of Parents/Children Aboard |
| Fare       | Ticket Fare                       |
| Embarked   | Port of Embarkation               |
| FamilySize | Engineered Feature                |
| IsAlone    | Engineered Feature                |
| AgeGroup   | Engineered Feature                |

### Target Variable

| Value | Meaning         |
| ----- | --------------- |
| 0     | Did Not Survive |
| 1     | Survived        |

---

## 🔍 Exploratory Data Analysis (EDA)

The dataset was analyzed using:

* Distribution plots
* Count plots
* Box plots
* Correlation heatmaps
* Survival analysis by gender and passenger class

Key relationships between features and survival outcomes were identified through visualizations and statistical analysis.

---

## 🛠 Data Preprocessing

The following preprocessing techniques were applied:

### Missing Value Handling

* Age → Median Imputation
* Embarked → Mode Imputation
* Cabin → Removed due to excessive missing values

### Data Cleaning

* Duplicate removal
* Data type verification
* Outlier inspection

### Feature Engineering

* FamilySize
* IsAlone
* AgeGroup

### Feature Encoding

* Label Encoding for categorical variables

### Feature Scaling

* StandardScaler

---

## 🤖 Machine Learning Models

The following supervised learning algorithms were implemented:

### Logistic Regression

A baseline classification model used for binary prediction.

### Decision Tree Classifier

A tree-based model capable of capturing nonlinear relationships within the data.

---

## 📈 Model Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Cross Validation

These metrics provide a comprehensive assessment of classification performance.

---

## ⚙ Hyperparameter Tuning

GridSearchCV was applied to optimize Decision Tree parameters, including:

* Max Depth
* Min Samples Split
* Min Samples Leaf

This helped improve model generalization and reduce overfitting.

---

## 📌 Key Insights

* Female passengers had significantly higher survival rates than male passengers.
* First-class passengers were more likely to survive than passengers in lower classes.
* Higher ticket fares were associated with better survival outcomes.
* Family size influenced survival probability.
* Passengers traveling alone generally had lower survival rates.
* Age showed a moderate impact on survival likelihood.

---

## 🚀 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 📋 Workflow

1. Data Collection
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Engineering
6. Feature Encoding
7. Feature Scaling
8. Train-Test Split
9. Model Training
10. Model Evaluation
11. Hyperparameter Tuning
12. Cross Validation
13. Feature Importance Analysis
14. Insight Generation

---

## 🎓 Learning Outcomes

Through this project, the following Machine Learning concepts were applied:

* Data Preprocessing
* Feature Engineering
* Classification Algorithms
* Model Evaluation
* Hyperparameter Optimization
* Cross Validation
* Insight Generation

---

## 🔮 Future Improvements

Potential enhancements include:

* Random Forest Classifier
* XGBoost Classifier
* Ensemble Learning Techniques
* Advanced Feature Selection
* Model Deployment using Flask or Streamlit
* Interactive Dashboard Development

---


This project was developed as part of my Machine Learning learning journey to strengthen practical understanding of data preprocessing, supervised learning, model evaluation, and predictive analytics.
