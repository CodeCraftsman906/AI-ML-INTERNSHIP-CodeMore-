# The objective of this project was to build a supervised machine learning classification model capable of predicting the target variable using the available features in the dataset.
# The project follows the complete machine learning workflow, including model training, evaluation, comparison, and optimization.


# WORKFLOW OF THE PROJECT

 Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Processed Dataset
      │
      ▼
Train-Test Split (80:20)
      │
      ▼
Train Logistic Regression
      │
      ▼
Train Decision Tree
      │
      ▼
Evaluate Both Models
      │
      ▼
Compare Performance
      │
      ▼
Select Best Model
      │
      ▼
Hyperparameter Tuning (GridSearchCV)
      │
      ▼
Optimized Logistic Regression
      │
      ▼
Final Evaluation


# LIBRARIES USED
**pandas
numpy
matplotlib
scikit-learn**


# DATASET OVERVIEW
**Dataset Size**: 5,000 records
**Total Features**: 28 columns
**Target Variable**: Feature13
**Problem Type**: Classification

The processed datset was loaded from the previous feature engineering task
**processed_dataset.csv**


# Train-Test Split

The processed dataset was divided into training and testing sets.

Training Data → 80%
Testing Data → 20%

The training set is used to learn patterns, while the testing set evaluates the model's performance on unseen data.


# Selection of Supervised Learning Models

Since the target variable is categorical, the project used **Supervised Classification** algorithms.

Two baseline models were selected:

**Model 1 — Logistic Regression**

Chosen because:

Simple and interpretable
Fast to train
Works well as a baseline classifier
Performs well on linearly separable data

**Model 2 — Decision Tree Classifier**

Chosen because:

Captures nonlinear relationships
Easy to interpret
Requires minimal assumptions about the data
Serves as a strong baseline for comparison


# Model Training

Both models were trained using the training dataset.

During Logistic Regression training, the following issue occurred:

**Challenge:**
ValueError:
could not convert string to float: 'High'

Solution:
The categorical values in Feature19 were encoded into numerical values, and the dataset was split again before retraining the model.


# Convergence Warning

During Logistic Regression training, another issue occurred:

Total number of iterations reached limit

Cause: The optimization algorithm required more iterations to find the best solution.

Solution: The max_iter parameter was increased (e.g., to 5000), allowing the algorithm to converge successfully.


# Model Evaluation

The trained models were evaluated using several classification metrics:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

**Baseline Results**
| Model               |  Accuracy |
| ------------------- | --------: |
| Logistic Regression | **61.1%** |
| Decision Tree       | **44.7%** |

**Logistic Regression** outperformed the Decision Tree and was selected for further optimization.


# Confusion Matrix

A confusion matrix was generated for each model to evaluate classification performance.

It helped identify:

Correct predictions
Incorrect predictions
Misclassified classes
Overall prediction quality


# Hyperparameter Tuning

After identifying Logistic Regression as the better baseline model, hyperparameter tuning was performed using **GridSearchCV**.

**Hyperparameters Tuned**:
C
solver
penalty
max_iter


# Summary of Model Evaluation Results

Two baseline supervised learning algorithms **Logistic Regression** and **Decision Tree Classifier** were trained and evaluated on the processed dataset. Their performance was measured using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix**.

| Metric    | Logistic Regression | Decision Tree |
| --------- | ------------------: | ------------: |
| Accuracy  |               61.1% |         44.7% |
| Precision |               37.3% |         47.1% |
| Recall    |               61.1% |         44.7% |
| F1-Score  |               46.3% |         45.8% |

# Interpretation

-> **Logistic Regression** achieved the **highest overall accuracy (61.1%)**, along with better Recall and F1-Score, indicating stronger overall classification performance on the test dataset.
-> **Decision Tree Classifier** achieved a slightly higher Precision but performed worse in terms of Accuracy, Recall, and F1-Score, suggesting that it did not generalize as effectively as Logistic Regression.
-> **Confusion matrices** were generated for both models to visualize correct and incorrect predictions and to identify class-wise misclassifications.
-> Based on the evaluation metrics, **Logistic Regression** was selected as the best-performing baseline model.
-> To further improve its performance, **GridSearchCV** was applied for hyperparameter tuning, optimizing parameters such as C, solver, and max_iter through 5-fold cross-validation.
