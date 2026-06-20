# House Price Prediction Dataset – Data Preprocessing & Exploratory Data Analysis (EDA)

The project demonstrates the complete data preprocessing and exploratory data analysis (EDA) pipeline using Python.

# Project Objectives

The objectives of this project are:

Load the housing dataset into Python using Pandas.
Detect and handle missing values.
Identify and remove duplicate records.
Verify and correct incorrect data types.
Perform exploratory data analysis (EDA).
Calculate summary statistics.
Identify relationships between variables.
Visualize the dataset using different graphs.
Export the cleaned dataset for future machine learning tasks.

# Dataset Information

**Dataset Name** -  House Price Prediction Dataset

# Dataset Source

The dataset is obtained from **Kaggle**, a popular platform for data science and machine learning datasets.

Source:
**https://www.kaggle.com/**

The dataset contains information about residential properties such as:

**Property Area
Number of Rooms
Build Year
Location
Street Type
Furnishing Status
Property Type
Swimming Pool Availability
Property Price**


# Step-by-Step Workflow

**Step 1 — Import Required Libraries**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

**Step 2 — Load the Dataset**

df = pd.read_csv("raw_dataset.csv")

The dataset is stored inside a Pandas DataFrame named **df**.

**Step 3 — Explore the Dataset**

df.head()
df.shape
df.columns
df.dtypes

These commands provide:

**Number of rows
Number of columns
Column names
Data types**

**Step 4 — Identify Missing Values**

df.isnull().sum()

This helps identify incomplete records in each column.

Depending on the dataset, missing values can be:

removed
replaced by mean
replaced by median
replaced by mode

**Step 5 — Identify Duplicate Rows**

Duplicate records are detected using

df.duplicated()

**Step 6 — Remove Duplicate Rows**

Duplicate records are removed using

df.drop_duplicates()

**Step 8 — Verify Data Types**

Each column's data type is verified using

df.dtypes

**Step 8 — Exploratory Data Analysis (EDA)**

EDA is performed to understand hidden patterns inside the dataset.

This includes:

**summary statistics**
**distributions**
**trends**
**relationships**

**Step 9 — Summary Statistics**

df.describe()

Important statistical measures are calculated:

**Mean
Median
Standard Deviation
Minimum
Maximum**

**Step 10 — Correlation Analysis**

Correlation helps determine how strongly two numerical variables are related.

Calculated using

df.corr()

The results are visualized using a heatmap.


**Step 11 — Data Visualization**

Several visualizations are created.

**(a) Histograms**

Purpose:

Understand feature distribution.

**(b) Boxplots**

Purpose:

Detect outliers.

**(c) Scatter Plots**

Purpose:

Identify relationships between variables.

Example:

Area vs Price

**(d) Countplots**

Purpose:

Visualize categorical variables.

Example:

Property Type

Location

Has Pool

**(e) Heatmap**

Purpose:

Visualize correlations among numerical variables.

**Step 13 — Export Cleaned Dataset**

After preprocessing, the cleaned dataset is exported.

df.to_csv("cleaned_dataset.csv", index=False)


# How to Run the Project (Step-by-Step)

**Step 1: Install Python**

Download and install Python 3.10 or later from the official website:

https://www.python.org/downloads/

During installation, check **"Add Python to PATH"**.

**Step 2: Install Jupyter Notebook**

Open Command Prompt or PowerShell and run:

pip install notebook

Launch Jupyter Notebook:

jupyter notebook

**Step 3: Install Required Libraries**

Install all required libraries using:

pip install pandas numpy matplotlib seaborn

Or install them individually:

pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn

**Step 4: Prepare the Project Folder**

Create a folder, for example:

House_Price_Project/
│
├── raw_dataset.csv
├── HousePriceEDA.ipynb

Place **raw_dataset.csv** in the same folder as your Jupyter Notebook.

**Step 5: Open the Notebook**

Start Jupyter Notebook:

jupyter notebook

Navigate to your project folder and open **HousePriceEDA.ipynb**.

Step 6: Import Libraries

Run the first cell:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

If there are no errors, the environment is ready.

**Step 7: Load the Dataset**

Run:

df = pd.read_csv("raw_dataset.csv")

Verify the data:

df.head()

**Step 8: Execute the Notebook**

Run each cell in order:

Inspect dataset structure
Handle missing values
Remove duplicate rows
Verify data types
Generate summary statistics
Create visualizations
Analyze correlations

**Step 9: Export the Cleaned Dataset**

After preprocessing, save the cleaned dataset:

df.to_csv("cleaned_dataset.csv", index=False)

This creates a new file named:

**cleaned_dataset.csv**

in the same project folder.

Step 10: Verify the Export

Reload the cleaned dataset to confirm it was saved correctly:

cleaned_df = pd.read_csv("cleaned_dataset.csv")
print(cleaned_df.head())
print(cleaned_df.shape)

If the data loads successfully, the preprocessing and EDA pipeline has been completed.










