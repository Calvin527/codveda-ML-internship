# Codveda Machine Learning Internship

This repository contains machine learning tasks completed as part of my Codveda Technologies Machine Learning Internship.

## Internship Domain

Machine Learning

## Author

Maroka Musa Calvin

---

## Project Overview

The purpose of this project is to demonstrate practical machine learning skills using Python and popular data science libraries. The completed tasks cover important machine learning workflow stages such as data preprocessing, regression modelling, classification modelling, model evaluation, and result interpretation.

The project includes tasks from different internship levels, starting from basic data preparation and progressing to supervised machine learning models.

---

## Completed Tasks

## Level 1 Task 1: Data Preprocessing for Machine Learning

### Objective

The objective of this task was to preprocess a raw dataset and prepare it for machine learning.

### Dataset Used

`churn-bigml-80.csv`

### Description

In this task, I prepared the churn dataset for machine learning by cleaning and transforming the data. The dataset was selected because it contains both numerical and categorical features, making it suitable for demonstrating common preprocessing techniques.

### Steps Completed

- Loaded the dataset using pandas
- Displayed the first rows of the dataset
- Checked dataset shape, data types, and missing values
- Handled missing values
- Encoded categorical variables using one-hot encoding
- Standardized numerical features using `StandardScaler`
- Split the dataset into training and testing sets
- Saved the processed training and testing datasets

### Output Files

- `results/level1_task1/train_preprocessed.csv`
- `results/level1_task1/test_preprocessed.csv`

### Report

- `reports/level1_task1_report.md`

---

## Level 1 Task 2: Simple Linear Regression Model

### Objective

The objective of this task was to build a Simple Linear Regression model to predict a continuous value.

### Dataset Used

`house_prediction.csv`

### Description

In this task, I used a house prediction dataset to train a Linear Regression model. The goal was to predict house prices using available house-related features.

### Steps Completed

- Loaded the house prediction dataset
- Inspected the dataset structure
- Checked missing values and data types
- Removed duplicate rows
- Handled missing values
- Separated input features and target variable
- Encoded categorical variables if present
- Standardized numerical features using `StandardScaler`
- Split the dataset into training and testing sets
- Trained a Linear Regression model
- Made predictions on the test dataset
- Evaluated the model using Mean Squared Error and R-squared
- Saved model coefficients, predictions, and visualization

### Evaluation Metrics Used

- Mean Squared Error
- R-squared Score

### Output Files

- `results/level1_task2/model_coefficients.csv`
- `results/level1_task2/predictions.csv`
- `results/level1_task2/actual_vs_predicted.png`

### Report

- `reports/level1_task2_report.md`

---

## Level 2 Task 1: Logistic Regression for Binary Classification

### Objective

The objective of this task was to implement a Logistic Regression model for binary classification.

### Dataset Used

`churn-bigml-80.csv`

### Description

In this task, I used the churn dataset to predict whether a customer is likely to churn or not. This is a binary classification problem because the target variable has two possible outcomes:

- `0`: Customer did not churn
- `1`: Customer churned

### Steps Completed

- Loaded the churn dataset
- Inspected the dataset structure
- Checked missing values and data types
- Removed duplicate rows
- Handled missing values
- Separated input features and target variable
- Converted the `Churn` target column into numerical values
- Encoded categorical variables using one-hot encoding
- Standardized numerical features using `StandardScaler`
- Split the dataset into training and testing sets
- Trained a Logistic Regression model
- Made predictions on the test dataset
- Evaluated the model using classification metrics
- Generated a confusion matrix
- Generated a classification report
- Created a ROC curve
- Interpreted model coefficients and odds ratios

### Evaluation Metrics Used

- Accuracy
- Precision
- Recall
- Confusion Matrix
- Classification Report
- ROC-AUC Score

### Output Files

- `results/level2_task1/logistic_regression_coefficients.csv`
- `results/level2_task1/classification_report.txt`
- `results/level2_task1/confusion_matrix.csv`
- `results/level2_task1/roc_curve.png`

### Report

- `reports/level2_task1_report.md`

---

## Tools and Libraries Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn

---

## Project Structure

```text
codveda-ML-internship/
│
├── data/
│   ├── churn-bigml-80.csv
│   └── house_prediction.csv
│
├── notebooks/
│   ├── level1_task1_preprocessing.py
│   ├── level1_task2_linear_regression.py
│   └── level2_task1_logistic_regression.py
│
├── results/
│   ├── level1_task1/
│   │   ├── train_preprocessed.csv
│   │   └── test_preprocessed.csv
│   │
│   ├── level1_task2/
│   │   ├── model_coefficients.csv
│   │   ├── predictions.csv
│   │   └── actual_vs_predicted.png
│   │
│   └── level2_task1/
│       ├── logistic_regression_coefficients.csv
│       ├── classification_report.txt
│       ├── confusion_matrix.csv
│       └── roc_curve.png
│
├── reports/
│   ├── level1_task1_report.md
│   ├── level1_task2_report.md
│   └── level2_task1_report.md
│
├── requirements.txt
└── README.md
```
