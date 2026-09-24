# Codveda Machine Learning Internship

This repository contains machine learning tasks completed as part of my Codveda Technologies Machine Learning Internship.

## Internship Domain

Machine Learning

## Author

Maroka Musa Calvin

---

## Project Overview

This repository demonstrates practical machine learning skills developed through the Codveda Technologies Machine Learning Internship.

The internship required the completion of two selected tasks from each level. The completed work progresses from fundamental data preprocessing and regression techniques to intermediate classification models and advanced ensemble and kernel-based machine learning methods.

The project covers the complete machine learning workflow, including:

- Data loading and inspection
- Data cleaning and preprocessing
- Missing value handling
- Categorical feature encoding
- Feature standardization
- Train/test splitting
- Regression modelling
- Binary and multiclass classification
- Hyperparameter tuning
- Cross-validation
- Feature importance analysis
- Model evaluation
- ROC-AUC analysis
- Decision boundary visualization
- Model interpretation
- Result visualization
- Technical documentation

---

# Completed Tasks

## Level 1 — Basic

### Level 1 Task 1: Data Preprocessing for Machine Learning

#### Objective

The objective of this task was to preprocess a raw dataset and prepare it for machine learning.

#### Dataset Used

`churn-bigml-80.csv`

#### Description

The customer churn dataset was cleaned and transformed into a format suitable for machine learning. The dataset contains both numerical and categorical features, making it suitable for demonstrating common preprocessing techniques.

#### Steps Completed

- Loaded the dataset using pandas
- Displayed the first rows of the dataset
- Inspected dataset shape and data types
- Checked for missing values
- Handled missing values
- Encoded categorical variables using one-hot encoding
- Standardized numerical features using `StandardScaler`
- Separated features and target variable
- Split the dataset into training and testing sets
- Saved the processed datasets

#### Output Files

- `results/level1_task1/train_preprocessed.csv`
- `results/level1_task1/test_preprocessed.csv`

#### Report

- `reports/level1_task1_report.md`

---

### Level 1 Task 2: Simple Linear Regression Model

#### Objective

The objective of this task was to build a Linear Regression model to predict a continuous variable.

#### Dataset Used

`house_prediction.csv`

#### Description

A house prediction dataset was used to train a Linear Regression model for predicting house prices from available property-related features.

#### Steps Completed

- Loaded the house prediction dataset
- Inspected dataset structure and data types
- Checked for missing values
- Removed duplicate records
- Handled missing values
- Separated features and target variable
- Encoded categorical variables where required
- Standardized numerical features using `StandardScaler`
- Split the dataset into training and testing sets
- Trained a Linear Regression model
- Generated predictions on unseen test data
- Extracted and interpreted model coefficients
- Evaluated model performance
- Visualized actual versus predicted prices
- Saved model results

#### Evaluation Metrics

- Mean Squared Error (MSE)
- R-squared Score

#### Output Files

- `results/level1_task2/model_coefficients.csv`
- `results/level1_task2/predictions.csv`
- `results/level1_task2/actual_vs_predicted.png`

#### Report

- `reports/level1_task2_report.md`

---

# Level 2 — Intermediate

## Level 2 Task 1: Logistic Regression for Binary Classification

### Objective

The objective of this task was to implement a Logistic Regression model for binary classification.

### Dataset Used

`churn-bigml-80.csv`

### Description

The customer churn dataset was used to predict whether a customer would leave or remain with the service.

The target variable contains two possible outcomes:

- `0` — Customer did not churn
- `1` — Customer churned

### Steps Completed

- Loaded and inspected the churn dataset
- Checked data types and missing values
- Removed duplicate records
- Handled missing values
- Separated features and target variable
- Converted the `Churn` target to numerical values
- Encoded categorical variables using one-hot encoding
- Standardized numerical features using `StandardScaler`
- Split the dataset into training and testing sets
- Trained a Logistic Regression model
- Generated predictions
- Evaluated classification performance
- Generated a confusion matrix
- Generated a classification report
- Created a ROC curve
- Calculated ROC-AUC
- Extracted model coefficients
- Calculated and interpreted odds ratios

### Evaluation Metrics

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

## Level 2 Task 2: Decision Trees for Classification

### Objective

The objective of this task was to build, visualize, evaluate, and prune a Decision Tree classifier.

### Dataset Used

`iris.csv`

### Description

The Iris dataset was used to classify flower species based on their physical measurements.

Both an original Decision Tree and a pruned Decision Tree were created to examine the effect of limiting tree depth and reducing model complexity.

### Steps Completed

- Loaded the Iris dataset
- Inspected dataset structure
- Checked missing values and data types
- Removed duplicate records
- Handled missing values
- Separated features and target variable
- Split the dataset into training and testing sets
- Trained an original Decision Tree classifier
- Generated predictions
- Evaluated classification performance
- Generated a confusion matrix
- Generated a classification report
- Visualized the original tree
- Created a pruned tree using `max_depth=3`
- Evaluated the pruned model
- Compared original and pruned model performance
- Extracted feature importance values
- Saved tree visualizations and evaluation results

### Evaluation Metrics

- Accuracy
- F1-score
- Classification Report
- Confusion Matrix

### Output Files

- `results/level2_task2/classification_report.txt`
- `results/level2_task2/confusion_matrix.csv`
- `results/level2_task2/feature_importance.csv`
- `results/level2_task2/original_decision_tree.png`
- `results/level2_task2/pruned_decision_tree.png`
- `results/level2_task2/decision_tree_metrics.csv`

### Report

- `reports/level2_task2_report.md`

---

# Level 3 — Advanced

## Level 3 Task 1: Random Forest Classifier

### Objective

The objective of this task was to build and optimize a Random Forest classifier for customer churn prediction.

### Dataset Used

`churn-bigml-80.csv`

### Description

A Random Forest classifier was trained on the customer churn dataset.

The task focused on ensemble learning, hyperparameter tuning, cross-validation, classification evaluation, and feature importance analysis.

### Steps Completed

- Loaded and inspected the churn dataset
- Removed duplicate records
- Handled missing values
- Separated features and target variable
- Encoded categorical variables
- Split the dataset into training and testing sets
- Created a Random Forest classifier
- Applied balanced class weighting
- Tuned `n_estimators`
- Tuned `max_depth`
- Used `GridSearchCV` for hyperparameter selection
- Applied stratified cross-validation
- Evaluated cross-validation F1-score
- Evaluated the final model on unseen test data
- Generated a confusion matrix
- Generated a classification report
- Extracted feature importance values
- Visualized the most important features
- Saved model metrics and best hyperparameters

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Cross-Validation F1-score
- Confusion Matrix
- Classification Report

### Output Files

- `results/level3_task1/best_parameters.csv`
- `results/level3_task1/classification_report.txt`
- `results/level3_task1/confusion_matrix.csv`
- `results/level3_task1/feature_importance.csv`
- `results/level3_task1/feature_importance.png`
- `results/level3_task1/model_metrics.csv`

### Report

- `reports/level3_task1_report.md`

---

## Level 3 Task 2: Support Vector Machine Classification

### Objective

The objective of this task was to build Support Vector Machine models for binary classification and compare different kernel functions.

### Dataset Used

`churn-bigml-80.csv`

### Description

Support Vector Machine models were trained to predict customer churn.

Two kernels were evaluated:

- Linear Kernel
- Radial Basis Function (RBF) Kernel

The models were compared using several classification metrics and ROC-AUC analysis.

### Steps Completed

- Loaded and inspected the churn dataset
- Removed duplicate records
- Handled missing values
- Separated features and target variable
- Converted the target variable into numerical format
- Encoded categorical features
- Split the dataset into training and testing sets
- Standardized features using `StandardScaler`
- Trained a Linear SVM model
- Trained an RBF SVM model
- Generated predictions for both models
- Compared Linear and RBF kernel performance
- Evaluated accuracy, precision, recall and F1-score
- Calculated ROC-AUC scores
- Generated classification reports
- Generated confusion matrices
- Generated ROC curves
- Applied Principal Component Analysis for 2D visualization
- Visualized the SVM decision boundary
- Saved model comparison results

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

### Output Files

- `results/level3_task2/linear_svm_classification_report.txt`
- `results/level3_task2/linear_svm_confusion_matrix.csv`
- `results/level3_task2/rbf_svm_classification_report.txt`
- `results/level3_task2/rbf_svm_confusion_matrix.csv`
- `results/level3_task2/svm_model_comparison.csv`
- `results/level3_task2/svm_roc_curve.png`
- `results/level3_task2/svm_decision_boundary.png`

### Report

- `reports/level3_task2_report.md`

---

# Tools and Libraries Used

## Programming Language

- Python

## Data Processing

- pandas
- NumPy

## Machine Learning

- scikit-learn

### Models and Algorithms

- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- Principal Component Analysis

### Preprocessing

- `StandardScaler`
- One-Hot Encoding
- Train/Test Splitting

### Model Selection and Validation

- `GridSearchCV`
- `StratifiedKFold`
- Cross-Validation

### Evaluation

- Mean Squared Error
- R-squared
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

## Visualization

- matplotlib
- seaborn

## Development and Version Control

- pathlib
- VS Code
- Git
- GitHub

---

# Project Structure

```text
codveda-ML-internship/
│
├── data/
│   ├── churn-bigml-80.csv
│   ├── house_prediction.csv
│   └── iris.csv
│
├── notebooks/
│   ├── level1_task1_preprocessing.py
│   ├── level1_task2_linear_regression.py
│   ├── level2_task1_logistic_regression.py
│   ├── level2_task2_decision_tree.py
│   ├── level3_task1_random_forest.py
│   └── level3_task2_svm.py
│
├── results/
│   │
│   ├── level1_task1/
│   │   ├── train_preprocessed.csv
│   │   └── test_preprocessed.csv
│   │
│   ├── level1_task2/
│   │   ├── model_coefficients.csv
│   │   ├── predictions.csv
│   │   └── actual_vs_predicted.png
│   │
│   ├── level2_task1/
│   │   ├── logistic_regression_coefficients.csv
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.csv
│   │   └── roc_curve.png
│   │
│   ├── level2_task2/
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.csv
│   │   ├── feature_importance.csv
│   │   ├── original_decision_tree.png
│   │   ├── pruned_decision_tree.png
│   │   └── decision_tree_metrics.csv
│   │
│   ├── level3_task1/
│   │   ├── best_parameters.csv
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.csv
│   │   ├── feature_importance.csv
│   │   ├── feature_importance.png
│   │   └── model_metrics.csv
│   │
│   └── level3_task2/
│       ├── linear_svm_classification_report.txt
│       ├── linear_svm_confusion_matrix.csv
│       ├── rbf_svm_classification_report.txt
│       ├── rbf_svm_confusion_matrix.csv
│       ├── svm_model_comparison.csv
│       ├── svm_roc_curve.png
│       └── svm_decision_boundary.png
│
├── reports/
│   ├── level1_task1_report.md
│   ├── level1_task2_report.md
│   ├── level2_task1_report.md
│   ├── level2_task2_report.md
│   ├── level3_task1_report.md
│   └── level3_task2_report.md
│
├── requirements.txt
└── README.md
```

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Calvin527/codveda-ML-internship.git
```

## 2. Navigate to the Project Folder

```bash
cd codveda-ML-internship
```

## 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

## 4. Run Level 1 Task 1

```bash
python notebooks/level1_task1_preprocessing.py
```

## 5. Run Level 1 Task 2

```bash
python notebooks/level1_task2_linear_regression.py
```

## 6. Run Level 2 Task 1

```bash
python notebooks/level2_task1_logistic_regression.py
```

## 7. Run Level 2 Task 2

```bash
python notebooks/level2_task2_decision_tree.py
```

## 8. Run Level 3 Task 1

```bash
python notebooks/level3_task1_random_forest.py
```

## 9. Run Level 3 Task 2

```bash
python notebooks/level3_task2_svm.py
```

---

# Internship Task Summary

| Level   | Task                   | Machine Learning Area   | Status    |
| ------- | ---------------------- | ----------------------- | --------- |
| Level 1 | Data Preprocessing     | Data Preparation        | Completed |
| Level 1 | Linear Regression      | Regression              | Completed |
| Level 2 | Logistic Regression    | Binary Classification   | Completed |
| Level 2 | Decision Tree          | Classification          | Completed |
| Level 3 | Random Forest          | Ensemble Classification | Completed |
| Level 3 | Support Vector Machine | Advanced Classification | Completed |

---

# Key Skills Demonstrated

Through these tasks, I gained practical experience in:

- Machine learning workflow development
- Data preprocessing and feature preparation
- Regression modelling
- Binary and multiclass classification
- Ensemble learning
- Decision trees and pruning
- Support Vector Machines
- Linear and non-linear kernels
- Hyperparameter tuning
- Cross-validation
- Handling class imbalance
- Feature importance analysis
- Model evaluation
- ROC-AUC analysis
- PCA-based visualization
- Machine learning result interpretation
- Technical documentation
- Git and GitHub version control

---

# GitHub Repository

[Codveda Machine Learning Internship Repository](https://github.com/Calvin527/codveda-ML-internship)

---

# Conclusion

This repository documents the machine learning work completed during my Codveda Technologies Machine Learning Internship.

Two tasks were completed at each internship level, progressing from fundamental data preprocessing and regression to intermediate classification and advanced machine learning models.

The internship strengthened my practical understanding of the complete machine learning workflow, including preparing datasets, selecting features, training models, tuning hyperparameters, applying cross-validation, evaluating model performance, interpreting predictions, visualizing results, and documenting technical work.

The completed tasks provided practical experience with Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Git, and GitHub while strengthening my foundations in Data Science and Machine Learning.
