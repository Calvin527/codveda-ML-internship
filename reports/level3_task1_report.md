# Level 3 Task 1: Random Forest Classifier

## 1. Task Overview

The objective of this task was to build and evaluate a Random Forest classifier on a classification dataset.

For this task, I used the customer churn dataset to predict whether a customer would churn. The Random Forest model was selected because it combines multiple decision trees and can provide strong classification performance while also allowing feature importance analysis.

## 2. Dataset Used

`churn-bigml-80.csv`

The dataset contains customer account information, service usage, call statistics, and a target variable named `Churn`.

The target variable contains two classes:

- `0`: Customer did not churn
- `1`: Customer churned

## 3. Tools and Libraries Used

- Python
- pandas
- scikit-learn
- matplotlib
- pathlib
- RandomForestClassifier
- GridSearchCV
- StratifiedKFold
- cross_val_score
- train_test_split
- accuracy_score
- precision_score
- recall_score
- f1_score
- confusion_matrix
- classification_report

## 4. Steps Completed

### 4.1 Loading the Dataset

The churn dataset was loaded using pandas.

The dataset was inspected by displaying the first rows, checking its shape, and checking for missing values.

### 4.2 Data Cleaning

Duplicate records were removed from the dataset.

Missing values were handled based on the data type:

- Numerical missing values were filled using the median.
- Categorical missing values were filled using the mode.

### 4.3 Separating Features and Target

The `Churn` column was selected as the target variable.

- `X` contained the input features.
- `y` contained the churn target.

The churn target was converted into numerical form where necessary.

### 4.4 Encoding Categorical Variables

Categorical variables were converted into numerical format using one-hot encoding with `pandas.get_dummies()`.

This allowed the Random Forest classifier to process categorical information.

### 4.5 Splitting the Dataset

The dataset was divided into training and testing sets using `train_test_split`.

An 80/20 split was used.

Stratification was applied so that the distribution of churn and non-churn customers remained similar in both datasets.

### 4.6 Training the Random Forest Model

A Random Forest classifier was created using `RandomForestClassifier`.

The model used `class_weight="balanced"` to help account for differences in the number of customers belonging to each class.

### 4.7 Hyperparameter Tuning

`GridSearchCV` was used to test different Random Forest configurations.

The main hyperparameters evaluated were:

- `n_estimators`
- `max_depth`

Different combinations were tested to identify the model configuration that produced the strongest F1-score.

### 4.8 Cross-Validation

Stratified cross-validation was used to evaluate the model across multiple subsets of the training data.

The F1-score was used as the cross-validation scoring metric.

Cross-validation provided a more reliable indication of how well the model could generalize to unseen data.

### 4.9 Model Evaluation

The best Random Forest model was evaluated on the testing dataset using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

## 5. Evaluation Results

### Best Hyperparameters

- Number of Trees (`n_estimators`): [PASTE VALUE]
- Maximum Depth (`max_depth`): [PASTE VALUE]

### Cross-Validation

- Mean Cross-Validation F1-score: [PASTE VALUE]

### Test Set Results

- Accuracy: [PASTE VALUE]
- Precision: [PASTE VALUE]
- Recall: [PASTE VALUE]
- F1-score: [PASTE VALUE]

## 6. Confusion Matrix

A confusion matrix was generated to examine the number of correct and incorrect predictions for both churn and non-churn customers.

The matrix contains:

- True Negatives
- False Positives
- False Negatives
- True Positives

This provided more detail about the model's classification performance than accuracy alone.

## 7. Feature Importance

Feature importance values were extracted from the trained Random Forest model.

These values show which variables contributed the most to the model's predictions.

The most important features were saved and visualized using a horizontal bar chart.

### Top Important Features

[PASTE TOP FEATURES FROM TERMINAL OR `feature_importance.csv`]

Feature importance analysis helped explain which customer characteristics had the strongest influence on churn prediction.

## 8. Output Files

The following output files were generated:

- `results/level3_task1/best_parameters.csv`
- `results/level3_task1/classification_report.txt`
- `results/level3_task1/confusion_matrix.csv`
- `results/level3_task1/feature_importance.csv`
- `results/level3_task1/feature_importance.png`
- `results/level3_task1/model_metrics.csv`

## 9. Conclusion

This task successfully implemented a Random Forest classifier for customer churn prediction.

The dataset was cleaned and transformed before model training. Hyperparameter tuning was performed using GridSearchCV, and cross-validation was used to evaluate model stability.

The final model was evaluated using accuracy, precision, recall, F1-score, confusion matrix, and classification report.

Feature importance analysis was also performed to identify the variables that had the strongest influence on customer churn predictions.

This task strengthened my practical understanding of ensemble learning, hyperparameter tuning, cross-validation, classification evaluation, and model interpretation.
