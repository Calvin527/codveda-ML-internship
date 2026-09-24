## 1. Task Overview

The objective of this task was to build a Logistic Regression model for binary classification.

The model was trained to predict whether a customer would churn or not. This is a binary classification problem because the target column has two possible outcomes:

- 0: Customer did not churn
- 1: Customer churned

## 2. Dataset Used

The dataset used for this task was:

`churn-bigml-80.csv`

This dataset was selected because it contains customer-related features and a target column called `Churn`, making it suitable for binary classification.

## 3. Tools and Libraries Used

The following tools and libraries were used:

- Python
- pandas
- scikit-learn
- matplotlib
- LogisticRegression
- StandardScaler
- train_test_split

## 4. Steps Completed

### 4.1 Loading the Dataset

The dataset was loaded using pandas. The first few rows were displayed to understand the structure of the data.

### 4.2 Inspecting the Dataset

The dataset shape, column data types, and missing values were checked before preprocessing.

### 4.3 Data Cleaning

Duplicate rows were removed, and missing values were handled.

- Numerical missing values were filled using the median.
- Categorical missing values were filled using the mode.

### 4.4 Separating Features and Target

The `Churn` column was used as the target variable.

- `X` contained the input features.
- `y` contained the target values.

The target values were converted into numerical form:

- False became 0
- True became 1

### 4.5 Encoding Categorical Variables

Categorical variables were converted into numerical format using one-hot encoding. This allowed the Logistic Regression model to process the data.

### 4.6 Scaling the Features

The input features were standardized using `StandardScaler`. Scaling helps improve model performance by placing numerical values on a similar scale.

### 4.7 Splitting the Dataset

The dataset was split into training and testing sets.

- Training set: used to train the model
- Testing set: used to evaluate the model

### 4.8 Training the Logistic Regression Model

A Logistic Regression model was trained using the training data. Logistic Regression was used because it is suitable for binary classification problems.

### 4.9 Model Evaluation

The model was evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- Confusion matrix
- Classification report
- ROC-AUC score

These metrics helped measure how well the model predicted customer churn.

### 4.10 Coefficients and Odds Ratios

The model coefficients were extracted and converted into odds ratios. This helped interpret how each feature influenced the probability of customer churn.

## 5. Output Files

The task produced the following output files:

- `results/level2_task1/logistic_regression_coefficients.csv`
- `results/level2_task1/classification_report.txt`
- `results/level2_task1/confusion_matrix.csv`
- `results/level2_task1/roc_curve.png`

## 6. Conclusion

This task successfully implemented a Logistic Regression model for binary classification. The churn dataset was cleaned, encoded, scaled, and split into training and testing sets. The model was trained and evaluated using classification metrics, and the results were saved for reporting and submission.
