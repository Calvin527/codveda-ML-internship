# Level 1 Task 1: Data Preprocessing for Machine Learning

## 1. Task Overview

The objective of this task was to preprocess a raw dataset and prepare it for machine learning. The task required handling missing data, encoding categorical variables, standardizing numerical features, and splitting the dataset into training and testing sets.

## 2. Dataset Used

For this task, I used the churn dataset: `churn-bigml-80.csv`.

This dataset was selected because it contains both numerical and categorical features, making it suitable for demonstrating the main preprocessing steps required for machine learning. It also contains a target column, `Churn`, which can later be used for classification tasks.

## 3. Tools and Libraries Used

The following tools and libraries were used:

- Python
- pandas
- scikit-learn
- StandardScaler
- train_test_split

## 4. Preprocessing Steps Completed

### 4.1 Loading the Dataset

The dataset was loaded using pandas. I first displayed the first few rows of the dataset to understand its structure and columns.

### 4.2 Checking Dataset Information

I checked the dataset shape, column data types, and missing values. This helped me understand which columns were numerical and which were categorical.

### 4.3 Handling Missing Values

Missing values were handled based on the column type:

- Numerical columns were filled using the median value.
- Categorical columns were filled using the most frequent value, also known as the mode.

This ensured that the dataset did not contain empty values before model training.

### 4.4 Encoding Categorical Variables

Categorical variables were converted into numerical format using one-hot encoding. This step was necessary because machine learning models require numerical input.

### 4.5 Standardizing Numerical Features

The feature values were standardized using `StandardScaler`. This helps place numerical features on a similar scale, which improves machine learning performance for many algorithms.

### 4.6 Splitting the Dataset

The dataset was split into training and testing sets using `train_test_split`.

- Training set: used to train machine learning models.
- Testing set: used to evaluate how well the model performs on unseen data.

## 5. Output Files Generated

After preprocessing, the following files were created:

- `results/level1_task1/train_preprocessed.csv`
- `results/level1_task1/test_preprocessed.csv`

These files contain the processed training and testing data.

## 6. Conclusion

This task successfully prepared the churn dataset for machine learning. The dataset was cleaned, categorical values were encoded, numerical values were standardized, and the data was split into training and testing sets. The processed files can now be used in future machine learning tasks such as logistic regression, random forest classification, or support vector machine classification.
