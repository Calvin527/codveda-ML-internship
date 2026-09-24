# Level 1 Task 2: Simple Linear Regression Model

## 1. Task Overview

The objective of this task was to build a simple Linear Regression model to predict a continuous value.

For this task, I used the house prediction dataset. The goal was to train a regression model that can learn the relationship between house-related features and the house price.

## 2. Dataset Used

The dataset used for this task was:

`house_prediction.csv`

This dataset was selected because it is suitable for regression, where the target value is continuous. In this task, the model was trained to predict house prices.

## 3. Tools and Libraries Used

The following tools and libraries were used:

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- LinearRegression
- StandardScaler
- train_test_split
- mean_squared_error
- r2_score

## 4. Steps Completed

### 4.1 Loading the Dataset

The house prediction dataset was loaded using pandas. The dataset was inspected by displaying the first few rows to understand its structure.

### 4.2 Inspecting the Dataset

The dataset shape, column information, data types, and missing values were checked before training the model. This helped confirm that the dataset was loaded correctly and ready for preprocessing.

### 4.3 Data Cleaning

Duplicate rows were removed from the dataset.

Missing values were handled based on the column type:

- Numerical columns were filled using the median value.
- Categorical columns were filled using the mode, which is the most frequent value.

This ensured that the dataset did not contain empty values before model training.

### 4.4 Separating Features and Target

The target column was selected as the value to be predicted. In this task, the target represented the house price.

- `X` contained the input features.
- `y` contained the target value, which was the house price.

### 4.5 Encoding Categorical Variables

Categorical variables were converted into numerical format using one-hot encoding if any categorical columns were present. This was necessary because Linear Regression requires numerical input values.

### 4.6 Scaling the Features

The input features were standardized using `StandardScaler`. Scaling helped place all numerical features on a similar scale before training the model.

### 4.7 Splitting the Dataset

The dataset was split into training and testing sets using `train_test_split`.

- The training set was used to train the Linear Regression model.
- The testing set was used to evaluate the model on unseen data.

### 4.8 Training the Linear Regression Model

A Linear Regression model was created and trained using the training dataset. The model learned the relationship between the house features and the target price.

### 4.9 Making Predictions

After training, the model was used to predict house prices using the testing dataset. The predicted values were then compared with the actual house prices.

### 4.10 Model Evaluation

The model was evaluated using:

- Mean Squared Error
- R-squared Score

Mean Squared Error measured the average squared difference between the actual and predicted prices. A lower MSE means better prediction performance.

R-squared measured how well the model explained the variation in house prices. A higher R-squared score means the model explains more of the target variable.

## 5. Model Coefficients

The model coefficients were extracted and saved. Coefficients help explain how each feature contributes to the predicted house price.

- A positive coefficient means the feature increases the predicted price.
- A negative coefficient means the feature decreases the predicted price.

## 6. Output Files

The task produced the following output files:

- `results/level1_task2/model_coefficients.csv`
- `results/level1_task2/predictions.csv`
- `results/level1_task2/actual_vs_predicted.png`

## 7. Conclusion

This task successfully implemented a Simple Linear Regression model for house price prediction. The dataset was loaded, cleaned, preprocessed, split into training and testing sets, and used to train a Linear Regression model. The model was evaluated using Mean Squared Error and R-squared, and the results were saved for reporting and submission.
