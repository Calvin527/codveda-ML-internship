# Level 3 Task 2: Support Vector Machine Classification

## 1. Task Overview

The objective of this task was to build and evaluate Support Vector Machine models for binary classification.

For this task, I used the customer churn dataset to predict whether a customer would churn.

Two SVM kernels were evaluated:

- Linear kernel
- Radial Basis Function (RBF) kernel

The performance of the two models was compared using classification metrics.

## 2. Dataset Used

`churn-bigml-80.csv`

The dataset contains customer account information, service usage information, call activity, customer service interactions, and a target variable named `Churn`.

The target variable contains two classes:

- `0`: Customer did not churn
- `1`: Customer churned

## 3. Tools and Libraries Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- pathlib
- StandardScaler
- SVC
- PCA
- train_test_split
- accuracy_score
- precision_score
- recall_score
- f1_score
- roc_auc_score
- roc_curve
- confusion_matrix
- classification_report

## 4. Steps Completed

### 4.1 Loading the Dataset

The customer churn dataset was loaded using pandas.

The dataset structure and first few rows were inspected before preprocessing.

### 4.2 Data Cleaning

Duplicate records were removed.

Missing values were handled based on the column data type:

- Numerical missing values were filled using the median.
- Categorical missing values were filled using the mode.

### 4.3 Separating Features and Target

The `Churn` column was used as the target variable.

The remaining columns were used as input features.

The churn values were converted into numerical format where necessary.

### 4.4 Encoding Categorical Features

Categorical variables were transformed into numerical values using one-hot encoding.

This made the dataset suitable for Support Vector Machine training.

### 4.5 Splitting the Dataset

The dataset was split into training and testing datasets using an 80/20 split.

Stratification was used to maintain the same class distribution in both training and testing data.

### 4.6 Feature Scaling

The features were standardized using `StandardScaler`.

Scaling is particularly important for Support Vector Machines because the algorithm is sensitive to differences in feature magnitude.

### 4.7 Training the Linear SVM

A Support Vector Machine model using a linear kernel was trained.

The linear kernel attempts to separate the two classes using a linear decision boundary.

### 4.8 Training the RBF SVM

A second Support Vector Machine model was trained using the RBF kernel.

The RBF kernel can model more complex and non-linear relationships between the input features and the target classes.

### 4.9 Model Evaluation

Both SVM models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

## 5. Linear SVM Results

- Accuracy: [PASTE VALUE]
- Precision: [PASTE VALUE]
- Recall: [PASTE VALUE]
- F1-score: [PASTE VALUE]
- ROC-AUC: [PASTE VALUE]

## 6. RBF SVM Results

- Accuracy: [PASTE VALUE]
- Precision: [PASTE VALUE]
- Recall: [PASTE VALUE]
- F1-score: [PASTE VALUE]
- ROC-AUC: [PASTE VALUE]

## 7. Kernel Comparison

The Linear and RBF SVM models were compared using their evaluation metrics.

| Metric    | Linear SVM | RBF SVM |
| --------- | ---------: | ------: |
| Accuracy  |    [VALUE] | [VALUE] |
| Precision |    [VALUE] | [VALUE] |
| Recall    |    [VALUE] | [VALUE] |
| F1-score  |    [VALUE] | [VALUE] |
| ROC-AUC   |    [VALUE] | [VALUE] |

The comparison helped determine how the choice of kernel affected classification performance.

The stronger performing model can be identified based on the combined evaluation metrics rather than accuracy alone.

## 8. ROC Curve

ROC curves were created for both the Linear and RBF SVM models.

The ROC curve compares:

- True Positive Rate
- False Positive Rate

The Area Under the Curve (AUC) provides an overall measure of the model's ability to distinguish between churn and non-churn customers.

A higher AUC value indicates stronger discrimination between the two classes.

## 9. Decision Boundary Visualization

Because the original dataset contains many input features, the data cannot be directly visualized in two dimensions.

Principal Component Analysis (PCA) was therefore used to reduce the scaled feature set to two principal components.

An additional SVM model was trained on the two-dimensional PCA representation for visualization purposes.

The resulting plot shows how an SVM decision boundary separates the two customer classes.

The PCA-based SVM was used for visualization only. The primary Linear and RBF SVM models were trained using the complete feature set.

## 10. Output Files

The following output files were generated:

- `results/level3_task2/linear_svm_classification_report.txt`
- `results/level3_task2/linear_svm_confusion_matrix.csv`
- `results/level3_task2/rbf_svm_classification_report.txt`
- `results/level3_task2/rbf_svm_confusion_matrix.csv`
- `results/level3_task2/svm_model_comparison.csv`
- `results/level3_task2/svm_roc_curve.png`
- `results/level3_task2/svm_decision_boundary.png`

## 11. Conclusion

This task successfully implemented Support Vector Machine models for customer churn classification.

Both Linear and RBF kernels were trained and evaluated. The models were compared using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and classification reports.

ROC curve analysis was used to examine classification performance across different decision thresholds.

PCA was also applied to create a two-dimensional representation of the data so that the SVM decision boundary could be visualized.

This task strengthened my understanding of feature scaling, Support Vector Machines, kernel methods, classification evaluation, ROC analysis, PCA, and decision boundary visualization.
