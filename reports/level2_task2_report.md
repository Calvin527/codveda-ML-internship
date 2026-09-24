# Level 2 Task 2: Decision Trees for Classification

## 1. Task Overview

The objective of this task was to build a Decision Tree classifier for a classification problem.

For this task, I used the Iris dataset to classify flower species based on their measurements. The Decision Tree model was trained to learn patterns from the dataset and predict the correct class of each flower.

## 2. Dataset Used

The dataset used for this task was:

`iris.csv`

This dataset was selected because it is suitable for classification. It contains numerical flower measurements and a target column representing the flower species.

## 3. Tools and Libraries Used

The following tools and libraries were used:

- Python
- pandas
- scikit-learn
- matplotlib
- DecisionTreeClassifier
- train_test_split
- plot_tree
- accuracy_score
- f1_score
- classification_report
- confusion_matrix

## 4. Steps Completed

### 4.1 Loading the Dataset

The Iris dataset was loaded using pandas. The first few rows were displayed to understand the structure of the dataset and confirm that it loaded correctly.

### 4.2 Inspecting the Dataset

The dataset shape, column data types, and missing values were checked. This helped confirm that the dataset was ready for preprocessing and model training.

### 4.3 Data Cleaning

Duplicate rows were removed from the dataset. Missing values were handled based on column type:

- Numerical missing values were filled using the median.
- Categorical missing values were filled using the mode.

This ensured that the dataset was clean before training the model.

### 4.4 Separating Features and Target

The dataset was separated into input features and target values.

- `X` contained the flower measurement features.
- `y` contained the target class, which represented the flower species.

### 4.5 Splitting the Dataset

The dataset was split into training and testing sets using `train_test_split`.

- The training set was used to train the Decision Tree model.
- The testing set was used to evaluate the model on unseen data.

### 4.6 Training the Original Decision Tree Model

An original Decision Tree classifier was trained using the training data. This model was allowed to grow naturally based on the data.

### 4.7 Evaluating the Original Model

The original Decision Tree model was evaluated using:

- Accuracy
- F1-score
- Classification report
- Confusion matrix

These metrics helped measure how well the model classified the flower species.

### 4.8 Visualizing the Original Decision Tree

The original Decision Tree structure was visualized using `plot_tree`. This helped show how the model makes decisions by splitting the dataset based on feature values.

### 4.9 Pruning the Decision Tree

A pruned Decision Tree model was created using `max_depth=3`.

Pruning was applied to reduce overfitting by limiting how deep the tree could grow. This helps the model generalize better to unseen data.

### 4.10 Evaluating the Pruned Model

The pruned Decision Tree model was also evaluated using accuracy and F1-score. The results of the original and pruned models were compared to understand the effect of pruning.

## 5. Evaluation Results

The model produced the following evaluation results:

### Original Decision Tree

- Accuracy: [Paste original accuracy here]
- F1-score: [Paste original F1-score here]

### Pruned Decision Tree

- Accuracy: [Paste pruned accuracy here]
- F1-score: [Paste pruned F1-score here]

The comparison helped show whether pruning improved or maintained the model performance while reducing model complexity.

## 6. Feature Importance

Feature importance was extracted from the Decision Tree model. This helped identify which flower measurements contributed the most to the model’s classification decisions.

Features with higher importance values had a stronger influence on the model’s predictions.

## 7. Output Files

The task produced the following output files:

- `results/level2_task2/classification_report.txt`
- `results/level2_task2/confusion_matrix.csv`
- `results/level2_task2/feature_importance.csv`
- `results/level2_task2/original_decision_tree.png`
- `results/level2_task2/pruned_decision_tree.png`
- `results/level2_task2/decision_tree_metrics.csv`

## 8. Conclusion

This task successfully implemented a Decision Tree classifier for flower species classification. The Iris dataset was loaded, cleaned, split into training and testing sets, and used to train both an original and pruned Decision Tree model.

The model was evaluated using accuracy, F1-score, classification report, and confusion matrix. The tree structures were visualized, and feature importance analysis was performed to better understand how the model made predictions.
