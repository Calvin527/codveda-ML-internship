
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# Level 2 Task 2: Decision Tree Classification
# ==========================================

# Create results folder
os.makedirs("results/level2_task2", exist_ok=True)

# Load dataset
df = pd.read_csv("data/iris.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape before cleaning:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset shape after cleaning:")
print(df.shape)

# Target column
target_column = "species"

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

print("\nTarget classes:")
print(y.value_counts())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train original Decision Tree model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions using original tree
y_pred = tree_model.predict(X_test)

# Evaluate original tree
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nOriginal Decision Tree Results:")
print("Accuracy:", accuracy)
print("F1-score:", f1)

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

cm = confusion_matrix(y_test, y_pred, labels=tree_model.classes_)
print("\nConfusion Matrix:")
print(cm)

# Save original results
with open("results/level2_task2/classification_report.txt", "w") as file:
    file.write(report)

confusion_df = pd.DataFrame(
    cm,
    columns=[f"Predicted {label}" for label in tree_model.classes_],
    index=[f"Actual {label}" for label in tree_model.classes_]
)
confusion_df.to_csv("results/level2_task2/confusion_matrix.csv")

# Save feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_model.feature_importances_
})

feature_importance.to_csv("results/level2_task2/feature_importance.csv", index=False)

# Visualize original Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=tree_model.classes_,
    filled=True,
    rounded=True
)
plt.title("Original Decision Tree")
plt.tight_layout()
plt.savefig("results/level2_task2/original_decision_tree.png")
plt.show()

# Train pruned Decision Tree model
# max_depth=3 limits how deep the tree can grow to reduce overfitting.
pruned_tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
pruned_tree_model.fit(X_train, y_train)

# Make predictions using pruned tree
pruned_y_pred = pruned_tree_model.predict(X_test)

# Evaluate pruned tree
pruned_accuracy = accuracy_score(y_test, pruned_y_pred)
pruned_f1 = f1_score(y_test, pruned_y_pred, average="weighted")

print("\nPruned Decision Tree Results:")
print("Accuracy:", pruned_accuracy)
print("F1-score:", pruned_f1)

pruned_report = classification_report(y_test, pruned_y_pred)
print("\nPruned Classification Report:")
print(pruned_report)

# Save comparison metrics
metrics = pd.DataFrame({
    "Model": ["Original Decision Tree", "Pruned Decision Tree"],
    "Accuracy": [accuracy, pruned_accuracy],
    "F1-score": [f1, pruned_f1]
})

metrics.to_csv("results/level2_task2/decision_tree_metrics.csv", index=False)

# Visualize pruned Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(
    pruned_tree_model,
    feature_names=X.columns,
    class_names=pruned_tree_model.classes_,
    filled=True,
    rounded=True
)
plt.title("Pruned Decision Tree")
plt.tight_layout()
plt.savefig("results/level2_task2/pruned_decision_tree.png")
plt.show()

print("\nSaved files:")
print("results/level2_task2/classification_report.txt")
print("results/level2_task2/confusion_matrix.csv")
print("results/level2_task2/feature_importance.csv")
print("results/level2_task2/original_decision_tree.png")
print("results/level2_task2/pruned_decision_tree.png")
print("results/level2_task2/decision_tree_metrics.csv")