
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# ================================
# Level 2 Task 1: Logistic Regression
# ================================

# Create results folder
os.makedirs("results/level2_task1", exist_ok=True)

# Load dataset
df = pd.read_csv("data/churn-bigml-80.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
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

# Target column
target_column = "Churn"

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert target column to 0 and 1
if y.dtype == "bool":
    y = y.astype(int)
else:
    y = y.map({"False": 0, "True": 1, False: 0, True: 1})

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nModel Evaluation Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("AUC Score:", auc)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
report = classification_report(y_test, y_pred)
print(report)

# Coefficients and odds ratios
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients["Odds Ratio"] = coefficients["Coefficient"].apply(lambda x: round(2.71828 ** x, 4))

print("\nModel Coefficients and Odds Ratios:")
print(coefficients)

# Save coefficients
coefficients.to_csv("results/level2_task1/logistic_regression_coefficients.csv", index=False)

# Save classification report
with open("results/level2_task1/classification_report.txt", "w") as file:
    file.write(report)

# Save confusion matrix
confusion_df = pd.DataFrame(
    cm,
    columns=["Predicted No Churn", "Predicted Churn"],
    index=["Actual No Churn", "Actual Churn"]
)
confusion_df.to_csv("results/level2_task1/confusion_matrix.csv")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig("results/level2_task1/roc_curve.png")
plt.show()

print("\nSaved files:")
print("results/level2_task1/logistic_regression_coefficients.csv")
print("results/level2_task1/classification_report.txt")
print("results/level2_task1/confusion_matrix.csv")
print("results/level2_task1/roc_curve.png")