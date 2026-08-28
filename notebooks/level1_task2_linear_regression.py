
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ================================
# Level 1 Task 2: Linear Regression
# ================================

os.makedirs("results/level1_task2", exist_ok=True)

# Load dataset
file_path = "data/house_prediction.csv"

# Try normal CSV loading first
df = pd.read_csv(file_path)

# If the dataset loads as one column, reload as space-separated
if df.shape[1] == 1:
    df = pd.read_csv(file_path, sep=r"\s+", header=None)

# If the dataset has 14 columns and no proper headers, assign common house dataset column names
if df.shape[1] == 14 and all(isinstance(col, int) for col in df.columns):
    df.columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "PRICE"
    ]

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

# Fill missing values
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset shape after cleaning:")
print(df.shape)

# Find target column
possible_targets = ["PRICE", "price", "MEDV", "SalePrice", "target"]

target_column = None
for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

# If no obvious price column exists, use the last column as target
if target_column is None:
    target_column = df.columns[-1]

print("\nTarget column used:", target_column)

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Encode categorical columns if any exist
X = pd.get_dummies(X, drop_first=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Interpret coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coefficients)

# Save results
coefficients.to_csv("results/level1_task2/model_coefficients.csv", index=False)

predictions = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

predictions.to_csv("results/level1_task2/predictions.csv", index=False)

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("results/level1_task2/actual_vs_predicted.png")
plt.show()

print("\nSaved files:")
print("results/level1_task2/model_coefficients.csv")
print("results/level1_task2/predictions.csv")
print("results/level1_task2/actual_vs_predicted.png")