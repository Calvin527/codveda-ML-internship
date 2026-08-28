
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# This task prepares raw data for machine learning:
# missing values, categorical encoding, feature scaling, and train/test split.

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

# Handle missing values
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Separate target column
target_column = "Churn"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert target values to 0 and 1 if needed
if y.dtype == "bool":
    y = y.astype(int)
else:
    y = y.map({"False": 0, "True": 1, False: 0, True: 1})

# Encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

print("\nShape after encoding:")
print(X_encoded.shape)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\nPreprocessing completed successfully!")
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

# Save processed files
os.makedirs("results/level1_task1", exist_ok=True)

train_df = pd.DataFrame(X_train, columns=X_encoded.columns)
train_df["Churn"] = y_train.values

test_df = pd.DataFrame(X_test, columns=X_encoded.columns)
test_df["Churn"] = y_test.values

train_df.to_csv("results/level1_task1/train_preprocessed.csv", index=False)
test_df.to_csv("results/level1_task1/test_preprocessed.csv", index=False)

print("\nSaved files:")
print("results/level1_task1/train_preprocessed.csv")
print("results/level1_task1/test_preprocessed.csv")