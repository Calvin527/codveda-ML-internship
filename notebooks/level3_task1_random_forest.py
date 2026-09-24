
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "churn-bigml-80.csv"

RESULTS_DIR = BASE_DIR / "results" / "level3_task1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_FILE)

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================================
# 2. CLEAN DATA
# ==========================================================

df = df.drop_duplicates()

for column in df.columns:

    if df[column].dtype == "object":
        df[column] = df[column].fillna(df[column].mode()[0])

    else:
        df[column] = df[column].fillna(df[column].median())


# ==========================================================
# 3. FEATURES AND TARGET
# ==========================================================

X = df.drop("Churn", axis=1)

y = df["Churn"]

if y.dtype == "bool":
    y = y.astype(int)
else:
    y = y.replace({
        "False": 0,
        "True": 1,
        False: 0,
        True: 1
    }).astype(int)


# Convert categorical variables
X = pd.get_dummies(X, drop_first=True)


# ==========================================================
# 4. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================================
# 5. RANDOM FOREST
# ==========================================================

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)


# ==========================================================
# 6. HYPERPARAMETER TUNING
# ==========================================================

param_grid = {

    "n_estimators": [100, 200, 300],

    "max_depth": [
        None,
        5,
        10,
        20
    ]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)


print("\nTraining Random Forest...")

grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_


print("\nBest Parameters:")
print(grid_search.best_params_)


# ==========================================================
# 7. CROSS VALIDATION
# ==========================================================

cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1"
)

print("\nCross-validation F1 scores:")
print(cv_scores)

print("\nMean Cross-validation F1:")
print(cv_scores.mean())


# ==========================================================
# 8. TEST SET PREDICTIONS
# ==========================================================

y_pred = best_model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\nRandom Forest Results")
print("--------------------------")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)


# ==========================================================
# 9. CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\nClassification Report:")
print(report)


with open(
    RESULTS_DIR / "classification_report.txt",
    "w"
) as file:

    file.write(report)


# ==========================================================
# 10. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

confusion_df = pd.DataFrame(
    cm,
    columns=[
        "Predicted No Churn",
        "Predicted Churn"
    ],
    index=[
        "Actual No Churn",
        "Actual Churn"
    ]
)

confusion_df.to_csv(
    RESULTS_DIR / "confusion_matrix.csv"
)


# ==========================================================
# 11. FEATURE IMPORTANCE
# ==========================================================

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        best_model.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


importance_df.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False
)


print("\nTop 10 Important Features:")
print(importance_df.head(10))


# ==========================================================
# 12. FEATURE IMPORTANCE GRAPH
# ==========================================================

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Feature Importance")

plt.ylabel("Feature")

plt.title(
    "Top Random Forest Feature Importances"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "feature_importance.png"
)

plt.show()


# ==========================================================
# 13. SAVE METRICS
# ==========================================================

metrics = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Mean CV F1"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        cv_scores.mean()
    ]

})


metrics.to_csv(
    RESULTS_DIR / "model_metrics.csv",
    index=False
)


# Save best parameters

best_params = pd.DataFrame(
    [grid_search.best_params_]
)

best_params.to_csv(
    RESULTS_DIR / "best_parameters.csv",
    index=False
)


print("\nLevel 3 Task 1 completed successfully.")