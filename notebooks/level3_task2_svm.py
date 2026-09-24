
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "churn-bigml-80.csv"

RESULTS_DIR = BASE_DIR / "results" / "level3_task2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_FILE)

print("Dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ==========================================================
# 2. CLEAN DATA
# ==========================================================

df = df.drop_duplicates()

for column in df.columns:

    if df[column].dtype == "object":
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

    else:
        df[column] = df[column].fillna(
            df[column].median()
        )


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


X = pd.get_dummies(
    X,
    drop_first=True
)


# ==========================================================
# 4. SPLIT DATA
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# 5. SCALE FEATURES
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================================
# 6. CREATE SVM MODELS
# ==========================================================

linear_model = SVC(
    kernel="linear",
    class_weight="balanced",
    random_state=42
)

rbf_model = SVC(
    kernel="rbf",
    class_weight="balanced",
    random_state=42
)


# ==========================================================
# 7. TRAIN MODELS
# ==========================================================

print("\nTraining Linear SVM...")

linear_model.fit(
    X_train_scaled,
    y_train
)


print("Training RBF SVM...")

rbf_model.fit(
    X_train_scaled,
    y_train
)


# ==========================================================
# 8. EVALUATION FUNCTION
# ==========================================================

def evaluate_model(
    model,
    name,
    X_test_data,
    y_test_data
):

    predictions = model.predict(
        X_test_data
    )

    scores = model.decision_function(
        X_test_data
    )

    accuracy = accuracy_score(
        y_test_data,
        predictions
    )

    precision = precision_score(
        y_test_data,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test_data,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test_data,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test_data,
        scores
    )

    print(f"\n{name}")

    print("----------------------")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)
    print("ROC-AUC  :", auc)


    report = classification_report(
        y_test_data,
        predictions,
        zero_division=0
    )


    file_name = (
        name
        .lower()
        .replace(" ", "_")
    )


    with open(
        RESULTS_DIR /
        f"{file_name}_classification_report.txt",
        "w"
    ) as file:

        file.write(report)


    cm = confusion_matrix(
        y_test_data,
        predictions
    )


    pd.DataFrame(
        cm,
        columns=[
            "Predicted No Churn",
            "Predicted Churn"
        ],
        index=[
            "Actual No Churn",
            "Actual Churn"
        ]
    ).to_csv(
        RESULTS_DIR /
        f"{file_name}_confusion_matrix.csv"
    )


    return {

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-score": f1,

        "AUC": auc,

        "Scores": scores
    }


# ==========================================================
# 9. EVALUATE BOTH MODELS
# ==========================================================

linear_results = evaluate_model(

    linear_model,

    "Linear SVM",

    X_test_scaled,

    y_test

)


rbf_results = evaluate_model(

    rbf_model,

    "RBF SVM",

    X_test_scaled,

    y_test

)


# ==========================================================
# 10. MODEL COMPARISON
# ==========================================================

comparison = pd.DataFrame({

    "Model": [
        linear_results["Model"],
        rbf_results["Model"]
    ],

    "Accuracy": [
        linear_results["Accuracy"],
        rbf_results["Accuracy"]
    ],

    "Precision": [
        linear_results["Precision"],
        rbf_results["Precision"]
    ],

    "Recall": [
        linear_results["Recall"],
        rbf_results["Recall"]
    ],

    "F1-score": [
        linear_results["F1-score"],
        rbf_results["F1-score"]
    ],

    "AUC": [
        linear_results["AUC"],
        rbf_results["AUC"]
    ]

})


comparison.to_csv(
    RESULTS_DIR /
    "svm_model_comparison.csv",
    index=False
)


print("\nModel Comparison:")

print(comparison)


# ==========================================================
# 11. ROC CURVES
# ==========================================================

linear_fpr, linear_tpr, _ = roc_curve(

    y_test,

    linear_results["Scores"]

)


rbf_fpr, rbf_tpr, _ = roc_curve(

    y_test,

    rbf_results["Scores"]

)


plt.figure(figsize=(8, 6))

plt.plot(
    linear_fpr,
    linear_tpr,
    label=f"Linear SVM AUC = {linear_results['AUC']:.3f}"
)

plt.plot(
    rbf_fpr,
    rbf_tpr,
    label=f"RBF SVM AUC = {rbf_results['AUC']:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Linear vs RBF SVM"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "svm_roc_curve.png"
)

plt.show()


# ==========================================================
# 12. PCA FOR 2D VISUALIZATION
# ==========================================================

pca = PCA(
    n_components=2
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)


# Train visualization model

visual_model = SVC(
    kernel="rbf",
    class_weight="balanced"
)

visual_model.fit(
    X_train_pca,
    y_train
)


# ==========================================================
# 13. DECISION BOUNDARY
# ==========================================================

x_min = X_train_pca[:, 0].min() - 1
x_max = X_train_pca[:, 0].max() + 1

y_min = X_train_pca[:, 1].min() - 1
y_max = X_train_pca[:, 1].max() + 1


xx, yy = np.meshgrid(

    np.linspace(
        x_min,
        x_max,
        300
    ),

    np.linspace(
        y_min,
        y_max,
        300
    )

)


grid = np.c_[
    xx.ravel(),
    yy.ravel()
]


Z = visual_model.predict(
    grid
)

Z = Z.reshape(
    xx.shape
)


plt.figure(
    figsize=(9, 7)
)

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.25
)

plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "SVM Decision Boundary using PCA"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "svm_decision_boundary.png"
)

plt.show()


print(
    "\nLevel 3 Task 2 completed successfully."
)