import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
    confusion_matrix
)

DATASET_PATH = "dataset/Training Dataset.arff"

MODEL_DIR = "models"

RESULT_DIR = "results"

FIGURE_DIR = "figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

print("Loading dataset...")

data, meta = arff.loadarff(DATASET_PATH)

df = pd.DataFrame(data)

for col in df.columns:
    df[col] = df[col].apply(lambda x: int(x.decode("utf-8")))

df["Result"] = df["Result"].replace(-1, 0)

X = df.drop("Result", axis=1)

y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset Ready.")

results = pd.read_csv(
    "results/model_comparison.csv"
)

print(results)

plt.figure(figsize=(10,6))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "figures/accuracy_comparison.png",
    dpi=300
)

plt.close()

print("Accuracy chart saved.")

plt.figure(figsize=(10,6))

plt.bar(
    results["Model"],
    results["Training Time"]
)

plt.title("Model Training Time Comparison")

plt.ylabel("Seconds")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "figures/training_time.png",
    dpi=300
)

plt.close()

print("Training time chart saved.")


# ==========================================================
# ROC CURVE
# ==========================================================

from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(8,6))

model_files = {

    "Logistic Regression":"logistic_regression.pkl",

    "Random Forest":"random_forest.pkl",

    "XGBoost":"xgboost.pkl",

    "LightGBM":"lightgbm.pkl",

    "CatBoost":"catboost.pkl"

}

for model_name, file in model_files.items():

    model = joblib.load(
        os.path.join(MODEL_DIR,file)
    )

    probabilities = model.predict_proba(X_test)[:,1]

    fpr,tpr,_ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={roc_auc:.3f})"
    )

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "figures/roc_curve.png",
    dpi=300
)

plt.close()

print("ROC Curve Saved.")


# ==========================================================
# CONFUSION MATRIX (Best Model)
# ==========================================================

print("\nGenerating Confusion Matrix...")

best_model = joblib.load(
    os.path.join(MODEL_DIR, "lightgbm.pkl")
)

predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Legitimate",
        "Phishing"
    ]
)

plt.figure(figsize=(7,7))

disp.plot(cmap="Blues")

plt.title("Confusion Matrix (LightGBM)")

plt.tight_layout()

plt.savefig(
    "figures/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion Matrix Saved.")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

print("\nGenerating Feature Importance...")

best_model = joblib.load(
    os.path.join(MODEL_DIR, "lightgbm.pkl")
)

importance = best_model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

plt.figure(figsize=(10,8))

plt.barh(

    importance_df["Feature"],

    importance_df["Importance"]

)

plt.gca().invert_yaxis()

plt.title("LightGBM Feature Importance")

plt.xlabel("Importance Score")

plt.tight_layout()

plt.savefig(

    "figures/feature_importance.png",

    dpi=300

)

plt.close()

print("Feature Importance Saved.")