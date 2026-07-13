import os
import time
import warnings
import joblib
import pandas as pd
import numpy as np

from scipy.io import arff

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

DATASET_PATH = "dataset/Training Dataset.arff"

MODEL_DIR = "models"

RESULT_DIR = "results"

FIGURE_DIR = "figures"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

print("="*70)
print("PHISHING WEBSITE DETECTION")
print("="*70)

print("\nLoading Dataset...")

data, meta = arff.loadarff(DATASET_PATH)

df = pd.DataFrame(data)

for col in df.columns:
    df[col] = df[col].apply(lambda x: int(x.decode("utf-8")))

print("Dataset Loaded Successfully!")

df["Result"] = df["Result"].replace(-1, 0)

print("Labels Converted.")

print(df.head())

X = df.drop("Result", axis=1)

y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    ),

    "CatBoost": CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        verbose=False,
        random_seed=RANDOM_STATE
    )

}

# ==========================================================
# Train and Evaluate Models
# ==========================================================

results = []

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

for model_name, model in models.items():

    print("\n" + "="*70)
    print(f"Training {model_name}")
    print("="*70)

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = predictions

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc = roc_auc_score(
        y_test,
        probabilities
    )

    cv_score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    ).mean()

    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"ROC AUC       : {roc:.4f}")
    print(f"CV Accuracy   : {cv_score:.4f}")
    print(f"Training Time : {training_time:.2f} sec")

    filename = model_name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, filename)
    )

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC AUC": roc,

        "CV Accuracy": cv_score,

        "Training Time": training_time

    })



    # ==========================================================
# Save Results
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("="*80)
print("MODEL COMPARISON")
print("="*80)

print(results_df)

results_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "model_comparison.csv"
    ),
    index=False
)

print("\nResults saved successfully!")

print("\nBest Model:")

print(results_df.iloc[0])