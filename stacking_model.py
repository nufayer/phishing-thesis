"""
==============================================================
 Phishing Website Detection using Optimized Stacking Ensemble
==============================================================

Author : Nufayer Mahmud
Model  : Optimized Stacking Ensemble
Dataset: UCI Phishing Websites Dataset

Base Models
------------
1. Random Forest (Optimized)
2. XGBoost (Optimized)
3. LightGBM (Optimized)
4. CatBoost (Best Version)

Meta Learner
------------
Logistic Regression

==============================================================
"""

import time
import joblib
import warnings
import pandas as pd

from scipy.io import arff

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

print("=" * 70)
print(" PHISHING WEBSITE DETECTION ")
print(" OPTIMIZED STACKING ENSEMBLE ")
print("=" * 70)

###############################################################
# LOAD DATASET
###############################################################

print("\nLoading Dataset...")

data, meta = arff.loadarff(
    "dataset/Training Dataset.arff"
)

df = pd.DataFrame(data)

for column in df.columns:

    df[column] = (
        df[column]
        .str.decode("utf-8")
        .astype(int)
    )

df["Result"] = df["Result"].replace(
    {-1: 0}
)

print("Dataset Loaded Successfully!")

print("\nDataset Shape :", df.shape)

X = df.drop("Result", axis=1)

y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

###############################################################
# BUILD BASE MODELS
###############################################################

print("\nCreating Optimized Base Models...")

random_forest = RandomForestClassifier(

    n_estimators=300,

    min_samples_split=2,

    min_samples_leaf=1,

    max_features="log2",

    max_depth=None,

    random_state=42

)

xgboost = XGBClassifier(

    n_estimators=400,

    learning_rate=0.05,

    max_depth=10,

    subsample=0.8,

    colsample_bytree=0.9,

    gamma=0,

    random_state=42,

    eval_metric="logloss"

)

lightgbm = LGBMClassifier(

    n_estimators=300,

    learning_rate=0.1,

    num_leaves=63,

    max_depth=8,

    subsample=0.9,

    colsample_bytree=0.8,

    random_state=42,

    verbosity=-1

)

catboost = CatBoostClassifier(

    verbose=0,

    random_state=42

)

base_models = [

    ("Random Forest", random_forest),

    ("XGBoost", xgboost),

    ("LightGBM", lightgbm),

    ("CatBoost", catboost)

]

print("Base Models Ready!")

###############################################################
# META LEARNER
###############################################################

print("\nCreating Meta Learner...")

meta_model = LogisticRegression(

    max_iter=1000,

    random_state=42

)

print("Meta Learner Ready!")



###############################################################
# BUILD STACKING MODEL
###############################################################

print("\nBuilding Optimized Stacking Ensemble...")

stack_model = StackingClassifier(

    estimators=base_models,

    final_estimator=meta_model,

    cv=5,

    stack_method="predict_proba",

    passthrough=False,

    n_jobs=-1

)

print("Stacking Ensemble Created Successfully!")

###############################################################
# TRAIN MODEL
###############################################################

print("\n" + "=" * 70)
print("TRAINING STACKING ENSEMBLE")
print("=" * 70)

start_time = time.time()

stack_model.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print("\nTraining Completed!")

###############################################################
# PREDICTIONS
###############################################################

print("\nMaking Predictions...")

predictions = stack_model.predict(
    X_test
)

probabilities = stack_model.predict_proba(
    X_test
)[:,1]

print("Predictions Completed!")

###############################################################
# CALCULATE METRICS
###############################################################

accuracy = accuracy_score(
    y_test,
    predictions
)

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

print("\nPerforming 5-Fold Cross Validation...")

cv_scores = cross_val_score(

    stack_model,

    X,

    y,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

cv_accuracy = cv_scores.mean()

print("Cross Validation Completed!")

###############################################################
# CONFUSION MATRIX
###############################################################

cm = confusion_matrix(
    y_test,
    predictions
)


###############################################################
# DISPLAY RESULTS
###############################################################

print("\n" + "=" * 70)
print("FINAL STACKING ENSEMBLE RESULTS")
print("=" * 70)

print(f"\nAccuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"ROC AUC       : {roc:.4f}")
print(f"CV Accuracy   : {cv_accuracy:.4f}")
print(f"Training Time : {training_time:.2f} sec")

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(classification_report(
    y_test,
    predictions
))

###############################################################
# SAVE MODEL
###############################################################

joblib.dump(
    stack_model,
    "models/stacking_model_optimized.pkl"
)

print("\nOptimized Stacking Model Saved Successfully!")

###############################################################
# SAVE RESULTS
###############################################################

results = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC",
        "Cross Validation Accuracy",
        "Training Time (sec)"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        roc,
        cv_accuracy,
        training_time
    ]

})

results.to_csv(
    "results/stacking_results.csv",
    index=False
)

print("Results saved to results/stacking_results.csv")

###############################################################
# FINAL SUMMARY
###############################################################

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
Dataset                : UCI Phishing Websites Dataset
Training Samples       : {len(X_train)}
Testing Samples        : {len(X_test)}

Base Models
-----------
✓ Random Forest (Optimized)
✓ XGBoost (Optimized)
✓ LightGBM (Optimized)
✓ CatBoost

Meta Learner
------------
✓ Logistic Regression

Evaluation Metrics
------------------
Accuracy              : {accuracy:.4f}
Precision             : {precision:.4f}
Recall                : {recall:.4f}
F1 Score              : {f1:.4f}
ROC AUC               : {roc:.4f}
Cross Validation      : {cv_accuracy:.4f}

Model Saved As
--------------
models/stacking_model_optimized.pkl
""")

print("=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)