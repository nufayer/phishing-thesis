import os
import warnings
import joblib
import pandas as pd

from scipy.io import arff

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

DATASET_PATH = "dataset/Training Dataset.arff"

MODEL_DIR = "models"

RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42

print("="*70)
print("LIGHTGBM HYPERPARAMETER OPTIMIZATION")
print("="*70)

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
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Dataset Ready!")

param_grid = {

    "n_estimators": [100, 200, 300, 500],

    "learning_rate": [0.01, 0.03, 0.05, 0.1],

    "num_leaves": [15, 31, 63],

    "max_depth": [-1, 5, 8, 10],

    "subsample": [0.8, 0.9, 1.0],

    "colsample_bytree": [0.8, 0.9, 1.0]

}

model = LGBMClassifier(
    random_state=RANDOM_STATE,
    verbose=-1
)

search = RandomizedSearchCV(

    estimator=model,

    param_distributions=param_grid,

    n_iter=20,

    scoring="accuracy",

    cv=5,

    random_state=RANDOM_STATE,

    verbose=2,

    n_jobs=-1

)

print("\nSearching for the best parameters...\n")

search.fit(X_train, y_train)

print("\nBest Parameters Found:\n")

print(search.best_params_)

print("\nBest Cross Validation Accuracy:")

print(search.best_score_)

best_model = search.best_estimator_

predictions = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nFinal Test Accuracy:")

print(accuracy)

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    predictions
))

joblib.dump(
    best_model,
    "models/lightgbm_optimized.pkl"
)

print("\nOptimized model saved successfully!")


