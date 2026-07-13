import time
import joblib
import pandas as pd

from scipy.io import arff

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
print("=" * 70)
print("OPTIMIZING MACHINE LEARNING MODELS")
print("=" * 70)

print("\nLoading Dataset...")

data, meta = arff.loadarff("dataset/Training Dataset.arff")

df = pd.DataFrame(data)

for column in df.columns:
    df[column] = df[column].str.decode("utf-8").astype(int)

df["Result"] = df["Result"].replace({-1: 0})

X = df.drop("Result", axis=1)
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset Ready!")
print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

def optimize_model(name, model, params):

    print("\n" + "=" * 60)
    print(f"Optimizing {name}")
    print("=" * 60)

    start = time.time()

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=30,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    end = time.time()

    print("\nBest Parameters:")

    print(search.best_params_)

    print(f"\nAccuracy : {accuracy:.4f}")

    print(f"Training Time : {end-start:.2f} sec")

    filename = f"models/{name.lower().replace(' ','_')}_optimized.pkl"

    joblib.dump(best_model, filename)

    print(f"Saved -> {filename}")

    return best_model

# ==========================================================
# RANDOM FOREST
# ==========================================================

rf_model = RandomForestClassifier(
    random_state=42
)

rf_params = {

    "n_estimators": [200,300,400,500],

    "max_depth": [10,15,20,25,None],

    "min_samples_split": [2,5,10],

    "min_samples_leaf": [1,2,4],

    "max_features": ["sqrt","log2",None]

}

optimized_rf = optimize_model(
    "Random Forest",
    rf_model,
    rf_params
)





# ==========================================================
# XGBOOST
# ==========================================================

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

xgb_params = {

    "n_estimators": [200,300,400,500],

    "learning_rate": [0.01,0.05,0.1,0.2],

    "max_depth": [4,6,8,10],

    "subsample": [0.7,0.8,0.9,1.0],

    "colsample_bytree": [0.7,0.8,0.9,1.0],

    "gamma": [0,0.1,0.2,0.3]

}

optimized_xgb = optimize_model(
    "XGBoost",
    xgb_model,
    xgb_params
)


# ==========================================================
# CATBOOST
# ==========================================================

cat_model = CatBoostClassifier(
    verbose=0,
    random_state=42
)

cat_params = {

    "iterations": [200,300,400,500],

    "learning_rate": [0.01,0.03,0.05,0.1],

    "depth": [4,6,8,10],

    "l2_leaf_reg": [1,3,5,7,9]

}

optimized_cat = optimize_model(
    "CatBoost",
    cat_model,
    cat_params
)