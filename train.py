import pandas as pd
import joblib

from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATASET
# ==========================================

data, meta = arff.loadarff("dataset/Training Dataset.arff")

df = pd.DataFrame(data)

print("Dataset Loaded Successfully!")

# ==========================================
# CONVERT BYTE VALUES TO INTEGER
# ==========================================

for column in df.columns:
    df[column] = df[column].apply(
        lambda x: int(x.decode("utf-8"))
    )

print("\nDataset Converted Successfully!")

print(df.head())

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Result", axis=1)

y = df["Result"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# RESULTS
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\n===================================")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("===================================")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/phishing_model.pkl"
)

print("\nModel Saved Successfully!")