import pandas as pd
import scipy.sparse as sp
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from lightgbm import LGBMClassifier

print("=" * 70)
print("HYBRID PHISHING DETECTION MODEL")
print("=" * 70)

# ----------------------------------------------------
# Load URL Dataset
# ----------------------------------------------------

print("\nLoading datasets...")

url_df = pd.read_csv("dataset/url_dataset.csv")

feature_df = pd.read_csv("dataset/generated_features.csv")

print("Datasets Loaded Successfully!")

# ----------------------------------------------------
# Labels
# ----------------------------------------------------

y = url_df["label"]

# ----------------------------------------------------
# TF-IDF
# ----------------------------------------------------

print("\nGenerating TF-IDF Features...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5),
    max_features=5000
)

X_text = vectorizer.fit_transform(url_df["URL"])

print("TF-IDF Shape :", X_text.shape)

# ----------------------------------------------------
# Handcrafted Features
# ----------------------------------------------------

X_numeric = feature_df.drop(columns=["label"])

print("Numeric Feature Shape :", X_numeric.shape)

print(X_numeric.columns.tolist())

# ----------------------------------------------------
# Combine Features
# ----------------------------------------------------

print("\nCombining Features...")

X = sp.hstack(
    [
        sp.csr_matrix(X_numeric.values),
        X_text
    ]
)

print("Combined Shape :", X.shape)

# ----------------------------------------------------
# Train/Test Split
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", len(y_train))

print("Testing Samples :", len(y_test))

# ----------------------------------------------------
# LightGBM
# ----------------------------------------------------

print("\nTraining Hybrid Model...")

model = LGBMClassifier(

    n_estimators=500,

    learning_rate=0.05,

    random_state=42

)

model.fit(X_train, y_train)

print("Training Complete!")

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

pred = model.predict(X_test)

prob = model.predict_proba(X_test)[:,1]

# ----------------------------------------------------
# Results
# ----------------------------------------------------

accuracy = accuracy_score(y_test,pred)

precision = precision_score(y_test,pred)

recall = recall_score(y_test,pred)

f1 = f1_score(y_test,pred)

roc = roc_auc_score(y_test,prob)

print("\n"+"="*70)

print("HYBRID MODEL RESULTS")

print("="*70)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print(f"ROC AUC   : {roc:.4f}")

print("\nClassification Report\n")

print(classification_report(y_test,pred))

# ----------------------------------------------------
# Save
# ----------------------------------------------------

joblib.dump(model,"models/hybrid_model.pkl")

joblib.dump(vectorizer,"models/hybrid_vectorizer.pkl")

print("\nHybrid Model Saved!")

print("Vectorizer Saved!")