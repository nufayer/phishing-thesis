import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("=" * 70)
print("URL NLP MODEL")
print("=" * 70)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv("dataset/url_dataset.csv")

print(f"\nTotal URLs : {len(df)}")

X = df["URL"]

y = df["label"]

# ---------------------------------------------------
# TF-IDF
# ---------------------------------------------------

print("\nGenerating TF-IDF Features...")

vectorizer = TfidfVectorizer(

    analyzer="char",

    ngram_range=(3,5),

    max_features=5000

)

X_tfidf = vectorizer.fit_transform(X)

print("TF-IDF Shape :", X_tfidf.shape)

# ---------------------------------------------------
# Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X_tfidf,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# ---------------------------------------------------
# Model
# ---------------------------------------------------

print("\nTraining LightGBM...")

model = LGBMClassifier(

    n_estimators=300,

    learning_rate=0.05,

    random_state=42

)

model.fit(X_train, y_train)

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, pred))

print("\n")

print(classification_report(y_test, pred))

# ---------------------------------------------------
# Save
# ---------------------------------------------------

joblib.dump(model, "models/nlp_model.pkl")

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel Saved!")

print("Vectorizer Saved!")