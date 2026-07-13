import pandas as pd
import numpy as np
from feature_extractor import extract_features
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("=" * 70)
print("BUILDING DIVERSE PHISHING DETECTION DATASET")
print("=" * 70)

# -------------------------------------------------
# 1. Load Tranco Top 1M (Legitimate)
# -------------------------------------------------
print("\n[1/5] Loading Tranco Top 1M (legitimate URLs)...")
tranco_df = pd.read_csv("dataset/top-1m.csv", header=None, names=["rank", "domain"])
# Take top 100k for faster processing, add https://
tranco_domains = tranco_df["domain"].head(100000).tolist()
legitimate_urls = [f"https://{d}" for d in tranco_domains]
print(f"     Loaded {len(legitimate_urls)} legitimate URLs")

# -------------------------------------------------
# 2. Load OpenPhish Feed (Phishing)
# -------------------------------------------------
print("\n[2/5] Loading OpenPhish feed (phishing URLs)...")
with open("dataset/openphish_feed.txt", "r") as f:
    phishing_urls = [line.strip() for line in f if line.strip()]
print(f"     Loaded {len(phishing_urls)} phishing URLs")

# -------------------------------------------------
# 3. Extract Features
# -------------------------------------------------
print("\n[3/5] Extracting features...")

def extract_features_batch(urls, label, batch_size=5000):
    features_list = []
    for i, url in enumerate(urls):
        if i % batch_size == 0:
            print(f"     Processed {i}/{len(urls)} ({label})...")
        try:
            feats = extract_features(url)
            feats["label"] = label
            features_list.append(feats)
        except Exception as e:
            print(f"     Error on {url}: {e}")
    return pd.DataFrame(features_list)

legit_features = extract_features_batch(legitimate_urls, 1)
phish_features = extract_features_batch(phishing_urls, 0)

# Combine
combined_df = pd.concat([legit_features, phish_features], ignore_index=True)
print(f"\n     Total samples: {len(combined_df)}")
print(f"     Legitimate: {len(legit_features)}")
print(f"     Phishing: {len(phish_features)}")

# Save combined features
combined_df.to_csv("dataset/combined_features.csv", index=False)
print("     Saved to dataset/combined_features.csv")

# -------------------------------------------------
# 4. Train Model
# -------------------------------------------------
print("\n[4/5] Training LightGBM model...")
X = combined_df.drop(columns=["label"])
y = combined_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------------------------------------------------
# 5. Evaluate
# -------------------------------------------------
print("\n[5/5] Evaluating...")
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred))

# Feature importance
importances = list(zip(X.columns, model.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
print("\nTop 15 Features:")
for f, i in importances[:15]:
    print(f"  {f}: {i}")

# Save model
joblib.dump(model, "models/diverse_model.pkl")
print("\nModel saved to models/diverse_model.pkl")

# -------------------------------------------------
# Quick Test
# -------------------------------------------------
print("\n" + "=" * 70)
print("QUICK TEST ON COMMON URLS")
print("=" * 70)
test_urls = [
    "https://google.com",
    "https://github.com",
    "https://facebook.com",
    "https://microsoft.com",
    "https://amazon.com",
    "http://paypal-login-security.xyz",
    "http://verify-account-paypal.com",
    "http://fake-bank-login.tk",
]

for url in test_urls:
    feats = extract_features(url)
    df = pd.DataFrame([feats])
    p = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    label = "Legitimate" if p == 1 else "Phishing"
    conf = prob.max() * 100
    print(f"  {url:<50} -> {label} ({conf:.1f}%)")

print("\nDone!")