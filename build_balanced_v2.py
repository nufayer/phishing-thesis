import pandas as pd
import numpy as np
from feature_extractor import extract_features
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

print("=" * 70)
print("BUILDING PROPERLY BALANCED PHISHING DETECTION DATASET")
print("=" * 70)

# -------------------------------------------------
# 1. Load Tranco Top 1M (Legitimate) - sample 30k
# -------------------------------------------------
print("\n[1/6] Loading Tranco Top 1M (legitimate URLs)...")
tranco_df = pd.read_csv("dataset/top-1m.csv", header=None, names=["rank", "domain"])
tranco_domains = tranco_df["domain"].head(30000).tolist()
legitimate_urls = [f"https://{d}" for d in tranco_domains]
print(f"     Loaded {len(legitimate_urls)} legitimate URLs")

# -------------------------------------------------
# 2. Load PhiUSIIL dataset - sample equally
# -------------------------------------------------
print("\n[2/6] Loading PhiUSIIL dataset...")
phiusil = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
print(f"     PhiUSIIL shape: {phiusil.shape}")
print(f"     Label dist: {phiusil['label'].value_counts().to_dict()}")

# Sample equally from PhiUSIIL
phishing_phi = phiusil[phiusil['label'] == 0]['URL'].head(25000).tolist()
legit_phi = phiusil[phiusil['label'] == 1]['URL'].head(25000).tolist()
print(f"     Using {len(phishing_phi)} phishing + {len(legit_phi)} legitimate from PhiUSIIL")

# -------------------------------------------------
# 3. Load OpenPhish Feed (Phishing)
# -------------------------------------------------
print("\n[3/6] Loading OpenPhish feed...")
with open("dataset/openphish_feed.txt", "r") as f:
    openphish_urls = [line.strip() for line in f if line.strip()]
print(f"     Loaded {len(openphish_urls)} phishing URLs")

# -------------------------------------------------
# 4. Combine all URLs - BALANCE EXACTLY
# -------------------------------------------------
all_legitimate = legitimate_urls + legit_phi
all_phishing = phishing_phi + openphish_urls

# Balance to minimum
min_count = min(len(all_legitimate), len(all_phishing))
all_legitimate = all_legitimate[:min_count]
all_phishing = all_phishing[:min_count]

print(f"\n[4/6] Balanced dataset: {min_count} legitimate + {min_count} phishing = {min_count*2} total")

# -------------------------------------------------
# 5. Extract Features
# -------------------------------------------------
print("\n[5/6] Extracting features...")

def extract_features_batch(urls, label, desc):
    features_list = []
    for i, url in enumerate(urls):
        if i % 5000 == 0 and i > 0:
            print(f"     {desc}: {i}/{len(urls)}")
        try:
            feats = extract_features(url)
            feats["label"] = label
            features_list.append(feats)
        except Exception as e:
            pass
    return pd.DataFrame(features_list)

legit_features = extract_features_batch(all_legitimate, 1, "Legitimate")
phish_features = extract_features_batch(all_phishing, 0, "Phishing")

combined_df = pd.concat([legit_features, phish_features], ignore_index=True)
print(f"     Total samples: {len(combined_df)}")
print(f"     Label dist: {combined_df['label'].value_counts().to_dict()}")

# Save
combined_df.to_csv("dataset/balanced_features_v2.csv", index=False)
print("     Saved to dataset/balanced_features_v2.csv")

# -------------------------------------------------
# 6. Train Model with proper class weighting
# -------------------------------------------------
print("\n[6/6] Training LightGBM model...")
X = combined_df.drop(columns=["label"])
y = combined_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compute class weights
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print(f"     Class weights: {class_weight_dict}")

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    random_state=42,
    n_jobs=-1,
    class_weight=class_weight_dict,
    verbose=-1
)
model.fit(X_train, y_train)

# Evaluate
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
joblib.dump(model, "models/balanced_diverse_model_v2.pkl")
print("\nModel saved to models/balanced_diverse_model_v2.pkl")

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
    "https://apple.com",
    "https://netflix.com",
    "https://twitter.com",
    "http://paypal-login-security.xyz",
    "http://verify-account-paypal.com",
    "http://fake-bank-login.tk",
    "http://secure-banking-update.ml",
    "http://paypal-login.verify.account-update.com/login.php",
    "http://apple-id-verification.com",
    "https://www.southbankmosaics.com",
    "http://www.teramill.com",
]

for url in test_urls:
    feats = extract_features(url)
    df = pd.DataFrame([feats])
    p = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    label = "Legitimate" if p == 1 else "Phishing"
    conf = prob.max() * 100
    print(f"  {url:<60} -> {label} ({conf:.1f}%)")

print("\nDone!")