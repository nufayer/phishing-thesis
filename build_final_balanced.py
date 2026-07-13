import pandas as pd
import numpy as np
from feature_extractor import extract_features
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("=" * 70)
print("BUILDING BALANCED DATASET: PhiUSIIL Phishing + Tranco Legitimate")
print("=" * 70)

# -------------------------------------------------
# 1. Load Tranco Top 1M (Legitimate) - sample 100k
# -------------------------------------------------
print("\n[1/5] Loading Tranco Top 1M (legitimate URLs)...")
tranco_df = pd.read_csv("dataset/top-1m.csv", header=None, names=["rank", "domain"])
tranco_domains = tranco_df["domain"].head(100000).tolist()
legitimate_urls = [f"https://{d}" for d in tranco_domains]
print(f"     Loaded {len(legitimate_urls)} legitimate URLs")

# -------------------------------------------------
# 2. Load PhiUSIIL Phishing URLs (real phishing)
# -------------------------------------------------
print("\n[2/5] Loading PhiUSIIL phishing URLs...")
phi_df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
phishing_urls = phi_df[phi_df["label"] == 0]["URL"].head(50000).tolist()  # 50k phishing
print(f"     Loaded {len(phishing_urls)} phishing URLs")

# -------------------------------------------------
# 3. Extract Features
# -------------------------------------------------
print("\n[3/5] Extracting features...")

def extract_batch(urls, label, batch_size=5000):
    feats = []
    for i, url in enumerate(urls):
        if i % batch_size == 0:
            print(f"     Processed {i}/{len(urls)}...")
        try:
            f = extract_features(url)
            f["label"] = label
            feats.append(f)
        except Exception as e:
            pass
    return pd.DataFrame(feats)

legit_df = extract_batch(legitimate_urls, 1)
phish_df = extract_batch(phishing_urls, 0)

combined = pd.concat([legit_df, phish_df], ignore_index=True)
print(f"\n     Total samples: {len(combined)}")
print(f"     Legitimate: {len(legit_df)}")
print(f"     Phishing: {len(phish_df)}")

combined.to_csv("dataset/balanced_features.csv", index=False)
print("     Saved to dataset/balanced_features.csv")

# -------------------------------------------------
# 4. Train Model
# -------------------------------------------------
print("\n[4/5] Training LightGBM...")
X = combined.drop(columns=["label"])
y = combined["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# -------------------------------------------------
# 5. Evaluate
# -------------------------------------------------
print("\n[5/5] Evaluating...")
pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred))

importances = list(zip(X.columns, model.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
print("\nTop 15 Features:")
for f, i in importances[:15]:
    print(f"  {f}: {i}")

joblib.dump(model, "models/balanced_final_model.pkl")
print("\nModel saved to models/balanced_final_model.pkl")

# -------------------------------------------------
# Quick Test
# -------------------------------------------------
print("\n" + "=" * 70)
print("QUICK TEST")
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
    "http://paypal-login.verify.account-update.com/login.php",
    "http://apple-id-verification.com",
    "http://www.teramill.com",
    "http://www.f0519141.xsph.ru",
    "https://service-mitld.firebaseapp.com/",
]

for url in test_urls:
    f = extract_features(url)
    df = pd.DataFrame([f])
    p = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    label = "Legitimate" if p == 1 else "Phishing"
    conf = prob.max() * 100
    print(f"  {url:<60} -> {label} ({conf:.1f}%)")

print("\nDone!")