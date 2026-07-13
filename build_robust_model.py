import pandas as pd
import numpy as np
from feature_extractor import extract_features
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

print("=" * 70)
print("BUILDING ROBUST PHISHING DETECTION MODEL")
print("=" * 70)

# -------------------------------------------------
# 1. Load Tranco Top 1M (Legitimate) - diverse sample
# -------------------------------------------------
print("\n[1/6] Loading Tranco Top 1M (legitimate URLs)...")
tranco_df = pd.read_csv("dataset/top-1m.csv", header=None, names=["rank", "domain"])

# Sample across different ranks for diversity
np.random.seed(42)
sampled_indices = np.random.choice(len(tranco_df), 40000, replace=False)
tranco_domains = tranco_df.iloc[sampled_indices]["domain"].tolist()
legitimate_urls = [f"https://{d}" for d in tranco_domains]

# Add URLs with paths/params (real-world patterns)
legit_with_paths = [
    "https://google.com/search?q=test",
    "https://github.com/user/repo",
    "https://api.github.com/users/octocat",
    "https://docs.google.com/document/d/abc123/edit",
    "https://drive.google.com/file/d/123/view",
    "https://facebook.com/profile.php?id=123",
    "https://twitter.com/user/status/123",
    "https://linkedin.com/in/username",
    "https://youtube.com/watch?v=abc123",
    "https://amazon.com/dp/B012345678",
    "https://microsoft.com/en-us/store",
    "https://apple.com/iphone",
    "https://netflix.com/browse",
    "https://stackoverflow.com/questions/12345",
    "https://reddit.com/r/python/comments/abc123",
    "https://wikipedia.org/wiki/Python",
    "https://dropbox.com/s/abc123/file.pdf",
    "https://slack.com/app_redirect",
    "https://zoom.us/j/123456789",
    "https://notion.so/workspace/page",
    "https://figma.com/file/abc123/design",
    "https://vercel.com/dashboard",
    "https://cloudflare.com/a/example.com",
    "https://paypal.com/home",
    "https://stripe.com/docs/api",
    "https://api.stripe.com/v1/customers",
    "https://twilio.com/console",
    "https://sendgrid.com/ui",
    "https://mailchimp.com/campaigns",
    "https://shopify.com/admin",
    "https://wordpress.com/wp-admin",
    "https://medium.com/@user/post",
] * 200  # Repeat for balance

legitimate_urls.extend(legit_with_paths)
print(f"     Total legitimate URLs: {len(legitimate_urls)}")

# -------------------------------------------------
# 2. Load PhiUSIIL Phishing URLs
# -------------------------------------------------
print("\n[2/6] Loading PhiUSIIL phishing URLs...")
phi_df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
phishing_urls = phi_df[phi_df["label"] == 0]["URL"].head(40000).tolist()
print(f"     Phishing URLs: {len(phishing_urls)}")

# -------------------------------------------------
# 3. Load OpenPhish Feed
# -------------------------------------------------
print("\n[3/6] Loading OpenPhish feed...")
with open("dataset/openphish_feed.txt", "r") as f:
    openphish_urls = [line.strip() for line in f if line.strip()]
phishing_urls.extend(openphish_urls)
print(f"     Total phishing URLs: {len(phishing_urls)}")

# -------------------------------------------------
# 4. Balance datasets
# -------------------------------------------------
min_count = min(len(legitimate_urls), len(phishing_urls))
legitimate_urls = legitimate_urls[:min_count]
phishing_urls = phishing_urls[:min_count]
print(f"\n[4/6] Balanced: {min_count} legitimate + {min_count} phishing = {min_count*2} total")

# -------------------------------------------------
# 5. Extract Features
# -------------------------------------------------
print("\n[5/6] Extracting features...")

def extract_batch(urls, label, desc, batch_size=5000):
    features = []
    for i, url in enumerate(urls):
        if i % batch_size == 0 and i > 0:
            print(f"     {desc}: {i}/{len(urls)}")
        try:
            f = extract_features(url)
            f["label"] = label
            features.append(f)
        except Exception as e:
            pass
    return pd.DataFrame(features)

legit_df = extract_batch(legitimate_urls, 1, "Legitimate")
phish_df = extract_batch(phishing_urls, 0, "Phishing")

combined = pd.concat([legit_df, phish_df], ignore_index=True)
print(f"     Total samples: {len(combined)}")
print(f"     Features: {len(combined.columns)-1}")
print(f"     Label dist: {combined['label'].value_counts().to_dict()}")

# Save
combined.to_csv("dataset/robust_features.csv", index=False)
print("     Saved to dataset/robust_features.csv")

# -------------------------------------------------
# 6. Train Model
# -------------------------------------------------
print("\n[6/6] Training LightGBM model...")
X = combined.drop(columns=["label"])
y = combined["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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
    verbose=-1,
    max_depth=10,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.9
)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred))

# Feature importance
importances = list(zip(X.columns, model.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
print("\nTop 25 Features:")
for f, i in importances[:25]:
    print(f"  {f:<35} {i}")

# Save
joblib.dump(model, "models/robust_phishing_model.pkl")
print("\nModel saved to models/robust_phishing_model.pkl")

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
    "https://api.github.com",
    "https://google.com/search?q=test",
    "https://docs.google.com/document/d/abc123",
    "https://api.stripe.com/v1/customers",
    "https://shop.example.com/product?id=123&cat=electronics",
    "http://paypal-login-security.xyz",
    "http://verify-account-paypal.com",
    "http://fake-bank-login.tk",
    "http://secure-banking-update.ml",
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