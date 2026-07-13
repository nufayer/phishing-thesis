import pandas as pd
import numpy as np
from feature_extractor import extract_features
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

print("=" * 70)
print("BUILDING ROBUST PHISHING DETECTION MODEL (v2)")
print("=" * 70)

# -------------------------------------------------
# 1. Load Tranco Top 1M (Legitimate) - diverse sample
# -------------------------------------------------
print("\n[1/6] Loading Tranco Top 1M (legitimate URLs)...")
tranco_df = pd.read_csv("dataset/top-1m.csv", header=None, names=["rank", "domain"])

# Sample across different ranks for diversity
np.random.seed(42)
sampled_indices = np.random.choice(len(tranco_df), 50000, replace=False)
tranco_domains = tranco_df.iloc[sampled_indices]["domain"].tolist()
legitimate_urls = [f"https://{d}" for d in tranco_domains]

# Add extensive legitimate URLs with subdomains/paths
legit_patterns = [
    # Subdomains
    "https://api.{domain}",
    "https://www.{domain}",
    "https://mail.{domain}",
    "https://drive.{domain}",
    "https://docs.{domain}",
    "https://calendar.{domain}",
    "https://photos.{domain}",
    "https://maps.{domain}",
    "https://translate.{domain}",
    "https://play.{domain}",
    "https://news.{domain}",
    "https://finance.{domain}",
    "https://shop.{domain}",
    "https://store.{domain}",
    "https://blog.{domain}",
    "https://dev.{domain}",
    "https://app.{domain}",
    "https://dashboard.{domain}",
    "https://console.{domain}",
    "https://admin.{domain}",
    "https://secure.{domain}",
    "https://login.{domain}",
    "https://account.{domain}",
    "https://profile.{domain}",
    "https://settings.{domain}",
    "https://help.{domain}",
    "https://support.{domain}",
    "https://status.{domain}",
    "https://cdn.{domain}",
    "https://static.{domain}",
    "https://assets.{domain}",
    "https://media.{domain}",
    "https://images.{domain}",
    "https://api-v2.{domain}",
    "https://v2.{domain}",
    "https://beta.{domain}",
    "https://staging.{domain}",
    "https://test.{domain}",
    "https://sandbox.{domain}",
]

# Apply to top domains
top_brands = ["google.com", "github.com", "facebook.com", "microsoft.com", "apple.com", 
              "amazon.com", "netflix.com", "twitter.com", "linkedin.com", "youtube.com",
              "instagram.com", "wikipedia.org", "reddit.com", "stackoverflow.com", "dropbox.com",
              "adobe.com", "salesforce.com", "slack.com", "zoom.us", "notion.so", "figma.com",
              "vercel.com", "cloudflare.com", "paypal.com", "stripe.com", "twilio.com"]

for brand in top_brands:
    for pattern in legit_patterns:
        legitimate_urls.append(pattern.format(domain=brand))

# Paths and query params
legit_with_paths = [
    "https://{domain}/search?q=test",
    "https://{domain}/user/profile",
    "https://{domain}/settings",
    "https://{domain}/dashboard",
    "https://{domain}/admin",
    "https://{domain}/api/v1/users",
    "https://{domain}/api/v2/products",
    "https://{domain}/blog/post/123",
    "https://{domain}/help/faq",
    "https://{domain}/support/ticket/456",
    "https://{domain}/product/item?id=123",
    "https://{domain}/category/electronics?page=1",
    "https://{domain}/checkout?cart=abc",
    "https://{domain}/login?redirect=/home",
    "https://{domain}/signup?plan=pro",
    "https://{domain}/reset-password?token=xyz",
    "https://{domain}/verify?code=123",
    "https://{domain}/oauth/authorize?client_id=abc",
    "https://{domain}/webhook/stripe",
    "https://{domain}/callback?state=abc",
] * 200

for brand in top_brands[:15]:  # Top 15 brands
    for path in legit_with_paths:
        legitimate_urls.append(path.format(domain=brand))

print(f"     Total legitimate URLs: {len(legitimate_urls)}")

# -------------------------------------------------
# 2. Load PhiUSIIL Phishing URLs
# -------------------------------------------------
print("\n[2/6] Loading PhiUSIIL phishing URLs...")
phi_df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
phishing_urls = phi_df[phi_df["label"] == 0]["URL"].head(50000).tolist()
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

combined.to_csv("dataset/robust_features_v2.csv", index=False)
print("     Saved to dataset/robust_features_v2.csv")

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
    n_estimators=1500,
    learning_rate=0.02,
    random_state=42,
    n_jobs=-1,
    class_weight=class_weight_dict,
    verbose=-1,
    max_depth=12,
    num_leaves=100,
    min_child_samples=30,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1
)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred))

# Feature importance
importances = list(zip(X.columns, model.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
print("\nTop 30 Features:")
for f, i in importances[:30]:
    print(f"  {f:<35} {i}")

# Save
joblib.dump(model, "models/robust_phishing_model_v2.pkl")
print("\nModel saved to models/robust_phishing_model_v2.pkl")

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
    "https://mail.google.com",
    "https://drive.google.com",
    "https://docs.google.com/document/d/abc123",
    "https://google.com/search?q=test",
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