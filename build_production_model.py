import pandas as pd
import numpy as np
import random
from feature_extractor import extract_features
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

print("=" * 70)
print("BUILDING PRODUCTION PHISHING DETECTION MODEL")
print("=" * 70)

# -------------------------------------------------
# 1. Generate diverse legitimate URLs
# -------------------------------------------------
print("\n[1/6] Generating diverse legitimate URLs...")

tranco_df = pd.read_csv("dataset/top-1m.csv", header=None, names=["rank", "domain"])

top_brands = [
    "google.com", "github.com", "facebook.com", "microsoft.com", "apple.com",
    "amazon.com", "netflix.com", "twitter.com", "linkedin.com", "youtube.com",
    "instagram.com", "wikipedia.org", "reddit.com", "stackoverflow.com", "dropbox.com",
    "adobe.com", "salesforce.com", "slack.com", "zoom.us", "notion.so", "figma.com",
    "vercel.com", "cloudflare.com", "paypal.com", "stripe.com", "twilio.com",
    "sendgrid.com", "mailchimp.com", "shopify.com", "squarespace.com", "wix.com",
    "wordpress.com", "medium.com", "nytimes.com", "wsj.com", "bloomberg.com",
    "reuters.com", "cnn.com", "bbc.com", "theguardian.com", "npr.org", "pbs.org",
    "firebaseapp.com", "vercel.app", "netlify.app", "herokuapp.com",
]

legitimate_urls = []

# Base domains
for brand in top_brands:
    legitimate_urls.append(f"https://{brand}")
    legitimate_urls.append(f"https://www.{brand}")

# Common subdomains
subdomains = [
    "api", "www", "mail", "drive", "docs", "calendar", "photos", "maps",
    "translate", "play", "news", "finance", "shop", "store", "blog", "dev",
    "app", "dashboard", "console", "admin", "secure", "login", "account",
    "profile", "settings", "help", "support", "status", "cdn", "static",
    "assets", "media", "images", "api-v2", "v2", "beta", "staging", "test",
    "sandbox", "preview", "development", "qa", "uat", "prod", "graphql", "rest",
    "cdn2", "cdn3", "img", "video", "stream", "live",
]

for brand in top_brands:
    for sub in subdomains:
        legitimate_urls.append(f"https://{sub}.{brand}")

# Common paths
paths = [
    "/", "/home", "/dashboard", "/profile", "/settings", "/account",
    "/login", "/signup", "/signin", "/register", "/auth", "/oauth",
    "/logout", "/password/reset", "/password/forgot", "/verify/email",
    "/verify/phone", "/two-factor", "/2fa", "/security", "/privacy",
    "/terms", "/legal", "/cookies", "/about", "/team", "/careers",
    "/jobs", "/press", "/blog", "/news", "/updates", "/changelog",
    "/releases", "/docs", "/documentation", "/api-docs", "/swagger",
    "/openapi", "/redoc", "/api", "/v1", "/v2", "/v3", "/graphql",
    "/rest", "/webhooks", "/callbacks", "/notifications", "/events",
    "/integrations", "/marketplace", "/apps", "/extensions", "/plugins",
    "/store", "/shop", "/products", "/services", "/solutions", "/pricing",
    "/plans", "/billing", "/invoices", "/subscription", "/usage", "/limits",
    "/quotas", "/analytics", "/reports", "/metrics", "/logs", "/audit",
    "/activity", "/history", "/search", "/explore", "/discover", "/trending",
    "/popular", "/recommended", "/featured", "/categories", "/tags",
    "/collections", "/folders", "/projects", "/repositories", "/repos",
    "/issues", "/pulls", "/commits", "/branches", "/tags", "/releases",
    "/wiki", "/pages", "/actions", "/workflows", "/pipelines", "/builds",
    "/deployments", "/environments", "/users", "/groups", "/roles",
    "/permissions", "/policies", "/auditlogs", "/events", "/metrics",
    "/alerts", "/incidents", "/oncall", "/schedules", "/escalations",
]

for brand in top_brands:
    for path in paths:
        legitimate_urls.append(f"https://{brand}{path}")

# Paths with dynamic IDs
id_paths = [
    "/users/{id}", "/users/{id}/profile", "/users/{id}/settings",
    "/projects/{id}", "/projects/{id}/settings", "/projects/{id}/members",
    "/repos/{id}", "/repos/{id}/issues", "/repos/{id}/pulls",
    "/repos/{id}/commits", "/repos/{id}/branches", "/repos/{id}/tags",
    "/repos/{id}/releases", "/repos/{id}/wiki", "/repos/{id}/pages",
    "/issues/{id}", "/issues/{id}/comments", "/pulls/{id}",
    "/pulls/{id}/files", "/pulls/{id}/commits", "/commits/{id}",
    "/branches/{id}", "/tags/{id}", "/releases/{id}", "/milestones/{id}",
    "/labels/{id}", "/teams/{id}", "/teams/{id}/members", "/orgs/{id}",
    "/orgs/{id}/members", "/orgs/{id}/repos", "/orgs/{id}/teams",
    "/hooks/{id}", "/apps/{id}", "/installations/{id}", "/marketplace/{id}",
    "/topics/{id}", "/gists/{id}", "/emojis", "/licenses", "/rate_limit",
]

for brand in top_brands:
    for path in id_paths:
        filled = path.format(id=random.randint(1, 10000))
        legitimate_urls.append(f"https://{brand}{filled}")

# Query parameters
query_params = [
    "?q=test", "?q=search&page=1", "?q=query&sort=date", "?search=python&page=2",
    "?id=12345", "?id=12345&action=view", "?page=1&limit=20",
    "?category=electronics&page=1", "?tag=python&page=1", "?sort=date&order=desc",
    "?utm_source=google&utm_medium=cpc&utm_campaign=summer",
    "?ref=homepage", "?redirect=/dashboard", "?return_to=/profile",
    "?state=abc123&code=xyz789", "?token=abc123", "?format=json&pretty=true",
    "?fields=id,name,email", "?include=comments,reactions", "?expand=repository",
    "?lang=en", "?locale=en_US", "?timezone=UTC", "?currency=USD",
    "?country=US", "?region=NA", "?city=New_York",
]

for brand in top_brands:
    for q in query_params:
        legitimate_urls.append(f"https://{brand}{q}")

# Random Tranco domains
np.random.seed(42)
for domain in tranco_df.sample(15000, random_state=42)["domain"]:
    legitimate_urls.append(f"https://{domain}")

# Remove duplicates
legitimate_urls = list(dict.fromkeys(legitimate_urls))
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

combined.to_csv("dataset/production_features.csv", index=False)
print("     Saved to dataset/production_features.csv")

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
    n_estimators=2000,
    learning_rate=0.015,
    random_state=42,
    n_jobs=-1,
    class_weight=class_weight_dict,
    verbose=-1,
    max_depth=14,
    num_leaves=150,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=0.2,
    min_split_gain=0.01,
    min_child_weight=1,
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
joblib.dump(model, "models/production_phishing_model.pkl")
print("\nModel saved to models/production_phishing_model.pkl")

# -------------------------------------------------
# Quick Test
# -------------------------------------------------
print("\n" + "=" * 70)
print("QUICK TEST")
print("=" * 70)
test_cases = [
    ("https://google.com", "Legitimate"),
    ("https://github.com", "Legitimate"),
    ("https://facebook.com", "Legitimate"),
    ("https://microsoft.com", "Legitimate"),
    ("https://amazon.com", "Legitimate"),
    ("https://api.github.com", "Legitimate"),
    ("https://mail.google.com", "Legitimate"),
    ("https://drive.google.com", "Legitimate"),
    ("https://docs.google.com/document/d/abc123", "Legitimate"),
    ("https://google.com/search?q=test", "Legitimate"),
    ("https://api.stripe.com/v1/customers", "Legitimate"),
    ("https://shop.example.com/product?id=123&cat=electronics", "Legitimate"),
    ("https://example.com/blog/post/123", "Legitimate"),
    ("https://myapp.firebaseapp.com", "Legitimate"),
    ("https://project.vercel.app", "Legitimate"),
    ("https://site.netlify.app", "Legitimate"),
    ("http://paypal-login-security.xyz", "Phishing"),
    ("http://verify-account-paypal.com", "Phishing"),
    ("http://fake-bank-login.tk", "Phishing"),
    ("http://secure-banking-update.ml", "Phishing"),
    ("http://paypal-login.verify.account-update.com/login.php", "Phishing"),
    ("http://apple-id-verification.com", "Phishing"),
    ("http://www.teramill.com", "Phishing"),
    ("http://www.f0519141.xsph.ru", "Phishing"),
    ("https://service-mitld.firebaseapp.com/", "Phishing"),
]

correct = 0
for url, expected in test_cases:
    f = extract_features(url)
    df = pd.DataFrame([f])
    p = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    label = "Legitimate" if p == 1 else "Phishing"
    conf = prob.max() * 100
    is_correct = label == expected
    if is_correct:
        correct += 1
    status = "OK" if is_correct else "FAIL"
    if not is_correct:
        print(f"  {status} {url:<60} -> {label} ({conf:.1f}%) [Expected: {expected}]")

accuracy = correct / len(test_cases) * 100
print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")

print("\nDone!")