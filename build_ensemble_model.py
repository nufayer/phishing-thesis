import pandas as pd
import numpy as np
import random
import joblib
import warnings
warnings.filterwarnings('ignore')

from feature_extractor import extract_features
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

print("=" * 70)
print("ENSEMBLE PHISHING MODEL TRAINING")
print("=" * 70)

# -------------------------------------------------
# 1. Load Phishing URLs
# -------------------------------------------------
print("\n[1/5] Loading phishing URLs...")
phi_df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
phishing_urls = phi_df[phi_df["label"] == 0]["URL"].head(30000).tolist()

with open("dataset/openphish_feed.txt", "r") as f:
    openphish_urls = [line.strip() for line in f if line.strip()]
phishing_urls.extend(openphish_urls)
print(f"     Total phishing URLs: {len(phishing_urls)}")

# -------------------------------------------------
# 2. Generate Legitimate URLs with Network Features
# -------------------------------------------------
print("\n[2/5] Generating legitimate URLs...")

tranco_df = pd.read_csv("dataset/top-1m.csv", header=None, names=["rank", "domain"])

ALL_BRANDS = [
    "google.com", "github.com", "facebook.com", "microsoft.com", "apple.com",
    "amazon.com", "netflix.com", "twitter.com", "linkedin.com", "youtube.com",
    "instagram.com", "wikipedia.org", "reddit.com", "stackoverflow.com", "dropbox.com",
    "adobe.com", "salesforce.com", "slack.com", "zoom.us", "notion.so", "figma.com",
    "vercel.com", "cloudflare.com", "paypal.com", "stripe.com", "twilio.com",
    "sendgrid.com", "mailchimp.com", "shopify.com", "squarespace.com", "wix.com",
    "wordpress.com", "medium.com", "nytimes.com", "wsj.com", "bloomberg.com",
    "reuters.com", "cnn.com", "bbc.com", "theguardian.com", "npr.org", "pbs.org",
    "gov.uk", "nih.gov", "cdc.gov", "who.int", "un.org", "mit.edu", "stanford.edu",
    "harvard.edu", "berkeley.edu", "cmu.edu", "caltech.edu", "princeton.edu",
    "yale.edu", "columbia.edu", "upenn.edu", "cornell.edu", "uchicago.edu",
    "example.com", "stripe.com", "twilio.com", "sendgrid.com", "mailchimp.com",
    "slack.com", "zoom.us", "notion.so", "figma.com", "vercel.app", "netlify.app",
    "herokuapp.com", "firebaseapp.com", "cloudflare.com", "cloudflare.workers.dev",
    "supabase.co", "planetscale.dev", "railway.app", "render.com", "fly.dev",
    "deno.dev", "begin.app", "azion.app", "pages.dev", "pages.github.io", "gitlab.io",
    "bitbucket.io", "surge.sh", "now.sh", "zeit.co", "glitch.me", "glitch.com",
    "replit.com", "repl.co", "codesandbox.io", "stackblitz.io",
]

legitimate_urls = []

# 1. Base domains
for brand in ALL_BRANDS:
    legitimate_urls.append(f"https://{brand}")
    legitimate_urls.append(f"https://www.{brand}")

# 2. Subdomains for ALL brands
SUBDOMAINS = [
    "api", "www", "mail", "drive", "docs", "calendar", "photos", "maps",
    "translate", "play", "news", "finance", "shop", "store", "blog", "dev",
    "app", "dashboard", "console", "admin", "secure", "login", "account",
    "profile", "settings", "help", "support", "status", "cdn", "static",
    "assets", "media", "images", "api-v2", "v2", "beta", "staging", "test",
    "sandbox", "preview", "development", "qa", "uat", "prod", "api2", "api3",
    "v3", "v4", "graphql", "rest", "grpc", "ws", "wss", "cdn2", "cdn3",
    "cdn-eu", "cdn-us", "cdn-asia", "img", "images2", "video", "stream",
    "live", "cdn-media", "cdn-assets", "cdn-static", "auth", "auth0",
    "okta", "sso", "saml", "oidc", "oauth", "openid", "passport", "jwt",
    "identity", "directory", "groups", "teams", "organizations", "tenants",
]

for brand in ALL_BRANDS:
    for sub in SUBDOMAINS:
        legitimate_urls.append(f"https://{sub}.{brand}")

# 3. Long paths for ALL brands (massive diversity)
LONG_PATHS = [
    "/document/d/{id}/edit", "/document/d/{id}/view", "/document/d/{id}/export",
    "/drive/folders/{id}", "/drive/files/{id}", "/drive/files/{id}/view",
    "/spreadsheets/d/{id}/edit", "/spreadsheets/d/{id}/view",
    "/presentation/d/{id}/edit", "/presentation/d/{id}/present",
    "/forms/d/{id}/edit", "/forms/d/{id}/view", "/forms/d/{id}/responses",
    "/sites/{id}/edit", "/sites/{id}/preview", "/sites/{id}/publish",
    "/blog/posts/{id}", "/blog/posts/{id}/edit", "/blog/posts/{id}/comments",
    "/blog/categories/{cat}", "/blog/tags/{tag}", "/blog/author/{author}",
    "/shop/products/{id}", "/shop/products/{id}/edit", "/shop/products/{id}/reviews",
    "/shop/categories/{cat}", "/shop/collections/{id}", "/shop/brands/{brand}",
    "/shop/cart", "/shop/checkout", "/shop/checkout/shipping", "/shop/checkout/payment",
    "/shop/checkout/review", "/shop/orders/{id}", "/shop/orders/{id}/details",
    "/shop/orders/{id}/invoice", "/shop/orders/{id}/tracking",
    "/account/orders", "/account/orders/{id}", "/account/addresses",
    "/account/payment-methods", "/account/subscriptions", "/account/wishlist",
    "/account/reviews", "/account/settings", "/account/profile", "/account/security",
    "/api/v1/users", "/api/v1/users/{id}", "/api/v1/users/{id}/profile",
    "/api/v1/users/{id}/posts", "/api/v1/users/{id}/comments", "/api/v1/users/{id}/settings",
    "/api/v1/posts", "/api/v1/posts/{id}", "/api/v1/posts/{id}/comments",
    "/api/v1/comments", "/api/v1/comments/{id}", "/api/v1/categories",
    "/api/v1/tags", "/api/v1/search", "/api/v1/upload", "/api/v1/download/{id}",
    "/dashboard", "/dashboard/analytics", "/dashboard/reports", "/dashboard/settings",
    "/dashboard/team", "/dashboard/team/members", "/dashboard/team/invites",
    "/dashboard/billing", "/dashboard/billing/invoices", "/dashboard/billing/payment-methods",
    "/dashboard/usage", "/dashboard/usage/api", "/dashboard/usage/storage",
    "/dashboard/logs", "/dashboard/logs/api", "/dashboard/logs/errors",
    "/settings/profile", "/settings/notifications", "/settings/security",
    "/settings/api-keys", "/settings/integrations", "/settings/webhooks",
    "/settings/billing", "/settings/plan", "/settings/usage", "/settings/domain",
    "/projects/{id}", "/projects/{id}/settings", "/projects/{id}/members",
    "/projects/{id}/environments", "/projects/{id}/deployments", "/projects/{id}/logs",
    "/repos/{id}", "/repos/{id}/issues", "/repos/{id}/pulls", "/repos/{id}/actions",
    "/repos/{id}/branches", "/repos/{id}/commits", "/repos/{id}/releases",
    "/repos/{id}/packages", "/repos/{id}/settings", "/repos/{id}/collaborators",
    "/repos/{id}/deploy-keys", "/repos/{id}/webhooks", "/repos/{id}/pages",
    "/repos/{id}/wiki", "/repos/{id}/graphs", "/repos/{id}/insights",
    "/issues/{id}", "/issues/{id}/comments", "/issues/{id}/events", "/issues/{id}/timeline",
    "/pulls/{id}", "/pulls/{id}/files", "/pulls/{id}/commits", "/pulls/{id}/reviews",
    "/pulls/{id}/comments", "/pulls/{id}/checks", "/actions/runs/{id}",
    "/actions/runs/{id}/jobs", "/actions/runs/{id}/logs", "/actions/artifacts/{id}",
    "/packages/{id}", "/packages/{id}/versions", "/packages/{id}/files",
    "/teams/{id}", "/teams/{id}/members", "/teams/{id}/repos", "/teams/{id}/settings",
    "/orgs/{id}", "/orgs/{id}/members", "/orgs/{id}/repos", "/orgs/{id}/teams",
    "/orgs/{id}/settings", "/orgs/{id}/audit-log", "/orgs/{id}/hooks",
    "/marketplace/{id}", "/marketplace/{id}/install", "/marketplace/{id}/reviews",
    "/settings/profile", "/settings/emails", "/settings/keys", "/settings/tokens",
    "/settings/applications", "/settings/organizations", "/settings/repositories",
    "/settings/billing", "/settings/plan", "/settings/security", "/settings/access",
    "/admin/users", "/admin/organizations", "/admin/repositories", "/admin/reports",
    "/admin/hooks", "/admin/logs", "/admin/stats", "/admin/config",
    "/enterprise/settings", "/enterprise/organizations", "/enterprise/users",
    "/enterprise/audit-log", "/enterprise/policies", "/enterprise/licensing",
]

print("     Generating massive path diversity for ALL brands...")
for brand in ALL_BRANDS:
    for path in LONG_PATHS:
        for _ in range(60):  # 60 variations per path per brand
            filled = path.format(
                id=random.randint(1, 100000),
                cat=random.choice(["electronics", "books", "clothing", "home", "sports"]),
                brand=random.choice(["nike", "adidas", "sony", "samsung", "apple"]),
                author=random.choice(["john", "jane", "admin", "editor", "author"]),
                tag=random.choice(["python", "javascript", "api", "web", "mobile"]),
            )
            legitimate_urls.append(f"https://{brand}{filled}")

# 4. Query params
QUERY_PATTERNS = [
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

for brand in ALL_BRANDS:
    for q in QUERY_PATTERNS:
        legitimate_urls.append(f"https://{brand}{q}")

# 5. Platform domains
PLATFORM_DOMAINS = [
    "firebaseapp.com", "web.app", "vercel.app", "netlify.app", "herokuapp.com",
    "glitch.me", "pages.dev", "cloudflare.workers.dev", "supabase.co",
    "planetscale.dev", "railway.app", "render.com", "fly.dev", "deno.dev",
    "begin.app", "azion.app", "pages.dev", "pages.github.io", "gitlab.io",
    "bitbucket.io", "surge.sh", "now.sh", "zeit.co", "glitch.me", "glitch.com",
    "replit.com", "repl.co", "codesandbox.io", "stackblitz.io",
]

for platform in PLATFORM_DOMAINS:
    legitimate_urls.append(f"https://myapp.{platform}")
    legitimate_urls.append(f"https://app.{platform}")
    legitimate_urls.append(f"https://dashboard.{platform}")
    legitimate_urls.append(f"https://api.{platform}")
    for path in ["/", "/api", "/health", "/ready", "/metrics", "/debug", "/graphql", "/swagger", "/docs"]:
        legitimate_urls.append(f"https://myapp.{platform}{path}")

# 6. Random Tranco
np.random.seed(42)
for domain in tranco_df.sample(10000, random_state=42)["domain"]:
    legitimate_urls.append(f"https://{domain}")
    legitimate_urls.append(f"https://www.{domain}")

# Deduplicate
legitimate_urls = list(dict.fromkeys(legitimate_urls))
print(f"     Total legitimate URLs: {len(legitimate_urls)}")

# -------------------------------------------------
# 3. Balance datasets
# -------------------------------------------------
min_count = min(len(legitimate_urls), len(phishing_urls))
legitimate_urls = legitimate_urls[:min_count]
phishing_urls = phishing_urls[:min_count]
print(f"\n[3/5] Balanced: {min_count} legitimate + {min_count} phishing = {min_count*2} total")

# -------------------------------------------------
# 4. Extract Features
# -------------------------------------------------
print("\n[4/5] Extracting features (including network features for first 5000)...")

def extract_batch(urls, label, desc, batch_size=1000):
    features = []
    for i, url in enumerate(urls):
        if i % batch_size == 0 and i > 0:
            print(f"     {desc}: {i}/{len(urls)}")
        try:
            # Disable network features during training for speed
            include_net = False
            f = extract_features(url, include_network=include_net)
            f["label"] = label
            features.append(f)
        except Exception as e:
            pass
    return pd.DataFrame(features)

# Extract for legitimate
legit_df = extract_batch(legitimate_urls, 1, "Legitimate")

# Extract for phishing
phish_df = extract_batch(phishing_urls, 0, "Phishing")

combined = pd.concat([legit_df, phish_df], ignore_index=True)
print(f"     Total samples: {len(combined)}")
print(f"     Features: {len(combined.columns)-1}")
print(f"     Label dist: {combined['label'].value_counts().to_dict()}")

combined.to_csv("dataset/ensemble_features.csv", index=False)
print("     Saved to dataset/ensemble_features.csv")

# -------------------------------------------------
# 5. Train Ensemble Model
# -------------------------------------------------
print("\n[5/5] Training Ensemble Model...")

# Define feature columns (exclude label)
feature_cols = [c for c in combined.columns if c != "label"]
X = combined[feature_cols]
y = combined["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"     Train: {len(X_train)}, Test: {len(X_test)}")

# Define base models
lgbm = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.015,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
    verbose=-1,
    max_depth=14,
    num_leaves=180,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=0.2,
    min_split_gain=0.01,
    min_child_weight=1,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
)

xgb = XGBClassifier(
    n_estimators=1500,
    learning_rate=0.02,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=1,
    max_depth=12,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    eval_metric='logloss',
    verbosity=0,
)

cat = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.02,
    random_seed=42,
    class_weights=[1, 1],
    verbose=False,
    depth=10,
    l2_leaf_reg=3,
    bagging_temperature=1,
    random_strength=1,
)

# Voting ensemble (soft voting)
ensemble = VotingClassifier(
    estimators=[
        ('lgbm', lgbm),
        ('xgb', xgb),
        ('cat', cat),
    ],
    voting='soft',
    n_jobs=-1
)

print("     Training ensemble...")
ensemble.fit(X_train, y_train)

# Evaluate
pred = ensemble.predict(X_test)
prob = ensemble.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred))

# Feature importance (average across models)
print("\nTop 30 Features (LGBM importance):")
importances = list(zip(feature_cols, lgbm.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
for f, i in importances[:30]:
    print(f"  {f:<35} {i}")

# Save model
joblib.dump(ensemble, "models/ensemble_phishing_model.pkl")
print("\nModel saved to models/ensemble_phishing_model.pkl")

# Save feature columns
joblib.dump(feature_cols, "models/ensemble_feature_cols.pkl")
print("Feature columns saved to models/ensemble_feature_cols.pkl")

# -------------------------------------------------
# Quick Test
# -------------------------------------------------
print("\n" + "=" * 70)
print("QUICK TEST")
print("=" * 70)

test_cases = [
    ("https://google.com", "Legitimate"),
    ("https://github.com", "Legitimate"),
    ("https://chatgpt.com", "Legitimate"),
    ("https://chat.openai.com", "Legitimate"),
    ("https://docs.google.com/document/d/abc123", "Legitimate"),
    ("https://api.stripe.com/v1/customers", "Legitimate"),
    ("https://myapp.firebaseapp.com", "Legitimate"),
    ("http://paypal-login-security.xyz", "Phishing"),
    ("http://fake-bank-login.tk", "Phishing"),
    ("http://verify-account-paypal.com", "Phishing"),
]

for url, expected in test_cases:
    f = extract_features(url, include_network=False)
    df = pd.DataFrame([f])
    p = ensemble.predict(df)[0]
    prob = ensemble.predict_proba(df)[0]
    label = "Legitimate" if p == 1 else "Phishing"
    conf = prob.max() * 100
    status = "OK" if label == expected else "FAIL"
    print(f"  {status} {url:<55} -> {label} ({conf:.1f}%) [Expected: {expected}]")

print("\nDone!")