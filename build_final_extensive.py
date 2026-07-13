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
print("BUILDING FINAL PHISHING MODEL - EXTENSIVE LEGIT PATHS")
print("=" * 70)

# -------------------------------------------------
# 1. Generate EXTENSIVE legitimate URLs with LONG PATHS
# -------------------------------------------------
print("\n[1/6] Generating extensive legitimate URLs with long paths...")

tranco_df = pd.read_csv("dataset/top-1m.csv", header=None, names=["rank", "domain"])

CORE_BRANDS = [
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
]

# Long path patterns (real-world app structures) - use simple {id} only
LONG_PATHS = [
    "/document/d/{id}/edit", "/document/d/{id}/view", "/document/d/{id}/export",
    "/drive/folders/{id}", "/drive/files/{id}", "/drive/files/{id}/view",
    "/spreadsheets/d/{id}/edit", "/spreadsheets/d/{id}/view",
    "/presentation/d/{id}/edit", "/presentation/d/{id}/present",
    "/forms/d/{id}/edit", "/forms/d/{id}/view", "/forms/d/{id}/responses",
    "/sites/{id}/edit", "/sites/{id}/preview", "/sites/{id}/publish",
    "/blog/posts/{id}", "/blog/posts/{id}/edit", "/blog/posts/{id}/comments",
    "/blog/categories/{id}", "/blog/tags/{id}", "/blog/author/{id}",
    "/shop/products/{id}", "/shop/products/{id}/edit", "/shop/products/{id}/reviews",
    "/shop/categories/{id}", "/shop/collections/{id}", "/shop/brands/{id}",
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

# Query patterns
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
    "?q=api&page=1", "?q=docs&sort=relevance", "?search=javascript&page=3",
    "?query=python&filter=recent", "?keyword=web&page=2",
    "?id=999&action=edit", "?id=888&action=delete", "?page=5&limit=50",
    "?offset=100&limit=25", "?start=0&end=100", "?since=2024-01-01",
    "?filter=active&value=true", "?category=books&page=2", "?tag=api&page=1",
    "?label=bug&page=1", "?sort=name&order=asc", "?sort=price&order=desc",
    "?utm_source=facebook&utm_medium=social&utm_campaign=promo",
    "?ref=dashboard", "?redirect=/home", "?return_to=/settings",
    "?next=/profile", "?continue=/products", "?callback=/callback",
    "?state=xyz789", "?code=abc123", "?token=secret123",
    "?access_token=xyz&refresh_token=abc", "?id_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
    "?authorization_code=auth123", "?session_id=sess123", "?session=abc",
    "?csrf=token123", "?xsrf=token456", "_csrf=token789", "_xsrf=token999",
    "?format=json", "?format=xml", "?format=csv", "?format=yaml",
    "?pretty=true", "?pretty=1", "?indent=2", "?indent=4",
]

legitimate_urls = []

# 1. Base domains
for brand in CORE_BRANDS:
    legitimate_urls.append(f"https://{brand}")
    legitimate_urls.append(f"https://www.{brand}")

# 2. Subdomains for core brands
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

for brand in CORE_BRANDS:
    for sub in SUBDOMAINS:
        legitimate_urls.append(f"https://{sub}.{brand}")

# 3. Long paths for core brands
for brand in CORE_BRANDS:
    for path in LONG_PATHS:
        filled = path.format(id=random.randint(1, 10000))
        legitimate_urls.append(f"https://{brand}{filled}")

# 4. Query params
for brand in CORE_BRANDS:
    for q in QUERY_PATTERNS:
        legitimate_urls.append(f"https://{brand}{q}")

# 5. Platform domains
PLATFORM_DOMAINS = [
    "firebaseapp.com", "web.app", "vercel.app", "netlify.app", "herokuapp.com",
    "glitch.me", "pages.dev", "cloudflare.workers.dev", "supabase.co",
    "planetscale.dev", "railway.app", "render.com", "fly.dev", "deno.dev",
    "begin.app", "azion.app", "pages.github.io", "gitlab.io", "bitbucket.io",
    "surge.sh", "now.sh", "zeit.co",
]

for platform in PLATFORM_DOMAINS:
    legitimate_urls.append(f"https://myapp.{platform}")
    legitimate_urls.append(f"https://app.{platform}")
    legitimate_urls.append(f"https://dashboard.{platform}")
    legitimate_urls.append(f"https://api.{platform}")
    for path in ["/", "/api", "/health", "/ready", "/metrics", "/debug", "/graphql", "/swagger", "/docs"]:
        legitimate_urls.append(f"https://myapp.{platform}{path}")

# 6. Random Tranco domains
np.random.seed(42)
for domain in tranco_df.sample(15000, random_state=42)["domain"]:
    legitimate_urls.append(f"https://{domain}")
    legitimate_urls.append(f"https://www.{domain}")

# Deduplicate
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

combined.to_csv("dataset/final_extensive_features.csv", index=False)
print("     Saved to dataset/final_extensive_features.csv")

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
    n_estimators=3000,
    learning_rate=0.01,
    random_state=42,
    n_jobs=-1,
    class_weight=class_weight_dict,
    verbose=-1,
    max_depth=16,
    num_leaves=200,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.3,
    reg_lambda=0.3,
    min_split_gain=0.01,
    min_child_weight=1,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
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
joblib.dump(model, "models/final_extensive_model.pkl")
print("\nModel saved to models/final_extensive_model.pkl")

# -------------------------------------------------
# Comprehensive Test
# -------------------------------------------------
print("\n" + "=" * 70)
print("COMPREHENSIVE TEST")
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
    ("https://docs.google.com/document/d/abc123/edit", "Legitimate"),
    ("https://drive.google.com/file/d/abc123/view", "Legitimate"),
    ("https://google.com/search?q=test", "Legitimate"),
    ("https://api.stripe.com/v1/customers", "Legitimate"),
    ("https://dashboard.stripe.com", "Legitimate"),
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