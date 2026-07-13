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
print("BUILDING FINAL PRODUCTION PHISHING MODEL")
print("=" * 70)

# -------------------------------------------------
# 1. Generate EXTENSIVE diverse legitimate URLs
# -------------------------------------------------
print("\n[1/6] Generating extensive diverse legitimate URLs...")

tranco_df = pd.read_csv("dataset/top-1m.csv", header=None, names=["rank", "domain"])

# Core brands that commonly use subdomains/paths
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

# Platform/hosting domains (legitimate but often used by developers)
PLATFORM_DOMAINS = [
    "firebaseapp.com", "web.app", "vercel.app", "netlify.app", "herokuapp.com",
    "glitch.me", "pages.dev", "cloudflare.workers.dev", "supabase.co", "planetscale.dev",
    "railway.app", "render.com", "fly.dev", "deno.dev", "begin.app", "azion.app",
    "pages.github.io", "gitlab.io", "bitbucket.io", "surge.sh", "now.sh", "zeit.co",
    "cloudflare-ipfs.com", "ipfs.io", "ipfs.dweb.link", "dweb.link",
]

# Extended brand list for similarity
ALL_BRANDS = CORE_BRANDS + PLATFORM_DOMAINS + [
    "example.com", "example.org", "example.net", "test.com", "test.org", "test.net",
    "localhost", "local", "dev", "staging", "qa", "uat", "sandbox", "demo",
]

legitimate_urls = []

# 1. Base domains
for brand in CORE_BRANDS:
    legitimate_urls.append(f"https://{brand}")
    legitimate_urls.append(f"https://www.{brand}")

# 2. Common subdomains for all core brands
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

# 3. Common paths for all core brands
COMMON_PATHS = [
    "/", "/home", "/dashboard", "/profile", "/settings", "/account",
    "/login", "/signup", "/signin", "/register", "/auth", "/oauth",
    "/logout", "/password/reset", "/password/forgot", "/verify/email",
    "/verify/phone", "/two-factor", "/2fa", "/security", "/privacy",
    "/terms", "/legal", "/cookies", "/gdpr", "/ccpa", "/about", "/team",
    "/careers", "/jobs", "/press", "/blog", "/news", "/updates",
    "/changelog", "/releases", "/docs", "/documentation", "/api-docs",
    "/swagger", "/openapi", "/redoc", "/api", "/v1", "/v2", "/v3",
    "/graphql", "/rest", "/webhooks", "/callbacks", "/notifications",
    "/events", "/webhooks/github", "/webhooks/stripe", "/webhooks/slack",
    "/integrations", "/marketplace", "/apps", "/extensions", "/plugins",
    "/store", "/shop", "/products", "/services", "/solutions", "/pricing",
    "/plans", "/billing", "/invoices", "/subscription", "/usage", "/limits",
    "/quotas", "/analytics", "/reports", "/metrics", "/logs", "/audit",
    "/activity", "/history", "/search", "/explore", "/discover", "/trending",
    "/popular", "/recommended", "/featured", "/categories", "/tags",
    "/collections", "/folders", "/projects", "/repositories", "/repos",
    "/issues", "/pulls", "/commits", "/branches", "/tags", "/releases",
    "/wiki", "/pages", "/actions", "/workflows", "/pipelines", "/builds",
    "/deployments", "/environments", "/clusters", "/namespaces", "/pods",
    "/services", "/ingress", "/configmaps", "/secrets", "/volumes",
    "/pvc", "/sc", "/ingressclass", "/networkpolicy", "/rbac",
    "/users", "/groups", "/roles", "/permissions", "/policies",
    "/auditlogs", "/events", "/metrics", "/alerts", "/incidents",
    "/oncall", "/schedules", "/escalations", "/notifications",
    "/webhooks", "/api-keys", "/tokens", "/certificates", "/keys",
    "/ssh-keys", "/gpg-keys", "/deploy-keys", "/access-tokens",
]

for brand in CORE_BRANDS:
    for path in COMMON_PATHS:
        legitimate_urls.append(f"https://{brand}{path}")

# 4. ID-based paths (very common in real apps)
ID_PATHS = [
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
    "/topics/{id}", "/gists/{id}", "/emojis", "/gitignore/templates",
    "/licenses", "/licenses/{id}", "/rate_limit", "/zen", "/meta",
]

for brand in CORE_BRANDS:
    for path in ID_PATHS:
        filled = path.format(id=random.randint(1, 10000))
        legitimate_urls.append(f"https://{brand}{filled}")

# 5. Query parameters - extensive
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
    "?fields=id,name,email", "?include=comments,reactions", "?expand=repository",
    "?embed=repository", "?populate=user", "?select=id,name",
    "?exclude=description,content", "?only=id", "?except=description",
    "?lang=en", "?locale=en_US", "?language=en", "?timezone=UTC",
    "?tz=America/New_York", "?currency=EUR", "?country=GB",
    "?region=EU", "?city=London",
]

for brand in CORE_BRANDS:
    for q in QUERY_PATTERNS:
        legitimate_urls.append(f"https://{brand}{q}")

# 6. Platform/hosting URLs (developer tools)
for platform in PLATFORM_DOMAINS:
    legitimate_urls.append(f"https://myapp.{platform}")
    legitimate_urls.append(f"https://app.{platform}")
    legitimate_urls.append(f"https://dashboard.{platform}")
    legitimate_urls.append(f"https://api.{platform}")
    for path in ["/", "/api", "/health", "/ready", "/metrics", "/debug", "/graphql", "/swagger", "/docs"]:
        legitimate_urls.append(f"https://myapp.{platform}{path}")

# 7. Random Tranco domains
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

combined.to_csv("dataset/final_production_features.csv", index=False)
print("     Saved to dataset/final_production_features.csv")

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
joblib.dump(model, "models/final_phishing_model.pkl")
print("\nModel saved to models/final_phishing_model.pkl")

# -------------------------------------------------
# Comprehensive Test
# -------------------------------------------------
print("\n" + "=" * 70)
print("COMPREHENSIVE TEST")
print("=" * 70)

test_cases = [
    # Legitimate - major brands
    ("https://google.com", "Legitimate"),
    ("https://github.com", "Legitimate"),
    ("https://facebook.com", "Legitimate"),
    ("https://microsoft.com", "Legitimate"),
    ("https://amazon.com", "Legitimate"),
    ("https://apple.com", "Legitimate"),
    ("https://netflix.com", "Legitimate"),
    ("https://twitter.com", "Legitimate"),
    ("https://linkedin.com", "Legitimate"),
    ("https://youtube.com", "Legitimate"),
    ("https://instagram.com", "Legitimate"),
    ("https://wikipedia.org", "Legitimate"),
    ("https://reddit.com", "Legitimate"),
    ("https://stackoverflow.com", "Legitimate"),
    ("https://dropbox.com", "Legitimate"),
    ("https://adobe.com", "Legitimate"),
    ("https://salesforce.com", "Legitimate"),
    ("https://slack.com", "Legitimate"),
    ("https://zoom.us", "Legitimate"),
    ("https://notion.so", "Legitimate"),
    
    # Legitimate - with subdomains
    ("https://api.github.com", "Legitimate"),
    ("https://mail.google.com", "Legitimate"),
    ("https://drive.google.com", "Legitimate"),
    ("https://docs.google.com/document/d/abc123", "Legitimate"),
    ("https://calendar.google.com", "Legitimate"),
    ("https://photos.google.com", "Legitimate"),
    ("https://maps.google.com", "Legitimate"),
    ("https://translate.google.com", "Legitimate"),
    ("https://play.google.com", "Legitimate"),
    ("https://news.google.com", "Legitimate"),
    ("https://finance.google.com", "Legitimate"),
    ("https://api.stripe.com/v1/customers", "Legitimate"),
    ("https://dashboard.stripe.com", "Legitimate"),
    ("https://api.twilio.com", "Legitimate"),
    ("https://api.sendgrid.com", "Legitimate"),
    ("https://api.mailchimp.com", "Legitimate"),
    ("https://api.shopify.com", "Legitimate"),
    ("https://admin.shopify.com", "Legitimate"),
    ("https://app.netlify.com", "Legitimate"),
    ("https://app.vercel.com", "Legitimate"),
    ("https://app.heroku.com", "Legitimate"),
    ("https://console.firebase.google.com", "Legitimate"),
    ("https://console.cloud.google.com", "Legitimate"),
    ("https://console.aws.amazon.com", "Legitimate"),
    ("https://portal.azure.com", "Legitimate"),
    ("https://myapp.firebaseapp.com", "Legitimate"),
    ("https://myapp.vercel.app", "Legitimate"),
    ("https://myapp.netlify.app", "Legitimate"),
    ("https://myapp.herokuapp.com", "Legitimate"),
    ("https://myapp.glitch.me", "Legitimate"),
    ("https://myapp.pages.dev", "Legitimate"),
    ("https://myapp.supabase.co", "Legitimate"),
    ("https://myapp.planetscale.dev", "Legitimate"),
    ("https://myapp.railway.app", "Legitimate"),
    ("https://myapp.render.com", "Legitimate"),
    ("https://myapp.fly.dev", "Legitimate"),
    ("https://myapp.deno.dev", "Legitimate"),
    
    # Legitimate - with paths
    ("https://google.com/search?q=test", "Legitimate"),
    ("https://github.com/user/repo", "Legitimate"),
    ("https://github.com/user/repo/issues/123", "Legitimate"),
    ("https://github.com/user/repo/pull/456", "Legitimate"),
    ("https://docs.google.com/document/d/abc123/edit", "Legitimate"),
    ("https://drive.google.com/file/d/abc123/view", "Legitimate"),
    ("https://stackoverflow.com/questions/12345", "Legitimate"),
    ("https://stackoverflow.com/questions/12345/how-to-do-x", "Legitimate"),
    ("https://stackoverflow.com/questions/tagged/python", "Legitimate"),
    ("https://reddit.com/r/python", "Legitimate"),
    ("https://reddit.com/r/python/comments/abc123", "Legitimate"),
    ("https://medium.com/@user/article-title-abc123", "Legitimate"),
    ("https://dev.to/user/post-title", "Legitimate"),
    ("https://shop.example.com/product/abc123", "Legitimate"),
    ("https://shop.example.com/product?id=123&cat=electronics", "Legitimate"),
    ("https://shop.example.com/category/electronics?page=2", "Legitimate"),
    ("https://shop.example.com/cart", "Legitimate"),
    ("https://shop.example.com/checkout?cart=abc123", "Legitimate"),
    ("https://shop.example.com/account/orders", "Legitimate"),
    ("https://shop.example.com/account/settings", "Legitimate"),
    ("https://api.example.com/v1/users", "Legitimate"),
    ("https://api.example.com/v1/users/123", "Legitimate"),
    ("https://api.example.com/v1/users/123/posts", "Legitimate"),
    ("https://api.example.com/graphql", "Legitimate"),
    ("https://api.example.com/health", "Legitimate"),
    ("https://api.example.com/metrics", "Legitimate"),
    ("https://api.example.com/docs", "Legitimate"),
    ("https://api.example.com/swagger", "Legitimate"),
    ("https://api.example.com/openapi.json", "Legitimate"),
    
    # Phishing - obvious
    ("http://paypal-login-security.xyz", "Phishing"),
    ("http://verify-account-paypal.com", "Phishing"),
    ("http://fake-bank-login.tk", "Phishing"),
    ("http://secure-banking-update.ml", "Phishing"),
    ("http://paypal-login.verify.account-update.com/login.php", "Phishing"),
    ("http://apple-id-verification.com", "Phishing"),
    ("http://www.teramill.com", "Phishing"),
    ("http://www.f0519141.xsph.ru", "Phishing"),
    ("https://service-mitld.firebaseapp.com/", "Phishing"),
    ("http://secure-login.banking.update.ml", "Phishing"),
    ("http://verify.account.security.update.cf", "Phishing"),
    ("http://microsoft-account.verify-login.xyz", "Phishing"),
    ("http://apple-support.verify-account.tk", "Phishing"),
    ("http://google-security.verify-login.ml", "Phishing"),
    ("http://amazon-account.update-billing.cf", "Phishing"),
    ("http://facebook-security.verify-identity.ga", "Phishing"),
    ("http://instagram-support.verify-login.gq", "Phishing"),
    ("http://twitter-verify.account.suspended.ml", "Phishing"),
    ("http://linkedin-security.verify-profile.cf", "Phishing"),
    ("http://github-verify.account.suspended.tk", "Phishing"),
    ("http://paypal-support.verify-payment.ga", "Phishing"),
    ("http://stripe-support.verify-account.cf", "Phishing"),
    ("http://coinbase-verify.identity.ml", "Phishing"),
    ("http://binance-support.security-check.tk", "Phishing"),
    ("http://discord-verify.account.claim.gq", "Phishing"),
]

correct = 0
failures = []

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
    else:
        failures.append((url, expected, label, conf))
        print(f"  FAIL: {url:<60} -> {label} ({conf:.1f}%) [Expected: {expected}]")

accuracy = correct / len(test_cases) * 100
print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")

if failures:
    print(f"\nFailures ({len(failures)}):")
    for url, expected, got, conf in failures:
        print(f"  {url} -> {got} (expected {expected})")

print("\nDone!")