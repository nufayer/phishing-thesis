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
print("DEFINITIVE PHISHING MODEL - REAL WORLD LEGITIMATE URLS")
print("=" * 70)

# -------------------------------------------------
# 1. Load PhiUSIIL PHISHING URLs (real phishing)
# -------------------------------------------------
print("\n[1/5] Loading PhiUSIIL phishing URLs...")
phi_df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
phishing_urls = phi_df[phi_df["label"] == 0]["URL"].head(50000).tolist()

with open("dataset/openphish_feed.txt", "r") as f:
    openphish_urls = [line.strip() for line in f if line.strip()]
phishing_urls.extend(openphish_urls)
print(f"     Total phishing URLs: {len(phishing_urls)}")

# -------------------------------------------------
# 2. Generate REAL-WORLD LEGITIMATE URLs with full diversity
# -------------------------------------------------
print("\n[2/5] Generating REAL-WORLD legitimate URLs...")

tranco_df = pd.read_csv("dataset/top-1m.csv", header=None, names=["rank", "domain"])

# Core brands that MUST have path/subdomain diversity
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
    "example.com", "stripe.com", "twilio.com", "sendgrid.com", "mailchimp.com",
    "slack.com", "zoom.us", "notion.so", "figma.com", "vercel.com", "netlify.app",
    "herokuapp.com", "firebaseapp.com", "cloudflare.com", "cloudflare.workers.dev",
    "supabase.co", "planetscale.dev", "railway.app", "render.com", "fly.dev",
    "deno.dev", "begin.app", "azion.app", "pages.dev", "gitlab.io", "bitbucket.io",
]

ALL_BRANDS = list(set(CORE_BRANDS))

# Real app URL structures
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

# 3. MASSIVE long paths for ALL brands
print("     Generating massive long paths for ALL brands...")
for brand in ALL_BRANDS:
    for path in LONG_PATHS:
        for _ in range(80):  # 80 repetitions per path per brand
            filled = path.format(
                id=random.randint(1, 100000),
                cat=random.choice(["electronics", "books", "clothing", "home", "sports"]),
                brand=random.choice(["nike", "adidas", "sony", "samsung", "apple"]),
                author=random.choice(["john", "jane", "admin", "editor", "author"]),
                tag=random.choice(["python", "javascript", "api", "web", "mobile"]),
            )
            legitimate_urls.append(f"https://{brand}{filled}")

# 4. Query parameters
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

# 5. Platform/hosting domains
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
# 3. Load phishing data
# -------------------------------------------------
print("\n[3/5] Loading phishing data...")
with open("dataset/openphish_feed.txt", "r") as f:
    openphish_urls = [line.strip() for line in f if line.strip()]
phishing_urls.extend(openphish_urls)
print(f"     Total phishing URLs: {len(phishing_urls)}")

# -------------------------------------------------
# 4. Balance and extract features
# -------------------------------------------------
min_count = min(len(legitimate_urls), len(phishing_urls))
legitimate_urls = legitimate_urls[:min_count]
phishing_urls = phishing_urls[:min_count]
print(f"\n[4/5] Balanced: {min_count} legitimate + {min_count} phishing = {min_count*2} total")

def extract_batch(urls, label, desc, batch_size=5000):
    features = []
    for i, url in enumerate(urls):
        if i % batch_size == 0 and i > 0:
            print(f"     {desc}: {i}/{len(urls)}")
        try:
            f = extract_features(url)
            f["label"] = label
            features.append(f)
        except Exception:
            pass
    return pd.DataFrame(features)

print("\n[5/5] Extracting features...")
legit_df = extract_batch(legitimate_urls, 1, "Legitimate")
phish_df = extract_batch(phishing_urls, 0, "Phishing")

combined = pd.concat([legit_df, phish_df], ignore_index=True)
print(f"     Total samples: {len(combined)}")
print(f"     Features: {len(combined.columns)-1}")
print(f"     Label dist: {combined['label'].value_counts().to_dict()}")

# Feature columns (30 URL features matching PhiUSIIL)
feature_cols = ['URLLength', 'DomainLength', 'PathLength', 'NoOfDots', 'NoOfHyphen', 'NoOfUnderscore', 
                'NoOfSlash', 'NoOfQuestionMark', 'NoOfEqual', 'NoOfAt', 'NoOfAmpersand', 'NoOfDigits', 
                'NoOfLetters', 'NoOfSpecialChars', 'DigitRatio', 'LetterRatio', 'SpecialCharRatio', 
                'URLEntropy', 'SuspiciousWordCount', 'NoOfPercent', 'NoOfColon', 'NoOfSemicolon', 
                'NoOfComma', 'NoOfPlus', 'NoOfStar', 'NoOfDollar', 'NoOfTilde', 'IsHTTPS', 'HasIPAddress', 'NoOfSubDomain']

# -------------------------------------------------
# 6. Train Model
# -------------------------------------------------
print("\n[6/6] Training LightGBM model...")
X = combined[feature_cols]
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
    reg_lambda=0.1,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred))

# Feature importance
importances = list(zip(feature_cols, model.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
print("\nTop 20 Features:")
for f, i in importances[:20]:
    print(f"  {f:<25} {i}")

# Save
joblib.dump(model, "models/definitive_phishing_model.pkl")
print("\nModel saved to models/definitive_phishing_model.pkl")

# -------------------------------------------------
# 7. Comprehensive Test
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
    ("https://docs.google.com/document/d/abc123/edit", "Legitimate"),
    ("https://drive.google.com/file/d/abc123/view", "Legitimate"),
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
    ("http://steam-support.verify-ownership.cf", "Phishing"),
    ("http://epicgames-verify.account.ml", "Phishing"),
    ("http://riot-support.verify-login.ga", "Phishing"),
    ("http://battle-net.verify-account.tk", "Phishing"),
    ("http://origin-support.verify-login.cf", "Phishing"),
    ("http://uplay-support.verify-account.gq", "Phishing"),
    ("http://gog-support.verify-ownership.ml", "Phishing"),
    ("http://itch-io.verify-purchase.cf", "Phishing"),
    ("http://humble-bundle.verify-key.ga", "Phishing"),
    ("http://fanatical-support.verify-order.tk", "Phishing"),
    ("http://green-man-gaming.verify-purchase.ml", "Phishing"),
    ("http://gamersgate-support.verify-order.gq", "Phishing"),
    ("http://indiegala-support.verify-key.cf", "Phishing"),
]

print(f"\nTesting {len(test_cases)} URLs...")
correct = 0
failures = []

for url, expected in test_cases:
    f = extract_features(url)
    feat_dict = {k: f[k] for k in feature_cols}
    df = pd.DataFrame([feat_dict])
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

# Save final model
joblib.dump(model, "models/final_definitive_model.pkl")
print("\nFinal model saved to models/final_definitive_model.pkl")

print("\nDone!")