import pandas as pd
import numpy as np
import random
import joblib
import warnings
warnings.filterwarnings('ignore')

from feature_extractor import extract_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

print("=" * 70)
print("MANUAL ENSEMBLE PHISHING MODEL TRAINING (v4 - Skewed Legit)")
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
# 2. Generate Legitimate URLs with EXTENDED Modern Patterns
# -------------------------------------------------
print("\n[2/5] Generating legitimate URLs with extended modern patterns...")

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

MODERN_TLDS = [
    "io", "dev", "app", "co", "ai", "tech", "xyz", "online", "site", "store",
    "tech", "store", "shop", "online", "space", "website", "digital",
    "cloud", "systems", "network", "software", "solutions", "services", "tools",
    "api", "dev", "io", "co", "ai", "ml", "gg", "sh", "tv", "fm", "me", "ly",
]

EXTENDED_BRANDS = []
for brand in ALL_BRANDS:
    base = brand.split(".")[0]
    for tld in MODERN_TLDS:
        if tld not in ["com", "org", "net", "edu", "gov"]:
            EXTENDED_BRANDS.append(f"{base}.{tld}")

ALL_BRANDS_EXTENDED = list(set(ALL_BRANDS + EXTENDED_BRANDS))
print(f"     Total brands (including modern TLDs): {len(ALL_BRANDS_EXTENDED)}")

legitimate_urls = []

# 1. Base domains
for brand in ALL_BRANDS_EXTENDED:
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
    "app", "dashboard", "admin", "console", "panel", "manage", "control",
    "portal", "platform", "client", "customer", "partner", "affiliate",
    "reseller", "whitelabel", "private", "internal", "corp", "enterprise",
    "saas", "cloud", "hosted", "managed", "dedicated", "shared", "trial",
    "demo", "sandbox", "preview", "staging", "staging2", "stg", "uat",
    "qa", "test", "dev", "dev2", "dev3", "local", "localhost", "localnet",
    "internal", "intranet", "extranet", "vpn", "remote", "access", "ssh",
    "git", "ci", "cd", "jenkins", "gitlab", "github", "bitbucket", "svn",
    "jira", "confluence", "wiki", "docs", "doc", "help", "kb", "knowledge",
    "support", "helpdesk", "tickets", "status", "monitor", "metrics", "logs",
    "alerts", "incidents", "oncall", "pager", "ops", "infra", "deploy",
    "release", "build", "artifact", "package", "registry", "docker", "k8s",
    "kubernetes", "helm", "argo", "flux", "tekton", "jenkins", "circleci",
    "travis", "github-actions", "gitlab-ci", "azure-devops", "bitrise",
]

for brand in ALL_BRANDS_EXTENDED:
    for sub in SUBDOMAINS:
        legitimate_urls.append(f"https://{sub}.{brand}")

# Long paths for ALL brands
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
    "/health", "/ready", "/live", "/metrics", "/ping", "/status", "/version",
    "/api/health", "/api/ready", "/api/live", "/api/metrics", "/api/status",
    "/api/v1/health", "/api/v1/ready", "/api/v1/live", "/api/v1/metrics",
    "/graphql", "/graphql/playground", "/graphql/voyager",
    "/swagger", "/swagger-ui", "/redoc", "/openapi.json", "/openapi.yaml",
    "/docs", "/docs/", "/documentation", "/reference", "/api-docs",
    "/.well-known/jwks.json", "/.well-known/openid-configuration",
    "/.well-known/security.txt", "/.well-known/assetlinks.json",
    "/.well-known/apple-app-site-association",
    "/robots.txt", "/sitemap.xml", "/sitemap_index.xml",
    "/ads.txt", "/app-ads.txt", "/sellers.json",
    "/security.txt", "/humans.txt",
]

print("     Generating massive path diversity for ALL brands...")
for brand in ALL_BRANDS_EXTENDED:
    for path in LONG_PATHS:
        for _ in range(60):
            filled = path.format(
                id=random.randint(1, 100000),
                cat=random.choice(["electronics", "books", "clothing", "home", "sports"]),
                brand=random.choice(["nike", "adidas", "sony", "samsung", "apple"]),
                author=random.choice(["john", "jane", "admin", "editor", "author"]),
                tag=random.choice(["python", "javascript", "api", "web", "mobile"]),
            )
            legitimate_urls.append(f"https://{brand}{filled}")

# Query params
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
    "?cursor=abc123", "?after=xyz789", "?before=def456", "?first=10", "?last=20",
    "?filter=active", "?filter[status]=active", "?filter[created_at][gt]=2024-01-01",
    "?include=author,comments,tags", "?fields=id,title,body,created_at",
    "?expand=author,repository,organization", "?sort=-created_at,updated_at",
    "?page[cursor]=abc123", "?page[number]=1&page[size]=20", "?page[offset]=0&page[limit]=50",
]

for brand in ALL_BRANDS_EXTENDED:
    for q in QUERY_PATTERNS:
        legitimate_urls.append(f"https://{brand}{q}")

# Platform domains
PLATFORM_DOMAINS = [
    "firebaseapp.com", "web.app", "vercel.app", "netlify.app", "herokuapp.com",
    "glitch.me", "pages.dev", "cloudflare.workers.dev", "supabase.co",
    "planetscale.dev", "railway.app", "render.com", "fly.dev", "deno.dev",
    "begin.app", "azion.app", "pages.dev", "pages.github.io", "gitlab.io",
    "bitbucket.io", "surge.sh", "now.sh", "zeit.co", "glitch.me", "glitch.com",
    "replit.com", "repl.co", "codesandbox.io", "stackblitz.io",
    "supabase.io", "planetscale.io", "neon.tech", "turso.io", "upstash.io",
    "redis.io", "momento.cloud", "momento.io", "mongoatlas.com", "mongodb.net",
    "cockroachlabs.cloud", "planetscale.com", "neon.tech", "turso.io",
    "cloudflare.dev", "workers.dev", "cfpages.dev", "edge.workers.dev",
    "deno.cloud", "deno.land", "esm.sh", "skypack.dev", "jspm.io",
    "unpkg.com", "cdn.jsdelivr.net", "cdn.statically.io", "raw.githubusercontent.com",
    "github.io", "gitlab.io", "bitbucket.io", "surge.sh", "netlify.com",
    "vercel.com", "netlify.app", "vercel.app", "heroku.com", "herokuapp.com",
    "render.com", "onrender.com", "fly.io", "fly.dev", "railway.app", "up.railway.app",
    "supabase.com", "supabase.io", "planetscale.com", "planetscale.dev",
    "neon.tech", "turso.io", "upstash.com", "upstash.io", "redis.com", "redis.io",
    "mongoatlas.com", "cloud.mongodb.com", "cockroachlabs.cloud", "cockroach.cloud",
    "timescale.cloud", "citusdata.com", "planetscale.com", "planetscale.dev",
    "neon.tech", "neon.cloud", "turso.io", "libsql.org", "sqlite.cloud",
    "deta.sh", "deta.dev", "deta.space", "railway.com", "railway.app", "up.railway.app",
    "render.com", "onrender.com", "fly.io", "fly.dev", "deno.com", "deno.land", "deno.deploy",
    "supabase.co", "supabase.io", "supabase.com", "planetscale.dev", "planetscale.com",
    "neon.tech", "neon.cloud", "neon.tech", "turso.io", "libsql.org",
    "deta.sh", "deta.dev", "deta.space", "railway.com", "railway.app", "up.railway.app",
    "render.com", "onrender.com", "fly.io", "fly.dev", "deno.com", "deno.land", "deno.deploy",
    "supabase.co", "supabase.io", "supabase.com", "planetscale.dev", "planetscale.com",
    "neon.tech", "neon.cloud", "neon.tech", "turso.io", "libsql.org",
    "deta.sh", "deta.dev", "deta.space", "railway.com", "railway.app", "up.railway.app",
    "render.com", "onrender.com", "fly.io", "fly.dev", "deno.com", "deno.land", "deno.deploy",
    "supabase.co", "supabase.io", "supabase.com", "planetscale.dev", "planetscale.com",
    "neon.tech", "neon.cloud", "neon.tech", "turso.io", "libsql.org",
    "deta.sh", "deta.dev", "deta.space", "railway.com", "railway.app", "up.railway.app",
    "render.com", "onrender.com", "fly.io", "fly.dev", "deno.com", "deno.land", "deno.deploy",
    "supabase.co", "supabase.io", "supabase.com", "planetscale.dev", "planetscale.com",
    "neon.tech", "neon.cloud", "neon.tech", "turso.io", "libsql.org",
    "deta.sh", "deta.dev", "deta.space", "railway.com", "railway.app", "up.railway.app",
    "render.com", "onrender.com", "fly.io", "fly.dev", "deno.com", "deno.land", "deno.deploy",
]

for platform in PLATFORM_DOMAINS:
    legitimate_urls.append(f"https://myapp.{platform}")
    legitimate_urls.append(f"https://app.{platform}")
    legitimate_urls.append(f"https://dashboard.{platform}")
    legitimate_urls.append(f"https://api.{platform}")
    legitimate_urls.append(f"https://admin.{platform}")
    legitimate_urls.append(f"https://console.{platform}")
    legitimate_urls.append(f"https://panel.{platform}")
    legitimate_urls.append(f"https://manage.{platform}")
    legitimate_urls.append(f"https://portal.{platform}")
    legitimate_urls.append(f"https://control.{platform}")
    for path in ["/", "/api", "/health", "/ready", "/metrics", "/debug", "/graphql", "/swagger", "/docs", "/openapi.json", "/redoc"]:
        legitimate_urls.append(f"https://myapp.{platform}{path}")

# Modern startup-like domains
STARTUP_PATTERNS = [
    "get{name}.com", "try{name}.com", "use{name}.com", "join{name}.com",
    "app{name}.com", "go{name}.com", "my{name}.com", "the{name}.com",
    "{name}hq.com", "{name}labs.com", "{name}inc.com", "{name}hq.io",
    "{name}labs.io", "{name}inc.io", "{name}hq.co", "{name}labs.co",
    "{name}.app", "{name}.dev", "{name}.ai", "{name}.tech", "{name}.co",
    "{name}.xyz", "{name}.online", "{name}.site", "{name}.tech",
]

STARTUP_NAMES = [
    "acme", "apex", "atlas", "bolt", "cipher", "delta", "echo", "flux", "gamma",
    "helix", "ion", "jolt", "kilo", "lambda", "meridian", "nexus", "orbit", "pulse",
    "quantum", "radix", "sigma", "titan", "vector", "zenith", "zen", "core", "base",
    "prime", "apex", "zen", "nova", "luna", "solar", "stellar", "orbital", "cosmic",
    "atomic", "molecular", "cellular", "organic", "synthetic", "digital", "analog",
    "virtual", "real", "true", "false", "null", "void", "empty", "full", "complete",
    "partial", "total", "sum", "diff", "delta", "gradient", "vector", "matrix",
    "tensor", "scalar", "array", "list", "set", "map", "hash", "tree", "graph",
    "node", "edge", "path", "route", "link", "connect", "bridge", "gateway", "proxy",
    "load", "balance", "scale", "auto", "manual", "auto", "smart", "intelligent",
    "ai", "ml", "dl", "nn", "gpt", "llm", "bert", "transformer", "attention",
]

for pattern in STARTUP_PATTERNS:
    for name in STARTUP_NAMES[:20]:
        if "{name}" in pattern:
            url = pattern.format(name=name)
            legitimate_urls.append(f"https://{url}")
            legitimate_urls.append(f"https://www.{url}")
            legitimate_urls.append(f"https://app.{url}")
            legitimate_urls.append(f"https://api.{url}")

# Modern app paths
MODERN_APP_PATHS = [
    "/api/v1/users", "/api/v1/users/me", "/api/v1/users/me/profile",
    "/api/v1/auth/login", "/api/v1/auth/register", "/api/v1/auth/logout",
    "/api/v1/auth/refresh", "/api/v1/auth/forgot-password", "/api/v1/auth/reset-password",
    "/api/v1/oauth/authorize", "/api/v1/oauth/token", "/api/v1/oauth/callback",
    "/api/v1/webhooks", "/api/v1/webhooks/github", "/api/v1/webhooks/stripe",
    "/graphql", "/graphql/playground", "/graphql/voyager",
    "/swagger", "/swagger-ui", "/redoc", "/openapi.json", "/openapi.yaml",
    "/health", "/healthz", "/ready", "/readyz", "/live", "/livez",
    "/metrics", "/metrics/prometheus", "/metrics/statsd",
    "/debug/pprof", "/debug/vars", "/debug/events",
    "/.well-known/jwks.json", "/.well-known/openid-configuration",
    "/.well-known/security.txt", "/.well-known/assetlinks.json",
    "/.well-known/apple-app-site-association",
    "/robots.txt", "/sitemap.xml", "/sitemap_index.xml",
    "/ads.txt", "/app-ads.txt", "/sellers.json",
    "/security.txt", "/humans.txt",
]

for brand in ALL_BRANDS_EXTENDED:
    for path in MODERN_APP_PATHS:
        legitimate_urls.append(f"https://{brand}{path}")

# Random Tranco
np.random.seed(42)
for domain in tranco_df.sample(15000, random_state=42)["domain"]:
    legitimate_urls.append(f"https://{domain}")
    legitimate_urls.append(f"https://www.{domain}")

# Deduplicate
legitimate_urls = list(dict.fromkeys(legitimate_urls))
print(f"     Total legitimate URLs: {len(legitimate_urls)}")

# -------------------------------------------------
# 3. Skew toward legitimate (2:1 ratio)
# -------------------------------------------------
target_legit = len(phishing_urls) * 2  # 2:1 ratio
legitimate_urls = legitimate_urls[:target_legit]
print(f"\n[3/5] Skewed: {len(legitimate_urls)} legitimate + {len(phishing_urls)} phishing = {len(legitimate_urls) + len(phishing_urls)} total")

# -------------------------------------------------
# 4. Extract Features (NO NETWORK)
# -------------------------------------------------
print("\n[4/5] Extracting features (no network)...")

def extract_batch(urls, label, desc, batch_size=5000):
    features = []
    for i, url in enumerate(urls):
        if i % batch_size == 0 and i > 0:
            print(f"     {desc}: {i}/{len(urls)}")
        try:
            f = extract_features(url, include_network=False)
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

combined.to_csv("dataset/ensemble_features_final.csv", index=False)
print("     Saved to dataset/ensemble_features_final.csv")

# -------------------------------------------------
# 5. Train Individual Models + Manual Ensemble
# -------------------------------------------------
print("\n[5/5] Training Individual Models + Manual Ensemble...")

feature_cols = [c for c in combined.columns if c != "label"]
X = combined[feature_cols]
y = combined["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"     Train: {len(X_train)}, Test: {len(X_test)}")

# Train individual models
print("     Training LGBM...")
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
lgbm.fit(X_train, y_train)

print("     Training XGBoost...")
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
xgb.fit(X_train, y_train)

print("     Training CatBoost...")
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
cat.fit(X_train, y_train)

# Train meta-learner (Stacking)
print("     Training Stacking Meta-learner...")
lgbm_probs = lgbm.predict_proba(X_train)[:, 1].reshape(-1, 1)
xgb_probs = xgb.predict_proba(X_train)[:, 1].reshape(-1, 1)
cat_probs = cat.predict_proba(X_train)[:, 1].reshape(-1, 1)
stack_train = np.hstack([lgbm_probs, xgb_probs, cat_probs])

meta_learner = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
meta_learner.fit(stack_train, y_train)

# Evaluate individual models
print("\nIndividual Model Performance:")
for name, model in [("LGBM", lgbm), ("XGBoost", xgb), ("CatBoost", cat)]:
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    print(f"  {name}: Accuracy={accuracy_score(y_test, pred):.4f}, AUC={roc_auc_score(y_test, prob):.4f}")

# Evaluate ensemble
lgbm_test = lgbm.predict_proba(X_test)[:, 1]
xgb_test = xgb.predict_proba(X_test)[:, 1]
cat_test = cat.predict_proba(X_test)[:, 1]
ensemble_probs = (lgbm_test + xgb_test + cat_test) / 3
ensemble_pred = (ensemble_probs > 0.5).astype(int)

print(f"\nSimple Average Ensemble: Accuracy={accuracy_score(y_test, ensemble_pred):.4f}, AUC={roc_auc_score(y_test, ensemble_probs):.4f}")

stack_test = np.hstack([lgbm_test.reshape(-1, 1), xgb_test.reshape(-1, 1), cat_test.reshape(-1, 1)])
stack_pred = meta_learner.predict(stack_test)
stack_probs = meta_learner.predict_proba(stack_test)[:, 1]
print(f"Stacking Ensemble: Accuracy={accuracy_score(y_test, stack_pred):.4f}, AUC={roc_auc_score(y_test, stack_probs):.4f}")

print("\nClassification Report (Stacking):")
print(classification_report(y_test, stack_pred))

# Feature importance
print("\nTop 30 Features (LGBM importance):")
importances = list(zip(feature_cols, lgbm.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
for f, i in importances[:30]:
    print(f"  {f:<35} {i}")

# Save models
ensemble_package = {
    'lgbm': lgbm,
    'xgb': xgb,
    'cat': cat,
    'meta': meta_learner,
    'feature_cols': feature_cols,
    'method': 'stacking'
}
joblib.dump(ensemble_package, "models/ensemble_final.pkl")
print("\nEnsemble saved to models/ensemble_final.pkl")

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
    ("https://example.com/blog/post/123", "Legitimate"),
    ("https://shop.example.com/product?id=123&cat=electronics", "Legitimate"),
    ("https://api.example.com/v1/users", "Legitimate"),
    ("https://myapp.vercel.app", "Legitimate"),
    ("https://myapp.netlify.app", "Legitimate"),
    ("https://myapp.herokuapp.com", "Legitimate"),
    ("https://myapp.pages.dev", "Legitimate"),
    ("https://myapp.supabase.co", "Legitimate"),
    ("https://myapp.planetscale.dev", "Legitimate"),
    ("https://myapp.railway.app", "Legitimate"),
    ("https://myapp.render.com", "Legitimate"),
    ("https://myapp.fly.dev", "Legitimate"),
    ("https://myapp.deno.dev", "Legitimate"),
    ("http://paypal-login-security.xyz", "Phishing"),
    ("http://fake-bank-login.tk", "Phishing"),
    ("http://verify-account-paypal.com", "Phishing"),
    ("http://apple-id-verification.com", "Phishing"),
    ("http://secure-banking-update.ml", "Phishing"),
]

print("\nTesting ensemble (Stacking):")
correct = 0
for url, expected in test_cases:
    f = extract_features(url, include_network=False)
    df = pd.DataFrame([f])
    
    lgbm_p = lgbm.predict_proba(df)[:, 1][0]
    xgb_p = xgb.predict_proba(df)[:, 1][0]
    cat_p = cat.predict_proba(df)[:, 1][0]
    avg_p = (lgbm_p + xgb_p + cat_p) / 3
    stack_p = meta_learner.predict_proba(np.array([[lgbm_p, xgb_p, cat_p]]))[:, 1][0]
    
    pred = "Legitimate" if stack_p > 0.5 else "Phishing"
    conf = max(stack_p, 1-stack_p) * 100
    is_correct = pred == expected
    if is_correct:
        correct += 1
    status = "OK" if is_correct else "FAIL"
    print(f"  {status} {url:<55} -> {pred} ({conf:.1f}%) [Expected: {expected}]")

accuracy = correct / len(test_cases) * 100
print(f"\nOverall Test Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")

print("\nDone!")