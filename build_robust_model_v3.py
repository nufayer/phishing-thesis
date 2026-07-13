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
print("BUILDING ROBUST PHISHING DETECTION MODEL (v3 - Final)")
print("=" * 70)

# -------------------------------------------------
# 1. Load Tranco + Generate EXTENSIVE legitimate URLs
# -------------------------------------------------
print("\n[1/6] Generating diverse legitimate URLs...")
tranco_df = pd.read_csv("dataset/top-1m.csv", header=None, names=["rank", "domain"])

# Top 200 brands for pattern generation
top_brands = [
    "google.com", "github.com", "facebook.com", "microsoft.com", "apple.com",
    "amazon.com", "netflix.com", "twitter.com", "linkedin.com", "youtube.com",
    "instagram.com", "wikipedia.org", "reddit.com", "stackoverflow.com", "dropbox.com",
    "adobe.com", "salesforce.com", "slack.com", "zoom.us", "notion.so", "figma.com",
    "vercel.com", "cloudflare.com", "paypal.com", "stripe.com", "twilio.com",
    "sendgrid.com", "mailchimp.com", "shopify.com", "squarespace.com", "wix.com",
    "wordpress.com", "medium.com", "nytimes.com", "wsj.com", "bloomberg.com",
    "reuters.com", "cnn.com", "bbc.com", "theguardian.com", "npr.org", "pbs.org",
    "gov.uk", "nih.gov", "cdc.gov", "who.int", "un.org",
    "atlassian.com", "jira.com", "confluence.com", "bitbucket.org", "gitlab.com",
    "docker.com", "kubernetes.io", "nginx.org", "apache.org", "linux.org",
    "python.org", "nodejs.org", "npmjs.com", "pypi.org", "maven.org",
    "rubygems.org", "crates.io", "packagist.org", "nuget.org", "bower.io",
    "firebase.google.com", "cloud.google.com", "aws.amazon.com", "azure.microsoft.com",
    "cloudflare.com", "fastly.com", "akamai.com", "cdn.jsdelivr.net", "unpkg.com",
    "fonts.googleapis.com", "fonts.gstatic.com", "ajax.googleapis.com",
    "maps.googleapis.com", "maps.gstatic.com", "www.gstatic.com",
    "accounts.google.com", "myaccount.google.com", "security.google.com",
    "admin.google.com", "workspace.google.com", "cloud.google.com",
    "console.cloud.google.com", "monitoring.google.com", "logging.google.com",
    "appengine.google.com", "bigquery.cloud.google.com", "storage.googleapis.com",
    "drive.google.com", "docs.google.com", "sheets.google.com", "slides.google.com",
    "forms.google.com", "sites.google.com", "keep.google.com", "photos.google.com",
    "calendar.google.com", "contacts.google.com", "groups.google.com",
    "chat.google.com", "meet.google.com", "voice.google.com", "translate.google.com",
    "books.google.com", "scholar.google.com", "patents.google.com", "trends.google.com",
    "analytics.google.com", "ads.google.com", "tagmanager.google.com",
    "optimize.google.com", "surveys.google.com", "datastudio.google.com",
    "firebase.google.com", "cloudfunctions.net", "cloudrun.app", "run.app",
    "cloudfunctions.net", "appspot.com", "web.app", "firebaseapp.com",
    "github.io", "githubusercontent.com", "githubassets.com", "github.com",
    "api.github.com", "raw.githubusercontent.com", "gist.githubusercontent.com",
    "githubusercontent.com", "github.com", "github.io", "github.dev",
    "vscode.dev", "github.dev", "codespaces.dev", "githubpreview.dev",
    "gitlab.io", "gitlab.com", "gitlab.io", "gitlab-static.net",
    "bitbucket.io", "bitbucket.org", "bitbucket.io", "atlassian.net",
    "jira.com", "confluence.com", "confluence.io", "trello.com",
    "slack.com", "slack-files.com", "slack-imgs.com", "slack-edge.com",
    "slack-redir.net", "slack.com", "slackhq.com", "slackapi.com",
    "discord.com", "discordapp.com", "discord.gg", "discord.media",
    "discord.new", "discord.gift", "discord.store", "discord.bio",
    "notion.so", "notion.site", "notion.new", "notion.ai",
    "figma.com", "figma.io", "figma-files.com", "figma-images.com",
    "vercel.app", "vercel.com", "vercel-dns.com", "vercel-inspect.com",
    "netlify.app", "netlify.com", "netlify.global", "netlify-cdn.com",
    "cloudflare.com", "cloudflare.net", "cloudflare-dns.com", "cloudflare-ipfs.com",
    "pages.dev", "workers.dev", "r2.dev", "d1.dev", "kv.dev",
    "stripe.com", "api.stripe.com", "dashboard.stripe.com", "js.stripe.com",
    "paypal.com", "www.paypal.com", "api.paypal.com", "api-m.paypal.com",
    "paypalobjects.com", "paypal-communication.com",
    "microsoft.com", "www.microsoft.com", "login.microsoft.com", "account.microsoft.com",
    "office.com", "www.office.com", "portal.office.com", "outlook.office.com",
    "teams.microsoft.com", "sharepoint.com", "sharepointonline.com",
    "onedrive.com", "onedrive.live.com", "1drv.ms", "1drv.ws",
    "azure.com", "portal.azure.com", "management.azure.com", "login.microsoftonline.com",
    "microsoftonline.com", "microsoft.com", "msn.com", "bing.com",
    "apple.com", "www.apple.com", "icloud.com", "me.com", "mac.com",
    "itunes.com", "apps.apple.com", "music.apple.com", "tv.apple.com",
    "developer.apple.com", "appstore.com", "icloud.com", "idmsa.apple.com",
    "amazon.com", "www.amazon.com", "aws.amazon.com", "console.aws.amazon.com",
    "s3.amazonaws.com", "cloudfront.net", "amazonaws.com", "amazon.com",
    "facebook.com", "www.facebook.com", "fb.com", "m.facebook.com",
    "instagram.com", "www.instagram.com", "cdninstagram.com",
    "whatsapp.com", "web.whatsapp.com", "api.whatsapp.com",
    "twitter.com", "www.twitter.com", "api.twitter.com", "t.co",
    "linkedin.com", "www.linkedin.com", "api.linkedin.com", "media.licdn.com",
    "youtube.com", "www.youtube.com", "youtu.be", "ytimg.com", "googlevideo.com",
    "netflix.com", "www.netflix.com", "api.netflix.com", "netflix.net",
    "spotify.com", "www.spotify.com", "api.spotify.com", "scdn.co",
    "airbnb.com", "www.airbnb.com", "api.airbnb.com", "a0.muscache.com",
    "uber.com", "www.uber.com", "api.uber.com", "uberstatic.com",
    "lyft.com", "www.lyft.com", "api.lyft.com",
    "doordash.com", "www.doordash.com", "api.doordash.com",
    "instacart.com", "www.instacart.com", "api.instacart.com",
    "robinhood.com", "www.robinhood.com", "api.robinhood.com",
    "coinbase.com", "www.coinbase.com", "api.coinbase.com", "pro.coinbase.com",
    "binance.com", "www.binance.com", "api.binance.com",
    "kraken.com", "www.kraken.com", "api.kraken.com",
    "gemini.com", "www.gemini.com", "api.gemini.com",
    "openai.com", "www.openai.com", "api.openai.com", "chat.openai.com",
    "anthropic.com", "www.anthropic.com", "api.anthropic.com",
    "huggingface.co", "www.huggingface.co", "api.huggingface.co",
    "github.com", "github.io", "github.dev", "githubassets.com",
    "gitlab.com", "gitlab.io", "bitbucket.org", "bitbucket.io",
    "sourceforge.net", "codeberg.org", "sr.ht", "git.sr.ht",
    "npmjs.com", "www.npmjs.com", "registry.npmjs.org", "github.com/npm",
    "pypi.org", "www.pypi.org", "files.pythonhosted.org", "pypi.org",
    "crates.io", "www.crates.io", "doc.rust-lang.org", "static.crates.io",
    "maven.org", "repo.maven.apache.org", "central.maven.org",
    "nuget.org", "www.nuget.org", "api.nuget.org",
    "packagist.org", "repo.packagist.org", "packagist.org",
    "rubygems.org", "www.rubygems.org", "rubygems.org",
    "cocoapods.org", "cocoapods.org", "trunk.cocoapods.org",
    "pub.dev", "pub.dartlang.org", "api.pub.dev",
    "bower.io", "bower.herokuapp.com", "registry.bower.io",
    "jspm.io", "jspm.io", "github.jspm.io",
    "deno.land", "deno.land", "raw.githubusercontent.com",
    "esm.sh", "esm.sh", "cdn.jsdelivr.net",
    "skypack.dev", "cdn.skypack.dev", "cdn.pika.dev",
    "unpkg.com", "unpkg.com", "cdn.jsdelivr.net",
    "cdnjs.cloudflare.com", "cdnjs.cloudflare.com", "cdn.jsdelivr.net",
    "cdn.jsdelivr.net", "cdn.jsdelivr.net", "cdn.jsdelivr.net",
]

# Remove duplicates
top_brands = list(dict.fromkeys(top_brands))

# Generate extensive legitimate URLs
legitimate_urls = []

# Base domains
for domain in top_brands:
    legitimate_urls.append(f"https://{domain}")

# Subdomains
subdomain_patterns = [
    "api", "www", "mail", "drive", "docs", "calendar", "photos", "maps",
    "translate", "play", "news", "finance", "shop", "store", "blog", "dev",
    "app", "dashboard", "console", "admin", "secure", "login", "account",
    "profile", "settings", "help", "support", "status", "cdn", "static",
    "assets", "media", "images", "api-v2", "v2", "beta", "staging", "test",
    "sandbox", "mobile", "m", "wap", "touch", "developer", "developers",
    "docs", "documentation", "reference", "guides", "tutorials", "examples",
    "demo", "try", "preview", "staging", "prod", "production", "live",
    "www2", "www3", "ww1", "ww2", "ww3", "secure2", "login2", "auth",
    "sso", "oauth", "oidc", "saml", "idp", "identity", "access", "portal",
    "my", "your", "user", "client", "customer", "partner", "vendor",
    "api-gateway", "gateway", "edge", "cdn", "static-files", "uploads",
    "downloads", "files", "storage", "blob", "bucket", "s3", "gs", "azure"
]

for brand in top_brands[:80]:  # Top 80 brands
    for sub in subdomain_patterns:
        legitimate_urls.append(f"https://{sub}.{brand}")

# Paths
path_patterns = [
    "/search?q={query}",
    "/user/profile",
    "/settings",
    "/dashboard",
    "/admin",
    "/api/v1/users",
    "/api/v2/products",
    "/api/v3/orders",
    "/graphql",
    "/rest/v1/items",
    "/blog/post/{id}",
    "/help/faq",
    "/support/ticket/{id}",
    "/product/item?id={id}",
    "/category/{cat}?page={page}",
    "/checkout?cart={cart}",
    "/login?redirect={url}",
    "/signup?plan={plan}",
    "/reset-password?token={token}",
    "/verify?code={code}",
    "/oauth/authorize?client_id={id}",
    "/webhook/stripe",
    "/callback?state={state}",
    "/health",
    "/metrics",
    "/status",
    "/ping",
    "/version",
    "/robots.txt",
    "/sitemap.xml",
    "/ads.txt",
    "/security.txt",
    "/.well-known/acme-challenge/{token}",
    "/.well-known/jwks.json",
    "/.well-known/openid-configuration",
    "/privacy",
    "/terms",
    "/cookies",
    "/gdpr",
    "/ccpa",
    "/contact",
    "/about",
    "/careers",
    "/press",
    "/investors",
    "/legal",
    "/compliance",
    "/accessibility",
    "/sitemap",
    "/rss",
    "/atom.xml",
    "/feed",
    "/newsletter",
    "/subscribe",
    "/unsubscribe",
    "/preferences",
    "/notifications",
    "/messages",
    "/inbox",
    "/sent",
    "/drafts",
    "/trash",
    "/spam",
    "/archive",
    "/labels",
    "/filters",
    "/forwarding",
    "/signature",
    "/vacation",
    "/offline",
    "/sync",
    "/backup",
    "/restore",
    "/import",
    "/export",
    "/print",
    "/pdf",
    "/download",
    "/upload",
    "/attachments",
    "/files",
    "/documents",
    "/spreadsheets",
    "/presentations",
    "/forms",
    "/surveys",
    "/polls",
    "/quizzes",
    "/tests",
    "/exams",
    "/assignments",
    "/grades",
    "/reports",
    "/analytics",
    "/insights",
    "/metrics",
    "/logs",
    "/audit",
    "/events",
    "/activities",
    "/timeline",
    "/history",
    "/versions",
    "/revisions",
    "/diffs",
    "/commits",
    "/branches",
    "/tags",
    "/releases",
    "/packages",
    "/artifacts",
    "/builds",
    "/deployments",
    "/environments",
    "/clusters",
    "/nodes",
    "/pods",
    "/containers",
    "/services",
    "/ingresses",
    "/configmaps",
    "/secrets",
    "/volumes",
    "/pvcs",
    "/pvs",
    "/namespaces",
    "/deployments",
    "/statefulsets",
    "/daemonsets",
    "/jobs",
    "/cronjobs",
    "/horizontalpodautoscalers",
    "/verticalpodautoscalers",
    "/poddisruptionbudgets",
    "/networkpolicies",
    "/resourcequotas",
    "/limitranges",
    "/serviceaccounts",
    "/roles",
    "/rolebindings",
    "/clusterroles",
    "/clusterrolebindings",
    "/customresourcedefinitions",
    "/mutatingwebhookconfigurations",
    "/validatingwebhookconfigurations",
    "/apiservices",
    "/controllerrevisions",
    "/endpoints",
    "/endpointslices",
    "/ingressclasses",
    "/ingresses",
    "/networkpolicies",
    "/poddisruptionbudgets",
    "/priorityclasses",
    "/resourcequotas",
    "/runtimeclasses",
    "/storageclasses",
    "/volumeattachments",
    "/csidrivers",
    "/csinodes",
    "/storageversions",
    "/flowschemas",
    "/prioritylevelconfigurations",
    "/certificatesigningrequests",
    "/leases",
    "/events",
    "/configmaps",
    "/secrets",
    "/serviceaccounts",
    "/pods",
    "/podtemplates",
    "/replicationcontrollers",
    "/replicasets",
    "/statefulsets",
    "/daemonsets",
    "/jobs",
    "/cronjobs",
    "/horizontalpodautoscalers",
    "/poddisruptionbudgets",
    "/endpoints",
    "/endpointslices",
    "/services",
    "/ingresses",
    "/networkpolicies",
    "/resourcequotas",
    "/limitranges",
]

for brand in top_brands[:50]:
    for path in path_patterns:
        # Fill placeholders
        filled_path = path.format(
            query=random.choice(["test", "search", "query", "find", "lookup"]),
            id=random.randint(1, 10000),
            cat=random.choice(["electronics", "books", "clothing", "home", "sports"]),
            page=random.randint(1, 100),
            cart=random.randint(1000, 9999),
            url=random.choice(["home", "dashboard", "profile", "settings"]),
            plan=random.choice(["free", "pro", "enterprise", "basic", "premium"]),
            token=random.randint(100000, 999999),
            code=random.randint(100000, 999999),
            state=random.randint(100000, 999999),
        )
        legitimate_urls.append(f"https://{brand}{filled_path}")

# Query params
query_patterns = [
    "?id={id}",
    "?page={page}&limit=20",
    "?q={query}&sort=relevance",
    "?category={cat}&page={page}",
    "?search={query}&filter=new",
    "?id={id}&action=view",
    "?ref=homepage&utm_source=web",
    "?utm_source=google&utm_medium=cpc",
    "?fbclid={id}&ref=facebook",
    "?gclid={id}&ref=google",
    "?token={token}&redirect=/dashboard",
    "?code={code}&state={state}",
    "?verification={code}&email=user@domain.com",
    "?reset={token}&email=user@domain.com",
    "?invite={code}&referrer={id}",
    "?share={id}&platform=web",
    "?embed=true&autoplay=false",
    "?format=json&pretty=true",
    "?api_key={key}&version=v1",
    "?access_token={token}&scope=read",
    "?client_id={id}&redirect_uri={url}",
]

for brand in top_brands[:30]:
    for q in query_patterns:
        filled = q.format(
            id=random.randint(10000, 99999),
            page=random.randint(1, 50),
            query=random.choice(["test", "search", "find", "query"]),
            cat=random.choice(["electronics", "books", "clothing", "home"]),
            token=random.randint(100000, 999999),
            code=random.randint(100000, 999999),
            state=random.randint(100000, 999999),
            key=random.randint(1000000, 9999999),
            url=random.choice(["home", "dashboard", "profile"]),
        )
        legitimate_urls.append(f"https://{brand}{filled}")

# Random Tranco domains
np.random.seed(42)
sampled = tranco_df.sample(20000, random_state=42)
for domain in sampled["domain"]:
    legitimate_urls.append(f"https://{domain}")

# Remove duplicates
legitimate_urls = list(dict.fromkeys(legitimate_urls))
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

combined.to_csv("dataset/robust_features_v3.csv", index=False)
print("     Saved to dataset/robust_features_v3.csv")

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
joblib.dump(model, "models/robust_phishing_model_v3.pkl")
print("\nModel saved to models/robust_phishing_model_v3.pkl")

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
    "https://example.com/blog/post/123",
    "https://myapp.firebaseapp.com",
    "https://project.vercel.app",
    "https://site.netlify.app",
    "https://app.herokuapp.com",
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
    status = "✓" if (label == "Legitimate" and "http://" not in url or label == "Phishing" and "http://" in url and "security" in url) else "✗"
    print(f"  {url:<60} -> {label} ({conf:.1f}%)")

print("\nDone!")