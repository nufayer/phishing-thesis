import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlparse

from feature_extractor import extract_features

# -------------------------------------------------
# Load Ensemble Model
# -------------------------------------------------

ensemble = joblib.load("models/ensemble_final.pkl")
lgbm = ensemble['lgbm']
xgb = ensemble['xgb']
cat = ensemble['cat']
meta = ensemble['meta']
feature_cols = ensemble['feature_cols']

# Whitelist of legitimate domains (comprehensive)
LEGITIMATE_DOMAINS = {
    "google.com", "github.com", "github.io", "facebook.com", "fb.com", "microsoft.com",
    "apple.com", "icloud.com", "amazon.com", "aws.amazon.com", "netflix.com",
    "twitter.com", "x.com", "t.co", "linkedin.com", "youtube.com", "ytimg.com",
    "instagram.com", "wikipedia.org", "reddit.com", "stackoverflow.com",
    "stackexchange.com", "dropbox.com", "adobe.com", "salesforce.com",
    "atlassian.com", "slack.com", "zoom.us", "notion.so", "notion.site",
    "figma.com", "vercel.com", "vercel.app", "cloudflare.com", "cloudflare.net",
    "paypal.com", "stripe.com", "twilio.com", "sendgrid.com", "mailchimp.com",
    "shopify.com", "squarespace.com", "wix.com", "wordpress.com", "medium.com",
    "nytimes.com", "wsj.com", "bloomberg.com", "reuters.com", "cnn.com",
    "bbc.com", "theguardian.com", "npr.org", "pbs.org", "gov.uk",
    "whitehouse.gov", "nih.gov", "cdc.gov", "who.int", "un.org",
    "mit.edu", "stanford.edu", "harvard.edu", "berkeley.edu", "cmu.edu",
    "caltech.edu", "princeton.edu", "yale.edu", "columbia.edu", "upenn.edu",
    "cornell.edu", "uchicago.edu",
    "gitlab.com", "gitlab.io", "bitbucket.org", "bitbucket.io",
    "dev.to", "hashnode.com", "medium.com", "substack.com", "ghost.io",
    "vercel.app", "netlify.app", "herokuapp.com", "firebaseapp.com",
    "web.app", "pages.dev", "cloudflare.workers.dev", "supabase.co",
    "planetscale.dev", "railway.app", "render.com", "fly.dev", "deno.dev",
    "begin.app", "azion.app", "pages.github.io", "gitlab.io", "bitbucket.io",
    "surge.sh", "now.sh", "zeit.co", "glitch.me", "glitch.com",
    "replit.com", "repl.co", "codesandbox.io", "stackblitz.io",
    "openai.com", "chatgpt.com", "chat.openai.com", "anthropic.com", "huggingface.co",
    "perplexity.ai", "claude.ai", "bard.google.com", "copilot.github.com",
    "copilot.microsoft.com", "gemini.google.com", "vertex.ai",
    "coinbase.com", "binance.com", "kraken.com", "gemini.com",
    "spotify.com", "twitch.tv", "discord.com", "discordapp.com",
    "virustotal.com", "hybrid-analysis.com", "any.run",
    "gov.uk", "gov.us", "europa.eu", "un.org", "who.int",
    "duckduckgo.com", "bing.com", "search.brave.com",
    "gmail.com", "outlook.com", "protonmail.com", "proton.me",
    "cloudflare.net", "cloudflare.com", "fastly.net", "akamai.net",
    "akamaihd.net", "cloudfront.net", "amazonaws.com", "s3.amazonaws.com",
    "api.github.com", "api.stripe.com", "api.twilio.com", "api.sendgrid.com",
    "api.mailchimp.com", "api.shopify.com", "api.slack.com", "api.notion.com",
    "api.figma.com", "api.vercel.com", "api.cloudflare.com",
    "docs.google.com", "drive.google.com", "sheets.google.com", "slides.google.com",
    "calendar.google.com", "meet.google.com", "keep.google.com",
    "onedrive.live.com", "sharepoint.com", "office.com",
    "notion.so", "notion.site", "airtable.com", "coda.io",
    "vercel.com", "vercel.app", "netlify.com", "netlify.app",
    "heroku.com", "herokuapp.com", "render.com", "onrender.com",
    "fly.io", "fly.dev", "railway.app", "up.railway.app",
    "supabase.co", "planetscale.dev", "neon.tech", "turso.io",
    "mongoatlas.com", "mongodb.net", "redis.io", "upstash.io",
    "npmjs.com", "pypi.org", "crates.io", "maven.org", "nuget.org",
    "pub.dev", "registry.npmjs.org", "github.com", "raw.githubusercontent.com",
    "datadoghq.com", "newrelic.com", "sentry.io", "logrocket.io",
    "grafana.com", "prometheus.io",
    "auth0.com", "okta.com", "auth.ory.sh", "clerk.dev", "supabase.co",
    "figma.com", "figma.io", "sketch.com", "invisionapp.com",
    "framer.com", "webflow.io", "webflow.com",
    "slack.com", "slack-files.com", "slack-imgs.com", "slack-edge.com",
    "discord.com", "discordapp.com", "discord.gg", "discord.media",
    "teams.microsoft.com", "teams.live.com",
    "shopify.com", "myshopify.com", "bigcommerce.com", "magento.com",
    "woocommerce.com", "squareup.com", "square.site",
    "stripe.com", "paypal.com", "braintreepayments.com", "adyen.com",
    "checkout.com", "wise.com", "transferwise.com",
    "google-analytics.com", "googletagmanager.com", "segment.io",
    "mixpanel.com", "amplitude.com", "heap.io", "fullstory.com",
    "algolia.com", "typesense.io", "meilisearch.com",
    "usa.gov", "gov.uk", "gov.ca", "gov.au", "service.gov.uk",
    "nih.gov", "cdc.gov", "fda.gov", "who.int", "ema.europa.eu",
    "coursera.org", "edx.org", "udemy.com", "khanacademy.org",
    "mit.edu", "stanford.edu", "harvard.edu", "berkeley.edu",
    "apnews.com", "reuters.com", "bloomberg.com", "ft.com",
    "wsj.com", "economist.com", "nytimes.com", "washingtonpost.com",
    "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "tiktok.com", "snapchat.com", "pinterest.com", "youtube.com",
    "stackoverflow.com", "stackexchange.com", "github.com", "gitlab.com",
    "bitbucket.org", "dev.to", "medium.com", "hashnode.com", "substack.com",
    "vercel.com", "netlify.com", "heroku.com", "render.com", "fly.io",
    "railway.app", "supabase.co", "planetscale.dev", "neon.tech",
    "mongodb.com", "atlas.mongodb.com", "redis.io", "upstash.io",
    "huggingface.co", "wandb.ai", "mlflow.org", "dvc.org", "clear.ml",
    "weightsandbiases.com", "comet.ml", "neptune.ai", "mlflow.org",
    "etherscan.io", "polygonscan.com", "bscscan.com", "solscan.io",
    "alchemy.com", "infura.io", "quicknode.com", "moralis.io",
    "1password.com", "bitwarden.com", "lastpass.com", "keepersecurity.com",
    "virustotal.com", "hybrid-analysis.com", "any.run", "joesandbox.cloud",
    "figma.com", "figma.io", "sketch.com", "invisionapp.com", "framer.com",
    "webflow.io", "webflow.com", "framer.dev", "vercel.app",
    "youtube.com", "ytimg.com", "twitch.tv", "vimeo.com", "dailymotion.com",
    "notion.so", "airtable.com", "coda.io", "roamresearch.com", "obsidian.md",
    "linear.app", "height.app", "shortcut.com", "clickup.com", "asana.com",
    "trello.com", "monday.com", "jira.com", "confluence.com",
    "aws.amazon.com", "console.aws.amazon.com", "cloud.google.com",
    "console.cloud.google.com", "azure.microsoft.com", "portal.azure.com",
    "digitalocean.com", "cloud.digitalocean.com", "linode.com",
    "vultr.com", "hetzner.com", "scaleway.com", "upcloud.com",
    "protonvpn.com", "nordvpn.com", "expressvpn.com", "surfshark.com",
    "mullvad.net", "ivpn.net", "protonmail.com", "proton.me", "tutanota.com",
    "docker.com", "hub.docker.com", "kubernetes.io", "helm.sh",
    "prometheus.io", "grafana.com", "jaegertracing.io", "zipkin.io",
    "istio.io", "linkerd.io", "envoyproxy.io", "coredns.io",
    "mongodb.com", "redis.io", "postgresql.org", "mysql.com",
    "cockroachlabs.com", "planetscale.com", "neon.tech", "turso.io",
    "supabase.com", "firebase.google.com", "planetscale.dev",
    "huggingface.co", "wandb.ai", "mlflow.org", "dvc.org", "clear.ml",
    "weightsandbiases.com", "comet.ml", "neptune.ai", "mlflow.org",
    "etherscan.io", "polygonscan.com", "bscscan.com", "solscan.io",
    "alchemy.com", "infura.io", "quicknode.com", "moralis.io",
}


def is_known_legitimate(url):
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        return any(domain == d or domain.endswith("." + d) for d in LEGITIMATE_DOMAINS)
    except Exception:
        return False


# -------------------------------------------------
# Predict URL
# -------------------------------------------------

def predict_url(url):

    # Quick whitelist check for known legitimate domains
    if is_known_legitimate(url):
        return {"prediction": "Legitimate", "confidence": 100.0}

    # -----------------------------
    # Enhanced Features
    # -----------------------------

    feature_dict = extract_features(url, include_network=False)

    # Ensure all features are present (fill missing with defaults)
    for col in feature_cols:
        if col not in feature_dict:
            feature_dict[col] = 0

    df = pd.DataFrame([feature_dict])[feature_cols]

    # -----------------------------
    # Ensemble Prediction (Stacking)
    # -----------------------------

    lgbm_prob = lgbm.predict_proba(df)[:, 1][0]
    xgb_prob = xgb.predict_proba(df)[:, 1][0]
    cat_prob = cat.predict_proba(df)[:, 1][0]

    # Stacking meta-learner
    stack_input = np.array([[lgbm_prob, xgb_prob, cat_prob]])
    final_prob = meta.predict_proba(stack_input)[:, 1][0]

    confidence = max(final_prob, 1 - final_prob) * 100

    if final_prob > 0.5:
        result = "Legitimate"
    else:
        result = "Phishing"

    return {
        "prediction": result,
        "confidence": round(confidence, 2)
    }