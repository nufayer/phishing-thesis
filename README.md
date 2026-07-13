# PhishShield AI - Advanced Phishing Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-orange.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.1+-orange.svg)](https://catboost.ai/)

> **Production-ready phishing detection system** with ensemble ML model and multi-layer zero-day defense.

---

## 🎯 Overview

PhishShield AI is a comprehensive phishing detection system that combines:

- **Ensemble ML Model** (99.92% accuracy) - Stacking: LGBM + XGBoost + CatBoost → Logistic Regression
- **5-Layer Zero-Day Defense** - Threat intel, heuristics, brand monitoring, dynamic analysis, DNS monitoring
- **72 Features** - URL structure, SSL/TLS, WHOIS, DNS, brand similarity, entropy, ratios
- **Real-time Web Interface** - Flask-based UI for instant URL analysis

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Ensemble ML Model** | Stacking ensemble (LGBM + XGBoost + CatBoost → Logistic Regression) |
| **72 Rich Features** | URL entropy, SSL/TLS cert analysis, WHOIS domain age, DNS records, brand similarity, suspicious patterns |
| **Zero-Day Defense** | 5-layer defense: threat intel + heuristics + brand monitoring + dynamic analysis + DNS monitoring |
| **Brand Protection** | Typosquatting detection, brand impersonation in subdomains, suspicious TLD detection |
| **Real-time UI** | Clean Flask web interface with confidence scoring and risk visualization |
| **API Ready** | Simple `predict_url()` function for integration |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHISHSHIELD AI                               │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Threat Intelligence (OpenPhish, URLhaus, PhishTank)   │
│  Layer 2: Heuristic Rules (Domain age, brand impersonation,     │
│           suspicious TLD, SSL/TLS, SPF/DMARC, subdomains)       │
│  Layer 3: Brand Monitoring (Typosquatting, fuzzy matching)      │
│  Layer 4: Dynamic Analysis (Headless browser - optional)        │
│  Layer 5: DNS Monitoring (Passive DNS, new registrations)       │
├─────────────────────────────────────────────────────────────────┤
│  Ensemble ML Model (Stacking)                                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐    ┌──────────────┐     │
│  │  LGBM   │  │  XGBoost│  │ CatBoost│───▶│Logistic Reg. │     │
│  └─────────┘  └─────────┘  └─────────┘    │  (Meta)      │     │
│                                            └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| LGBM | 99.91% | 1.0000 |
| XGBoost | 99.91% | 1.0000 |
| CatBoost | 99.92% | 1.0000 |
| **Stacking Ensemble** | **99.92%** | **1.0000** |

**Training Data:** 90K samples (60K legitimate + 30K phishing)
- **Legitimate:** Tranco Top 1M + modern platform URLs + real app patterns
- **Phishing:** PhiUSIIL (30K) + OpenPhish + URLhaus feeds

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/yourusername/phishing-thesis.git
cd phishing-thesis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Required Data

```bash
# Download Tranco Top 1M (for training)
wget https://tranco-list.eu/top-1m.csv.zip -O dataset/top-1m.csv.zip
unzip dataset/top-1m.csv.zip -d dataset/

# Download PhiUSIIL dataset
# (Place PhiUSIIL_Phishing_URL_Dataset.csv in dataset/)

# OpenPhish feed (auto-downloaded on first run)
```

### Run Web Application

```bash
python app.py
# Server starts at http://127.0.0.1:5000
```

---

## 💻 Usage

### Web Interface

1. Start server: `python app.py`
2. Open `http://127.0.0.1:5000`
3. Enter URL and click "Analyze Website"
4. View prediction with confidence and risk score

### Python API

```python
from predictor import predict_url

# Simple prediction
result = predict_url("https://google.com")
# {'prediction': 'Legitimate', 'confidence': 100.0}

result = predict_url("http://paypal-login-security.xyz")
# {'prediction': 'Phishing', 'confidence': 93.6}
```

### Zero-Day Defense (Advanced)

```python
from zero_day_defense import ZeroDayDefense, quick_scan, deep_scan

# Quick scan (heuristics + threat intel)
result = quick_scan("https://example.com")

# Deep scan (includes dynamic browser analysis)
result = deep_scan("https://suspicious-site.com")

# Result structure:
{
    "url": "https://...",
    "final_verdict": "Phishing|Suspicious|Likely Legitimate",
    "risk_score": 85.5,
    "confidence": 92.3,
    "layers": {
        "threat_intel": {"known_phishing": False, "verdict": "Unknown"},
        "heuristics": {"risk_score": 85, "reasons": [...]},
        "brand_monitor": {"typosquatting": True, "matched_brand": "paypal"},
        "dynamic_analysis": {...}  # if deep_analysis=True
    }
}
```

---

## 🔧 Configuration

### Feature Extraction Options

```python
from feature_extractor import extract_features

# Without network features (fast, ~1ms)
features = extract_features("https://example.com", include_network=False)

# With network features (slower, ~500ms)
features = extract_features("https://example.com", include_network=True)
```

### Key Features (72 total)

| Category | Features |
|----------|----------|
| **URL Structure** | URLLength, DomainLength, PathLength, NoOfSubDomain, NoOfDots, NoOfHyphen, NoOfSlash |
| **Ratios** | DigitRatio, LetterRatio, SpecialCharRatio, HyphenRatio |
| **Entropy** | URLEntropy, DomainEntropy, PathEntropy |
| **Character Stats** | CharContinuationRate, URLCharProb, NoOfDigits, NoOfLetters |
| **SSL/TLS** | HasSSL, SSLCertValid, SSLDaysToExpiry, SSLIsSelfSigned |
| **WHOIS** | DomainAgeDays, DomainAgeMonths, RegistrarReputation |
| **DNS** | HasMXRecord, HasARecord, HasNSRecord, HasSPFRecord, HasDMARCRecord |
| **Brand** | BrandSimilarity, HasSuspiciousPrefixSuffix |
| **Suspicious** | IsSuspiciousTLD, HasSuspiciousPrefixSuffix, SuspiciousWordCount |

---

## 📁 Project Structure

```
phishing-thesis/
├── app.py                      # Flask web application
├── predictor.py                # Main prediction API
├── feature_extractor.py        # 72-feature extraction
├── zero_day_defense.py         # 5-layer zero-day defense
├── predictor.py                # Ensemble model wrapper
├── models/
│   └── ensemble_final.pkl      # Trained ensemble model
├── dataset/
│   ├── top-1m.csv              # Tranco top 1M domains
│   ├── openphish_feed.txt      # OpenPhish feed (auto-downloaded)
│   └── PhiUSIIL_Phishing_URL_Dataset.csv  # Training data
├── templates/
│   └── index.html              # Web UI template
├── static/
│   └── style.css               # UI styling
├── build_ensemble_v4.py        # Training script
├── zero_day_defense.py         # Zero-day defense layers
├── requirements.txt
└── README.md
```

---

## 🔬 Training Your Own Model

```bash
# Run training with custom parameters
python build_ensemble_v4.py

# Or use the balanced training script
python build_ensemble_manual.py
```

### Training Parameters

```python
# In build_ensemble_v4.py
lgbm = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.015,
    max_depth=14,
    num_leaves=180,
    class_weight='balanced',
    ...
)

xgb = XGBClassifier(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=12,
    ...
)

cat = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.02,
    depth=10,
    class_weights=[1, 2],  # Skew toward legitimate
    ...
)
```

---

## 🛡️ Zero-Day Defense Layers

| Layer | Description | Detection Time |
|-------|-------------|----------------|
| **1. Threat Intel** | OpenPhish, URLhaus, PhishTank feeds | Real-time |
| **2. Heuristics** | Domain age, brand impersonation, TLD, SSL, DNS | < 1ms |
| **3. Brand Monitor** | Typosquatting, fuzzy matching, DNS variations | < 5ms |
| **4. Dynamic Analysis** | Headless browser, form analysis, credential detection | ~2-5s |
| **5. DNS Monitor** | New registrations, passive DNS, historical records | API-dependent |

### Heuristic Rules (Key)

| Pattern | Risk Score | Description |
|---------|------------|-------------|
| New domain (< 30 days) + brand + suspicious TLD | +40 | Zero-day pattern |
| Brand in subdomain of another domain | +40 | `paypal.login.evil.com` |
| Brand + suspicious TLD | +35 | `paypal-login.xyz` |
| Brand with prefix/suffix in subdomain | +30 | `secure-paypal.evil.com` |
| New domain (< 30 days) | +25 | Recently registered |
| No SSL/HTTPS | +20 | Missing encryption |
| Self-signed SSL | +20 | Untrusted certificate |
| Missing SPF/DMARC | +10 each | Poor email security |
| Excessive subdomains (>4) | +20 | `a.b.c.d.evil.com` |

---

## 🧪 Testing

```bash
# Run built-in tests
python test_zero_day.py

# Test web API
python -c "
from app import app
with app.test_client() as c:
    r = c.post('/predict', data={'url': 'https://google.com'})
    print('Safe' if 'safe' in r.data.decode() else 'Phishing')
"
```

---

## 📦 Dependencies

```txt
# Core ML
lightgbm>=3.3.0
xgboost>=1.7.0
catboost>=1.1.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0

# Web
flask>=2.0.0

# Features
tldextract>=5.0.0
python-whois>=0.9.0
dnspython>=2.4.0
requests>=2.31.0

# Optional: Dynamic Analysis
playwright>=1.40.0  # pip install playwright && playwright install chromium
```

---

## 🌐 Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t phishshield .
docker run -p 5000:5000 phishshield
```

### Production Considerations

- Use `gunicorn` or `uwsgi` instead of Flask dev server
- Enable HTTPS with reverse proxy (nginx + certbot)
- Add rate limiting (Flask-Limiter)
- Cache threat intel feeds (Redis)
- Monitor with Prometheus/Grafana

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **PhiUSIIL Dataset** - Phishing URL dataset
- **OpenPhish** - Real-time phishing feed
- **URLhaus** - Malware URL database
- **Tranco** - Top domains list
- **LightGBM/XGBoost/CatBoost** - Gradient boosting frameworks

---

## 📧 Contact

**Author:** [Your Name]  
**Email:** your.email@example.com  
**GitHub:** [github.com/yourusername/phishing-thesis](https://github.com/yourusername/phishing-thesis)

---

⭐ **Star this repo if you find it useful!**