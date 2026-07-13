"""
Zero-Day Phishing Defense Layers
Multi-layer defense system for detecting novel phishing attacks
"""

import re
import socket
import ssl
import whois
import dns.resolver
import requests
import hashlib
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse
from difflib import SequenceMatcher
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Optional: for dynamic analysis
try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


# ============================================================
# LAYER 1: THREAT INTELLIGENCE FEEDS
# ============================================================

class ThreatIntelligence:
    """Real-time threat intelligence feed integration"""
    
    FEEDS = {
        "openphish": "https://openphish.com/feed.txt",
        "phishtank": "https://data.phishtank.com/data/online-valid.csv",
        "urlhaus": "https://urlhaus.abuse.ch/downloads/csv_recent/",
        "phishing_database": "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-NEW-today.txt",
    }
    
    def __init__(self, cache_ttl=3600):
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.known_phishing = set()
        self.last_update = {}
    
    def fetch_feed(self, name, url):
        """Fetch and parse a threat feed"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Failed to fetch {name}: {e}")
            return None
    
    def parse_openphish(self, content):
        return [line.strip() for line in content.splitlines() if line.strip()]
    
    def parse_phishtank(self, content):
        urls = []
        for line in content.splitlines()[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) > 1:
                url = parts[1].strip('"')
                if url.startswith('http'):
                    urls.append(url)
        return urls
    
    def parse_urlhaus(self, content):
        urls = []
        for line in content.splitlines():
            if line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) > 2:
                url = parts[2].strip('"')
                if url.startswith('http'):
                    urls.append(url)
        return urls
    
    def update_all(self):
        """Update all threat feeds"""
        parsers = {
            "openphish": self.parse_openphish,
            "phishtank": self.parse_phishtank,
            "urlhaus": self.parse_urlhaus,
            "phishing_database": self.parse_openphish,
        }
        
        all_urls = set()
        for name, url in self.FEEDS.items():
            content = self.fetch_feed(name, url)
            if content:
                parser = parsers.get(name)
                if parser:
                    urls = parser(content)
                    all_urls.update(urls)
                    print(f"  {name}: {len(urls)} URLs")
        
        self.known_phishing = all_urls
        self.last_update = {name: datetime.now() for name in self.FEEDS}
        print(f"Total unique phishing URLs: {len(self.known_phishing)}")
        return self.known_phishing
    
    def is_known_phishing(self, url):
        """Check if URL is in known phishing database"""
        # Normalize URL
        normalized = self._normalize_url(url)
        return normalized in self.known_phishing
    
    def _normalize_url(self, url):
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace("www.", "")
            path = parsed.path.rstrip('/')
            return f"{parsed.scheme}://{domain}{path}"
        except:
            return url


# ============================================================
# LAYER 2: ZERO-DAY HEURISTIC RULES
# ============================================================

class ZeroDayHeuristics:
    """Heuristic rules for detecting zero-day phishing patterns"""
    
    # Common phishing patterns
    PHISHING_KEYWORDS = {
        "login", "signin", "sign-in", "signin", "logon", "logon",
        "verify", "verification", "confirm", "confirmation",
        "secure", "security", "account", "update", "upgrade",
        "billing", "payment", "invoice", "receipt", "transaction",
        "suspend", "suspended", "restricted", "limited", "locked",
        "unlock", "restore", "recover", "reset", "validate",
        "authenticate", "authentication", "credential", "credential",
        "support", "helpdesk", "service", "customer", "client",
        "wallet", "crypto", "bitcoin", "ethereum", "binance",
        "coinbase", "kraken", "metamask", "walletconnect",
    }
    
    BRAND_KEYWORDS = {
        "paypal", "apple", "microsoft", "google", "amazon", "facebook",
        "instagram", "whatsapp", "netflix", "spotify", "steam",
        "github", "gitlab", "bitbucket", "atlassian", "jira",
        "slack", "discord", "telegram", "signal", "zoom",
        "teams", "outlook", "office365", "azure", "aws", "gcp",
        "stripe", "square", "venmo", "cashapp", "zelle",
        "bank", "chase", "wellsfargo", "bankofamerica", "citibank",
        "capitalone", "amex", "discover", "visa", "mastercard",
    }
    
    SUSPICIOUS_TLDS = {
        "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work",
        "party", "loan", "download", "racing", "stream", "science",
        "bid", "win", "accountant", "faith", "date", "review",
        "trade", "webcam", "wang", "online", "site", "space",
        "website", "tech", "store", "shop", "online", "party",
    }
    
    LEGITIMATE_TLDS = {
        "com", "org", "net", "edu", "gov", "io", "co", "ai", "dev",
        "app", "net", "org", "com", "gov", "edu", "mil", "int",
    }
    
    def __init__(self):
        self.threat_intel = ThreatIntelligence()
        self.cache = {}
    
    def analyze(self, url, features=None):
        """
        Comprehensive zero-day heuristic analysis
        Returns (is_suspicious, risk_score, reasons)
        """
        if features is None:
            features = self._extract_basic_features(url)
        
        reasons = []
        risk_score = 0
        
        # 1. Known phishing check
        if self.threat_intel.is_known_phishing(url):
            return True, 100, ["Known phishing URL in threat intelligence feeds"]
        
        # 2. Domain age check
        domain_age = features.get("DomainAgeDays", 999)
        if domain_age < 30:
            risk_score += 25
            reasons.append(f"New domain ({domain_age} days old)")
        elif domain_age < 90:
            risk_score += 10
            reasons.append(f"Recent domain ({domain_age} days old)")
        
        # 3. Brand impersonation
        brand_score, matched_brand = self._check_brand_impersonation(url, features)
        if brand_score > 0:
            risk_score += brand_score
            reasons.append(f"Brand impersonation detected: {matched_brand}")
        
        # 4. Suspicious TLD
        if features.get("IsSuspiciousTLD", 0):
            risk_score += 20
            reasons.append("Suspicious TLD detected")
        
        # 5. Suspicious prefix/suffix (brand in subdomain)
        if features.get("HasSuspiciousPrefixSuffix", 0):
            risk_score += 30
            reasons.append("Brand name in suspicious position (subdomain/prefix)")
        
        # 6. SSL/TLS issues
        if not features.get("HasSSL", 0):
            risk_score += 20
            reasons.append("No SSL/HTTPS")
        elif not features.get("SSLCertValid", 1):
            risk_score += 15
            reasons.append("Invalid/expired SSL certificate")
        elif features.get("SSLIsSelfSigned", 0):
            risk_score += 20
            reasons.append("Self-signed SSL certificate")
        
        # 6. Missing security records
        if not features.get("HasSPFRecord", 0):
            risk_score += 10
            reasons.append("Missing SPF record")
        if not features.get("HasDMARCRecord", 0):
            risk_score += 10
            reasons.append("Missing DMARC record")
        
        # 7. Excessive subdomains
        subdomain_count = features.get("NoOfSubDomain", 0)
        if subdomain_count > 4:
            risk_score += 20
            reasons.append(f"Excessive subdomains ({subdomain_count})")
        elif subdomain_count > 2:
            risk_score += 10
            reasons.append(f"Multiple subdomains ({subdomain_count})")
        
        # 7. URL shortener
        if features.get("IsURLShortener", 0):
            risk_score += 15
            reasons.append("URL shortener detected")
        
        # 8. IP address as domain
        if features.get("HasIPAddress", 0):
            risk_score += 25
            reasons.append("IP address used as domain")
        
        # 9. Abnormal URL (@ symbol, etc.)
        if features.get("AbnormalURL", 0):
            risk_score += 20
            reasons.append("Abnormal URL structure (@ symbol)")
        
        # 10. Brand + suspicious TLD combo
        if features.get("BrandSimilarity", 0) > 0.5 and features.get("IsSuspiciousTLD", 0):
            risk_score += 35
            reasons.append("Brand name with suspicious TLD")
        
        # 11. New domain + brand + suspicious TLD (high confidence zero-day)
        if (features.get("DomainAgeDays", 999) < 30 and
            features.get("BrandSimilarity", 0) > 0.6 and
            features.get("IsSuspiciousTLD", 0)):
            risk_score += 40
            reasons.append("ZERO-DAY PATTERN: New domain + brand + suspicious TLD")
        
        # Normalize score (0-100)
        risk_score = min(risk_score, 100)
        
        is_suspicious = risk_score >= 40
        
        return is_suspicious, risk_score, reasons
    
    def _extract_basic_features(self, url):
        """Extract basic features without full feature extractor"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        
        # Extract TLD
        tld = domain.split(".")[-1] if "." in domain else ""
        
        features = {
            "DomainAgeDays": 999,  # Will be filled by WHOIS if available
            "IsSuspiciousTLD": 1 if tld in self.SUSPICIOUS_TLDS else 0,
            "IsSuspiciousTLDPattern": 0,  # Will be computed
            "HasSuspiciousPrefixSuffix": self._check_prefix_suffix(domain),
            "BrandSimilarity": self._compute_brand_similarity(domain),
            "HasSSL": 1 if parsed.scheme == "https" else 0,
            "SSLCertValid": 1,  # Default optimistic
            "SSLIsSelfSigned": 0,
            "HasSPFRecord": 0,
            "HasDMARCRecord": 0,
            "NoOfSubDomain": max(len(domain.split(".")) - 2, 0),
            "IsURLShortener": 1 if domain in self._get_url_shorteners() else 0,
            "HasIPAddress": 1 if re.match(r"^\d+\.\d+\.\d+\.\d+$", domain) else 0,
            "AbnormalURL": 1 if "@" in url else 0,
            "DomainAgeDays": 999,
            "BrandSimilarity": self._compute_brand_similarity(domain),
            "IsSuspiciousTLD": 1 if tld in self.SUSPICIOUS_TLDS else 0,
            "HasSuspiciousPrefixSuffix": self._check_prefix_suffix(domain),
        }
        return features
    
    def _get_url_shorteners(self):
        return {
            "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
            "buff.ly", "adf.ly", "bc.vc", "tiny.cc", "short.to", "s.id",
            "cutt.ly", "rb.gy", "v.gd", "x.co", "tr.im", "u.to", "yourls.org",
        }
    
    def _check_prefix_suffix(self, domain):
        """Check if domain has brand name as prefix/suffix with extra chars"""
        domain = domain.lower().replace("www.", "")
        parts = domain.split(".")
        main = parts[0] if parts else ""
        
        for brand in self.BRAND_KEYWORDS:
            if brand in main and main != brand:
                if main.startswith(brand) or main.endswith(brand):
                    return 1
        return 0
    
    def _compute_brand_similarity(self, domain):
        domain = domain.lower().replace("www.", "").split(".")[0]
        max_sim = 0
        for brand in self.BRAND_KEYWORDS:
            if brand in domain or domain in brand:
                common = sum(1 for c in brand if c in domain)
                sim = common / max(len(brand), len(domain))
                max_sim = max(max_sim, sim)
        return max_sim
    
    def _check_brand_impersonation(self, url, features):
        """Check for brand impersonation attempts - ONLY flags suspicious contexts"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        parts = domain.split(".")
        
        for brand in self.BRAND_KEYWORDS:
            if brand in domain:
                # Case 1: LEGITIMATE - Brand IS the root domain with subdomain
                # e.g., docs.google.com (parts=['docs', 'google', 'com']), api.stripe.com
                # For standard TLDs (.com, .org, .net, .io, etc.), root domain is parts[-2]
                if len(parts) >= 2 and parts[-2] == brand:
                    # Brand is the root domain (e.g., google.com, stripe.com)
                    # Subdomains like docs.google.com, api.stripe.com are LEGITIMATE
                    continue
                
                # Case 2: LEGITIMATE - Brand IS the domain (no subdomain)
                # e.g., google.com, stripe.com
                if len(parts) >= 2 and parts[-2] == brand:
                    continue
                
                # Case 3: SUSPICIOUS - Brand in subdomain of ANOTHER domain
                # e.g., paypal.login.evil.com, google-login.evil.com
                if len(parts) >= 3 and brand in parts[-2]:
                    # Brand in subdomain of another domain
                    return 40, brand
                
                # Case 4: SUSPICIOUS - Brand in domain with suspicious TLD
                # e.g., google-login.xyz, paypal-security.tk
                if len(parts) >= 2 and parts[-1] in self.SUSPICIOUS_TLDS:
                    if brand in parts[-2]:
                        return 35, brand
                
                # Case 5: SUSPICIOUS - Brand with suspicious prefix/suffix in subdomain
                # e.g., secure-paypal.evil.com, login-google.fake.com
                if len(parts) >= 3:
                    for part in parts[:-2]:  # Check subdomains only
                        if brand in part and part != brand:
                            if part.startswith(brand) or part.endswith(brand):
                                return 30, brand
        
        return 0, None
    
    def enrich_with_whois(self, url):
        """Enrich features with WHOIS data"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        
        try:
            w = whois.whois(domain)
            features = {}
            
            if w.creation_date:
                if isinstance(w.creation_date, list):
                    creation = w.creation_date[0]
                else:
                    creation = w.creation_date
                if creation:
                    age_days = (datetime.now() - creation).days
                    features["DomainAgeDays"] = age_days
            
            # Check registrar reputation
            if w.registrar:
                reputable = {
                    "godaddy", "namecheap", "google domains", "cloudflare",
                    "amazon", "gandi", "hover", "name.com", "porkbun",
                    "dynadot", "epik", "porkbun", "namecheap", "godaddy",
                }
                registrar_lower = str(w.registrar).lower()
                features["RegistrarReputation"] = 1 if any(r in registrar_lower for r in reputable) else 0
            
            return features
        except Exception:
            return {"DomainAgeDays": 999, "RegistrarReputation": 0}


# ============================================================
# LAYER 3: BRAND MONITORING / TYPOSQUATTING
# ============================================================

class BrandMonitor:
    """Monitor for typosquatting and brand impersonation domains"""
    
    def __init__(self):
        self.known_brands = {
            "google", "github", "facebook", "microsoft", "apple", "amazon",
            "netflix", "twitter", "linkedin", "youtube", "instagram",
            "wikipedia", "reddit", "stackoverflow", "dropbox", "adobe",
            "salesforce", "slack", "zoom", "notion", "figma", "vercel",
            "cloudflare", "paypal", "stripe", "twilio", "sendgrid",
            "mailchimp", "shopify", "squarespace", "wix", "wordpress",
            "medium", "nytimes", "wsj", "bloomberg", "reuters", "cnn",
            "bbc", "theguardian", "npr", "pbs", "gov", "nih", "cdc",
            "who", "un", "openai", "anthropic", "huggingface", "vercel",
            "netlify", "heroku", "firebase", "supabase", "planetscale",
            "railway", "render", "fly", "deno", "begin", "azion",
            "coinbase", "binance", "kraken", "gemini", "metamask",
            "uniswap", "opensea", "foundation", "rarible", "magiceden",
            "discord", "telegram", "signal", "zoom", "teams", "slack",
            "atlassian", "jira", "confluence", "bitbucket", "gitlab",
            "docker", "kubernetes", "nginx", "apache", "linux",
            "python", "nodejs", "npm", "pypi", "maven", "gradle",
        }
    
    def check_typosquatting(self, domain):
        """
        Check if domain is a typosquatting attempt
        Returns (is_typosquat, matched_brand, similarity, variations)
        """
        domain = domain.lower().replace("www.", "").split(".")[0]
        
        for brand in self.known_brands:
            # Exact substring match
            if brand in domain and domain != brand:
                similarity = SequenceMatcher(None, domain, brand).ratio()
                
                # Check common typosquatting patterns
                variations = self._get_typosquat_variations(brand)
                for var in variations:
                    if var in domain:
                        return True, brand, 0.95, [var]
                
                if SequenceMatcher(None, domain, brand).ratio() > 0.85:
                    return True, brand, SequenceMatcher(None, domain, brand).ratio(), ["similar"]
        
        return False, None, 0.0, []
    
    def _get_typosquat_variations(self, brand):
        """Generate common typosquatting variations for a brand"""
        variations = set()
        
        # Common prefixes/suffixes used in phishing
        prefixes = ["secure-", "login-", "verify-", "account-", "update-",
                   "security-", "support-", "help-", "service-", "billing-",
                   "payment-", "account-", "user-", "customer-", "client-"]
        suffixes = ["-security", "-login", "-verify", "-account", "-update",
                   "-support", "-help", "-service", "-billing", "-payment",
                   "-online", "-official", "-center", "-portal", "-manage"]
        
        # Prefix variations
        for p in prefixes:
            variations.add(p + brand)
            variations.add(p + brand + "s")
        
        # Suffix variations
        for s in suffixes:
            variations.add(brand + s)
            variations.add(brand + "s" + s)
        
        # Character substitution (common typos)
        substitutions = {
            'a': '@', 'e': '3', 'i': '1', 'o': '0', 'l': '1',
            's': '$', 't': '7', 'g': '9', 'b': '6', 'z': '2',
        }
        for char, sub in substitutions.items():
            if char in brand:
                variations.add(brand.replace(char, sub))
        
        # Missing/extra characters
        for i in range(len(brand)):
            # Missing char
            variations.add(brand[:i] + brand[i+1:])
            # Extra char
            for c in 'abcdefghijklmnopqrstuvwxyz0123456789':
                variations.add(brand[:i] + c + brand[i:])
        
        # Transposition
        for i in range(len(brand) - 1):
            variations.add(brand[:i] + brand[i+1] + brand[i] + brand[i+2:])
        
        return variations
    
    def check_dns_variations(self, brand):
        """Check DNS for registered typosquat domains"""
        variations = self._get_typosquat_variations(brand)
        found = []
        
        for var in list(variations)[:50]:  # Limit to avoid too many queries
            for tld in [".com", ".net", ".org", ".io", ".co", ".xyz"]:
                domain = var + tld
                try:
                    answers = dns.resolver.resolve(domain, 'A')
                    if answers:
                        found.append(domain)
                except:
                    pass
        
        return found


# ============================================================
# LAYER 4: DYNAMIC ANALYSIS (HEADLESS BROWSER)
# ============================================================

class DynamicAnalyzer:
    """Headless browser analysis for credential harvesting detection"""
    
    def __init__(self, timeout=30):
        self.timeout = timeout
    
    async def analyze(self, url):
        """Analyze page behavior for credential harvesting"""
        if not HAS_PLAYWRIGHT:
            return {"error": "Playwright not installed", "risk": 0}
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    viewport={"width": 1280, "height": 720},
                )
                page = await context.new_page()
                
                # Track network requests
                requests_made = []
                page.on("request", lambda req: requests_made.append({
                    "url": req.url,
                    "method": req.method,
                    "resource_type": req.resource_type,
                }))
                
                # Navigate
                response = await page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
                
                # Analyze page
                analysis = {
                    "url": url,
                    "final_url": page.url,
                    "status_code": response.status if response else None,
                    "title": await page.title(),
                    "has_password_field": await self._has_password_field(page),
                    "has_login_form": await self._has_login_form(page),
                    "forms": await self._analyze_forms(page),
                    "external_resources": await self._check_external_resources(page),
                    "credential_fields": await self._find_credential_fields(page),
                    "brand_impersonation": await self._check_brand_impersonation(page),
                    "redirects": await self._check_redirects(page),
                    "requests_count": len(requests_made),
                    "external_requests": len([r for r in requests_made if not self._is_same_domain(r["url"], page.url)]),
                }
                
                # Calculate risk
                analysis["risk_score"] = self._calculate_risk(analysis)
                analysis["is_phishing"] = analysis["risk_score"] >= 50
                
                await browser.close()
                return analysis
                
        except Exception as e:
            return {"error": str(e), "risk_score": 0, "is_phishing": False}
    
    def _is_same_domain(self, url, reference_url):
        try:
            return urlparse(url).netloc == urlparse(reference_url).netloc
        except:
            return False
    
    async def _has_password_field(self, page):
        return await page.query_selector('input[type="password"]') is not None
    
    async def _has_login_form(self, page):
        form = await page.query_selector('form')
        if form:
            inputs = await form.query_selector_all('input')
            has_user = any(await inp.get_attribute('type') in ['text', 'email', 'username'] for inp in inputs)
            has_pass = any(await inp.get_attribute('type') == 'password' for inp in inputs)
            return has_user and has_pass
        return False
    
    async def _analyze_forms(self, page):
        forms = await page.query_selector_all('form')
        form_data = []
        for form in forms:
            action = await form.get_attribute('action') or ''
            method = await form.get_attribute('method') or 'GET'
            inputs = await form.query_selector_all('input')
            input_types = []
            for inp in inputs:
                itype = await inp.get_attribute('type') or 'text'
                iname = await inp.get_attribute('name') or ''
                input_types.append({"type": itype, "name": iname})
            form_data.append({
                "action": action,
                "method": method,
                "is_external": self._is_external_action(action, page.url),
                "inputs": input_types,
            })
        return form_data
    
    def _is_external_action(self, action, page_url):
        if not action:
            return False
        try:
            action_domain = urlparse(action).netloc
            page_domain = urlparse(page_url).netloc
            return action_domain and action_domain != page_domain
        except:
            return False
    
    async def _check_external_resources(self, page):
        scripts = await page.query_selector_all('script[src]')
        external = []
        page_domain = urlparse(page.url).netloc
        for script in scripts:
            src = await script.get_attribute('src')
            if src:
                src_domain = urlparse(src).netloc
                if src_domain and src_domain != page_domain:
                    external.append(src)
        return external
    
    async def _find_credential_fields(self, page):
        fields = []
        selectors = [
            'input[type="password"]',
            'input[name*="password"]',
            'input[name*="pass"]',
            'input[name*="pwd"]',
            'input[name*="secret"]',
            'input[name*="token"]',
            'input[name*="otp"]',
            'input[name*="pin"]',
            'input[name*="cvv"]',
            'input[name*="ssn"]',
        ]
        for selector in selectors:
            elements = await page.query_selector_all(selector)
            for el in elements:
                name = await el.get_attribute('name') or ''
                id_attr = await el.get_attribute('id') or ''
                fields.append({"selector": selector, "name": name, "id": id_attr})
        return fields
    
    async def _check_brand_impersonation(self, page):
        """Check if page impersonates a known brand"""
        title = await page.title()
        body_text = await page.inner_text('body') if await page.query_selector('body') else ''
        content = (title + ' ' + body_text).lower()
        
        brands = {
            "paypal", "apple", "microsoft", "google", "amazon", "facebook",
            "instagram", "whatsapp", "netflix", "spotify", "steam",
            "github", "gitlab", "bitbucket", "atlassian", "jira",
            "slack", "discord", "telegram", "signal", "zoom",
            "teams", "outlook", "office365", "azure", "aws", "gcp",
            "stripe", "square", "venmo", "cashapp", "zelle",
            "coinbase", "binance", "kraken", "metamask",
        }
        
        found = []
        for brand in brands:
            if brand in content:
                # Check if brand appears in suspicious context
                if any(word in content for word in ["login", "signin", "verify", "account", "security"]):
                    found.append(brand)
        return found
    
    async def _check_redirects(self, page):
        # Check for suspicious redirects
        return {"final_url": page.url, "redirect_count": 0}
    
    def _calculate_risk(self, analysis):
        risk = 0
        
        # Credential harvesting indicators
        if analysis.get("has_password_field"):
            risk += 15
        if analysis.get("has_login_form"):
            risk += 20
        if analysis.get("credential_fields"):
            risk += len(analysis["credential_fields"]) * 5
        
        # Form analysis
        for form in analysis.get("forms", []):
            if form.get("is_external"):
                risk += 30
            if form.get("method", "").upper() == "POST":
                risk += 5
        
        # Brand impersonation
        if analysis.get("brand_impersonation"):
            risk += len(analysis["brand_impersonation"]) * 15
        
        # External resources
        if len(analysis.get("external_resources", [])) > 5:
            risk += 10
        
        # Credential fields
        risk += len(analysis.get("credential_fields", [])) * 8
        
        # External form actions
        external_forms = [f for f in analysis.get("forms", []) if f.get("is_external")]
        risk += len(external_forms) * 25
        
        return min(risk, 100)


# ============================================================
# LAYER 5: DNS MONITORING / BRAND PROTECTION
# ============================================================

class DNSMonitor:
    """Monitor DNS for new phishing domain registrations"""
    
    def __init__(self):
        self.monitored_brands = {
            "paypal", "apple", "microsoft", "google", "amazon", "facebook",
            "instagram", "whatsapp", "netflix", "spotify", "steam",
            "github", "gitlab", "bitbucket", "atlassian", "jira",
            "slack", "discord", "telegram", "signal", "zoom",
            "teams", "outlook", "office365", "azure", "aws", "gcp",
            "stripe", "square", "venmo", "cashapp", "zelle",
            "coinbase", "binance", "kraken", "metamask",
        }
    
    def check_new_registrations(self, hours=24):
        """Check for newly registered domains matching brands"""
        # This would integrate with domain registration monitoring services
        # like DomainTools, SecurityTrails, or Censys
        # For now, return empty - would need API integration
        return []
    
    def passive_dns_lookup(self, domain):
        """Check passive DNS for historical records"""
        # Would integrate with SecurityTrails, Censys, Farsight
        return {"first_seen": None, "last_seen": None, "historical_ips": []}


# ============================================================
# UNIFIED ZERO-DAY DEFENSE
# ============================================================

class ZeroDayDefense:
    """Unified zero-day phishing defense system"""
    
    def __init__(self):
        self.threat_intel = ThreatIntelligence()
        self.heuristics = ZeroDayHeuristics()
        self.brand_monitor = BrandMonitor()
        self.dynamic_analyzer = DynamicAnalyzer()
        self.dns_monitor = DNSMonitor()
        
        # Initialize threat intel
        print("Loading threat intelligence feeds...")
        self.threat_intel.update_all()
    
    def analyze(self, url, features=None, deep_analysis=False):
        """
        Complete zero-day analysis
        Returns comprehensive result
        """
        result = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "layers": {},
            "final_verdict": "Unknown",
            "risk_score": 0,
            "confidence": 0,
        }
        
        # Layer 1: Threat Intelligence
        known_phishing = self.threat_intel.is_known_phishing(url)
        result["layers"]["threat_intel"] = {
            "known_phishing": known_phishing,
            "verdict": "Phishing" if known_phishing else "Unknown",
        }
        if known_phishing:
            result["final_verdict"] = "Phishing"
            result["risk_score"] = 100
            result["confidence"] = 100
            return result
        
        # Layer 2: Heuristics
        if features is None:
            # Import feature extractor
            from feature_extractor import extract_features
            features = extract_features(url, include_network=True)
        
        is_suspicious, risk_score, reasons = self.heuristics.analyze(url, features)
        result["layers"]["heuristics"] = {
            "suspicious": is_suspicious,
            "risk_score": risk_score,
            "reasons": reasons,
            "verdict": "Phishing" if is_suspicious else "Suspicious" if risk_score > 20 else "Likely Legitimate",
        }
        
        # Layer 3: Brand Monitoring
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "").split(".")[0]
        is_typosquat, brand, similarity, variations = self.brand_monitor.check_typosquatting(domain)
        result["layers"]["brand_monitor"] = {
            "typosquatting": is_typosquat,
            "matched_brand": brand,
            "similarity": similarity,
            "variations_found": variations,
            "verdict": "Typosquatting" if is_typosquat else "Clean",
        }
        
        # Layer 4: Dynamic Analysis (if requested)
        if deep_analysis:
            print(f"Running dynamic analysis on {url}...")
            import asyncio
            dynamic_result = asyncio.run(self.dynamic_analyzer.analyze(url))
            result["layers"]["dynamic_analysis"] = dynamic_result
        
        # Combine verdicts
        result = self._combine_verdicts(result)
        return result
    
    def _combine_verdicts(self, result):
        """Combine all layer verdicts into final decision"""
        scores = []
        weights = []
        
        # Threat Intel (highest confidence)
        if result["layers"]["threat_intel"]["known_phishing"]:
            result["final_verdict"] = "Phishing"
            result["risk_score"] = 100
            result["confidence"] = 100
            return result
        
        # Heuristics
        h_score = result["layers"]["heuristics"]["risk_score"]
        scores.append(h_score)
        weights.append(0.4)
        
        # Brand Monitor
        if result["layers"]["brand_monitor"]["typosquatting"]:
            scores.append(85)
            weights.append(0.3)
        else:
            scores.append(0)
            weights.append(0.1)
        
        # Dynamic Analysis (if available)
        if "dynamic_analysis" in result["layers"]:
            dyn_score = result["layers"]["dynamic_analysis"].get("risk_score", 0)
            scores.append(dyn_score)
            weights.append(0.3)
        
        # Weighted average
        if sum(weights) > 0:
            final_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            final_score = 0
        
        # Decision thresholds
        if final_score >= 70:
            result["final_verdict"] = "Phishing"
            result["confidence"] = min(final_score, 95)
        elif final_score >= 40:
            result["final_verdict"] = "Suspicious"
            result["confidence"] = min(final_score + 10, 80)
        elif final_score >= 20:
            result["final_verdict"] = "Suspicious (Low Confidence)"
            result["confidence"] = min(final_score + 20, 70)
        else:
            result["final_verdict"] = "Likely Legitimate"
            result["confidence"] = 100 - min(final_score * 2, 80)
        
        result["risk_score"] = round(final_score, 1)
        return result


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def quick_scan(url):
    """Quick zero-day scan using heuristics only"""
    defense = ZeroDayDefense()
    return defense.analyze(url)

def deep_scan(url):
    """Deep scan with dynamic analysis"""
    defense = ZeroDayDefense()
    return defense.analyze(url, deep_analysis=True)

def batch_scan(urls):
    """Scan multiple URLs"""
    defense = ZeroDayDefense()
    results = []
    for url in urls:
        results.append(defense.analyze(url))
    return results


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Test URLs
    test_urls = [
        "https://google.com",
        "https://github.com",
        "http://paypal-login-security.xyz",
        "http://verify-account-paypal.com",
        "http://fake-bank-login.tk",
        "https://myapp.vercel.app",
        "https://docs.google.com/document/d/abc123",
    ]
    
    defense = ZeroDayDefense()
    
    print("=" * 70)
    print("ZERO-DAY PHISHING DEFENSE - TEST SCAN")
    print("=" * 70)
    
    for url in test_urls:
        print(f"\nScanning: {url}")
        result = defense.analyze(url)
        print(f"  Verdict: {result['final_verdict']}")
        print(f"  Risk Score: {result['risk_score']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Heuristics: {result['layers']['heuristics']['reasons']}")
        if result['layers']['brand_monitor']['typosquatting']:
            print(f"  Typosquatting: {result['layers']['brand_monitor']['matched_brand']}")