import re
import math
import socket
import ssl
import whois
import dns.resolver
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime
import tldextract
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------
# Suspicious Words
# ---------------------------------------------------

SUSPICIOUS_WORDS = [
    "login", "signin", "secure", "account", "verify", "update",
    "bank", "paypal", "confirm", "password", "token", "wallet",
    "crypto", "bonus", "free", "support", "service", "billing",
    "recovery", "reset", "unlock", "validate", "auth", "ssn",
    "social", "security", "credit", "card", "debit", "wire",
    "transfer", "payment", "invoice", "order", "shipping",
    "delivery", "tracking", "refund", "cancel", "suspend",
    "limited", "restricted", "urgent", "immediate", "action",
    "required", "alert", "warning", "notice", "official"
]

# Known URL shorteners
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bc.vc", "tiny.cc", "short.to", "s.id",
    "cutt.ly", "rb.gy", "v.gd", "x.co", "tr.im", "u.to", "yourls.org"
}

# Known legitimate brands (for similarity checking)
KNOWN_BRANDS = {
    "google", "github", "facebook", "microsoft", "apple", "amazon",
    "netflix", "twitter", "linkedin", "youtube", "instagram", "wikipedia",
    "reddit", "stackoverflow", "dropbox", "adobe", "salesforce", "slack",
    "zoom", "notion", "figma", "vercel", "cloudflare", "paypal", "stripe",
    "twilio", "sendgrid", "mailchimp", "shopify", "squarespace", "wix",
    "wordpress", "medium", "nytimes", "wsj", "bloomberg", "reuters", "cnn",
    "bbc", "theguardian", "npr", "pbs", "gov", "nih", "cdc", "who", "un"
}

# Suspicious TLDs
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "party",
    "loan", "download", "racing", "stream", "science", "bid", "win",
    "accountant", "faith", "date", "review", "trade", "webcam", "wang"
}

# ---------------------------------------------------
# URL Entropy
# ---------------------------------------------------

def calculate_entropy(text):
    counter = Counter(text)
    length = len(text)
    entropy = 0
    for count in counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy


# ---------------------------------------------------
# Character Continuation Rate
# ---------------------------------------------------

def char_continuation_rate(text):
    if len(text) < 2:
        return 0
    continuations = sum(1 for i in range(len(text)-1) if text[i] == text[i+1])
    return continuations / (len(text) - 1)


# ---------------------------------------------------
# Suspicious Word Counter
# ---------------------------------------------------

def suspicious_word_count(url):
    url = url.lower()
    count = 0
    for word in SUSPICIOUS_WORDS:
        if word in url:
            count += 1
    return count


# ---------------------------------------------------
# URL Shortener Detection
# ---------------------------------------------------

def is_url_shortener(domain):
    domain = domain.lower().replace("www.", "")
    return 1 if domain in URL_SHORTENERS else 0


# ---------------------------------------------------
# Brand Similarity (Levenshtein-like)
# ---------------------------------------------------

def brand_similarity(domain):
    domain = domain.lower().replace("www.", "")
    parts = domain.split(".")
    max_sim = 0
    for part in parts:
        for brand in KNOWN_BRANDS:
            # Exact match
            if part == brand:
                return 1.0
            # Substring match (brand in part or part in brand)
            if brand in part or part in brand:
                common = sum(1 for c in brand if c in part)
                sim = common / max(len(brand), len(part))
                max_sim = max(max_sim, sim)
            # Character overlap
            common = sum(1 for c in brand if c in part)
            sim = common / max(len(brand), len(part))
            max_sim = max(max_sim, sim)
    return max_sim


# ---------------------------------------------------
# Prefix/Suffix Analysis
# ---------------------------------------------------

def has_suspicious_prefix_suffix(domain):
    domain = domain.lower().replace("www.", "")
    parts = domain.split(".")
    main = parts[0] if parts else ""
    
    # Check for brand name with prefix/suffix
    suspicious = 0
    for brand in KNOWN_BRANDS:
        if brand in main and main != brand:
            # Check if brand is at start or end with extra chars
            if main.startswith(brand) or main.endswith(brand):
                suspicious = 1
                break
    return suspicious


# ---------------------------------------------------
# Obfuscation Detection
# ---------------------------------------------------

def has_obfuscation(url):
    # Hex encoding, unicode, excessive encoding
    obfuscation_patterns = [
        r"%[0-9a-fA-F]{2}",  # URL encoding
        r"\\x[0-9a-fA-F]{2}",  # Hex
        r"\\u[0-9a-fA-F]{4}",  # Unicode
        r"&#x?[0-9]+;",  # HTML entities
    ]
    count = sum(len(re.findall(p, url)) for p in obfuscation_patterns)
    return 1 if count > 3 else 0


def obfuscation_ratio(url):
    obf_chars = len(re.findall(r"%[0-9a-fA-F]{2}", url))
    return obf_chars / len(url) if url else 0


# ---------------------------------------------------
# TLD Analysis
# ---------------------------------------------------

def tld_features(domain):
    extracted = tldextract.extract(domain)
    tld = extracted.suffix.lower()
    
    return {
        "TLDLength": len(tld),
        "IsSuspiciousTLD": 1 if tld in SUSPICIOUS_TLDS else 0,
        "TLDLegitimateProb": 1.0 if tld in {"com", "org", "net", "edu", "gov", "io", "co", "ai"} else 0.3
    }


# ---------------------------------------------------
# URL Similarity Index
# ---------------------------------------------------

def url_similarity_index(url):
    # How similar is this URL to common patterns
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.lower()
    
    score = 0
    # Common legitimate patterns
    if domain.endswith((".com", ".org", ".net", ".edu", ".gov")):
        score += 0.3
    if "login" in path or "signin" in path:
        score -= 0.2
    if parsed.scheme == "https":
        score += 0.2
    return max(0, min(1, score + 0.5))


# ---------------------------------------------------
# SSL Certificate Features
# ---------------------------------------------------

def get_ssl_features(domain):
    """Extract SSL certificate features - returns BENIGN defaults on failure"""
    features = {
        "HasSSL": 0,
        "SSLCertValid": 1,  # Assume valid unless proven otherwise
        "SSLDaysToExpiry": 365,
        "SSLIsSelfSigned": 0,
        "SSLHasValidChain": 1,
        "SSLKeySize": 2048,
        "SSLSigAlgorithm": "",
    }
    
    try:
        # Create SSL context
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        with socket.create_connection((domain, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                features["HasSSL"] = 1
                
                # Check validity
                not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                days_to_expiry = (not_after - datetime.now()).days
                features["SSLDaysToExpiry"] = max(days_to_expiry, 0)
                features["SSLCertValid"] = 1 if days_to_expiry > 0 else 0
                
                # Check key size
                pubkey = ssl._ssl._test_decode_cert(cert)
                if 'subjectPublicKeyInfo' in cert:
                    # Extract key size from certificate
                    pass
                
                # Check if self-signed (simplified)
                issuer = dict(x[0] for x in cert.get('issuer', []))
                subject = dict(x[0] for x in cert.get('subject', []))
                features["SSLIsSelfSigned"] = 1 if issuer == subject else 0
                features["SSLHasValidChain"] = 1 if issuer != subject else 0
                
    except Exception:
        # Return BENIGN defaults on any failure
        pass
    
    return features


# ---------------------------------------------------
# WHOIS Features
# ---------------------------------------------------

def get_whois_features(domain):
    """Extract WHOIS/domain age features - returns BENIGN defaults on failure"""
    features = {
        "DomainAgeDays": 365,    # Assume 1 year old if unknown
        "DomainAgeMonths": 12,
        "WhoisAvailable": 0,
        "RegistrarReputation": 1,  # Assume reputable if unknown
    }
    
    try:
        w = whois.whois(domain)
        features["WhoisAvailable"] = 1
        
        if w.creation_date:
            if isinstance(w.creation_date, list):
                creation = w.creation_date[0]
            else:
                creation = w.creation_date
            if creation:
                age_days = (datetime.now() - creation).days
                features["DomainAgeDays"] = max(age_days, 0)
                features["DomainAgeMonths"] = age_days / 30
        
        # Check registrar reputation
        if w.registrar:
            reputable_registrars = {
                "godaddy", "namecheap", "google domains", "cloudflare", "amazon",
                "gandi", "hover", "name.com", "porkbun", "dynadot", "epik"
            }
            registrar_lower = str(w.registrar).lower()
            features["RegistrarReputation"] = 1 if any(r in str(w.registrar).lower() for r in reputable_registrars) else 0
            
    except Exception:
        # Return BENIGN defaults on any failure
        pass
    
    return features


# ---------------------------------------------------
# DNS Features
# ---------------------------------------------------

def get_dns_features(domain):
    """Extract DNS features - returns BENIGN defaults on failure"""
    features = {
        "HasMXRecord": 1,  # Assume exists if unknown
        "HasARecord": 1,
        "HasNSRecord": 1,
        "HasTXTRecord": 1,
        "DNSServersCount": 2,
        "HasSPFRecord": 1,   # Assume exists if unknown
        "HasDMARCRecord": 1, # Assume exists if unknown
    }
    
    try:
        # A record
        try:
            answers = dns.resolver.resolve(domain, 'A')
            features["HasARecord"] = 1 if answers else 0
        except:
            features["HasARecord"] = 0
        
        # MX record
        try:
            answers = dns.resolver.resolve(domain, 'MX')
            features["HasMXRecord"] = 1 if answers else 0
        except:
            features["HasMXRecord"] = 0
        
        # NS record
        try:
            answers = dns.resolver.resolve(domain, 'NS')
            features["HasNSRecord"] = 1 if answers else 0
            features["DNSServersCount"] = len(answers) if answers else 0
        except:
            features["HasNSRecord"] = 0
            features["DNSServersCount"] = 0
        
        # TXT record (check for SPF/DMARC)
        try:
            answers = dns.resolver.resolve(domain, 'TXT')
            if answers:
                features["HasTXTRecord"] = 1
                for rdata in answers:
                    txt = str(rdata).lower()
                    if "v=spf1" in txt:
                        features["HasSPFRecord"] = 1
                    if "v=dmarc1" in txt:
                        features["HasDMARCRecord"] = 1
            else:
                features["HasTXTRecord"] = 0
                features["HasSPFRecord"] = 0
                features["HasDMARCRecord"] = 0
        except:
            features["HasTXTRecord"] = 0
            features["HasSPFRecord"] = 0
            features["HasDMARCRecord"] = 0
            
    except Exception:
        # Return BENIGN defaults on any failure
        pass
    
    return features


# ---------------------------------------------------
# Feature Extraction
# ---------------------------------------------------

def extract_features(url, include_network=False):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query
    
    features = {}
    
    # ---------------------------------------------------
    # Basic URL Features
    # ---------------------------------------------------
    
    features["URLLength"] = len(url)
    features["DomainLength"] = len(domain)
    features["PathLength"] = len(path)
    features["QueryLength"] = len(query)
    
    features["NoOfDots"] = url.count(".")
    features["NoOfHyphen"] = url.count("-")
    features["NoOfUnderscore"] = url.count("_")
    features["NoOfSlash"] = url.count("/")
    features["NoOfQuestionMark"] = url.count("?")
    features["NoOfEqual"] = url.count("=")
    features["NoOfAt"] = url.count("@")
    features["NoOfAmpersand"] = url.count("&")
    
    features["NoOfDigits"] = sum(c.isdigit() for c in url)
    features["NoOfLetters"] = sum(c.isalpha() for c in url)
    features["NoOfSpecialChars"] = len(re.findall(r"[^A-Za-z0-9]", url))
    
    # ---------------------------------------------------
    # Ratios
    # ---------------------------------------------------
    
    url_len = features["URLLength"]
    features["DigitRatio"] = features["NoOfDigits"] / url_len if url_len else 0
    features["LetterRatio"] = features["NoOfLetters"] / url_len if url_len else 0
    features["SpecialCharRatio"] = features["NoOfSpecialChars"] / url_len if url_len else 0
    features["HyphenRatio"] = features["NoOfHyphen"] / url_len if url_len else 0
    features["SubDomainRatio"] = features.get("NoOfSubDomain", 0) / max(features["NoOfDots"], 1)
    
    # ---------------------------------------------------
    # URL Entropy
    # ---------------------------------------------------
    
    features["URLEntropy"] = calculate_entropy(url)
    features["DomainEntropy"] = calculate_entropy(domain) if domain else 0
    features["PathEntropy"] = calculate_entropy(path) if path else 0
    
    # ---------------------------------------------------
    # Character Continuation Rate
    # ---------------------------------------------------
    
    features["CharContinuationRate"] = char_continuation_rate(url)
    
    # ---------------------------------------------------
    # Suspicious Words
    # ---------------------------------------------------
    
    features["SuspiciousWordCount"] = suspicious_word_count(url)
    features["SuspiciousWordInDomain"] = suspicious_word_count(domain)
    features["SuspiciousWordInPath"] = suspicious_word_count(path)
    
    # ---------------------------------------------------
    # Suspicious Characters
    # ---------------------------------------------------
    
    features["NoOfPercent"] = url.count("%")
    features["NoOfColon"] = url.count(":")
    features["NoOfSemicolon"] = url.count(";")
    features["NoOfComma"] = url.count(",")
    features["NoOfPlus"] = url.count("+")
    features["NoOfStar"] = url.count("*")
    features["NoOfDollar"] = url.count("$")
    features["NoOfTilde"] = url.count("~")
    features["NoOfHash"] = url.count("#")
    features["NoOfBracket"] = url.count("[") + url.count("]")
    features["NoOfParenthesis"] = url.count("(") + url.count(")")
    
    # ---------------------------------------------------
    # HTTPS
    # ---------------------------------------------------
    
    features["IsHTTPS"] = 1 if parsed.scheme == "https" else 0
    
    # ---------------------------------------------------
    # IP Address
    # ---------------------------------------------------
    
    ip_pattern = r"^\d+\.\d+\.\d+\.\d+$"
    features["HasIPAddress"] = 1 if re.match(ip_pattern, domain) else 0
    
    # ---------------------------------------------------
    # Subdomains
    # ---------------------------------------------------
    
    if domain:
        features["NoOfSubDomain"] = max(len(domain.split(".")) - 2, 0)
        features["MaxSubDomainLength"] = max(len(p) for p in domain.split(".")) if domain else 0
    else:
        features["NoOfSubDomain"] = 0
        features["MaxSubDomainLength"] = 0
    
    # ---------------------------------------------------
    # TLD Features
    # ---------------------------------------------------
    
    tld_feats = tld_features(domain)
    features.update(tld_feats)
    
    # ---------------------------------------------------
    # URL Shortener
    # ---------------------------------------------------
    
    features["IsURLShortener"] = is_url_shortener(domain)
    
    # ---------------------------------------------------
    # Brand Similarity
    # ---------------------------------------------------
    
    features["BrandSimilarity"] = brand_similarity(domain)
    features["HasSuspiciousPrefixSuffix"] = has_suspicious_prefix_suffix(domain)
    
    # ---------------------------------------------------
    # Obfuscation
    # ---------------------------------------------------
    
    features["HasObfuscation"] = has_obfuscation(url)
    features["ObfuscationRatio"] = obfuscation_ratio(url)
    
    # ---------------------------------------------------
    # URL Similarity Index
    # ---------------------------------------------------
    
    features["URLSimilarityIndex"] = url_similarity_index(url)
    
    # ---------------------------------------------------
    # URL Char Probability
    # ---------------------------------------------------
    
    # Probability of random character sequence
    if url_len > 0:
        unique_chars = len(set(url.lower()))
        features["URLCharProb"] = unique_chars / url_len
    else:
        features["URLCharProb"] = 0
    
    # ---------------------------------------------------
    # Double Slash Redirect
    # ---------------------------------------------------
    
    features["DoubleSlashRedirect"] = 1 if "//" in path else 0
    
    # ---------------------------------------------------
    # Prefix/Suffix in Domain
    # ---------------------------------------------------
    
    features["PrefixSuffix"] = 1 if "-" in domain.split(".")[0] else 0
    
    # ---------------------------------------------------
    # Abnormal URL (IP in URL, @ symbol)
    # ---------------------------------------------------
    
    features["AbnormalURL"] = 1 if "@" in url or features["HasIPAddress"] else 0
    
    # ---------------------------------------------------
    # Network Features (optional - slower)
    # ---------------------------------------------------
    
    if include_network and domain:
        # SSL features
        ssl_feats = get_ssl_features(domain)
        features.update(ssl_feats)
        
        # WHOIS features
        whois_feats = get_whois_features(domain)
        features.update(whois_feats)
        
        # DNS features
        dns_feats = get_dns_features(domain)
        features.update(dns_feats)
    else:
        # Default values for network features when not fetched
        default_network = {
            "HasSSL": features["IsHTTPS"],
            "SSLCertValid": features["IsHTTPS"],
            "SSLDaysToExpiry": 365,
            "SSLIsSelfSigned": 0,
            "SSLHasValidChain": 1,
            "SSLKeySize": 2048,
            "DomainAgeDays": 365,
            "DomainAgeMonths": 12,
            "WhoisAvailable": 0,
            "RegistrarReputation": 0,
            "HasMXRecord": 0,
            "HasARecord": 1 if features["IsHTTPS"] else 0,
            "HasNSRecord": 0,
            "HasTXTRecord": 0,
            "DNSServersCount": 0,
            "HasSPFRecord": 0,
            "HasDMARCRecord": 0,
        }
        features.update(default_network)
    
    return features