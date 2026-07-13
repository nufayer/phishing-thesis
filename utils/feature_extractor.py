import re
import math
from collections import Counter
from urllib.parse import urlparse


# ---------------------------------------------------
# URL Entropy
# ---------------------------------------------------

def calculate_entropy(text):

    if len(text) == 0:
        return 0

    counter = Counter(text)

    entropy = 0

    for count in counter.values():

        p = count / len(text)

        entropy -= p * math.log2(p)

    return entropy


# ---------------------------------------------------
# Suspicious Words
# ---------------------------------------------------

SUSPICIOUS_WORDS = [

    "login",
    "signin",
    "secure",
    "account",
    "verify",
    "update",
    "bank",
    "paypal",
    "confirm",
    "password",
    "token",
    "wallet",
    "crypto",
    "bonus",
    "free"

]


def suspicious_word_count(url):

    url = url.lower()

    count = 0

    for word in SUSPICIOUS_WORDS:

        if word in url:
            count += 1

    return count


# ---------------------------------------------------
# Feature Extraction
# ---------------------------------------------------

def extract_features(url):

    parsed = urlparse(url)

    domain = parsed.netloc

    path = parsed.path

    features = {}

    features["URLLength"] = len(url)

    features["DomainLength"] = len(domain)

    features["PathLength"] = len(path)

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

    features["NoOfSpecialChars"] = len(
        re.findall(r"[^A-Za-z0-9]", url)
    )

    features["DigitRatio"] = (
        features["NoOfDigits"] / features["URLLength"]
        if features["URLLength"] else 0
    )

    features["LetterRatio"] = (
        features["NoOfLetters"] / features["URLLength"]
        if features["URLLength"] else 0
    )

    features["SpecialCharRatio"] = (
        features["NoOfSpecialChars"] / features["URLLength"]
        if features["URLLength"] else 0
    )

    features["URLEntropy"] = calculate_entropy(url)

    features["SuspiciousWordCount"] = suspicious_word_count(url)

    features["NoOfPercent"] = url.count("%")

    features["NoOfColon"] = url.count(":")

    features["NoOfSemicolon"] = url.count(";")

    features["NoOfComma"] = url.count(",")

    features["NoOfPlus"] = url.count("+")

    features["NoOfStar"] = url.count("*")

    features["NoOfDollar"] = url.count("$")

    features["NoOfTilde"] = url.count("~")

    features["IsHTTPS"] = int(parsed.scheme == "https")

    ip_pattern = r"^\d+\.\d+\.\d+\.\d+$"

    features["HasIPAddress"] = int(
        re.match(ip_pattern, domain) is not None
    )

    if domain:

        features["NoOfSubDomain"] = max(
            len(domain.split(".")) - 2,
            0
        )

    else:

        features["NoOfSubDomain"] = 0

    return features