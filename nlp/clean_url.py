import re
from urllib.parse import urlparse


def clean_url(url):
    """
    Cleans a URL for NLP processing.
    Returns a string of meaningful words.
    """

    # Convert to lowercase
    url = url.lower()

    # Remove protocol
    url = re.sub(r"https?://", "", url)

    # Remove www
    url = re.sub(r"www\.", "", url)

    # Remove query parameters
    url = url.split("?")[0]

    # Remove fragments
    url = url.split("#")[0]

    # Replace separators with spaces
    url = re.sub(r"[-_/.:=]", " ", url)

    # Remove digits
    url = re.sub(r"\d+", " ", url)

    # Remove special characters
    url = re.sub(r"[^a-zA-Z ]", " ", url)

    # Remove extra spaces
    url = re.sub(r"\s+", " ", url)

    return url.strip()

if __name__ == "__main__":

    test_url = "https://www.paypal-login-security-update.com/login.php?id=123"

    print(clean_url(test_url))