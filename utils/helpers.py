import socket
import requests
import whois
from urllib.parse import urlparse


# ---------------------------------------------------
# Get Domain
# ---------------------------------------------------

def get_domain(url):
    return urlparse(url).netloc


# ---------------------------------------------------
# HTTPS
# ---------------------------------------------------

def uses_https(url):
    return url.lower().startswith("https://")


# ---------------------------------------------------
# DNS Lookup
# ---------------------------------------------------

def dns_exists(url):

    try:

        socket.gethostbyname(get_domain(url))

        return True

    except:

        return False


# ---------------------------------------------------
# WHOIS Lookup
# ---------------------------------------------------

def get_domain_info(url):

    try:

        domain = get_domain(url)

        return whois.whois(domain)

    except:

        return None


# ---------------------------------------------------
# Count Redirects
# ---------------------------------------------------

def redirect_count(url):

    try:

        response = requests.get(

            url,

            timeout=5,

            allow_redirects=True

        )

        return len(response.history)

    except:

        return 0