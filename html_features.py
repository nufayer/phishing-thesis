import requests
import urllib3

from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Ignore SSL warnings
urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


def extract_html_features(url):

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/137.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=3,
            allow_redirects=True,
            verify=False
        )

        soup = BeautifulSoup(response.text, "html.parser")

        features = {}

        # ------------------------------------------------
        # Basic Page Information
        # ------------------------------------------------

        features["StatusCode"] = response.status_code
        features["FinalURLLength"] = len(response.url)
        features["RedirectCount"] = len(response.history)

        # ------------------------------------------------
        # HTML Elements
        # ------------------------------------------------

        features["TitleLength"] = (
            len(soup.title.string.strip())
            if soup.title and soup.title.string
            else 0
        )

        features["NoOfForms"] = len(soup.find_all("form"))
        features["NoOfInputs"] = len(soup.find_all("input"))
        features["NoOfPasswordFields"] = len(
            soup.find_all("input", {"type": "password"})
        )
        features["NoOfHiddenFields"] = len(
            soup.find_all("input", {"type": "hidden"})
        )
        features["NoOfButtons"] = len(soup.find_all("button"))
        features["NoOfLinks"] = len(soup.find_all("a"))
        features["NoOfImages"] = len(soup.find_all("img"))
        features["NoOfScripts"] = len(soup.find_all("script"))
        features["NoOfCSS"] = len(
            soup.find_all("link", rel="stylesheet")
        )
        features["NoOfIframes"] = len(soup.find_all("iframe"))

        # ------------------------------------------------
        # Meta Information
        # ------------------------------------------------

        features["HasDescription"] = int(
            soup.find("meta", attrs={"name": "description"}) is not None
        )

        features["HasFavicon"] = int(
            soup.find(
                "link",
                rel=lambda x: x and "icon" in x.lower()
            ) is not None
        )

        # ------------------------------------------------
        # Security
        # ------------------------------------------------

        parsed = urlparse(response.url)

        features["UsesHTTPS"] = int(parsed.scheme == "https")
        features["HasLoginForm"] = int(
            features["NoOfPasswordFields"] > 0
        )

        return features

    except Exception:
        return None