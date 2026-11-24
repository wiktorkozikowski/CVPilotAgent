"""Reading offers from job boards: the active-jobs sitemap and single offer pages.

The same rules as src/agent_project/fetch_offers.py (kept as a copy so the
server does not depend on the agent package): only the sitemap and offer pages,
which robots.txt allows; never /api/; one request at a time with a pause.
"""

import json
import re
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser

USER_AGENT = "CVPilotAgent-student-project"
TIMEOUT_SECONDS = 30

SITES = {
    "justjoin": "https://justjoin.it",
    "rocketjobs": "https://rocketjobs.pl",
}

JSON_LD = re.compile(
    r'<script type="application/ld\+json"[^>]*>(.*?)</script>', re.DOTALL
)
LOC = re.compile(r"<loc>([^<]+)</loc>")
# The level the site shows on the offer ("junior", "mid", ...). It is not in the
# JobPosting block, only in the page data, where quotes may be escaped.
OFFER_LEVEL = re.compile(r'experienceLevel\\*":\{\\*"label\\*":\\*"(\w+)')
BLOCK_TAGS = {"p", "li", "br", "ul", "ol", "h1", "h2", "h3", "h4", "div"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(raw: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw)
    lines = (line.strip() for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def fetch(url: str, attempts: int = 2) -> str:
    """GET a page; network errors are retried, HTTP errors are raised at once."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def offer_urls(base: str, get=fetch) -> list[str]:
    """Follow the active-jobs sitemap index down to the offer URLs."""
    index = get(f"{base}/sitemaps/active-jobs.xml")
    urls: list[str] = []
    for part in LOC.findall(index):
        urls.extend(LOC.findall(get(part)))
    return urls


def _first_place(job_location) -> dict:
    if isinstance(job_location, list):
        job_location = job_location[0] if job_location else {}
    return (job_location or {}).get("address", {}) or {}


def parse_offer(url: str, page: str, source: str) -> dict | None:
    """Map the schema.org JobPosting block of an offer page to a flat record."""
    for block in JSON_LD.findall(page):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or data.get("@type") != "JobPosting":
            continue
        salary = (data.get("baseSalary") or {}).get("value") or {}
        kinds = data.get("employmentType")
        if isinstance(kinds, str):
            kinds = [kinds]
        level = OFFER_LEVEL.search(page)
        return {
            "id": url.rstrip("/").rsplit("/", 1)[-1],
            "source": source,
            "url": url,
            "title": data.get("title") or "",
            "company": (data.get("hiringOrganization") or {}).get("name") or "",
            "location": _first_place(data.get("jobLocation")).get("addressLocality")
            or "",
            "remote": data.get("jobLocationType") == "TELECOMMUTE",
            "employment_type": kinds,
            "salary_min": salary.get("minValue"),
            "salary_max": salary.get("maxValue"),
            "salary_currency": (data.get("baseSalary") or {}).get("currency"),
            "salary_unit": salary.get("unitText"),
            "date_posted": data.get("datePosted"),
            "valid_through": data.get("validThrough"),
            "description": html_to_text(data.get("description") or ""),
            "experience_level": level.group(1) if level else None,
        }
    return None
