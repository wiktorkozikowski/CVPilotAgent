"""Download a random sample of real job offers from a job board's public sitemap.

Usage: python -m agent_project.fetch_offers justjoin|rocketjobs [count]

Only the sitemap and single offer pages are requested (robots.txt disallows
/api/). Requests are sequential, with a pause between them.
"""

import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
USER_AGENT = "CVPilotAgent-student-project"
DELAY_SECONDS = 1.0
SEED = 42

SITES = {
    "justjoin": "https://justjoin.it",
    "rocketjobs": "https://rocketjobs.pl",
}

JSON_LD = re.compile(
    r'<script type="application/ld\+json"[^>]*>(.*?)</script>', re.DOTALL
)
LOC = re.compile(r"<loc>([^<]+)</loc>")
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


def fetch(url: str, attempts: int = 4) -> str:
    """GET a page, retrying with a growing pause on network errors."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def offer_urls(base: str) -> list[str]:
    """Follow the active-jobs sitemap index down to the offer URLs."""
    index = fetch(f"{base}/sitemaps/active-jobs.xml")
    urls: list[str] = []
    for part in LOC.findall(index):
        urls.extend(LOC.findall(fetch(part)))
    return urls


def first_place(job_location) -> dict:
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
        if data.get("@type") != "JobPosting":
            continue
        salary = (data.get("baseSalary") or {}).get("value") or {}
        return {
            "id": url.rstrip("/").rsplit("/", 1)[-1],
            "source": source,
            "url": url,
            "title": data.get("title"),
            "company": (data.get("hiringOrganization") or {}).get("name"),
            "location": first_place(data.get("jobLocation")).get("addressLocality"),
            "remote": data.get("jobLocationType") == "TELECOMMUTE",
            "employment_type": data.get("employmentType"),
            "salary_min": salary.get("minValue"),
            "salary_max": salary.get("maxValue"),
            "salary_currency": (data.get("baseSalary") or {}).get("currency"),
            "salary_unit": salary.get("unitText"),
            "date_posted": data.get("datePosted"),
            "valid_through": data.get("validThrough"),
            "description": html_to_text(data.get("description") or ""),
        }
    return None


def main() -> None:
    site = sys.argv[1] if len(sys.argv) > 1 else ""
    if site not in SITES:
        raise SystemExit(f"Usage: fetch_offers {'|'.join(SITES)} [count]")
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    base = SITES[site]

    urls = offer_urls(base)
    sample = random.Random(SEED).sample(urls, min(count, len(urls)))
    print(f"{site}: {len(urls)} offers in sitemap, fetching {len(sample)}", flush=True)

    out_path = DATA_DIR / f"offers_sample_{site}.json"
    offers: list[dict] = []
    if out_path.exists():
        wanted = set(sample)
        offers = [
            o
            for o in json.loads(out_path.read_text(encoding="utf-8"))
            if o["url"] in wanted
        ]
        print(f"  resuming: {len(offers)} already saved", flush=True)
    done = {offer["url"] for offer in offers}
    for number, url in enumerate(sample, 1):
        if url in done:
            continue
        try:
            offer = parse_offer(url, fetch(url), site)
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"  skip {url}: {error}", flush=True)
            offer = None
        if offer:
            offers.append(offer)
        if number % 25 == 0 or number == len(sample):
            out_path.write_text(
                json.dumps(offers, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"  {number}/{len(sample)} fetched, {len(offers)} saved", flush=True)
        time.sleep(DELAY_SECONDS)
    print(f"Saved {len(offers)} offers to {out_path}")


if __name__ == "__main__":
    main()
