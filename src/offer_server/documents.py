"""Reads raw offers (offers_sample_*.json or fetch_all JSON Lines), cleans them
and cuts each one into the text pieces that get embedded."""

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from offer_server.text import city_key, plain

# Short lines containing these words open a section of the description.
REQUIREMENT_HEADERS = (
    "wymagani",
    "oczekuj",
    "oczekiwani",
    "kogo szukamy",
    "szukamy osoby",
    "szukamy wlasnie ciebie",
    "poszukiwane kompetencje",
    "mile widziane",
    "dodatkowym atutem",
    "twoj profil",
    "requirement",
    "must have",
    "nice to have",
    "your profile",
    "qualification",
    "looking for",
    "tech stack",
    "what you bring",
    "skills",
)
DUTY_HEADERS = (
    "obowiazk",
    "zadani",
    "zadan",
    "czym bedziesz",
    "co bedziesz",
    "opis stanowiska",
    "responsibilit",
    "your role",
    "about the role",
    "what you'll do",
    "what you will do",
    "your tasks",
    "job description",
)
OTHER_HEADERS = (
    "oferujemy",
    "oferta",
    "benefit",
    "o nas",
    "o firmie",
    "opis firmy",
    "kim jestesmy",
    "otrzymasz",
    "zyskasz",
    "we offer",
    "what we offer",
    "in it for you",
    "about us",
    "who we are",
    "recruitment",
    "rekrutac",
    "other details",
    "o projekcie",
    "about the project",
)
MAX_HEADER_CHARS = 60
MAX_HEADER_WORDS = 8
# The small embedding model reads about 128 tokens, so offers are cut into
# pieces of this many words, each starting with the offer title.
CHUNK_WORDS = 70
MAX_CHUNKS = 6
EXCERPT_CHARS = 900


class RawOffer(BaseModel):
    """One record in the format of data/offers_data_source.md."""

    model_config = ConfigDict(extra="ignore")

    id: str
    source: str
    url: str
    title: str
    company: str
    location: str = ""
    remote: bool = False
    employment_type: list[str] | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_unit: str | None = None
    date_posted: str | None = None
    valid_through: str | None = None
    description: str = ""

    # Older downloaders (fetch_offers.py, fetch_all.py) keep the page's values
    # as they were: null for a missing title or city, a single string for one
    # form of employment.
    @field_validator("title", "company", "location", "description", mode="before")
    @classmethod
    def _none_is_empty(cls, value: Any) -> Any:
        return "" if value is None else value

    @field_validator("remote", mode="before")
    @classmethod
    def _none_is_false(cls, value: Any) -> Any:
        return False if value is None else value

    @field_validator("employment_type", mode="before")
    @classmethod
    def _text_is_list(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class OfferDoc(BaseModel):
    """A cleaned offer with the parts of its description that matter."""

    id: str
    source: str
    url: str
    title: str
    company: str
    location: str
    remote: bool
    employment_type: list[str]
    salary_min: float | None
    salary_max: float | None
    salary_currency: str | None
    salary_unit: str | None
    date_posted: str | None
    valid_through: str | None
    requirements: str
    duties: str
    description: str


def _json_lines(text: str) -> tuple[list, int]:
    records, broken = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            broken += 1
    return records, broken


def load_raw_offers(paths: list[Path]) -> list[RawOffer]:
    """Read offer files: a JSON list, or one offer per line (fetch_all.py).

    The format is recognised from the content, not the extension, so a renamed
    file still loads. A folder means every *.json and *.jsonl file in it.
    A broken line (an interrupted download) or a record without id, URL or
    source is skipped and counted, so one bad offer does not stop the import.
    A file that holds no offers at all (e.g. options.json) is skipped.
    """
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted([*path.glob("*.jsonl"), *path.glob("*.json")]))
        else:
            files.append(path)
    offers = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        skipped = 0
        try:
            data = json.loads(text)
            records = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            records, skipped = _json_lines(text)
        if records and isinstance(records[0], dict) and "url" not in records[0]:
            print(f"{path}: to nie są oferty, pomijam.", flush=True)
            continue
        for record in records:
            try:
                offers.append(RawOffer.model_validate(record))
            except ValidationError:
                skipped += 1
        if skipped:
            print(f"{path}: pominięto {skipped} uszkodzonych rekordów.", flush=True)
    return offers


def _header_kind(line: str) -> str | None:
    stripped = line.strip().rstrip(":?")
    if not stripped or len(stripped) > MAX_HEADER_CHARS:
        return None
    if len(stripped.split()) > MAX_HEADER_WORDS:
        return None
    key = plain(stripped).replace("’", "'")
    for kind, headers in (
        ("other", OTHER_HEADERS),
        ("requirements", REQUIREMENT_HEADERS),
        ("duties", DUTY_HEADERS),
    ):
        if any(header in key for header in headers):
            return kind
    return None


def split_sections(description: str) -> dict[str, str]:
    """Requirements and duties from a description; empty strings if not found."""
    sections: dict[str, list[str]] = {"requirements": [], "duties": [], "other": []}
    current = "other"
    for line in description.splitlines():
        kind = _header_kind(line)
        if kind is not None:
            current = kind
            continue
        if line.strip():
            sections[current].append(line.strip())
    return {key: "\n".join(lines) for key, lines in sections.items()}


def _city_spellings(offers: list[RawOffer]) -> dict[str, str]:
    """For each city, its most common spelling that keeps Polish letters."""
    counts: dict[str, Counter[str]] = {}
    for offer in offers:
        if offer.location.strip():
            name = offer.location.strip()
            counts.setdefault(city_key(name), Counter())[name] += 1
    best = {}
    for key, names in counts.items():
        # Prefer a spelling with diacritics ("Kraków" over "Krakow"), then the
        # most common one.
        best[key] = max(names, key=lambda n: (not n.isascii(), names[n]))
    best.setdefault("warszawa", "Warszawa")
    return best


def clean_offers(offers: list[RawOffer], today: date) -> tuple[list[OfferDoc], dict]:
    """Drop expired offers and duplicates; unify city names. Returns stats too."""
    spellings = _city_spellings(offers)
    kept: dict[tuple[str, str, str], OfferDoc] = {}
    expired = 0
    for offer in offers:
        if offer.valid_through and offer.valid_through[:10] < today.isoformat():
            expired += 1
            continue
        location = offer.location.strip()
        if location:
            location = spellings.get(city_key(location), location)
        sections = split_sections(offer.description)
        doc = OfferDoc(
            id=f"{offer.source}:{offer.id}",
            source=offer.source,
            url=offer.url,
            title=offer.title.strip(),
            company=offer.company.strip(),
            location=location,
            remote=offer.remote,
            employment_type=offer.employment_type or [],
            salary_min=offer.salary_min,
            salary_max=offer.salary_max,
            salary_currency=offer.salary_currency,
            salary_unit=offer.salary_unit,
            date_posted=offer.date_posted,
            valid_through=offer.valid_through,
            requirements=sections["requirements"],
            duties=sections["duties"],
            description=offer.description,
        )
        # The same job is often listed under several addresses (categories).
        key = (plain(doc.company), plain(doc.title), city_key(doc.location))
        previous = kept.get(key)
        if previous is None or (doc.date_posted or "") > (previous.date_posted or ""):
            kept[key] = doc
    docs = list(kept.values())
    stats = {
        "read": len(offers),
        "expired": expired,
        "duplicates": len(offers) - expired - len(docs),
        "kept": len(docs),
        "with_requirements": sum(bool(doc.requirements) for doc in docs),
    }
    return docs, stats


def excerpt(doc: OfferDoc) -> str:
    """What the offer asks for, short enough to show the agent and the CV writer."""
    text = doc.requirements or doc.duties or doc.description
    if len(text) <= EXCERPT_CHARS:
        return text
    return text[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " …"


def chunks(doc: OfferDoc) -> list[str]:
    """Texts to embed for one offer: the title, then pieces of its requirements.

    Each piece starts with the title, so a piece about "SQL" still says which
    job it belongs to.
    """
    head = f"{doc.title}. {doc.company}."
    body = (
        " ".join(part for part in (doc.requirements, doc.duties) if part)
        or doc.description
    )
    tokens = re.split(r"\s+", body.strip()) if body.strip() else []
    pieces = [head]
    for start in range(0, len(tokens), CHUNK_WORDS):
        if len(pieces) > MAX_CHUNKS:
            break
        pieces.append(f"{doc.title}: " + " ".join(tokens[start : start + CHUNK_WORDS]))
    return pieces
