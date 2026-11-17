"""Client of the offer server (CVPilotAgent/server): sends the parts of a profile
needed for the search and turns the returned candidates into OfferMatch objects.

Name, e-mail, phone and links never leave this program.

Settings in .env:
    OFFERS_SERVER_URL    e.g. http://127.0.0.1:8100; unset = local search
    OFFERS_SERVER_TOKEN  the server's OFFER_SERVER_TOKEN, if it has one
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, get_args

from dotenv import load_dotenv

from agent_project.models import EmploymentType, Offer, OfferMatch, Profile

MAX_LIMIT = 10
TIMEOUT_SECONDS = 30
# Only these profile fields are sent; the server rejects anything else.
SEARCH_FIELDS = (
    "location",
    "target_roles",
    "skills",
    "seniority",
    "work_modes",
    "employment_types",
    "salary_expectation",
    "summary",
)
KNOWN_EMPLOYMENT_TYPES = set(get_args(EmploymentType))


class OffersServerError(RuntimeError):
    """The server could not be reached or refused the request."""


def search_payload(profile: Profile) -> dict[str, Any]:
    data = profile.model_dump(include=set(SEARCH_FIELDS), exclude_none=True)
    data["experience"] = [
        {"position": job.position, "description": job.description}
        for job in profile.experience
    ]
    return data


def _to_match(candidate: dict[str, Any]) -> OfferMatch:
    offer = Offer(
        id=candidate["id"],
        title=candidate["title"],
        company=candidate["company"],
        location=candidate.get("location")
        or ("Zdalnie" if candidate.get("remote") else ""),
        source=candidate["source"],
        url=candidate["url"],
        remote=candidate.get("remote"),
        employment_type=[
            kind
            for kind in candidate.get("employment_type") or []
            if kind in KNOWN_EMPLOYMENT_TYPES
        ]
        or None,
        salary_min=candidate.get("salary_min"),
        salary_max=candidate.get("salary_max"),
        salary_currency=candidate.get("salary_currency"),
        salary_unit=candidate.get("salary_unit"),
        requirements=candidate.get("requirements") or None,
    )
    return OfferMatch(
        offer=offer,
        matched_skills=candidate.get("matched_skills", []),
        score=candidate["score"],
        conflicts=candidate.get("conflicts", []),
    )


class OffersServerClient:
    def __init__(
        self, url: str, token: str | None = None, timeout: float = TIMEOUT_SECONDS
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OffersServerClient | None":
        """A client when OFFERS_SERVER_URL is set, otherwise None (local search)."""
        load_dotenv()
        url = os.getenv("OFFERS_SERVER_URL")
        if not url:
            return None
        return cls(url, os.getenv("OFFERS_SERVER_TOKEN") or None)

    def search(
        self, profile: Profile, limit: int = 10, query: str | None = None
    ) -> list[OfferMatch]:
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit musi być od 1 do {MAX_LIMIT}, podano {limit}")
        payload: dict[str, Any] = {"profile": search_payload(profile), "limit": limit}
        if query:
            payload["query"] = query
        request = urllib.request.Request(
            self.url + "/search",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                message = json.loads(error.read().decode("utf-8"))["message"]
            except (ValueError, KeyError, UnicodeDecodeError):
                message = f"HTTP {error.code}"
            raise OffersServerError(
                f"Serwer ofert odrzucił zapytanie: {message}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise OffersServerError("Serwer ofert jest niedostępny.") from error
        return [_to_match(candidate) for candidate in data.get("candidates", [])]


def load_seen_offers(path: Path) -> list[Offer]:
    """Offers the server returned earlier, so the shortlist can show and check them."""
    if not path.exists():
        return []
    return [Offer.model_validate(item) for item in json.loads(path.read_text("utf-8"))]


def remember_offers(path: Path, offers: list[Offer]) -> None:
    by_id = {offer.id: offer for offer in load_seen_offers(path)}
    by_id.update({offer.id: offer for offer in offers})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [offer.model_dump(exclude_none=True) for offer in by_id.values()],
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
