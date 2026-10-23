"""Read-only tools for the CV agent: load the profile and find matching offers."""

import json
from pathlib import Path

from pydantic import BaseModel, Field


class Profile(BaseModel):
    name: str
    location: str
    target_roles: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)


class Offer(BaseModel):
    id: str
    title: str
    company: str
    location: str
    required_skills: list[str]
    source: str
    url: str


class OfferMatch(BaseModel):
    offer: Offer
    matched_skills: list[str]
    score: float


def load_profile(path: Path) -> Profile:
    """Read the user's profile from a JSON file and validate it."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Profile.model_validate(data)


def load_offers(path: Path) -> list[Offer]:
    """Read saved job offers from a JSON file and validate them."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Offer.model_validate(item) for item in data]


def search_offers(
    profile: Profile, offers: list[Offer], limit: int = 10
) -> list[OfferMatch]:

    known = {skill.casefold() for skill in profile.skills}
    matches = []
    for offer in offers:
        required = {skill.casefold(): skill for skill in offer.required_skills}
        matched = [required[key] for key in required if key in known]
        if not matched:
            continue
        score = len(matched) / len(required)
        matches.append(OfferMatch(offer=offer, matched_skills=matched, score=score))
    matches.sort(
        key=lambda match: (match.score, len(match.matched_skills)), reverse=True
    )
    return matches[:limit]
