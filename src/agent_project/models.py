"""Data models for the CV agent and loaders for the JSON files that hold them."""

import json
from pathlib import Path
from typing import Literal

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


class ShortlistResult(BaseModel):
    status: Literal[
        "added", "removed", "not_found", "already_exists", "confirmation_required"
    ]
    offer_id: str


def load_profile(path: Path) -> Profile:
    """Read the user's profile from a JSON file and validate it."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Profile.model_validate(data)


def load_offers(path: Path) -> list[Offer]:
    """Read saved job offers from a JSON file and validate them."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Offer.model_validate(item) for item in data]
