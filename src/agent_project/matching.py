"""Scoring of job offers against a user's profile.

Skills, role and location build the score. The user's preferences (level, form
of employment, pay) never hide an offer: offer data is too incomplete for that.
An offer that goes against a preference loses part of its score and says why
in `conflicts`.
"""

import re
import unicodedata

from agent_project.models import Offer, OfferMatch, Profile, SalaryUnit

MAX_LIMIT = 30
SKILL_WEIGHT = 0.7
ROLE_WEIGHT = 0.2
LOCATION_WEIGHT = 0.1
# Each preference the offer goes against multiplies the score by this.
CONFLICT_FACTOR = 0.5
REMOTE_MARKERS = ("remote", "zdaln")
# Offers spell the same city in several ways ("Kraków", "Krakow", "Warsaw").
CITY_ALIASES = {"warsaw": "warszawa", "cracow": "krakow"}
# Words in an offer title that name its level; checked in this order.
LEVEL_WORDS = {
    "intern": ("intern", "internship", "staz", "stazysta", "stazystka", "praktykant"),
    "junior": ("junior", "mlodszy", "mlodsza"),
    "senior": ("senior", "starszy", "starsza"),
    "lead": ("lead", "principal", "head"),
    "mid": ("mid", "regular"),
}
# Rough working time, used only to compare pay given in different units.
MONTHLY_FACTOR: dict[SalaryUnit, float] = {
    "HOUR": 168,
    "DAY": 21,
    "MONTH": 1,
    "YEAR": 1 / 12,
}


def _plain(text: str) -> str:
    """Lower case without Polish diacritics, so "Kraków" equals "Krakow"."""
    text = text.casefold().replace("ł", "l")
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def _city(text: str) -> str:
    plain = _plain(text).strip()
    return CITY_ALIASES.get(plain, plain)


def _role_matches(profile: Profile, offer: Offer) -> bool:
    title = offer.title.casefold()
    return any(role.casefold() in title for role in profile.target_roles)


def _is_remote(offer: Offer) -> bool:
    location = _plain(offer.location)
    return bool(offer.remote) or any(marker in location for marker in REMOTE_MARKERS)


def _has_no_office(offer: Offer) -> bool:
    """A remote offer that names no city to work from."""
    location = _plain(offer.location)
    return any(marker in location for marker in REMOTE_MARKERS) or (
        bool(offer.remote) and not location.strip()
    )


def _location_matches(profile: Profile, offer: Offer) -> bool:
    modes = profile.work_modes
    if _is_remote(offer):
        return not modes or "remote" in modes
    if modes == ["remote"]:
        return False  # the user's own city does not help if they will not commute
    return bool(profile.location.strip()) and _city(profile.location) in _city(
        offer.location
    )


def offer_levels(title: str) -> list[str]:
    """Levels named in an offer title; empty when the title names none."""
    words = set(re.findall(r"[a-z]+", _plain(title)))
    return [level for level, names in LEVEL_WORDS.items() if words & set(names)]


def _pay_too_low(profile: Profile, offer: Offer) -> bool:
    wanted = profile.salary_expectation
    if wanted is None or offer.salary_max is None or offer.salary_unit is None:
        return False
    if offer.salary_currency != wanted.currency:
        return False  # no exchange rates here, so different currencies are not judged
    offered = offer.salary_max * MONTHLY_FACTOR[offer.salary_unit]
    return offered < wanted.min * MONTHLY_FACTOR[wanted.unit]


def _conflicts(profile: Profile, offer: Offer) -> list[str]:
    """Reasons, in Polish, why the offer goes against what the user asked for."""
    conflicts = []
    levels = offer_levels(offer.title)
    if profile.seniority and levels and not set(levels) & set(profile.seniority):
        conflicts.append(f"poziom: {', '.join(levels)}")
    modes = profile.work_modes
    if modes == ["remote"] and offer.remote is False:
        conflicts.append("praca stacjonarna")
    if modes and "remote" not in modes and _has_no_office(offer):
        conflicts.append("tylko praca zdalna")
    if (
        profile.employment_types
        and offer.employment_type
        and not set(profile.employment_types) & set(offer.employment_type)
    ):
        conflicts.append(f"forma zatrudnienia: {', '.join(offer.employment_type)}")
    if _pay_too_low(profile, offer):
        conflicts.append(
            f"wynagrodzenie do {offer.salary_max:g} {offer.salary_currency}"
            f"/{offer.salary_unit}, poniżej oczekiwań"
        )
    return conflicts


def search_offers(
    profile: Profile, offers: list[Offer], limit: int = 10
) -> list[OfferMatch]:
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit musi być od 1 do {MAX_LIMIT}, podano {limit}")

    known = {skill.casefold() for skill in profile.skills}
    matches = []
    for offer in offers:
        required = {skill.casefold(): skill for skill in offer.required_skills}
        matched = [required[key] for key in required if key in known]
        if not matched:
            continue
        skill_score = len(matched) / len(required)
        score = (
            SKILL_WEIGHT * skill_score
            + ROLE_WEIGHT * float(_role_matches(profile, offer))
            + LOCATION_WEIGHT * float(_location_matches(profile, offer))
        )
        conflicts = _conflicts(profile, offer)
        score *= CONFLICT_FACTOR ** len(conflicts)
        matches.append(
            OfferMatch(
                offer=offer, matched_skills=matched, score=score, conflicts=conflicts
            )
        )
    matches.sort(
        key=lambda match: (match.score, len(match.matched_skills)), reverse=True
    )
    return matches[:limit]
