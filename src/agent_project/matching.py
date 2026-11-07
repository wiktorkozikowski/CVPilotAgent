"""Scoring of job offers against a user's profile."""

from agent_project.models import Offer, OfferMatch, Profile

MAX_LIMIT = 10
SKILL_WEIGHT = 0.7
ROLE_WEIGHT = 0.2
LOCATION_WEIGHT = 0.1
REMOTE_MARKERS = ("remote", "zdaln")


def _role_matches(profile: Profile, offer: Offer) -> bool:
    title = offer.title.casefold()
    return any(role.casefold() in title for role in profile.target_roles)


def _location_matches(profile: Profile, offer: Offer) -> bool:
    location = offer.location.casefold()
    if any(marker in location for marker in REMOTE_MARKERS):
        return True
    return bool(profile.location) and profile.location.casefold() in location


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
        matches.append(OfferMatch(offer=offer, matched_skills=matched, score=score))
    matches.sort(
        key=lambda match: (match.score, len(match.matched_skills)), reverse=True
    )
    return matches[:limit]
