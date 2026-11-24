"""Finds offers for a profile: vector search first, then the user's preferences.

The level is a hard requirement: an offer above (or beside) the user's level,
by the site's level tag, the title or the years it asks for, is left out.
Other preferences (work mode, form of employment, pay) never hide an offer,
because offer data is incomplete. An offer that goes against one loses half of
its score and says why in `conflicts`.
"""

import re
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from offer_server.documents import OfferDoc, excerpt
from offer_server.embedder import Embedder
from offer_server.index import OfferIndex
from offer_server.text import city_key, contains_term, plain, words

# Score = vector similarity plus three plain-text signals, each from 0 to 1.
# The small model alone ranks a remote "Camunda Engineer" close to "Magazynier",
# so the title and the skills found in the text keep it on track.
SIMILARITY_WEIGHT = 0.6
TITLE_WEIGHT = 0.2
SKILL_WEIGHT = 0.1
LOCATION_WEIGHT = 0.1
SKILLS_FOR_FULL_SCORE = 5
TITLE_MATCH_TO_RETRIEVE = 0.5
CONFLICT_FACTOR = 0.5
# The ranking hands the agent up to this many candidates.
MAX_LIMIT = 100
MIN_RETRIEVED = 100
REMOTE_MARKERS = ("remote", "zdaln")
# Words in an offer title that name its level.
LEVEL_WORDS = {
    "intern": ("intern", "internship", "staz", "stazysta", "stazystka", "praktykant"),
    "junior": ("junior", "mlodszy", "mlodsza"),
    "senior": ("senior", "starszy", "starsza"),
    "lead": ("lead", "principal", "head"),
    "mid": ("mid", "regular"),
}
# Rough working time, used only to compare pay given in different units.
MONTHLY_FACTOR = {"HOUR": 168, "DAY": 21, "MONTH": 1, "YEAR": 1 / 12}
# Most titles name no level, but the text often asks for years of experience
# ("3+ years of experience", "minimum 2-3 lata doświadczenia"). The number found
# is the lower bound of a range; a match counts only near a word about
# experience, so "firma od 20 lat na rynku" is ignored.
YEARS_PATTERN = re.compile(
    r"(?<![\d.,])(\d{1,2})\s*(?:\+|(?:-|–|—|to|do)\s*\d{1,2})?\s*\+?\s*"
    r"(?:years?|yrs|lata|lat|rok|roku)\b"
)
EXPERIENCE_WORDS = re.compile(r"experien|doswiadcz|stazu pracy")
EXPERIENCE_CONTEXT_CHARS = 80
MAX_PLAUSIBLE_YEARS = 15
# How level_conflicts starts its reasons; such offers are left out.
LEVEL_CONFLICT_PREFIXES = ("poziom:", "wymaga ")
# The most years an offer may ask for at each level the user names.
LEVEL_MAX_YEARS = {"intern": 0, "junior": 2, "mid": 5, "senior": 99, "lead": 99}

Seniority = Literal["intern", "junior", "mid", "senior", "lead"]
WorkMode = Literal["onsite", "hybrid", "remote"]
EmploymentType = Literal["FULL_TIME", "PART_TIME", "CONTRACTOR", "INTERN"]
Short = Field(max_length=200)


class Salary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float = Field(gt=0, le=10_000_000)
    currency: Literal["PLN", "EUR", "USD"] = "PLN"
    unit: Literal["HOUR", "DAY", "MONTH", "YEAR"] = "MONTH"


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: str = Short
    description: list[str] = Field(default_factory=list, max_length=15)


class SearchProfile(BaseModel):
    """What the server needs to search. Contact data is not accepted on purpose."""

    model_config = ConfigDict(extra="forbid")

    location: str = Field(default="", max_length=120)
    target_roles: list[str] = Field(default_factory=list, max_length=100)
    skills: list[str] = Field(default_factory=list, max_length=200)
    seniority: list[Seniority] = Field(default_factory=list)
    work_modes: list[WorkMode] = Field(default_factory=list)
    employment_types: list[EmploymentType] = Field(default_factory=list)
    salary_expectation: Salary | None = None
    summary: str | None = Field(default=None, max_length=600)
    experience: list[Job] = Field(default_factory=list, max_length=30)
    # Years of work from the profile's experience dates, if known.
    years_experience: float | None = Field(default=None, ge=0, le=60)


class Candidate(BaseModel):
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
    score: float
    similarity: float
    matched_skills: list[str]
    conflicts: list[str]
    requirements: str


def profile_text(profile: SearchProfile, query: str | None = None) -> str:
    """The profile as one text to embed. The most telling parts come first,
    because the model reads only the beginning of a long text."""
    parts = []
    if query:
        parts.append(query)
    if profile.target_roles:
        parts.append(", ".join(profile.target_roles))
    if profile.seniority:
        parts.append(" ".join(profile.seniority))
    if profile.skills:
        parts.append(", ".join(profile.skills))
    if profile.experience:
        parts.append(", ".join(job.position for job in profile.experience))
    if profile.summary:
        parts.append(profile.summary)
    return ". ".join(parts)


def offer_levels(title: str) -> list[str]:
    title_words = words(title)
    return [level for level, names in LEVEL_WORDS.items() if title_words & set(names)]


def doc_levels(offer: OfferDoc) -> list[str]:
    """The offer's level: the site's tag when it has one, else the title's.

    The tag wins: "Data Analyst (Junior / Mid)" tagged mid is a mid offer.
    """
    return [offer.level] if offer.level else offer_levels(offer.title)


def _is_remote(offer: OfferDoc) -> bool:
    location = plain(offer.location)
    return offer.remote or any(marker in location for marker in REMOTE_MARKERS)


def location_score(profile: SearchProfile, offer: OfferDoc) -> float:
    """1 for the user's city or a remote job they want; 0.5 for remote when the
    user did not say; 0 otherwise."""
    modes = profile.work_modes
    if _is_remote(offer):
        if "remote" in modes:
            return 1.0
        return 0.5 if not modes else 0.0
    if modes == ["remote"] or not profile.location.strip():
        return 0.0
    return float(city_key(profile.location) == city_key(offer.location))


def title_score(profile: SearchProfile, offer: OfferDoc) -> float:
    """Share of a target role's words found in the offer title (best role)."""
    title_words = words(offer.title)
    best = 0.0
    for role in profile.target_roles:
        role_words = words(role)
        if role_words:
            best = max(best, len(role_words & title_words) / len(role_words))
    return best


def _pay_too_low(profile: SearchProfile, offer: OfferDoc) -> bool:
    wanted = profile.salary_expectation
    if wanted is None or offer.salary_max is None or offer.salary_unit is None:
        return False
    if offer.salary_currency != wanted.currency:
        return False  # no exchange rates here
    factor = MONTHLY_FACTOR.get(offer.salary_unit)
    if factor is None:
        return False
    return offer.salary_max * factor < wanted.min * MONTHLY_FACTOR[wanted.unit]


def required_years(offer: OfferDoc) -> int | None:
    """The years of experience the offer asks for, if its text says so.

    With several numbers ("5+ years of Java, 2 years of Git") the largest
    counts, since all of them are required.
    """
    text = plain(offer.requirements or offer.description)
    found = []
    for match in YEARS_PATTERN.finditer(text):
        years = int(match.group(1))
        start = max(0, match.start() - EXPERIENCE_CONTEXT_CHARS)
        context = text[start : match.end() + EXPERIENCE_CONTEXT_CHARS]
        if 0 < years <= MAX_PLAUSIBLE_YEARS and EXPERIENCE_WORDS.search(context):
            found.append(years)
    return max(found) if found else None


def allowed_years(profile: SearchProfile) -> float | None:
    """The most years of experience an offer may ask of this user: the cap of
    the highest level they named, or their own years plus one if more."""
    if not profile.seniority and profile.years_experience is None:
        return None
    cap = max((LEVEL_MAX_YEARS[level] for level in profile.seniority), default=0)
    if profile.years_experience is not None:
        cap = max(cap, profile.years_experience + 1)
    return cap


def level_conflicts(profile: SearchProfile, offer: OfferDoc) -> list[str]:
    """The level named in the title or the years asked in the text do not fit."""
    found = []
    levels = doc_levels(offer)
    if profile.seniority and levels and not set(levels) & set(profile.seniority):
        found.append(f"poziom: {', '.join(levels)}")
    years = required_years(offer)
    allowed = allowed_years(profile)
    if years is not None and allowed is not None and years > allowed:
        found.append(f"wymaga {years}+ lat doświadczenia")
    return found


def conflicts(profile: SearchProfile, offer: OfferDoc) -> list[str]:
    """Reasons, in Polish, why the offer goes against what the user asked for."""
    found = level_conflicts(profile, offer)
    modes = profile.work_modes
    if modes == ["remote"] and not offer.remote:
        found.append("praca stacjonarna")
    if modes and "remote" not in modes and offer.remote and not offer.location:
        found.append("tylko praca zdalna")
    if (
        profile.employment_types
        and offer.employment_type
        and not set(profile.employment_types) & set(offer.employment_type)
    ):
        found.append(f"forma zatrudnienia: {', '.join(offer.employment_type)}")
    if _pay_too_low(profile, offer):
        found.append(
            f"wynagrodzenie do {offer.salary_max:g} {offer.salary_currency}"
            f"/{offer.salary_unit}, poniżej oczekiwań"
        )
    return found


def matched_skills(profile: SearchProfile, offer: OfferDoc) -> list[str]:
    text = f"{offer.title}\n{offer.description}"
    return [skill for skill in profile.skills if contains_term(text, skill)]


def search(
    index: OfferIndex,
    embedder: Embedder,
    profile: SearchProfile,
    limit: int = 10,
    query: str | None = None,
) -> list[Candidate]:
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit musi być od 1 do {MAX_LIMIT}")
    text = profile_text(profile, query)
    if not text.strip():
        raise ValueError("Profil nie ma stanowisk ani umiejętności do wyszukania.")
    vector: np.ndarray = embedder.embed([text])[0]
    similarities = index.offer_scores(vector)
    # The closest offers by meaning, plus every offer whose title names a target
    # role, so a weak model cannot hide an obvious match.
    positions = {i for i, _ in index.search(vector, max(MIN_RETRIEVED, limit * 5))}
    positions |= {
        i
        for i, offer in enumerate(index.offers)
        if title_score(profile, offer) >= TITLE_MATCH_TO_RETRIEVE
    }

    candidates = []
    for position in positions:
        similarity = float(similarities[position])
        offer = index.offers[position]
        reasons = conflicts(profile, offer)
        skills = matched_skills(profile, offer)
        skill_share = min(1.0, len(skills) / SKILLS_FOR_FULL_SCORE)
        score = (
            SIMILARITY_WEIGHT * max(similarity, 0.0)
            + TITLE_WEIGHT * title_score(profile, offer)
            + SKILL_WEIGHT * skill_share
            + LOCATION_WEIGHT * location_score(profile, offer)
        )
        score *= CONFLICT_FACTOR ** len(reasons)
        candidates.append(
            Candidate(
                **offer.model_dump(
                    include={
                        "id",
                        "source",
                        "url",
                        "title",
                        "company",
                        "location",
                        "remote",
                        "employment_type",
                        "salary_min",
                        "salary_max",
                        "salary_currency",
                        "salary_unit",
                    }
                ),
                score=round(score, 4),
                similarity=round(similarity, 4),
                matched_skills=skills,
                conflicts=reasons,
                requirements=re.sub(r"\n{2,}", "\n", excerpt(offer)),
            )
        )
    # The level is a hard requirement: offers above it are never shown.
    candidates = [
        c
        for c in candidates
        if not any(r.startswith(LEVEL_CONFLICT_PREFIXES) for r in c.conflicts)
    ]
    candidates.sort(key=lambda c: (c.score, len(c.matched_skills)), reverse=True)
    return candidates[:limit]
