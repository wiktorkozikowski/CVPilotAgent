"""The offer database the agent browses: the vector index from offer_server,
loaded into this process.

`match_profile` is the ranking for a whole profile (what the "Szukaj ofert"
button shows). `browse`, `details` and `compare` are what the model uses to look
around the database by itself: its own search phrases, filters, one offer in
full, and one offer against the profile.

Nothing here writes to the database; offer_server.update does that.
"""

from datetime import date
from pathlib import Path
from typing import Any, get_args

from agent_project.models import EmploymentType, Offer, OfferMatch, Profile
from offer_server.documents import OfferDoc, excerpt
from offer_server.embedder import FastEmbedder
from offer_server.holder import IndexHolder
from offer_server.index import load_index
from offer_server.paths import index_dir
from offer_server.ranking import (
    LEVEL_MAX_YEARS,
    MAX_LIMIT,
    SearchProfile,
    conflicts,
    level_conflicts,
    matched_skills,
    offer_levels,
    required_years,
    search,
    title_score,
)
from offer_server.text import city_key, plain

MAX_QUERY_CHARS = 200
DETAILS_CHARS = 2500  # how much of an offer's text the model reads at once
BROWSE_EXCERPT_CHARS = 200
KNOWN_EMPLOYMENT_TYPES = set(get_args(EmploymentType))
LEVELS = ("intern", "junior", "mid", "senior", "lead")
REMOTE_MARKERS = ("remote", "zdaln")


def search_profile(profile: Profile) -> SearchProfile:
    """The parts of a profile used for searching; contact data is left out."""
    data = profile.model_dump(
        include={
            "location",
            "target_roles",
            "skills",
            "seniority",
            "work_modes",
            "employment_types",
            "salary_expectation",
            "summary",
        },
        exclude_none=True,
    )
    data["experience"] = [
        {"position": job.position, "description": job.description}
        for job in profile.experience
    ]
    data["years_experience"] = years_of_experience(profile)
    return SearchProfile.model_validate(data)


def _month(value: str, today: date) -> float:
    """ "2021" or "2021-03" as a year with a fraction; "obecnie" is today."""
    if value == "obecnie":
        return today.year + (today.month - 1) / 12
    year, _, month = value.partition("-")
    return int(year) + ((int(month) - 1) / 12 if month else 0)


def years_of_experience(profile: Profile, today: date | None = None) -> float | None:
    """Years from the first job's start to the last job's end (or today).

    Gaps count too, so this leans generous: it only decides which offers ask
    for clearly more experience than the user has.
    """
    if not profile.experience:
        return None
    today = today or date.today()
    starts = [_month(job.start, today) for job in profile.experience]
    ends = [_month(job.end or "obecnie", today) for job in profile.experience]
    return round(max(0.0, max(ends) - min(starts)), 1)


def to_offer(doc: OfferDoc) -> Offer:
    return Offer(
        id=doc.id,
        title=doc.title,
        company=doc.company,
        location=doc.location or ("Zdalnie" if doc.remote else ""),
        source=doc.source,
        url=doc.url,
        remote=doc.remote,
        employment_type=[
            kind for kind in doc.employment_type if kind in KNOWN_EMPLOYMENT_TYPES
        ]
        or None,
        salary_min=doc.salary_min,
        salary_max=doc.salary_max,
        salary_currency=doc.salary_currency,
        salary_unit=doc.salary_unit,
        requirements=excerpt(doc) or None,
    )


def _is_remote(doc: OfferDoc) -> bool:
    return doc.remote or any(marker in plain(doc.location) for marker in REMOTE_MARKERS)


def _salary(doc: OfferDoc) -> dict[str, Any] | None:
    if doc.salary_min is None and doc.salary_max is None:
        return None
    return {
        "min": doc.salary_min,
        "max": doc.salary_max,
        "currency": doc.salary_currency,
        "unit": doc.salary_unit,
    }


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"


def _summary(doc: OfferDoc) -> dict[str, Any]:
    return {
        "id": doc.id,
        "title": doc.title,
        "company": doc.company,
        "location": doc.location,
        "remote": _is_remote(doc),
        "employment_type": doc.employment_type,
        "salary": _salary(doc),
        "years_required": required_years(doc),
    }


class OfferDB:
    def __init__(self, holder: IndexHolder) -> None:
        self.holder = holder
        self._lookup: dict[str, tuple[int, OfferDoc]] = {}
        self._lookup_for = None

    @classmethod
    def open(cls, folder: Path | None = None) -> "OfferDB":
        """Load the index from disk (var/index by default) with its own model."""
        folder = folder or index_dir()
        index = load_index(folder)
        embedder = FastEmbedder(index.meta.get("model"))
        return cls(IndexHolder(index, embedder, folder))

    def _by_id(self) -> dict[str, tuple[int, OfferDoc]]:
        index = self.holder.get()
        if self._lookup_for is not index:  # rebuilt once per loaded index
            self._lookup = {doc.id: (i, doc) for i, doc in enumerate(index.offers)}
            self._lookup_for = index
        return self._lookup

    def stats(self) -> dict[str, Any]:
        index = self.holder.get()
        return {
            "offers": len(index.offers),
            "built_at": index.meta.get("built_at"),
            "model": index.meta.get("model"),
        }

    def get(self, offer_id: str) -> OfferDoc | None:
        found = self._by_id().get(offer_id)
        return found[1] if found else None

    def offer(self, offer_id: str) -> Offer | None:
        doc = self.get(offer_id)
        return to_offer(doc) if doc else None

    def match_profile(
        self, profile: Profile, limit: int = 10, query: str | None = None
    ) -> list[OfferMatch]:
        """The ranking for a whole profile, best first."""
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit musi być od 1 do {MAX_LIMIT}, podano {limit}")
        found = search(
            self.holder.get(),
            self.holder.embedder,
            search_profile(profile),
            limit,
            query or None,
        )
        return [
            OfferMatch(
                offer=self.offer(candidate.id),
                matched_skills=candidate.matched_skills,
                score=candidate.score,
                conflicts=candidate.conflicts,
            )
            for candidate in found
        ]

    def browse(
        self,
        query: str,
        limit: int = 10,
        city: str | None = None,
        remote: bool | None = None,
        level: str | None = None,
        employment_type: str | None = None,
        exclude: set[str] | None = None,
        profile: Profile | None = None,
    ) -> list[dict[str, Any]]:
        """Offers closest in meaning to `query` that pass the filters.

        `level` drops offers whose title names another level or whose text asks
        for more years than that level has; titles without a level pass, since
        most do not say. With `profile`, offers that do not fit its level or
        experience are dropped too, and each result lists its conflicts.
        """
        query = query.strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"zapytanie to tekst od 1 do {MAX_QUERY_CHARS} znaków.")
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit musi być od 1 do {MAX_LIMIT}.")
        if level is not None and level not in LEVELS:
            raise ValueError(f"poziom to jedno z: {', '.join(LEVELS)}.")
        known = KNOWN_EMPLOYMENT_TYPES
        if employment_type is not None and employment_type not in known:
            kinds = ", ".join(sorted(KNOWN_EMPLOYMENT_TYPES))
            raise ValueError(f"forma_zatrudnienia to jedno z: {kinds}.")

        wanted = search_profile(profile) if profile is not None else None
        index = self.holder.get()
        vector = self.holder.embedder.embed([query])[0]
        similarities = index.offer_scores(vector)
        wanted_city = city_key(city) if city and city.strip() else None
        found = []
        for position in similarities.argsort()[::-1]:
            doc = index.offers[int(position)]
            if exclude and doc.id in exclude:
                continue
            if wanted_city and city_key(doc.location) != wanted_city:
                continue
            if remote is not None and _is_remote(doc) != remote:
                continue
            if level is not None:
                levels = offer_levels(doc.title)
                if levels and level not in levels:
                    continue
                years = required_years(doc)
                if years is not None and years > LEVEL_MAX_YEARS[level]:
                    continue
            if wanted is not None and level_conflicts(wanted, doc):
                continue
            if employment_type is not None and employment_type not in (
                doc.employment_type or []
            ):
                continue
            item = {
                **_summary(doc),
                "similarity": round(float(similarities[position]), 3),
                "requirements": _short(excerpt(doc), BROWSE_EXCERPT_CHARS),
            }
            if wanted is not None:
                item["conflicts"] = conflicts(wanted, doc)
            found.append(item)
            if len(found) == limit:
                break
        return found

    def level_problems(self, profile: Profile, offer_id: str) -> list[str]:
        """Why the offer is above (or beside) the user's level; empty if fine."""
        doc = self.get(offer_id)
        return [] if doc is None else level_conflicts(search_profile(profile), doc)

    def details(self, offer_id: str) -> dict[str, Any] | None:
        doc = self.get(offer_id)
        if doc is None:
            return None
        return {
            **_summary(doc),
            "url": doc.url,
            "source": doc.source,
            "date_posted": doc.date_posted,
            "valid_through": doc.valid_through,
            "requirements": _short(doc.requirements, DETAILS_CHARS),
            "duties": _short(doc.duties, DETAILS_CHARS),
            # The whole text only when the offer has no sections to cut out.
            "description": ""
            if doc.requirements or doc.duties
            else _short(doc.description, DETAILS_CHARS),
        }

    def compare(self, profile: Profile, offer_id: str) -> dict[str, Any] | None:
        """How one offer fits the profile, as facts for the model to weigh."""
        found = self._by_id().get(offer_id)
        if found is None:
            return None
        position, doc = found
        wanted = search_profile(profile)
        skills = matched_skills(wanted, doc)
        index = self.holder.get()
        text = ". ".join(
            part
            for part in (", ".join(wanted.target_roles), ", ".join(wanted.skills))
            if part
        )
        similarity = None
        if text:
            vector = self.holder.embedder.embed([text])[0]
            similarity = round(float(index.offer_scores(vector)[position]), 3)
        return {
            **_summary(doc),
            "similarity_to_profile": similarity,
            "title_matches_target_role": round(title_score(wanted, doc), 2),
            "profile_skills_in_offer": skills,
            "profile_skills_not_in_offer": [
                skill for skill in wanted.skills if skill not in skills
            ],
            "conflicts": conflicts(wanted, doc),
            "requirements": _short(doc.requirements or doc.description, DETAILS_CHARS),
        }
