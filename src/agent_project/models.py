"""Data models for the CV agent and loaders for the JSON files that hold them."""

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

Seniority = Literal["intern", "junior", "mid", "senior", "lead"]
# Offers carry only a `remote` flag, but their descriptions name hybrid work often.
WorkMode = Literal["onsite", "hybrid", "remote"]
# The same values as `employment_type` in offers (schema.org JobPosting).
EmploymentType = Literal["FULL_TIME", "PART_TIME", "CONTRACTOR", "INTERN"]
# The same values as `salary_currency` and `salary_unit` in offers.
Currency = Literal["PLN", "EUR", "USD"]
SalaryUnit = Literal["HOUR", "DAY", "MONTH", "YEAR"]
# CEFR levels, as offers state them ("angielski B2"), plus a native speaker.
LanguageLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2", "native"]
DrivingLicense = Literal[
    "AM", "A1", "A2", "A", "B1", "B", "BE", "C1", "C1E", "C", "CE",
    "D1", "D1E", "D", "DE", "T",
]  # fmt: skip

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
PHONE_PATTERN = r"^\+?[0-9][0-9 \-()]{6,18}$"
URL_PATTERN = r"^https?://[^\s/]+\.[^\s]+$"
# Dates as "2021" or "2021-03"; an experience or education entry may end "obecnie".
START_PATTERN = r"^\d{4}(-\d{2})?$"
END_PATTERN = r"^(\d{4}(-\d{2})?|obecnie)$"
MAX_SUMMARY_CHARS = 600
Short = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Url = Annotated[str, StringConstraints(max_length=300, pattern=URL_PATTERN)]


class Experience(BaseModel):
    """One job: what the user did there, as short bullet points."""

    model_config = ConfigDict(extra="forbid")

    company: Short = Field(max_length=120)
    position: Short = Field(max_length=120)
    start: str = Field(pattern=START_PATTERN)
    end: str | None = Field(default=None, pattern=END_PATTERN)
    description: list[Short] = Field(default_factory=list, max_length=15)


class Education(BaseModel):
    model_config = ConfigDict(extra="forbid")

    institution: Short = Field(max_length=120)
    degree: Short = Field(max_length=120)
    start: str | None = Field(default=None, pattern=START_PATTERN)
    end: str | None = Field(default=None, pattern=END_PATTERN)


class Language(BaseModel):
    """A language with its level; offers usually ask for a level, not just a name."""

    model_config = ConfigDict(extra="forbid")

    name: Short = Field(max_length=40)
    level: LanguageLevel | None = None


class Certificate(BaseModel):
    """A certificate, course or formal permit (e.g. AWS, UDT, książeczka sanepid)."""

    model_config = ConfigDict(extra="forbid")

    name: Short = Field(max_length=160)
    issuer: Short | None = Field(default=None, max_length=120)
    year: str | None = Field(default=None, pattern=START_PATTERN)


class Project(BaseModel):
    """An own, study or open-source project; it stands in for experience in a CV."""

    model_config = ConfigDict(extra="forbid")

    name: Short = Field(max_length=120)
    description: list[Short] = Field(default_factory=list, max_length=10)
    technologies: list[Short] = Field(default_factory=list, max_length=30)
    url: Url | None = None


class SalaryExpectation(BaseModel):
    """The lowest pay the user accepts, in the units that offers use."""

    model_config = ConfigDict(extra="forbid")

    min: float = Field(gt=0, le=10_000_000)
    currency: Currency = "PLN"
    unit: SalaryUnit = "MONTH"


def languages_from_names(value: Any) -> Any:
    """Accept the old format, where a language was only its name."""
    if not isinstance(value, list):
        return value
    return [{"name": item} if isinstance(item, str) else item for item in value]


class Profile(BaseModel):
    """What a CV and the offer search need to know about the user.

    Left out on purpose: date of birth, photo, marital status and full address.
    A CV does not need them and they invite bias. Years of experience are not
    stored either: they follow from the dates in `experience`.
    """

    # Who the user is and how to reach them.
    name: str
    location: str
    email: str | None = Field(default=None, max_length=254, pattern=EMAIL_PATTERN)
    phone: str | None = Field(default=None, pattern=PHONE_PATTERN)
    linkedin: Url | None = None
    github: Url | None = None
    website: Url | None = None
    summary: str | None = Field(default=None, max_length=MAX_SUMMARY_CHARS)

    # What job the user is looking for; these mirror the fields of an offer.
    target_roles: list[str] = Field(default_factory=list)
    seniority: list[Seniority] = Field(default_factory=list)
    work_modes: list[WorkMode] = Field(default_factory=list)
    employment_types: list[EmploymentType] = Field(default_factory=list)
    salary_expectation: SalaryExpectation | None = None

    # What the user can do and has done.
    skills: list[str] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certificates: list[Certificate] = Field(default_factory=list)
    driving_licenses: list[DrivingLicense] = Field(default_factory=list)

    _old_languages = field_validator("languages", mode="before")(languages_from_names)


class Offer(BaseModel):
    id: str
    title: str
    company: str
    location: str
    required_skills: list[str] = Field(default_factory=list)
    source: str
    url: str
    # Optional, named as in offers_sample_*.json; None means the offer does not say.
    remote: bool | None = None
    employment_type: list[EmploymentType] | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_unit: SalaryUnit | None = None
    # What the offer asks for, cut from its description by the offer server.
    requirements: str | None = None


class OfferMatch(BaseModel):
    offer: Offer
    matched_skills: list[str]
    score: float
    # Why the offer goes against the user's preferences; each one lowers the score.
    conflicts: list[str] = Field(default_factory=list)


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
