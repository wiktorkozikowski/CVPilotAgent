"""Builds a user profile in a chat: a draft file, typed tools, and their checks.

The model may request a tool call. Every check happens here in Python:
arguments are validated against ProfileDraft, the save needs a confirmation that
comes from the application (never from the model's arguments), and nothing is
saved while required fields are missing.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent_project.agent import MAX_RUN_COST_USD, _error, run_agent
from agent_project.models import (
    EMAIL_PATTERN,
    MAX_SUMMARY_CHARS,
    PHONE_PATTERN,
    Certificate,
    Currency,
    DrivingLicense,
    Education,
    EmploymentType,
    Experience,
    Language,
    LanguageLevel,
    Profile,
    Project,
    SalaryExpectation,
    SalaryUnit,
    Seniority,
    Url,
    WorkMode,
    languages_from_names,
)
from agent_project.profiles import save_profile

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DRAFT_PATH = DATA_DIR / "profile_draft.json"
PROFILES_DIR = DATA_DIR / "profiles"
REQUIRED_FIELDS = ("name", "location", "target_roles", "skills")

# Limits: a full CV can hold many roles and skills, but a single chat turn
# should add only a few at a time.
MAX_ROLES = 100
MAX_SKILLS = 200
MAX_LANGUAGES = 20
MAX_ENTRIES = 30
MAX_ITEMS_PER_CALL = 20

LIST_SECTIONS = ("target_roles", "skills")
ENTRY_MODELS: dict[str, type[BaseModel]] = {
    "languages": Language,
    "experience": Experience,
    "education": Education,
    "projects": Project,
    "certificates": Certificate,
}


class ProfileDraft(BaseModel):
    """A profile that may still be incomplete. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=120)
    location: str | None = Field(default=None, max_length=120)
    email: str | None = Field(default=None, max_length=254, pattern=EMAIL_PATTERN)
    phone: str | None = Field(default=None, pattern=PHONE_PATTERN)
    linkedin: Url | None = None
    github: Url | None = None
    website: Url | None = None
    summary: str | None = Field(default=None, max_length=MAX_SUMMARY_CHARS)
    target_roles: list[str] | None = Field(default=None, max_length=MAX_ROLES)
    seniority: list[Seniority] | None = Field(default=None, max_length=5)
    work_modes: list[WorkMode] | None = Field(default=None, max_length=3)
    employment_types: list[EmploymentType] | None = Field(default=None, max_length=4)
    salary_expectation: SalaryExpectation | None = None
    skills: list[str] | None = Field(default=None, max_length=MAX_SKILLS)
    languages: list[Language] | None = Field(default=None, max_length=MAX_LANGUAGES)
    experience: list[Experience] | None = Field(default=None, max_length=MAX_ENTRIES)
    education: list[Education] | None = Field(default=None, max_length=MAX_ENTRIES)
    projects: list[Project] | None = Field(default=None, max_length=MAX_ENTRIES)
    certificates: list[Certificate] | None = Field(default=None, max_length=MAX_ENTRIES)
    driving_licenses: list[DrivingLicense] | None = Field(default=None, max_length=16)

    _old_languages = field_validator("languages", mode="before")(languages_from_names)

    @field_validator("name", "location", "summary")
    @classmethod
    def _not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("nie może być puste")
        return value

    @field_validator("target_roles", "skills")
    @classmethod
    def _items_not_blank(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        items = [item.strip() for item in value]
        if any(not item for item in items):
            raise ValueError("pozycje nie mogą być puste")
        return items


def _load_draft(draft_path: Path) -> dict[str, Any]:
    if not draft_path.exists():
        return {}
    return json.loads(draft_path.read_text(encoding="utf-8"))


def _save_draft(draft_path: Path, draft: dict[str, Any]) -> None:
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(
        json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_profile_into_draft(profile: Profile, draft_path: Path) -> dict[str, Any]:
    """Replace the draft with a saved profile, so the chat can edit it."""
    _save_draft(draft_path, profile.model_dump(mode="json", exclude_none=True))
    return get_draft(draft_path)


def missing_fields(draft: dict[str, Any]) -> list[str]:
    return [field for field in REQUIRED_FIELDS if not draft.get(field)]


def get_draft(draft_path: Path) -> dict[str, Any]:
    """Read-only: the draft so far and the required fields still missing."""
    draft = _load_draft(draft_path)
    return {"status": "ok", "draft": draft, "missing_fields": missing_fields(draft)}


def update_draft(fields: dict[str, Any], draft_path: Path) -> dict[str, Any]:
    """Reversible write: merge validated fields into the draft.

    Nothing is written when any field is invalid, so a bad value never
    overwrites a good one. A list passed here replaces the previous list.
    """
    unknown = sorted(set(fields) - set(ProfileDraft.model_fields))
    if unknown:
        return {
            "status": "validation_error",
            "message": f"Nieznane pola: {', '.join(unknown)}",
        }

    merged = {**_load_draft(draft_path), **fields}
    try:
        validated = ProfileDraft.model_validate(merged)
    except ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return {
            "status": "validation_error",
            "message": f"Niepoprawna wartość pola {location}: {first['msg']}",
        }

    draft = validated.model_dump(exclude_none=True)
    _save_draft(draft_path, draft)
    return {"status": "ok", "draft": draft, "missing_fields": missing_fields(draft)}


def _merge_unique(existing: list[str], new: list[str]) -> list[str]:
    """Append items, skipping any already present (case-insensitive)."""
    seen = {item.casefold() for item in existing}
    merged = list(existing)
    for item in new:
        key = item.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            merged.append(item.strip())
    return merged


def add_items(section: Any, items: Any, draft_path: Path) -> dict[str, Any]:
    """Append roles or skills to the draft without resending the whole list."""
    if section not in LIST_SECTIONS:
        return _validation_error(f"Nieznana sekcja: {section}")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS_PER_CALL:
        return _validation_error(f"Podaj od 1 do {MAX_ITEMS_PER_CALL} pozycji.")
    if not all(isinstance(item, str) for item in items):
        return _validation_error("Pozycje muszą być tekstem.")

    current = _load_draft(draft_path).get(section, [])
    return update_draft({section: _merge_unique(current, items)}, draft_path)


def add_entry(section: str, entry: dict[str, Any], draft_path: Path) -> dict[str, Any]:
    """Append one entry (a job, a school, a language...), checked against its model.

    A language already in the draft is replaced, so its level can be corrected.
    """
    try:
        validated = ENTRY_MODELS[section].model_validate(entry)
    except ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return _validation_error(f"Niepoprawna wartość pola {location}: {first['msg']}")

    current = _load_draft(draft_path).get(section, [])
    if section == "languages":
        name = validated.name.casefold()
        current = [
            item
            for item in languages_from_names(current)
            if str(item.get("name", "")).casefold() != name
        ]
    if len(current) >= MAX_ENTRIES:
        return _validation_error(f"Można zapisać najwyżej {MAX_ENTRIES} wpisów.")
    entries = [*current, validated.model_dump(exclude_none=True)]
    return update_draft({section: entries}, draft_path)


def _bad_index(index: Any, items: list) -> bool:
    return (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(items)
    )


def replace_item(
    section: Any, index: Any, value: Any, draft_path: Path
) -> dict[str, Any]:
    """Correct one role or skill in place. Used by the page, not the model."""
    if section not in LIST_SECTIONS:
        return _validation_error(f"Nieznana sekcja: {section}")
    if not isinstance(value, str) or not value.strip():
        return _validation_error("Pozycja nie może być pusta.")
    items = list(_load_draft(draft_path).get(section, []))
    if _bad_index(index, items):
        return {"status": "not_found"}
    value = value.strip()
    if any(
        other.casefold() == value.casefold()
        for position, other in enumerate(items)
        if position != index
    ):
        return _validation_error("Ta pozycja już jest na liście.")
    items[index] = value
    return update_draft({section: items}, draft_path)


def save_entry(
    section: Any, entry: Any, draft_path: Path, index: Any = None
) -> dict[str, Any]:
    """Add an entry, or with `index` replace that one. Used by the page."""
    if section not in ENTRY_MODELS:
        return _validation_error(f"Nieznana sekcja: {section}")
    if not isinstance(entry, dict):
        return _validation_error("Wpis musi być obiektem JSON.")
    if index is None:
        return add_entry(section, entry, draft_path)
    try:
        validated = ENTRY_MODELS[section].model_validate(entry)
    except ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return _validation_error(f"Niepoprawna wartość pola {location}: {first['msg']}")

    current = list(_load_draft(draft_path).get(section, []))
    if section == "languages":
        current = languages_from_names(current)
    if _bad_index(index, current):
        return {"status": "not_found"}
    if section == "languages" and any(
        str(item.get("name", "")).casefold() == validated.name.casefold()
        for position, item in enumerate(current)
        if position != index
    ):
        return _validation_error("Ten język już jest na liście.")
    current[index] = validated.model_dump(exclude_none=True)
    return update_draft({section: current}, draft_path)


def remove_item(section: str, index: int, draft_path: Path) -> dict[str, Any]:
    """Remove one item by its position (0-based). Used by the page, not the model."""
    if section not in (*LIST_SECTIONS, *ENTRY_MODELS):
        return _validation_error(f"Nieznana sekcja: {section}")
    draft = _load_draft(draft_path)
    items = draft.get(section, [])
    if _bad_index(index, items):
        return {"status": "not_found"}
    del items[index]
    _save_draft(draft_path, draft)
    return {
        "status": "removed",
        "draft": draft,
        "missing_fields": missing_fields(draft),
    }


def finalize_profile(
    draft_path: Path,
    profiles_dir: Path,
    confirmed: bool,
    profile_id: str | None = None,
    keep_draft: bool = False,
) -> dict[str, Any]:
    """Sensitive: save the draft as a profile, only after the user confirmed.

    `confirmed` must come from the application, never from the model's arguments.
    With `profile_id` the draft overwrites that saved profile instead of adding a
    new one; with `keep_draft` the draft stays, so the chat can go on editing it.
    """
    draft = _load_draft(draft_path)
    missing = missing_fields(draft)
    if missing:
        return {"status": "incomplete", "missing_fields": missing}
    if not confirmed:
        return {"status": "confirmation_required"}

    profile = Profile.model_validate(draft)
    profile_id = save_profile(profiles_dir, profile, profile_id)
    if not keep_draft:
        draft_path.unlink(missing_ok=True)
    return {"status": "saved", "profile_id": profile_id}


def _validation_error(message: str) -> dict[str, Any]:
    return {"status": "validation_error", "message": message}


_STRING_LIST = {"type": "array", "items": {"type": "string"}}


def _enum_list(literal: Any, description: str) -> dict[str, Any]:
    """A JSON-schema list whose allowed values come from the model's Literal."""
    return {
        "type": "array",
        "items": {"type": "string", "enum": list(get_args(literal))},
        "description": description,
    }


PROFILE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "pobierz_profil",
            "description": (
                "Zwraca profil zbierany w rozmowie i listę brakujących pól "
                "(missing_fields). Użyj przed pytaniem o kolejną informację."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zapisz_pola_profilu",
            "description": (
                "Zapisuje dane kontaktowe, podsumowanie i preferencje pracy podane "
                "przez użytkownika. Podawaj tylko pola, które użytkownik faktycznie "
                "podał. Listy zastępują poprzednie wartości. Zwraca status: ok albo "
                "validation_error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Imię i nazwisko."},
                    "location": {"type": "string", "description": "Miasto."},
                    "email": {"type": "string", "description": "Adres e-mail."},
                    "phone": {"type": "string", "description": "Numer telefonu."},
                    "linkedin": {
                        "type": "string",
                        "description": "Pełny adres profilu LinkedIn (https://...).",
                    },
                    "github": {
                        "type": "string",
                        "description": "Pełny adres profilu GitHub (https://...).",
                    },
                    "website": {
                        "type": "string",
                        "description": "Adres portfolio albo strony (https://...).",
                    },
                    "summary": {
                        "type": "string",
                        "maxLength": MAX_SUMMARY_CHARS,
                        "description": (
                            "Podsumowanie zawodowe: 2-4 zdania o tym, kim użytkownik "
                            "jest zawodowo i czego szuka."
                        ),
                    },
                    "seniority": _enum_list(
                        Seniority,
                        "Poziom stanowisk, których szuka: intern (staż), junior, "
                        "mid, senior, lead.",
                    ),
                    "work_modes": _enum_list(
                        WorkMode,
                        "Akceptowany tryb pracy: onsite (stacjonarnie), hybrid, "
                        "remote (zdalnie).",
                    ),
                    "employment_types": _enum_list(
                        EmploymentType,
                        "Akceptowana forma: FULL_TIME (pełny etat), PART_TIME "
                        "(część etatu), CONTRACTOR (B2B, kontrakt), INTERN (staż).",
                    ),
                    "salary_expectation": {
                        "type": "object",
                        "description": "Najniższe wynagrodzenie, które akceptuje.",
                        "properties": {
                            "min": {"type": "number", "description": "Kwota."},
                            "currency": {
                                "type": "string",
                                "enum": list(get_args(Currency)),
                                "description": "Waluta, domyślnie PLN.",
                            },
                            "unit": {
                                "type": "string",
                                "enum": list(get_args(SalaryUnit)),
                                "description": "Za co: domyślnie MONTH (miesiąc).",
                            },
                        },
                        "required": ["min"],
                    },
                    "driving_licenses": _enum_list(
                        DrivingLicense, "Kategorie prawa jazdy, np. B."
                    ),
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dodaj_pozycje_profilu",
            "description": (
                "Dopisuje stanowiska (target_roles) albo umiejętności (skills) do "
                "profilu. Nie usuwa poprzednich pozycji i pomija duplikaty. Zwraca "
                "status: ok albo validation_error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": list(LIST_SECTIONS),
                        "description": "Która lista.",
                    },
                    "items": {
                        **_STRING_LIST,
                        "maxItems": MAX_ITEMS_PER_CALL,
                        "description": "Nowe pozycje, po jednej w każdym elemencie.",
                    },
                },
                "required": ["section", "items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dodaj_doswiadczenie",
            "description": (
                "Dodaje jedno miejsce pracy do doświadczenia. Opis to krótkie punkty: "
                "co użytkownik tam robił. Daty w formacie RRRR albo RRRR-MM; koniec "
                "to „obecnie”, jeśli użytkownik nadal tam pracuje."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Firma."},
                    "position": {"type": "string", "description": "Stanowisko."},
                    "start": {"type": "string", "description": "Początek, RRRR-MM."},
                    "end": {
                        "type": "string",
                        "description": "Koniec, RRRR-MM albo obecnie.",
                    },
                    "description": {
                        **_STRING_LIST,
                        "maxItems": 15,
                        "description": "Co tam robiłeś, po jednym punkcie.",
                    },
                },
                "required": ["company", "position", "start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dodaj_wyksztalcenie",
            "description": (
                "Dodaje jedną szkołę albo kierunek studiów. Daty w formacie RRRR; "
                "koniec to „obecnie”, jeśli użytkownik nadal się tam uczy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "institution": {"type": "string", "description": "Uczelnia."},
                    "degree": {
                        "type": "string",
                        "description": "Kierunek i stopień, np. Informatyka (lic.).",
                    },
                    "start": {"type": "string", "description": "Rok rozpoczęcia."},
                    "end": {
                        "type": "string",
                        "description": "Rok ukończenia albo obecnie.",
                    },
                },
                "required": ["institution", "degree"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dodaj_jezyk",
            "description": (
                "Dodaje jeden język z poziomem. Ponowne dodanie tego samego języka "
                "zmienia jego poziom. Nie zgaduj poziomu: jeśli użytkownik go nie "
                "podał, pomiń pole level."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Język, np. angielski."},
                    "level": {
                        "type": "string",
                        "enum": list(get_args(LanguageLevel)),
                        "description": "Poziom CEFR albo native (ojczysty).",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dodaj_projekt",
            "description": (
                "Dodaje jeden projekt własny, studencki albo open source. Ważne "
                "zwłaszcza, gdy użytkownik ma mało doświadczenia zawodowego."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Nazwa projektu."},
                    "description": {
                        **_STRING_LIST,
                        "maxItems": 10,
                        "description": "Co zrobił i z jakim efektem, po punkcie.",
                    },
                    "technologies": {
                        **_STRING_LIST,
                        "maxItems": 30,
                        "description": "Użyte technologie i narzędzia.",
                    },
                    "url": {
                        "type": "string",
                        "description": "Adres projektu (https://...).",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dodaj_certyfikat",
            "description": (
                "Dodaje jeden certyfikat, kurs albo uprawnienie zawodowe (np. AWS "
                "Cloud Practitioner, uprawnienia UDT, książeczka sanepidowska). "
                "Prawo jazdy zapisuj w zapisz_pola_profilu."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Nazwa."},
                    "issuer": {"type": "string", "description": "Kto wydał."},
                    "year": {"type": "string", "description": "Rok uzyskania, RRRR."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zatwierdz_profil",
            "description": (
                "Zapisuje profil po zgodzie użytkownika na podsumowanie. Działa tylko, "
                "gdy nie brakuje pól. Zwraca status: saved, incomplete albo "
                "confirmation_required."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

PROFILE_SYSTEM_PROMPT = (
    "Jesteś asystentem, który pomaga użytkownikowi przygotować profil (CV) do "
    "wyszukiwania pracy.\n"
    "Odpowiadaj krótko: najwyżej trzy zdania i jedno pytanie. Nie wypisuj długich "
    "list ani propozycji umiejętności.\n"
    "Pytaj o jedną rzecz naraz, w tej kolejności: pola z missing_fields; "
    "doświadczenie; wykształcenie; języki z poziomem; czego szuka (poziom, tryb "
    "pracy, forma zatrudnienia, oczekiwane wynagrodzenie); projekty; certyfikaty, "
    "uprawnienia i prawo jazdy; e-mail, telefon i linki; krótkie podsumowanie. "
    "Wszystko poza missing_fields jest dobrowolne: zapytaj raz i idź dalej, jeśli "
    "użytkownik nie chce odpowiadać. Zapisuj tylko to, co użytkownik podał, i nie "
    "wymyślaj wartości.\n"
    "W doświadczeniu i projektach dopytaj o konkrety: co użytkownik zrobił, czym "
    "i z jakim efektem (najlepiej z liczbą). Jeśli ma mało doświadczenia, zapytaj "
    "o projekty.\n"
    "Nie pytaj o datę urodzenia, stan cywilny, zdjęcie ani dokładny adres i ich "
    "nie zapisuj: w CV są zbędne.\n"
    "Stanowiska i umiejętności dopisuj narzędziem dodaj_pozycje_profilu. Miejsce "
    "pracy zapisuj dodaj_doswiadczenie, szkołę dodaj_wyksztalcenie, język "
    "dodaj_jezyk, projekt dodaj_projekt, certyfikat albo uprawnienie "
    "dodaj_certyfikat. Pozostałe pola zapisuj zapisz_pola_profilu.\n"
    "Jeśli narzędzie zwróci validation_error, poproś użytkownika o poprawną "
    "wartość.\n"
    "Zanim cokolwiek zapiszesz, sprawdź, czy odpowiedź ma sens dla pola, o które "
    "pytasz:\n"
    "- Miasto: jeśli to nie jest nazwa miejscowości (np. „pizza”, „asdf”), nic nie "
    "zapisuj; napisz, że nie znasz takiego miasta, i poproś o inne. Jeśli wygląda "
    "na literówkę albo brak polskich znaków (np. „gdańks”, „krakow”), nic nie "
    "zapisuj; zapytaj „Czy chodziło o Gdańsk?” i zapisz poprawną nazwę dopiero, "
    "gdy użytkownik potwierdzi.\n"
    "- Stanowiska, umiejętności i języki: oczywiste literówki poprawiaj tak samo, "
    "pytając najpierw (np. „pyton” → „Czy chodziło o Python?”). Jeśli coś nie "
    "jest umiejętnością ani stanowiskiem, dopytaj, zamiast zapisywać.\n"
    "- Imię i nazwisko: jeśli wpis nie wygląda na imię i nazwisko (cyfry, "
    "przypadkowe znaki, jedno słowo), dopytaj.\n"
    "- Gdy pytasz „Czy chodziło o…?”, w tej odpowiedzi nie wywołuj żadnego "
    "narzędzia dla tego pola. Poczekaj, aż użytkownik potwierdzi, i dopiero wtedy "
    "zapisz poprawioną wartość.\n"
    "- Zapisuj nazwy w poprawnej pisowni, z polskimi znakami i wielką literą "
    "(np. „Gdańsk”, „Wiktor Kozikowski”).\n"
    "Nie odrzucaj wartości tylko dlatego, że są nietypowe: zagraniczne miasta, "
    "rzadkie stanowiska i nowe technologie są w porządku. Pytaj tylko wtedy, gdy "
    "masz wątpliwość.\n"
    "Gdy wymagane pola są uzupełnione, napisz, że profil widać po lewej stronie i "
    "zapisze się po kliknięciu przycisku „Zatwierdź profil”. Sam nie wywołuj "
    "zatwierdz_profil."
)

# Added to the chat only when the app can search offers (see run_profile_agent).
CHAT_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "szukaj_ofert",
        "description": (
            "Wyszukuje oferty pracy dla zapisanego profilu użytkownika. Wyniki "
            "pojawiają się w tabeli po prawej stronie strony. Zwraca status: ok "
            "(z listą ofert), no_profile (profil nie jest jeszcze zapisany) albo "
            "error."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Ile ofert znaleźć, od 1 do 30.",
                }
            },
            "required": ["limit"],
        },
    },
}

CHAT_SEARCH_PROMPT = (
    "\nWyszukiwanie ofert: gdy użytkownik chce zobaczyć oferty (np. „szukaj "
    "ofert”), a nie podał, ile ich chce, zapytaj o liczbę od 1 do 30. Potem "
    "wywołaj szukaj_ofert z tą liczbą. Jeśli narzędzie zwróci no_profile, napisz, "
    "że najpierw trzeba kliknąć „Zatwierdź profil”. Po wyszukaniu napisz krótko, "
    "że oferty są w tabeli po prawej, i wymień najwyżej trzy najlepsze (stanowisko "
    "i firma). Jeśli któraś ma niepuste conflicts, powiedz, co nie pasuje. Nie "
    "wymyślaj ofert."
)

# Added when the chat can also browse the offer database itself.
CHAT_DB_PROMPT = (
    "\nPrzeglądanie bazy ofert: na pytania o konkretne oferty, firmy, miasta, "
    "technologie albo pokrewne stanowiska użyj przeszukaj_oferty z własnym "
    "zapytaniem i filtrami. Szczegóły jednej oferty da pokaz_oferte, a jej "
    "dopasowanie do zapisanego profilu porownaj_z_profilem. Podawaj tylko "
    "oferty z wyników narzędzi, ze stanowiskiem, firmą i id."
)


ENTRY_TOOLS = {
    "dodaj_doswiadczenie": "experience",
    "dodaj_wyksztalcenie": "education",
    "dodaj_jezyk": "languages",
    "dodaj_projekt": "projects",
    "dodaj_certyfikat": "certificates",
}


def _deny() -> bool:
    return False


def run_profile_tool(
    call: dict,
    draft_path: Path = DRAFT_PATH,
    profiles_dir: Path = PROFILES_DIR,
    confirm: Callable[[], bool] = _deny,
) -> str:
    """Check one model request in Python, then run the matching function."""
    try:
        arguments = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError:
        return _error("Argumenty narzędzia nie są poprawnym JSON-em.")
    if not isinstance(arguments, dict):
        return _error("Argumenty narzędzia muszą być obiektem JSON.")

    if call["name"] == "pobierz_profil":
        result = get_draft(draft_path)
    elif call["name"] == "zapisz_pola_profilu":
        result = update_draft(arguments, draft_path)
    elif call["name"] == "dodaj_pozycje_profilu":
        result = add_items(arguments.get("section"), arguments.get("items"), draft_path)
    elif call["name"] in ENTRY_TOOLS:
        result = add_entry(ENTRY_TOOLS[call["name"]], arguments, draft_path)
    elif call["name"] == "zatwierdz_profil":
        # Arguments from the model are ignored; only the app's confirm() counts.
        result = finalize_profile(draft_path, profiles_dir, confirmed=confirm())
    else:
        return _error(f"Nieznane narzędzie: {call['name']}")
    return json.dumps(result, ensure_ascii=False)


def run_profile_agent(
    client,
    messages: list[dict],
    trace_path: Path,
    draft_path: Path = DRAFT_PATH,
    profiles_dir: Path = PROFILES_DIR,
    confirm: Callable[[], bool] = _deny,
    max_steps: int = 12,
    max_cost_usd: float = MAX_RUN_COST_USD,
    search: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    extra_tools: list[dict[str, Any]] | None = None,
    extra_dispatch: Callable[[dict], str] | None = None,
) -> str:
    """Run the chat that builds a profile, using the shared agent loop.

    With `search` the model also gets szukaj_ofert; `search` receives the tool's
    arguments and returns the result the model sees. `extra_tools` (e.g. the
    offer database tools) are run by `extra_dispatch`.
    """
    tools = PROFILE_TOOLS if search is None else [*PROFILE_TOOLS, CHAT_SEARCH_TOOL]
    extra_names = {tool["function"]["name"] for tool in extra_tools or []}
    if extra_tools and extra_dispatch is not None:
        tools = [*tools, *extra_tools]

    def dispatch(call: dict) -> str:
        if extra_dispatch is not None and call["name"] in extra_names:
            return extra_dispatch(call)
        if search is not None and call["name"] == "szukaj_ofert":
            try:
                arguments = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                return _error("Argumenty narzędzia nie są poprawnym JSON-em.")
            if not isinstance(arguments, dict):
                return _error("Argumenty narzędzia muszą być obiektem JSON.")
            return json.dumps(search(arguments), ensure_ascii=False)
        return run_profile_tool(call, draft_path, profiles_dir, confirm)

    return run_agent(
        client,
        messages,
        profile=None,
        offers=None,
        trace_path=trace_path,
        max_steps=max_steps,
        max_cost_usd=max_cost_usd,
        tools=tools,
        dispatch=dispatch,
    )
