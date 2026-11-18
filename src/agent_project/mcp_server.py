"""Local MCP server exposing the CV agent's tools.

Each tool here calls the same function that agent.py uses, so the
logic lives in one place. The removal tool is not exposed: it needs a user
confirmation that an MCP client cannot give yet.
"""

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from agent_project.matching import search_offers
from agent_project.models import load_offers, load_profile
from agent_project.offer_db import OfferDB
from agent_project.profile_tools import DRAFT_PATH, get_draft, update_draft
from agent_project.shortlist import (
    add_to_shortlist,
    load_seen_offers,
    remember_offers,
)
from offer_server.index import index_path
from offer_server.paths import index_dir

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROFILE_PATH = DATA_DIR / "profile.json"
OFFERS_PATH = DATA_DIR / "job_offers.json"
SHORTLIST_PATH = DATA_DIR / "shortlist.json"
SEEN_OFFERS_PATH = DATA_DIR / "offers_seen.json"

_db: OfferDB | None = None


def offer_db() -> OfferDB | None:
    """The offer database when its index exists, else None (job_offers.json)."""
    global _db
    if _db is None and index_path(index_dir()).exists():
        _db = OfferDB.open()
    return _db


server = MCPServer(
    "cv-pilot",
    instructions=(
        "Wyszukiwanie ofert pracy, lista wybranych i budowanie profilu w rozmowie."
    ),
)


@server.tool()
def szukaj_ofert(limit: int = 10) -> list[dict]:
    """Znajduje oferty pasujące do profilu użytkownika, od najlepiej dopasowanych.

    limit: od 1 do 30.
    """
    profile = load_profile(PROFILE_PATH)
    db = offer_db()
    if db is None:
        matches = search_offers(profile, load_offers(OFFERS_PATH), limit=limit)
    else:
        matches = db.match_profile(profile, limit)
        remember_offers(SEEN_OFFERS_PATH, [match.offer for match in matches])
    return [match.model_dump(mode="json") for match in matches]


@server.tool()
def dodaj_do_listy(offer_id: str) -> dict:
    """Dodaje ofertę do listy wybranych.

    Zwraca status: added, already_exists albo not_found.
    """
    offers = load_offers(OFFERS_PATH) + load_seen_offers(SEEN_OFFERS_PATH)
    db = offer_db()
    found = db.offer(offer_id) if db is not None else None
    if found is not None:
        remember_offers(SEEN_OFFERS_PATH, [found])
        offers.append(found)
    result = add_to_shortlist(offer_id, offers, SHORTLIST_PATH)
    return result.model_dump(mode="json")


@server.tool()
def pobierz_profil() -> dict:
    """Zwraca profil zbierany w rozmowie i listę brakujących pól (missing_fields)."""
    return get_draft(DRAFT_PATH)


@server.tool()
def zapisz_pola_profilu(
    name: str | None = None,
    location: str | None = None,
    target_roles: list[str] | None = None,
    skills: list[str] | None = None,
    languages: list[str] | None = None,
    seniority: list[str] | None = None,
) -> dict:
    """Zapisuje pola podane przez użytkownika. Zwraca status: ok albo validation_error.

    Zapisuje tylko roboczy profil. Zatwierdzenie i zapis profilu wymagają zgody
    użytkownika w aplikacji, więc nie są dostępne przez MCP.
    """
    fields = {
        key: value
        for key, value in {
            "name": name,
            "location": location,
            "target_roles": target_roles,
            "skills": skills,
            "languages": languages,
            "seniority": seniority,
        }.items()
        if value is not None
    }
    return update_draft(fields, DRAFT_PATH)


if __name__ == "__main__":
    server.run()
