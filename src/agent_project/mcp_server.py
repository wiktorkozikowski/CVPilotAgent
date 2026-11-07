"""Local MCP server exposing the CV agent's tools.

Each tool here calls the same function that agent.py uses, so the
logic lives in one place. The removal tool is not exposed: it needs a user
confirmation that an MCP client cannot give yet.
"""

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from agent_project.matching import search_offers
from agent_project.models import load_offers, load_profile
from agent_project.shortlist import add_to_shortlist

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROFILE_PATH = DATA_DIR / "profile.json"
OFFERS_PATH = DATA_DIR / "job_offers.json"
SHORTLIST_PATH = DATA_DIR / "shortlist.json"

server = MCPServer(
    "cv-pilot",
    instructions="Wyszukiwanie ofert pracy dopasowanych do profilu i lista wybranych.",
)


@server.tool()
def szukaj_ofert(limit: int = 10) -> list[dict]:
    """Znajduje oferty pasujące do profilu użytkownika, od najlepiej dopasowanych.

    limit: od 1 do 10.
    """
    matches = search_offers(
        load_profile(PROFILE_PATH), load_offers(OFFERS_PATH), limit=limit
    )
    return [match.model_dump(mode="json") for match in matches]


@server.tool()
def dodaj_do_listy(offer_id: str) -> dict:
    """Dodaje ofertę do listy wybranych.

    Zwraca status: added, already_exists albo not_found.
    """
    result = add_to_shortlist(offer_id, load_offers(OFFERS_PATH), SHORTLIST_PATH)
    return result.model_dump(mode="json")


if __name__ == "__main__":
    server.run()
