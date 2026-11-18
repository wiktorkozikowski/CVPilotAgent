import json

import pytest

from agent_project.agent import (
    DB_TOOLS,
    PICK_REMINDER,
    TOOLS,
    run_agent,
    run_tool,
    tools_for,
)
from agent_project.llm import MockClient
from agent_project.models import Profile

PROFILE = Profile.model_validate(
    {
        "name": "Anna Nowak",
        "email": "anna@example.com",
        "location": "Kraków",
        "target_roles": ["Data Analyst"],
        "skills": ["SQL", "Power BI", "Python"],
    }
)


def _call(name, arguments):
    return {"id": f"call_{name}", "name": name, "arguments": json.dumps(arguments)}


def _reply(name=None, arguments=None, text=None):
    return {
        "text": text,
        "tool_calls": [] if name is None else [_call(name, arguments or {})],
        "usage": {"cost": 0.001},
        "latency_ms": 0,
    }


def test_browse_filters_by_city_and_level(offer_db) -> None:
    found = offer_db.browse("SQL raportowanie", city="Krakow")
    assert {item["id"] for item in found} == {
        "justjoin:bi-krakow",
        "justjoin:warehouse",
    }

    # A title without a level passes; "Senior Java Developer" does not.
    junior = offer_db.browse("Java Spring", level="junior")
    assert "justjoin:java" not in {item["id"] for item in junior}

    remote = offer_db.browse("analityk", remote=True)
    assert [item["id"] for item in remote] == ["justjoin:remote-analyst"]


def test_browse_rejects_bad_arguments(offer_db) -> None:
    with pytest.raises(ValueError):
        offer_db.browse("")
    with pytest.raises(ValueError):
        offer_db.browse("x" * 201)
    with pytest.raises(ValueError):
        offer_db.browse("SQL", level="guru")


def test_compare_lists_skills_in_and_missing(offer_db) -> None:
    facts = offer_db.compare(PROFILE, "justjoin:bi-krakow")

    assert facts["profile_skills_in_offer"] == ["SQL", "Power BI"]
    assert facts["profile_skills_not_in_offer"] == ["Python"]
    assert offer_db.compare(PROFILE, "justjoin:nie-ma") is None


def test_details_do_not_carry_profile_data(offer_db) -> None:
    details = offer_db.details("justjoin:java")
    assert details["requirements"] == "Java, Spring, Kafka"
    assert details["url"].startswith("https://justjoin.it/")


def test_model_gets_database_tools_only_with_a_database(offer_db) -> None:
    assert tools_for(None) == TOOLS
    names = {tool["function"]["name"] for tool in tools_for(offer_db)}
    assert {"przeszukaj_oferty", "pokaz_oferte", "wybierz_oferty"} <= names
    assert len(tools_for(offer_db)) == len(TOOLS) + len(DB_TOOLS)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"zapytanie": 5},
        {"zapytanie": "SQL", "limit": "10"},
        {"zapytanie": "SQL", "limit": 0},
        {"zapytanie": "SQL", "zdalnie": "tak"},
        {"zapytanie": "SQL", "poziom": "guru"},
        {"zapytanie": "SQL", "forma_zatrudnienia": "B2B"},
    ],
)
def test_browse_tool_validates_arguments(offer_db, tmp_path, arguments) -> None:
    result = json.loads(
        run_tool(
            _call("przeszukaj_oferty", arguments),
            PROFILE,
            [],
            tmp_path / "s.json",
            db=offer_db,
        )
    )
    assert result["status"] == "error"


def test_seen_offers_are_skipped_and_can_be_shortlisted(offer_db, tmp_path) -> None:
    offers = []
    shortlist = tmp_path / "s.json"
    first = json.loads(
        run_tool(
            _call("przeszukaj_oferty", {"zapytanie": "analityk SQL", "limit": 2}),
            PROFILE,
            offers,
            shortlist,
            db=offer_db,
        )
    )
    seen = [item["id"] for item in first["offers"]]
    again = json.loads(
        run_tool(
            _call("przeszukaj_oferty", {"zapytanie": "analityk SQL"}),
            PROFILE,
            offers,
            shortlist,
            db=offer_db,
        )
    )
    assert not set(seen) & {item["id"] for item in again["offers"]}

    added = json.loads(
        run_tool(
            _call("dodaj_do_listy", {"offer_id": seen[0]}),
            PROFILE,
            offers,
            shortlist,
            db=offer_db,
        )
    )
    assert added["status"] == "added"


def test_agent_browses_compares_and_picks(offer_db, tmp_path) -> None:
    client = MockClient(
        [
            _reply("szukaj_ofert", {"limit": 3}),
            _reply("przeszukaj_oferty", {"zapytanie": "Power BI", "miasto": "Kraków"}),
            _reply("porownaj_z_profilem", {"offer_id": "justjoin:bi-krakow"}),
            _reply(
                "wybierz_oferty",
                {
                    "oferty": [
                        {"offer_id": "justjoin:bi-krakow", "powod": "SQL i Power BI."},
                        {"offer_id": "justjoin:zmyslona", "powod": "?"},
                        {"offer_id": "justjoin:bi-krakow", "powod": "drugi raz"},
                    ]
                },
            ),
            _reply(text="Polecam raportowanie w Krakowie."),
        ]
    )
    picks = []
    trace = tmp_path / "trace.jsonl"

    answer = run_agent(
        client,
        [{"role": "user", "content": "Znajdź oferty"}],
        PROFILE,
        [],
        trace,
        shortlist_path=tmp_path / "s.json",
        db=offer_db,
        picks=picks,
    )

    assert answer == "Polecam raportowanie w Krakowie."
    assert picks == [{"offer_id": "justjoin:bi-krakow", "reason": "SQL i Power BI."}]
    results = [
        json.loads(line)["payload"]
        for line in trace.read_text("utf-8").splitlines()
        if json.loads(line)["event"] == "tool_result"
    ]
    picked = json.loads(results[-1]["result"])
    assert picked["unknown"] == ["justjoin:zmyslona"]
    # The contact data of the profile never reaches a tool result.
    assert "anna@example.com" not in trace.read_text("utf-8")


def test_model_that_skips_the_choice_is_reminded_once(offer_db, tmp_path) -> None:
    client = MockClient(
        [
            _reply("szukaj_ofert", {"limit": 2}),
            _reply(text="Oto oferty."),
            _reply(
                "wybierz_oferty",
                {"oferty": [{"offer_id": "justjoin:bi-krakow", "powod": "SQL."}]},
            ),
            _reply(text="Wybrałem jedną."),
        ]
    )
    messages = [{"role": "user", "content": "Wybierz oferty"}]
    picks = []

    answer = run_agent(
        client,
        messages,
        PROFILE,
        [],
        tmp_path / "trace.jsonl",
        shortlist_path=tmp_path / "s.json",
        db=offer_db,
        picks=picks,
    )

    assert answer == "Wybrałem jedną."
    assert [pick["offer_id"] for pick in picks] == ["justjoin:bi-krakow"]
    assert sum(m["content"] == PICK_REMINDER for m in messages) == 1
