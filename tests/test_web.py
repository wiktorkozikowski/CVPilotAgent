import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from conftest import DB_OFFERS, make_offer_db

from agent_project.llm import MockClient
from agent_project.models import Profile
from agent_project.profiles import save_profile
from agent_project.web import make_server

DATA = Path(__file__).resolve().parent.parent / "data"


def _reply(text=None, name=None, arguments=None):
    """A mock model reply: plain text, or one tool call when `name` is given."""
    calls = []
    if name is not None:
        calls.append(
            {
                "id": f"call_{name}",
                "name": name,
                "arguments": json.dumps(arguments or {}, ensure_ascii=False),
            }
        )
    return {
        "text": text,
        "tool_calls": calls,
        "usage": {"cost": 0.001},
        "latency_ms": 0,
        "provider": "TestProvider",
        "model": "test-model",
    }


@pytest.fixture
def start_server(tmp_path):
    servers = []

    def start(responses, offer_db=None, auth=None):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        shutil.copy(DATA / "job_offers.json", data_dir / "job_offers.json")
        (data_dir / "profiles").mkdir()
        shutil.copy(DATA / "profile.json", data_dir / "profiles" / "jan.json")
        # One client for all requests, so replies are used up in order.
        client = MockClient(responses)
        httpd = make_server(
            port=0,
            data_dir=data_dir,
            client_factory=lambda: client,
            offer_db=offer_db,
            auth=auth,
            secure_cookie=False,
        )
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_port}", tmp_path

    yield start
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def server(start_server):
    return start_server([_reply("Znalazłem dopasowania.")])


def _request(url, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_index_page_is_served(server) -> None:
    base, _ = server
    with urllib.request.urlopen(base + "/", timeout=10) as res:
        html = res.read().decode("utf-8")
    assert "Stwórz profil w rozmowie" in html
    assert 'id="chat-input"' in html
    assert "Dodaj profil" not in html
    assert "Szukaj dopasowań" not in html
    assert 'id="pv-profiles"' in html
    assert "Utwórz profil" in html


def test_lists_saved_profiles(server) -> None:
    base, _ = server
    status, data = _request(base + "/api/profiles")
    assert status == 200
    assert [p["id"] for p in data["profiles"]] == ["jan"]


def test_chat_message_saves_fields_to_draft(start_server) -> None:
    base, tmp_path = start_server(
        [
            _reply(
                name="zapisz_pola_profilu",
                arguments={"name": "Anna Nowak", "location": "Kraków"},
            ),
            _reply(text="Jakie stanowiska cię interesują?"),
        ]
    )
    status, data = _request(
        base + "/api/chat/message",
        "POST",
        {"message": "Nazywam się Anna Nowak, mieszkam w Krakowie."},
    )
    assert status == 200
    assert data["answer"] == "Jakie stanowiska cię interesują?"
    assert data["draft"]["name"] == "Anna Nowak"
    assert data["missing_fields"] == ["target_roles", "skills"]

    status, state = _request(base + "/api/chat")
    assert [m["role"] for m in state["messages"]] == ["user", "assistant"]
    assert state["missing_fields"] == ["target_roles", "skills"]


def test_chat_rejects_empty_message(server) -> None:
    base, _ = server
    status, data = _request(base + "/api/chat/message", "POST", {"message": "   "})
    assert status == 400
    assert data["error"]


def test_failed_chat_turn_leaves_no_half_turn(start_server) -> None:
    base, _ = start_server([])  # the mock has no replies, so the run fails
    status, _ = _request(base + "/api/chat/message", "POST", {"message": "Cześć"})
    assert status == 502
    _, state = _request(base + "/api/chat")
    assert state["messages"] == []


def test_confirm_is_refused_until_fields_are_complete(start_server) -> None:
    base, tmp_path = start_server(
        [
            _reply(name="zapisz_pola_profilu", arguments={"name": "Anna Nowak"}),
            _reply(text="Jaki masz adres?"),
        ]
    )
    _request(base + "/api/chat/message", "POST", {"message": "Jestem Anna Nowak."})

    status, data = _request(base + "/api/chat/confirm", "POST", {})

    assert status == 400
    assert "skills" in data["missing_fields"]
    assert not (tmp_path / "data" / "profiles" / "anna-nowak.json").exists()


def test_model_cannot_save_profile_by_itself(start_server) -> None:
    complete = {
        "name": "Anna Nowak",
        "location": "Kraków",
        "target_roles": ["Data Analyst"],
        "skills": ["SQL"],
    }
    base, tmp_path = start_server(
        [
            _reply(name="zapisz_pola_profilu", arguments=complete),
            # The model asks to save; the server must refuse without a click.
            _reply(name="zatwierdz_profil", arguments={"confirmed": True}),
            _reply(text="Kliknij „Zatwierdź profil”, żeby zapisać."),
        ]
    )

    status, data = _request(
        base + "/api/chat/message", "POST", {"message": "Tak, wszystko się zgadza."}
    )

    assert status == 200
    assert data["answer"] == "Kliknij „Zatwierdź profil”, żeby zapisać."
    assert not (tmp_path / "data" / "profiles" / "anna-nowak.json").exists()
    assert (tmp_path / "data" / "profile_draft.json").exists()


def test_page_can_remove_one_item_from_the_draft(start_server) -> None:
    base, tmp_path = start_server(
        [
            _reply(
                name="zapisz_pola_profilu",
                arguments={
                    "name": "Anna Nowak",
                    "location": "Kraków",
                    "target_roles": ["Data Analyst"],
                    "skills": ["SQL", "Excel"],
                },
            ),
            _reply(text="Dobrze."),
        ]
    )
    _request(base + "/api/chat/message", "POST", {"message": "SQL i Excel."})

    status, data = _request(
        base + "/api/profile/remove", "POST", {"section": "skills", "index": 0}
    )

    assert status == 200
    assert data["draft"]["skills"] == ["Excel"]
    status, _ = _request(
        base + "/api/profile/remove", "POST", {"section": "skills", "index": 9}
    )
    assert status == 404
    status, _ = _request(
        base + "/api/profile/remove", "POST", {"section": "name", "index": 0}
    )
    assert status == 400


ANNA = {
    "name": "Anna Nowak",
    "location": "Kraków",
    "target_roles": ["Data Analyst"],
    "skills": ["SQL"],
}


def test_confirm_button_saves_profile_and_keeps_chat(start_server) -> None:
    base, tmp_path = start_server(
        [
            _reply(name="zapisz_pola_profilu", arguments=ANNA),
            _reply(text="Oto podsumowanie profilu."),
        ]
    )
    _request(base + "/api/chat/message", "POST", {"message": "Anna, Kraków, SQL."})

    status, data = _request(base + "/api/chat/confirm", "POST", {})

    assert status == 200
    assert data["profile_id"] == "anna-nowak"
    saved = json.loads(
        (tmp_path / "data" / "profiles" / "anna-nowak.json").read_text("utf-8")
    )
    assert saved["skills"] == ["SQL"]
    _, state = _request(base + "/api/chat")
    # The chat goes on with an offer to search; the panel still shows the profile.
    assert [m["role"] for m in state["messages"]] == ["user", "assistant", "assistant"]
    assert "Szukaj ofert" in state["messages"][-1]["text"]
    assert state["profile_id"] == "anna-nowak"
    assert state["draft"]["name"] == "Anna Nowak"


def test_second_confirm_updates_the_same_profile(start_server) -> None:
    base, tmp_path = start_server(
        [
            _reply(name="zapisz_pola_profilu", arguments=ANNA),
            _reply(text="Gotowe."),
            _reply(
                name="dodaj_pozycje_profilu",
                arguments={"section": "skills", "items": ["Excel"]},
            ),
            _reply(text="Dodałem Excel."),
        ]
    )
    _request(base + "/api/chat/message", "POST", {"message": "Anna, Kraków, SQL."})
    _request(base + "/api/chat/confirm", "POST", {})
    _request(base + "/api/chat/message", "POST", {"message": "Znam też Excel."})

    status, data = _request(base + "/api/chat/confirm", "POST", {})

    assert status == 200
    assert data["profile_id"] == "anna-nowak"
    profiles = tmp_path / "data" / "profiles"
    assert not (profiles / "anna-nowak-2.json").exists()
    saved = json.loads((profiles / "anna-nowak.json").read_text("utf-8"))
    assert saved["skills"] == ["SQL", "Excel"]


def test_chat_model_searches_offers_for_saved_profile(start_server) -> None:
    fake = make_offer_db(DB_OFFERS[:1])
    base, _ = start_server(
        [
            _reply(name="zapisz_pola_profilu", arguments=ANNA),
            _reply(text="Gotowe."),
            _reply(name="szukaj_ofert", arguments={"limit": 3}),
            _reply(text="Oferty są w tabeli po prawej."),
        ],
        offer_db=fake,
    )
    _request(base + "/api/chat/message", "POST", {"message": "Anna, Kraków, SQL."})
    _request(base + "/api/chat/confirm", "POST", {})

    status, data = _request(base + "/api/chat/message", "POST", {"message": "3 oferty"})

    assert status == 200
    assert data["answer"] == "Oferty są w tabeli po prawej."
    assert data["search"]["profile_id"] == "anna-nowak"
    assert [m["id"] for m in data["search"]["matches"]] == ["justjoin:remote-analyst"]
    assert fake.limits == [3]
    # A reload shows the same offers.
    _, state = _request(base + "/api/chat")
    assert state["search"] == data["search"]


def test_loading_a_saved_profile_starts_a_chat_around_it(start_server) -> None:
    fake = make_offer_db(DB_OFFERS[:1])
    base, tmp_path = start_server(
        [
            _reply(name="szukaj_ofert", arguments={"limit": 2}),
            _reply(text="Oferty są po prawej."),
            _reply(
                name="dodaj_pozycje_profilu",
                arguments={"section": "skills", "items": ["Tableau"]},
            ),
            _reply(text="Dodałem Tableau."),
        ],
        offer_db=fake,
    )

    status, data = _request(base + "/api/chat/load", "POST", {"profile_id": "jan"})

    assert status == 200
    assert data["profile_id"] == "jan"
    assert data["draft"]["name"] == "Jan Przykładowy"
    assert data["missing_fields"] == []
    assert [m["role"] for m in data["messages"]] == ["assistant"]
    assert "Jan Przykładowy" in data["messages"][0]["text"]

    # The chat can search for the loaded profile right away.
    status, data = _request(base + "/api/chat/message", "POST", {"message": "2 oferty"})
    assert data["search"]["profile_id"] == "jan"
    assert fake.limits == [2]

    # Saving after an edit overwrites the loaded profile, not a new file.
    _request(base + "/api/chat/message", "POST", {"message": "Znam Tableau."})
    status, data = _request(base + "/api/chat/confirm", "POST", {})
    assert data["profile_id"] == "jan"
    saved = json.loads((tmp_path / "data" / "profiles" / "jan.json").read_text("utf-8"))
    assert "Tableau" in saved["skills"]
    assert sorted(p.name for p in (tmp_path / "data" / "profiles").iterdir()) == [
        "jan.json"
    ]


def test_page_edits_the_draft_by_hand(server) -> None:
    base, _ = server
    _request(base + "/api/chat/load", "POST", {"profile_id": "jan"})

    def edit(body):
        return _request(base + "/api/profile/edit", "POST", body)

    status, data = edit(
        {
            "action": "fields",
            "fields": {
                "location": "Poznań",
                "email": None,
                "seniority": ["junior", "mid"],
                "salary_expectation": {"min": 6000},
            },
        }
    )
    assert status == 200
    assert data["draft"]["location"] == "Poznań"
    assert "email" not in data["draft"]
    assert data["draft"]["seniority"] == ["junior", "mid"]
    assert data["draft"]["salary_expectation"]["currency"] == "PLN"

    status, data = edit({"action": "fields", "fields": {"email": "zly-adres"}})
    assert status == 400
    assert "email" in data["error"]

    skills = data and _request(base + "/api/chat")[1]["draft"]["skills"]
    status, data = edit(
        {"action": "replace_item", "section": "skills", "index": 0, "value": "Pyton 3"}
    )
    assert status == 200
    assert data["draft"]["skills"][0] == "Pyton 3"
    status, _ = edit(
        {"action": "replace_item", "section": "skills", "index": 1, "value": "pyton 3"}
    )
    assert status == 400  # already on the list
    status, _ = edit(
        {"action": "replace_item", "section": "skills", "index": 99, "value": "X"}
    )
    assert status == 404

    status, data = edit(
        {"action": "add_items", "section": "skills", "items": ["Figma"]}
    )
    assert data["draft"]["skills"][-1] == "Figma"
    assert len(data["draft"]["skills"]) == len(skills) + 1

    job = {"company": "Studio", "position": "Projektant", "start": "2024-01"}
    status, data = edit({"action": "entry", "section": "experience", "entry": job})
    assert status == 200
    index = len(data["draft"]["experience"]) - 1
    status, data = edit(
        {
            "action": "entry",
            "section": "experience",
            "index": index,
            "entry": {**job, "end": "obecnie", "description": ["Meble"]},
        }
    )
    assert status == 200
    assert data["draft"]["experience"][index]["description"] == ["Meble"]
    status, data = edit(
        {"action": "entry", "section": "experience", "entry": {"company": "X"}}
    )
    assert status == 400

    status, _ = edit({"action": "nope"})
    assert status == 400


def test_loading_unknown_profile_returns_404(server) -> None:
    base, _ = server
    status, _ = _request(base + "/api/chat/load", "POST", {"profile_id": "../x"})
    assert status == 404
    status, _ = _request(base + "/api/chat/load", "POST", {})
    assert status == 400


def test_chat_search_before_saving_asks_to_confirm(start_server) -> None:
    fake = make_offer_db(DB_OFFERS[:1])
    base, tmp_path = start_server(
        [
            _reply(name="szukaj_ofert", arguments={"limit": 5}),
            _reply(text="Najpierw zatwierdź profil."),
        ],
        offer_db=fake,
    )

    status, data = _request(base + "/api/chat/message", "POST", {"message": "Szukaj"})

    assert status == 200
    assert data["search"] is None
    assert fake.limits == []
    trace = sorted((tmp_path / "traces").glob("chat_*.jsonl"))[-1]
    assert "no_profile" in trace.read_text("utf-8")


def test_search_takes_a_limit(start_server) -> None:
    fake = make_offer_db(DB_OFFERS[:1])
    # The model answers without a choice; after the app's one reminder it
    # still does not pick, and the table falls back to the ranking.
    base, _ = start_server([_reply("Jedna oferta.")] * 2, offer_db=fake)

    status, _ = _request(
        base + "/api/search", "POST", {"profile_id": "jan", "limit": 30}
    )
    assert status == 200
    assert fake.limits == [30]

    status, data = _request(
        base + "/api/search", "POST", {"profile_id": "jan", "limit": 31}
    )
    assert status == 400
    assert data["error"]


def test_reset_discards_conversation_and_draft(start_server) -> None:
    base, tmp_path = start_server(
        [
            _reply(name="zapisz_pola_profilu", arguments={"name": "Anna Nowak"}),
            _reply(text="Jaki masz adres?"),
        ]
    )
    _request(base + "/api/chat/message", "POST", {"message": "Jestem Anna."})

    status, _ = _request(base + "/api/chat/reset", "POST", {})

    assert status == 200
    _, state = _request(base + "/api/chat")
    assert state["messages"] == []
    assert state["draft"] == {}
    assert not (tmp_path / "data" / "profile_draft.json").exists()


def test_search_runs_agent_for_selected_profile(server) -> None:
    base, tmp_path = server
    status, data = _request(base + "/api/search", "POST", {"profile_id": "jan"})
    assert status == 200
    assert data["answer"] == "Znalazłem dopasowania."
    assert data["profile"]["name"] == "Jan Przykładowy"
    assert data["matches"]
    trace_file = tmp_path / data["trace"]
    records = [json.loads(line) for line in trace_file.read_text("utf-8").splitlines()]
    assert records[0]["event"] == "model_decision"
    assert records[0]["payload"]["provider"] == "TestProvider"


def test_search_unknown_profile_returns_404(server) -> None:
    base, _ = server
    status, data = _request(base + "/api/search", "POST", {"profile_id": "../secret"})
    assert status == 404
    assert data["error"]


def test_search_without_profile_id_returns_400(server) -> None:
    base, _ = server
    status, _ = _request(base + "/api/search", "POST", {})
    assert status == 400


def test_shortlist_add_and_confirmed_remove(server) -> None:
    base, _ = server
    status, data = _request(
        base + "/api/shortlist/add",
        "POST",
        {"profile_id": "jan", "offer_id": "offer-001"},
    )
    assert status == 200
    assert data["status"] == "added"
    status, data = _request(base + "/api/shortlist?profile_id=jan")
    assert [o["id"] for o in data["offers"]] == ["offer-001"]

    status, data = _request(
        base + "/api/shortlist/remove",
        "POST",
        {"profile_id": "jan", "offer_id": "offer-001"},
    )
    assert data["status"] == "confirmation_required"
    status, data = _request(
        base + "/api/shortlist/remove",
        "POST",
        {"profile_id": "jan", "offer_id": "offer-001", "confirmed": True},
    )
    assert data["status"] == "removed"


def test_each_profile_has_its_own_shortlist(server) -> None:
    base, tmp_path = server
    other = save_profile(
        tmp_path / "data" / "profiles",
        Profile(name="Ola Druga", location="Łódź", skills=["SQL"]),
    )
    _request(
        base + "/api/shortlist/add",
        "POST",
        {"profile_id": "jan", "offer_id": "offer-002"},
    )
    _, jan_list = _request(base + "/api/shortlist?profile_id=jan")
    _, other_list = _request(base + f"/api/shortlist?profile_id={other}")
    assert [o["id"] for o in jan_list["offers"]] == ["offer-002"]
    assert other_list["offers"] == []


def test_shortlist_rejects_unknown_profile(server) -> None:
    base, _ = server
    status, _ = _request(base + "/api/shortlist?profile_id=../secret")
    assert status == 404
    status, _ = _request(
        base + "/api/shortlist/add",
        "POST",
        {"profile_id": "nie-ma", "offer_id": "offer-001"},
    )
    assert status == 404


def test_shortlist_add_unknown_offer(server) -> None:
    base, _ = server
    status, data = _request(
        base + "/api/shortlist/add",
        "POST",
        {"profile_id": "jan", "offer_id": "nope"},
    )
    assert data["status"] == "not_found"


def test_search_returns_run_steps(server) -> None:
    base, _ = server
    status, data = _request(base + "/api/search", "POST", {"profile_id": "jan"})
    assert status == 200
    kinds = [step["kind"] for step in data["steps"]]
    assert kinds == ["decyzja"]
    assert data["steps"][0]["provider"] == "TestProvider"


def test_search_in_offer_db_and_shortlist_its_offer(start_server) -> None:
    fake = make_offer_db(DB_OFFERS[:1])
    base, _ = start_server(
        [
            _reply(name="szukaj_ofert", arguments={"limit": 5}),
            _reply("Znalazłem jedną ofertę."),
            _reply("Nadal bez wyboru."),
        ],
        offer_db=fake,
    )

    status, data = _request(base + "/api/search", "POST", {"profile_id": "jan"})

    assert status == 200
    assert [m["id"] for m in data["matches"]] == ["justjoin:remote-analyst"]
    assert data["matches"][0]["requirements"] == "SQL, Excel"
    assert fake.limits == [10]  # the table and the agent's tool share one search
    tool_step = [s for s in data["steps"] if s["kind"] == "wynik"][0]
    assert "justjoin:remote-analyst" in tool_step["result"]

    status, data = _request(
        base + "/api/shortlist/add",
        "POST",
        {"profile_id": "jan", "offer_id": "justjoin:remote-analyst"},
    )
    assert data["status"] == "added"
    status, data = _request(base + "/api/shortlist?profile_id=jan")
    assert data["offers"] == [
        {
            "id": "justjoin:remote-analyst",
            "title": "Data Analyst",
            "company": "Zdalna Firma",
        }
    ]


def test_search_shows_the_offers_the_agent_picked(start_server) -> None:
    db = make_offer_db()
    base, _ = start_server(
        [
            _reply(name="szukaj_ofert", arguments={"limit": 5}),
            _reply(name="przeszukaj_oferty", arguments={"zapytanie": "Power BI"}),
            _reply(
                name="wybierz_oferty",
                arguments={
                    "oferty": [
                        {"offer_id": "justjoin:bi-krakow", "powod": "Power BI."},
                    ]
                },
            ),
            _reply("Polecam jedną ofertę."),
        ],
        offer_db=db,
    )

    status, data = _request(
        base + "/api/search", "POST", {"profile_id": "jan", "limit": 5}
    )

    assert status == 200
    assert data["picked"] is True
    assert [(m["id"], m["reason"]) for m in data["matches"]] == [
        ("justjoin:bi-krakow", "Power BI.")
    ]
    status, data = _request(
        base + "/api/shortlist/add",
        "POST",
        {"profile_id": "jan", "offer_id": "justjoin:bi-krakow"},
    )
    assert data["status"] == "added"
