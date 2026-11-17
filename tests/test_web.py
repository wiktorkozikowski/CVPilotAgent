import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

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

    def start(responses, offers_client=None):
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
            offers_client=offers_client,
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
    assert "Szukaj dopasowań" in html


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


def test_confirm_button_saves_profile_and_clears_chat(start_server) -> None:
    base, tmp_path = start_server(
        [
            _reply(
                name="zapisz_pola_profilu",
                arguments={
                    "name": "Anna Nowak",
                    "location": "Kraków",
                    "target_roles": ["Data Analyst"],
                    "skills": ["SQL"],
                },
            ),
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
    assert state["messages"] == []
    assert state["draft"] == {}


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


class FakeOffersClient:
    """Stands in for the offer server: returns one offer that is not local."""

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def search(self, profile, limit=10, query=None):
        from agent_project.models import Offer, OfferMatch
        from agent_project.offers_client import OffersServerError

        self.calls += 1
        if self.fail:
            raise OffersServerError("Serwer ofert jest niedostępny.")
        offer = Offer(
            id="justjoin:remote-analyst",
            title="Data Analyst",
            company="Zdalna Firma",
            location="Zdalnie",
            source="justjoin",
            url="https://justjoin.it/job-offer/remote-analyst",
            requirements="SQL, Excel",
        )
        return [OfferMatch(offer=offer, matched_skills=["SQL"], score=0.9)]


def test_search_through_offer_server_and_shortlist_its_offer(start_server) -> None:
    fake = FakeOffersClient()
    base, _ = start_server(
        [
            _reply(name="szukaj_ofert", arguments={"limit": 5}),
            _reply("Znalazłem jedną ofertę."),
        ],
        offers_client=fake,
    )

    status, data = _request(base + "/api/search", "POST", {"profile_id": "jan"})

    assert status == 200
    assert [m["id"] for m in data["matches"]] == ["justjoin:remote-analyst"]
    assert data["matches"][0]["requirements"] == "SQL, Excel"
    assert fake.calls == 1  # the table and the agent's tool share one server call
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


def test_search_reports_unreachable_offer_server(start_server) -> None:
    base, _ = start_server([], offers_client=FakeOffersClient(fail=True))

    status, data = _request(base + "/api/search", "POST", {"profile_id": "jan"})

    assert status == 502
    assert "niedostępny" in data["error"]
