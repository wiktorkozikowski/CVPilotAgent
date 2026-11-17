import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_project.agent import run_tool
from agent_project.models import Profile
from agent_project.offers_client import (
    OffersServerClient,
    OffersServerError,
    load_seen_offers,
    remember_offers,
    search_payload,
)

CANDIDATE = {
    "id": "justjoin:java-dev",
    "source": "justjoin",
    "url": "https://justjoin.it/job-offer/java-dev",
    "title": "Java Developer",
    "company": "Firma",
    "location": "",
    "remote": True,
    "employment_type": ["CONTRACTOR", "OTHER"],
    "salary_min": 100,
    "salary_max": 150,
    "salary_currency": "PLN",
    "salary_unit": "HOUR",
    "score": 0.81,
    "similarity": 0.77,
    "matched_skills": ["Java"],
    "conflicts": [],
    "requirements": "Java, Spring. " * 60,
}


def make_profile() -> Profile:
    return Profile.model_validate(
        {
            "name": "Anna Nowak",
            "location": "Kraków",
            "email": "anna@example.com",
            "phone": "+48 600 123 456",
            "github": "https://github.com/anna",
            "target_roles": ["Java Developer"],
            "skills": ["Java"],
            "seniority": ["mid"],
            "experience": [
                {"company": "Acme", "position": "Programista", "start": "2022"}
            ],
        }
    )


@pytest.fixture
def stub_server():
    """A fake offer server that records requests and answers from `reply`."""
    state = {
        "requests": [],
        "reply": (200, {"status": "ok", "candidates": [CANDIDATE]}),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["requests"].append(
                {
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "body": json.loads(body),
                }
            )
            status, payload = state["reply"]
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()


def test_payload_leaves_contact_data_at_home() -> None:
    payload = search_payload(make_profile())

    assert set(payload) == {
        "location",
        "target_roles",
        "skills",
        "seniority",
        "work_modes",
        "employment_types",
        "experience",
    }
    assert payload["experience"] == [{"position": "Programista", "description": []}]
    assert "Anna" not in json.dumps(payload)


def test_search_sends_token_and_converts_candidates(stub_server) -> None:
    client = OffersServerClient(stub_server["url"], token="sekret")

    matches = client.search(make_profile(), limit=3, query="Spring")

    sent = stub_server["requests"][0]
    assert sent["path"] == "/search"
    assert sent["auth"] == "Bearer sekret"
    assert sent["body"]["limit"] == 3
    assert sent["body"]["query"] == "Spring"
    offer = matches[0].offer
    assert offer.id == "justjoin:java-dev"
    assert offer.location == "Zdalnie"
    assert offer.employment_type == ["CONTRACTOR"]  # unknown kinds are dropped
    assert offer.requirements.startswith("Java, Spring.")
    assert matches[0].score == 0.81


def test_server_error_message_is_passed_on(stub_server) -> None:
    stub_server["reply"] = (400, {"status": "error", "message": "zły profil"})

    with pytest.raises(OffersServerError, match="zły profil"):
        OffersServerClient(stub_server["url"]).search(make_profile())


def test_unreachable_server_is_a_clear_error() -> None:
    client = OffersServerClient("http://127.0.0.1:9", timeout=2)

    with pytest.raises(OffersServerError, match="niedostępny"):
        client.search(make_profile())


def test_agent_tool_uses_server_and_can_shortlist_its_offer(stub_server, tmp_path):
    client = OffersServerClient(stub_server["url"])
    offers = []
    shortlist = tmp_path / "shortlist.json"

    def search(profile, limit):
        return client.search(profile, limit)

    found = json.loads(
        run_tool(
            {"name": "szukaj_ofert", "arguments": '{"limit": 5}'},
            make_profile(),
            offers,
            shortlist,
            search=search,
        )
    )
    added = json.loads(
        run_tool(
            {
                "name": "dodaj_do_listy",
                "arguments": '{"offer_id": "justjoin:java-dev"}',
            },
            make_profile(),
            offers,
            shortlist,
            search=search,
        )
    )

    assert found["status"] == "ok"
    # The model gets a shortened excerpt; the full one stays on the Offer.
    assert len(found["offers"][0]["offer"]["requirements"]) < 420
    assert len(offers[0].requirements) > 420
    assert added["status"] == "added"


def test_agent_tool_reports_server_failure_as_error(tmp_path) -> None:
    def search(profile, limit):
        raise OffersServerError("Serwer ofert jest niedostępny.")

    result = json.loads(
        run_tool(
            {"name": "szukaj_ofert", "arguments": "{}"},
            make_profile(),
            [],
            tmp_path / "s.json",
            search=search,
        )
    )

    assert result == {"status": "error", "message": "Serwer ofert jest niedostępny."}


def test_seen_offers_are_merged_by_id(stub_server, tmp_path) -> None:
    path = tmp_path / "seen.json"
    first = OffersServerClient(stub_server["url"]).search(make_profile())[0].offer

    remember_offers(path, [first])
    remember_offers(path, [first.model_copy(update={"title": "Nowy tytuł"})])

    seen = load_seen_offers(path)
    assert [offer.title for offer in seen] == ["Nowy tytuł"]


def test_client_is_off_without_url(monkeypatch) -> None:
    monkeypatch.setenv("OFFERS_SERVER_URL", "")
    monkeypatch.setattr("agent_project.offers_client.load_dotenv", lambda: None)

    assert OffersServerClient.from_env() is None
