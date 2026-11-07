import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agent_project.llm import MockClient
from agent_project.web import make_server

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def server(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(DATA / "job_offers.json", data_dir / "job_offers.json")
    shutil.copy(DATA / "options.json", data_dir / "options.json")
    (data_dir / "profiles").mkdir()
    shutil.copy(DATA / "profile.json", data_dir / "profiles" / "jan.json")

    def client_factory():
        return MockClient(
            [
                {
                    "text": "Znalazłem dopasowania.",
                    "tool_calls": [],
                    "usage": {"cost": 0.001},
                    "latency_ms": 0,
                    "provider": "TestProvider",
                    "model": "test-model",
                }
            ]
        )

    httpd = make_server(port=0, data_dir=data_dir, client_factory=client_factory)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", tmp_path
    httpd.shutdown()
    httpd.server_close()


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
    assert "Dodaj profil" in html
    assert "Szukaj dopasowań" in html


def test_returns_ready_roles_and_skills(server) -> None:
    base, _ = server
    status, data = _request(base + "/api/options")
    assert status == 200
    assert "Data Analyst" in data["roles"]
    assert not any(role.startswith("Junior") for role in data["roles"])
    assert "SQL" in data["skills"]
    assert data["seniority"] == ["junior", "mid", "senior"]


def test_lists_saved_profiles(server) -> None:
    base, _ = server
    status, data = _request(base + "/api/profiles")
    assert status == 200
    assert [p["id"] for p in data["profiles"]] == ["jan"]


def test_creates_profile_and_returns_id(server) -> None:
    base, tmp_path = server
    body = {
        "name": "Anna Nowak",
        "location": "Kraków",
        "target_roles": ["Data Analyst"],
        "skills": ["SQL"],
        "languages": ["polski"],
    }
    status, data = _request(base + "/api/profiles", "POST", body)
    assert status == 201
    assert data["id"] == "anna-nowak"
    saved = json.loads(
        (tmp_path / "data" / "profiles" / "anna-nowak.json").read_text("utf-8")
    )
    assert saved["skills"] == ["SQL"]


def test_saves_seniority_levels(server) -> None:
    base, tmp_path = server
    body = {"name": "Piotr Mid", "location": "Gdańsk", "seniority": ["junior", "mid"]}
    status, data = _request(base + "/api/profiles", "POST", body)
    assert status == 201
    assert data["seniority"] == ["junior", "mid"]
    saved = json.loads(
        (tmp_path / "data" / "profiles" / "piotr-mid.json").read_text("utf-8")
    )
    assert saved["seniority"] == ["junior", "mid"]


def test_rejects_unknown_seniority(server) -> None:
    base, _ = server
    body = {"name": "Ola", "location": "Łódź", "seniority": ["expert"]}
    status, _ = _request(base + "/api/profiles", "POST", body)
    assert status == 400


def test_rejects_profile_without_name(server) -> None:
    base, _ = server
    status, data = _request(base + "/api/profiles", "POST", {"location": "Gdańsk"})
    assert status == 400
    assert data["error"]


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
    base, _ = server
    body = {"name": "Ola Druga", "location": "Łódź", "skills": ["SQL"]}
    _, created = _request(base + "/api/profiles", "POST", body)
    other = created["id"]
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
