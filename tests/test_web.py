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
