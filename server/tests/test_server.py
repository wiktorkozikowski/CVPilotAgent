import json
import threading
import urllib.error
import urllib.request
from datetime import date

import pytest

from offer_server.app import make_server
from offer_server.documents import (
    RawOffer,
    chunks,
    clean_offers,
    excerpt,
    split_sections,
)
from offer_server.embedder import HashingEmbedder
from offer_server.index import build_index, load_index, save_index
from offer_server.ranking import SearchProfile, search
from offer_server.text import contains_term

TODAY = date(2026, 11, 12)


def raw(offer_id: str, **fields) -> RawOffer:
    data = {
        "id": offer_id,
        "source": "justjoin",
        "url": f"https://example.com/{offer_id}",
        "title": "Data Analyst",
        "company": "Firma",
        "location": "Warszawa",
        "remote": False,
        "employment_type": ["FULL_TIME"],
        "valid_through": "2026-12-31T00:00:00Z",
        "date_posted": "2026-11-05T00:00:00Z",
        "description": "Wymagania\nSQL i Excel\nOferujemy\nowocowe czwartki",
    }
    data.update(fields)
    return RawOffer.model_validate(data)


OFFERS = [
    raw(
        "analyst",
        title="Junior Data Analyst",
        description="Wymagania:\nSQL, Excel, Power BI",
    ),
    raw(
        "senior",
        title="Senior Data Analyst",
        description="Requirements\nSQL, Python, dbt",
    ),
    raw("java", title="Java Developer", description="Must have\nJava, Spring, Kafka"),
    raw(
        "warehouse",
        title="Magazynier",
        location="Krakow",
        description="Nasze wymagania\nuprawnienia UDT, wózek widłowy",
    ),
    raw(
        "remote",
        title="Data Analyst",
        location="",
        remote=True,
        employment_type=["CONTRACTOR"],
        salary_max=40,
        salary_currency="PLN",
        salary_unit="HOUR",
        description="Requirements\nSQL, Tableau",
    ),
]


@pytest.fixture()
def index():
    docs, _ = clean_offers(OFFERS, TODAY)
    return build_index(docs, HashingEmbedder())


def test_sections_split_requirements_from_benefits() -> None:
    sections = split_sections(
        "O nas\nJesteśmy firmą\nTwój zakres obowiązków:\nraporty\n"
        "Nasze wymagania\nSQL\nExcel\nOferujemy\nowoce"
    )
    assert sections["requirements"] == "SQL\nExcel"
    assert sections["duties"] == "raporty"
    assert "owoce" not in sections["requirements"]


def test_clean_drops_expired_and_duplicates_and_unifies_cities() -> None:
    offers = [
        raw("a", location="Kraków"),
        raw("b", location="Krakow", date_posted="2026-11-09T00:00:00Z"),
        raw("old", title="Inna", valid_through="2026-11-01T00:00:00Z"),
        raw("c", title="Tester", location="Krakow"),
    ]
    docs, stats = clean_offers(offers, TODAY)

    assert stats == {
        "read": 4,
        "expired": 1,
        "duplicates": 1,
        "kept": 2,
        "with_requirements": 2,
    }
    by_title = {doc.title: doc for doc in docs}
    assert by_title["Data Analyst"].id == "justjoin:b"  # the newer listing stays
    assert by_title["Tester"].location == "Kraków"


def test_chunks_start_with_title_and_excerpt_is_requirements() -> None:
    docs, _ = clean_offers([raw("a", description="Wymagania\n" + "SQL " * 200)], TODAY)

    pieces = chunks(docs[0])
    assert pieces[0] == "Data Analyst. Firma."
    assert all(piece.startswith("Data Analyst") for piece in pieces)
    assert len(pieces) <= 7
    assert excerpt(docs[0]).startswith("SQL")


def test_contains_term_respects_word_boundaries() -> None:
    assert contains_term("Znajomość C++ i SQL", "c++")
    assert not contains_term("CSS i HTML", "C")
    assert contains_term("wózek widłowy UDT", "Wózek widłowy")


def test_search_ranks_by_role_and_reports_conflicts(index) -> None:
    profile = SearchProfile(
        location="Warszawa",
        target_roles=["Data Analyst"],
        skills=["SQL", "Excel"],
        seniority=["junior"],
        employment_types=["FULL_TIME"],
        salary_expectation={"min": 9000},
    )

    found = search(index, HashingEmbedder(), profile, limit=5)
    by_id = {c.id: c for c in found}

    assert found[0].id == "justjoin:analyst"
    assert by_id["justjoin:analyst"].matched_skills == ["SQL", "Excel"]
    assert by_id["justjoin:senior"].conflicts == ["poziom: senior"]
    assert set(by_id["justjoin:remote"].conflicts) == {
        "forma zatrudnienia: CONTRACTOR",
        "wynagrodzenie do 40 PLN/HOUR, poniżej oczekiwań",
    }
    assert by_id["justjoin:analyst"].requirements == "SQL, Excel, Power BI"


def test_title_match_is_found_even_outside_the_closest_vectors(index) -> None:
    profile = SearchProfile(target_roles=["Magazynier"], location="Kraków")

    found = search(index, HashingEmbedder(), profile, limit=1)

    assert found[0].id == "justjoin:warehouse"


def test_profile_without_roles_or_skills_is_rejected(index) -> None:
    with pytest.raises(ValueError):
        search(index, HashingEmbedder(), SearchProfile(location="Warszawa"))


def test_contact_data_is_not_accepted() -> None:
    with pytest.raises(ValueError):
        SearchProfile.model_validate({"skills": ["SQL"], "email": "a@b.pl"})


def test_index_survives_save_and_load(index, tmp_path) -> None:
    save_index(index, tmp_path)
    loaded = load_index(tmp_path)

    assert [o.id for o in loaded.offers] == [o.id for o in index.offers]
    assert loaded.meta["model"] == "hashing-256"


@pytest.fixture()
def server(index):
    srv = make_server(index, HashingEmbedder(), port=0, token="sekret")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _post(url: str, payload, token: str | None = "sekret") -> tuple[int, dict]:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(url + "/search", data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_http_search_returns_candidates(server) -> None:
    status, data = _post(
        server, {"profile": {"target_roles": ["Java Developer"]}, "limit": 2}
    )

    assert status == 200
    assert data["candidates"][0]["id"] == "justjoin:java"
    assert len(data["candidates"]) == 2


def test_http_health_needs_no_token(server) -> None:
    with urllib.request.urlopen(server + "/health") as response:
        data = json.loads(response.read())
    assert data == {
        "status": "ok",
        "offers": 5,
        "model": "hashing-256",
        "built_at": data["built_at"],
    }


@pytest.mark.parametrize(
    ("payload", "token", "status"),
    [
        ({"profile": {"skills": ["SQL"]}}, None, 401),
        ({"profile": {"skills": ["SQL"]}}, "zly", 401),
        ({"profile": {"skills": ["SQL"]}, "limit": 0}, "sekret", 400),
        ({"profile": {"skills": ["SQL"]}, "limit": True}, "sekret", 400),
        ({"profile": {"skills": ["SQL"]}, "query": "x" * 201}, "sekret", 400),
        ({"profile": {"skills": ["SQL"], "phone": "600"}}, "sekret", 400),
        ({"profile": {}}, "sekret", 400),
        (b"{nie json", "sekret", 400),
        ([1, 2], "sekret", 400),
    ],
)
def test_http_search_rejects_bad_requests(server, payload, token, status) -> None:
    code, data = _post(server, payload, token)

    assert code == status
    assert data["status"] == "error"


@pytest.mark.parametrize("token", [None, "krotki", "change-me"])
def test_network_server_requires_a_real_token(index, token) -> None:
    with pytest.raises(ValueError, match="TOKEN"):
        make_server(index, HashingEmbedder(), host="0.0.0.0", port=0, token=token)


def test_server_refuses_index_built_with_another_model(index) -> None:
    with pytest.raises(ValueError, match="model"):
        make_server(index, HashingEmbedder(dim=64), port=0)
