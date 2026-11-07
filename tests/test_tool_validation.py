import json
from pathlib import Path

from agent_project.agent import run_tool
from agent_project.models import load_offers, load_profile

DATA = Path(__file__).resolve().parent.parent / "data"


def _call(name: str, arguments: str) -> dict:
    return {"id": "call_1", "name": name, "arguments": arguments}


def _run(call: dict, tmp_path: Path, confirm=lambda offer_id: False) -> dict:
    result = run_tool(
        call,
        load_profile(DATA / "profile.json"),
        load_offers(DATA / "job_offers.json"),
        tmp_path / "shortlist.json",
        confirm,
    )
    return json.loads(result)


def test_limit_must_be_integer(tmp_path) -> None:
    assert (
        _run(_call("szukaj_ofert", '{"limit": "abc"}'), tmp_path)["status"] == "error"
    )
    assert _run(_call("szukaj_ofert", '{"limit": true}'), tmp_path)["status"] == "error"


def test_limit_out_of_range_is_an_error(tmp_path) -> None:
    assert _run(_call("szukaj_ofert", '{"limit": 0}'), tmp_path)["status"] == "error"
    assert _run(_call("szukaj_ofert", '{"limit": 11}'), tmp_path)["status"] == "error"


def test_broken_json_arguments_are_an_error(tmp_path) -> None:
    assert _run(_call("szukaj_ofert", "{nie json"), tmp_path)["status"] == "error"
    assert _run(_call("szukaj_ofert", "[1, 2]"), tmp_path)["status"] == "error"


def test_valid_search_still_works(tmp_path) -> None:
    result = _run(_call("szukaj_ofert", '{"limit": 2}'), tmp_path)
    assert result["status"] == "ok"
    assert len(result["offers"]) == 2


def test_removal_without_confirmation_does_not_remove(tmp_path) -> None:
    shortlist = tmp_path / "shortlist.json"
    shortlist.write_text('["offer-001"]', encoding="utf-8")
    result = _run(_call("usun_z_listy", '{"offer_id": "offer-001"}'), tmp_path)
    assert result["status"] == "confirmation_required"
    assert json.loads(shortlist.read_text(encoding="utf-8")) == ["offer-001"]
