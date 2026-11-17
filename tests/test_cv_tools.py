import json

import pytest

from agent_project.agent import run_tool
from agent_project.matching import search_offers
from agent_project.models import Offer, Profile
from agent_project.shortlist import (
    add_to_shortlist,
    load_shortlist,
    remove_from_shortlist,
)


def make_profile(**overrides) -> Profile:
    data = {
        "name": "Anna",
        "location": "Warszawa",
        "target_roles": ["analityk"],
        "skills": ["Python", "SQL"],
        "languages": ["pl"],
    }
    data.update(overrides)
    return Profile(**data)


def make_offer(
    offer_id: str,
    title: str = "Junior Developer",
    location: str = "Warszawa",
    required_skills: list[str] | None = None,
    **extra,
) -> Offer:
    return Offer(
        **extra,
        id=offer_id,
        title=title,
        company="Firma",
        location=location,
        required_skills=required_skills or ["Python"],
        source="test",
        url=f"https://example.com/{offer_id}",
    )


def test_skips_offers_without_shared_skill():
    offers = [make_offer("java", required_skills=["Java"])]

    assert search_offers(make_profile(), offers) == []


def test_ranks_offer_with_more_shared_skills_first():
    full = make_offer("full", title="Analityk", required_skills=["Python", "SQL"])
    partial = make_offer(
        "partial", title="Analityk", required_skills=["Python", "Java"]
    )

    matches = search_offers(make_profile(), [partial, full])

    assert [match.offer.id for match in matches] == ["full", "partial"]
    assert matches[0].score == pytest.approx(1.0)
    assert matches[1].score == pytest.approx(0.7 * 0.5 + 0.2 + 0.1)


def test_matching_role_raises_score():
    analyst = make_offer("analyst", title="Analityk danych")
    developer = make_offer("developer", title="Developer")

    matches = search_offers(make_profile(), [developer, analyst])

    assert [match.offer.id for match in matches] == ["analyst", "developer"]


def test_remote_offer_counts_as_matching_location():
    remote = make_offer("remote", title="Developer", location="Zdalnie")
    krakow = make_offer("krakow", title="Developer", location="Kraków")

    matches = search_offers(make_profile(), [krakow, remote])

    assert [match.offer.id for match in matches] == ["remote", "krakow"]
    assert matches[1].score == pytest.approx(0.7)


def test_empty_profile_location_does_not_match_any_offer():
    offer = make_offer("any", title="Developer", location="Kraków")

    matches = search_offers(make_profile(location=""), [offer])

    assert matches[0].score == pytest.approx(0.7)


def test_city_spelling_without_polish_letters_still_matches():
    offer = make_offer("krakow", title="Developer", location="Krakow")
    warsaw = make_offer("warsaw", title="Developer", location="Warsaw")

    in_krakow = search_offers(make_profile(location="Kraków"), [offer])
    in_warsaw = search_offers(make_profile(location="Warszawa"), [warsaw])

    assert in_krakow[0].score == pytest.approx(0.8)
    assert in_warsaw[0].score == pytest.approx(0.8)


def test_offer_above_the_users_level_drops_below_a_fitting_one():
    senior = make_offer("senior", title="Senior Developer")
    junior = make_offer("junior", title="Młodszy Developer")
    unnamed = make_offer("unnamed", title="Developer")

    matches = search_offers(
        make_profile(seniority=["junior"]), [senior, junior, unnamed]
    )

    assert [match.offer.id for match in matches][-1] == "senior"
    assert matches[-1].conflicts == ["poziom: senior"]
    assert matches[-1].score == pytest.approx(0.8 * 0.5)
    assert matches[0].conflicts == []


def test_remote_preference_decides_whether_location_counts():
    remote = make_offer("remote", title="Developer", location="Kraków", remote=True)
    office = make_offer("office", title="Developer", remote=False)

    only_remote = search_offers(make_profile(work_modes=["remote"]), [office, remote])
    only_office = search_offers(make_profile(work_modes=["onsite"]), [office, remote])

    assert [match.offer.id for match in only_remote] == ["remote", "office"]
    assert only_remote[1].conflicts == ["praca stacjonarna"]
    assert [match.offer.id for match in only_office] == ["office", "remote"]
    assert only_office[1].score == pytest.approx(0.7)  # has a city, so no conflict


def test_employment_type_and_pay_conflicts_lower_the_score():
    profile = make_profile(
        employment_types=["FULL_TIME"],
        salary_expectation={"min": 10000, "currency": "PLN", "unit": "MONTH"},
    )
    low_contract = make_offer(
        "low",
        employment_type=["CONTRACTOR"],
        salary_max=50,
        salary_currency="PLN",
        salary_unit="HOUR",
    )
    good = make_offer(
        "good",
        employment_type=["FULL_TIME", "CONTRACTOR"],
        salary_max=80,
        salary_currency="PLN",
        salary_unit="HOUR",
    )
    in_euro = make_offer(
        "euro", salary_max=100, salary_currency="EUR", salary_unit="MONTH"
    )

    matches = {
        m.offer.id: m for m in search_offers(profile, [low_contract, good, in_euro])
    }

    assert len(matches["low"].conflicts) == 2
    assert matches["low"].score == pytest.approx(0.8 * 0.25)
    assert matches["good"].conflicts == []
    assert matches["euro"].conflicts == []  # another currency is not judged


def test_offer_in_sample_format_fields_loads():
    offer = make_offer(
        "sample",
        remote=True,
        employment_type=["FULL_TIME"],
        salary_min=150,
        salary_max=165,
        salary_currency="PLN",
        salary_unit="HOUR",
    )

    assert offer.salary_unit == "HOUR"


def test_limit_is_applied():
    offers = [make_offer(f"offer-{number}") for number in range(5)]

    assert len(search_offers(make_profile(), offers, limit=2)) == 2


@pytest.mark.parametrize("limit", [0, 11])
def test_limit_outside_range_is_rejected(limit):
    with pytest.raises(ValueError):
        search_offers(make_profile(), [make_offer("any")], limit=limit)


def test_run_tool_reports_invalid_limit_as_error(tmp_path):
    call = {"name": "szukaj_ofert", "arguments": '{"limit": 11}'}

    result = run_tool(call, make_profile(), [make_offer("any")], tmp_path / "s.json")

    assert json.loads(result)["status"] == "error"


def test_add_to_shortlist_saves_offer_once(tmp_path):
    path = tmp_path / "shortlist.json"
    offers = [make_offer("a")]

    first = add_to_shortlist("a", offers, path)
    second = add_to_shortlist("a", offers, path)

    assert first.status == "added"
    assert second.status == "already_exists"
    assert load_shortlist(path) == ["a"]


def test_add_to_shortlist_rejects_unknown_offer(tmp_path):
    path = tmp_path / "shortlist.json"

    result = add_to_shortlist("missing", [make_offer("a")], path)

    assert result.status == "not_found"
    assert not path.exists()


def test_run_tool_adds_offer_and_returns_json_status(tmp_path):
    call = {"name": "dodaj_do_listy", "arguments": '{"offer_id": "a"}'}

    result = run_tool(call, make_profile(), [make_offer("a")], tmp_path / "s.json")

    assert '"status":"added"' in result.replace(" ", "")


def test_remove_requires_confirmation(tmp_path):
    path = tmp_path / "shortlist.json"
    add_to_shortlist("a", [make_offer("a")], path)

    result = remove_from_shortlist("a", path, confirmed=False)

    assert result.status == "confirmation_required"
    assert load_shortlist(path) == ["a"]


def test_remove_with_confirmation_deletes_offer(tmp_path):
    path = tmp_path / "shortlist.json"
    add_to_shortlist("a", [make_offer("a")], path)

    result = remove_from_shortlist("a", path, confirmed=True)

    assert result.status == "removed"
    assert load_shortlist(path) == []


def test_remove_missing_offer_reports_not_found(tmp_path):
    result = remove_from_shortlist("a", tmp_path / "shortlist.json", confirmed=True)

    assert result.status == "not_found"


def test_model_cannot_confirm_removal_in_arguments(tmp_path):
    path = tmp_path / "shortlist.json"
    add_to_shortlist("a", [make_offer("a")], path)
    call = {
        "name": "usun_z_listy",
        "arguments": '{"offer_id": "a", "confirmed": true}',
    }

    result = run_tool(call, make_profile(), [make_offer("a")], path)

    assert '"status":"confirmation_required"' in result.replace(" ", "")
    assert load_shortlist(path) == ["a"]


def test_user_confirmation_callback_allows_removal(tmp_path):
    path = tmp_path / "shortlist.json"
    add_to_shortlist("a", [make_offer("a")], path)
    call = {"name": "usun_z_listy", "arguments": '{"offer_id": "a"}'}

    result = run_tool(
        call, make_profile(), [make_offer("a")], path, confirm=lambda _: True
    )

    assert '"status":"removed"' in result.replace(" ", "")
    assert load_shortlist(path) == []


def test_search_result_contains_offer_id_for_follow_up_call(tmp_path):
    call = {"name": "szukaj_ofert", "arguments": "{}"}

    result = run_tool(call, make_profile(), [make_offer("abc123")], tmp_path / "s.json")

    assert json.loads(result)["offers"][0]["offer"]["id"] == "abc123"
