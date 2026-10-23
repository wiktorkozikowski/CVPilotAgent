from pathlib import Path

from agent_project.cv_tools import load_offers, load_profile, search_offers

DATA = Path(__file__).resolve().parent.parent / "data"


def test_load_sample_profile() -> None:
    profile = load_profile(DATA / "profile.json")
    assert "Python" in profile.skills
    assert profile.target_roles


def test_load_sample_offers() -> None:
    offers = load_offers(DATA / "job_offers.json")
    assert len(offers) == 4
    assert len({offer.id for offer in offers}) == len(offers)


def test_search_ranks_by_share_of_required_skills() -> None:
    profile = load_profile(DATA / "profile.json")
    offers = load_offers(DATA / "job_offers.json")
    matches = search_offers(profile, offers)

    ids = [match.offer.id for match in matches]
    # offer-001: SQL, Excel, statistics = 3/3; offer-002: Python, Git, SQL = 3/3
    assert ids[:2] == ["offer-001", "offer-002"]
    # offer-003 matches only Python and SQL (2 of 4); offer-004 matches Excel (1 of 3)
    assert ids[2:] == ["offer-003", "offer-004"]
    assert matches[2].matched_skills == ["Python", "SQL"]
    assert matches[2].score == 0.5


def test_offers_without_matching_skills_are_skipped() -> None:
    profile = load_profile(DATA / "profile.json")
    offers = load_offers(DATA / "job_offers.json")
    profile.skills = ["Rust"]
    assert search_offers(profile, offers) == []


def test_skill_matching_ignores_case_and_limit_applies() -> None:
    profile = load_profile(DATA / "profile.json")
    profile.skills = ["python", "sql"]
    offers = load_offers(DATA / "job_offers.json")
    matches = search_offers(profile, offers, limit=1)
    # offer-002 covers 2 of 3 required skills, the best share for these skills
    assert len(matches) == 1
    assert matches[0].offer.id == "offer-002"
    assert matches[0].matched_skills == ["Python", "SQL"]
