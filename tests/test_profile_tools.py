import json
from pathlib import Path

from agent_project.profile_tools import (
    PROFILE_SYSTEM_PROMPT,
    PROFILE_TOOLS,
    run_profile_agent,
    run_profile_tool,
)


def _call(name: str, arguments: str = "{}") -> dict:
    return {"id": "call_1", "name": name, "arguments": arguments}


def _run(name, tmp_path: Path, arguments: str = "{}", confirm=lambda: False) -> dict:
    result = run_profile_tool(
        _call(name, arguments),
        draft_path=tmp_path / "draft.json",
        profiles_dir=tmp_path / "profiles",
        confirm=confirm,
    )
    return json.loads(result)


def _reply(name: str | None = None, arguments: dict | None = None, text=None) -> dict:
    calls = []
    if name is not None:
        calls.append(
            {"id": f"call_{name}", "name": name, "arguments": json.dumps(arguments)}
        )
    return {"text": text, "tool_calls": calls, "usage": {}, "latency_ms": 0}


def test_update_fills_draft_and_shrinks_missing_fields(tmp_path) -> None:
    first = _run(
        "zapisz_pola_profilu",
        tmp_path,
        '{"name": "Anna Nowak", "location": "Kraków"}',
    )
    assert first["status"] == "ok"
    assert first["missing_fields"] == ["target_roles", "skills"]

    second = _run(
        "zapisz_pola_profilu",
        tmp_path,
        '{"target_roles": ["Data Analyst"], "skills": ["SQL"]}',
    )
    assert second["missing_fields"] == []
    assert second["draft"]["name"] == "Anna Nowak"


def test_unknown_field_is_rejected_and_draft_is_untouched(tmp_path) -> None:
    result = _run("zapisz_pola_profilu", tmp_path, '{"salary": 10000}')

    assert result["status"] == "validation_error"
    assert not (tmp_path / "draft.json").exists()


def test_invalid_seniority_keeps_previous_good_value(tmp_path) -> None:
    _run("zapisz_pola_profilu", tmp_path, '{"name": "Anna Nowak"}')

    result = _run("zapisz_pola_profilu", tmp_path, '{"seniority": ["guru"]}')

    assert result["status"] == "validation_error"
    draft = json.loads((tmp_path / "draft.json").read_text(encoding="utf-8"))
    assert draft == {"name": "Anna Nowak"}


def test_blank_name_and_blank_skill_are_rejected(tmp_path) -> None:
    assert (
        _run("zapisz_pola_profilu", tmp_path, '{"name": "   "}')["status"]
        == "validation_error"
    )
    assert (
        _run("zapisz_pola_profilu", tmp_path, '{"skills": ["SQL", ""]}')["status"]
        == "validation_error"
    )


def test_finalize_is_incomplete_even_when_confirmed(tmp_path) -> None:
    _run("zapisz_pola_profilu", tmp_path, '{"name": "Anna Nowak"}')

    result = _run("zatwierdz_profil", tmp_path, confirm=lambda: True)

    assert result["status"] == "incomplete"
    assert "skills" in result["missing_fields"]
    assert not (tmp_path / "profiles").exists()


def test_finalize_without_confirmation_saves_nothing(tmp_path) -> None:
    _run(
        "zapisz_pola_profilu",
        tmp_path,
        '{"name": "Anna Nowak", "location": "Kraków", '
        '"target_roles": ["Data Analyst"], "skills": ["SQL"]}',
    )

    result = _run("zatwierdz_profil", tmp_path, confirm=lambda: False)

    assert result["status"] == "confirmation_required"
    assert not (tmp_path / "profiles").exists()
    assert (tmp_path / "draft.json").exists()


def test_model_cannot_confirm_by_its_own_arguments(tmp_path) -> None:
    _run(
        "zapisz_pola_profilu",
        tmp_path,
        '{"name": "Anna Nowak", "location": "Kraków", '
        '"target_roles": ["Data Analyst"], "skills": ["SQL"]}',
    )

    result = _run("zatwierdz_profil", tmp_path, '{"confirmed": true}')

    assert result["status"] == "confirmation_required"
    assert not (tmp_path / "profiles").exists()


def test_confirmed_finalize_saves_profile_and_clears_draft(tmp_path) -> None:
    _run(
        "zapisz_pola_profilu",
        tmp_path,
        '{"name": "Anna Nowak", "location": "Kraków", '
        '"target_roles": ["Data Analyst"], "skills": ["SQL"]}',
    )

    result = _run("zatwierdz_profil", tmp_path, confirm=lambda: True)

    assert result["status"] == "saved"
    assert result["profile_id"] == "anna-nowak"
    saved = json.loads((tmp_path / "profiles" / "anna-nowak.json").read_text("utf-8"))
    assert saved["skills"] == ["SQL"]
    assert not (tmp_path / "draft.json").exists()


def test_bad_json_arguments_are_an_error(tmp_path) -> None:
    assert _run("zapisz_pola_profilu", tmp_path, "{nie json")["status"] == "error"
    assert _run("zapisz_pola_profilu", tmp_path, "[1]")["status"] == "error"


class ProfileChatClient:
    """Chooses each next call from the last tool result, as a model would."""

    def complete(self, messages, tools) -> dict:
        del tools
        last = messages[-1]
        if last["role"] != "tool":
            # The first try has a wrong seniority value on purpose.
            return _reply(
                "zapisz_pola_profilu",
                {"name": "Anna Nowak", "location": "Kraków", "seniority": ["guru"]},
            )

        result = json.loads(last["content"])
        if result["status"] == "validation_error":
            return _reply(
                "zapisz_pola_profilu",
                {"name": "Anna Nowak", "location": "Kraków", "skills": ["SQL"]},
            )
        if result["status"] == "ok" and result["missing_fields"] == ["target_roles"]:
            return _reply("zapisz_pola_profilu", {"target_roles": ["Data Analyst"]})
        if result["status"] == "ok" and not result["missing_fields"]:
            return _reply("zatwierdz_profil", {})
        return _reply(text="Profil nie został zapisany, bo nie potwierdziłeś zapisu.")


def test_profile_chat_changes_plan_after_observations(tmp_path) -> None:
    messages = [
        {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": "Jestem Anna Nowak z Krakowa, umiem SQL."},
    ]

    answer = run_profile_agent(
        ProfileChatClient(),
        messages,
        trace_path=tmp_path / "trace.jsonl",
        draft_path=tmp_path / "draft.json",
        profiles_dir=tmp_path / "profiles",
        confirm=lambda: False,
    )

    results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    assert [r["status"] for r in results] == [
        "validation_error",
        "ok",
        "ok",
        "confirmation_required",
    ]
    assert results[1]["missing_fields"] == ["target_roles"]
    assert answer == "Profil nie został zapisany, bo nie potwierdziłeś zapisu."
    assert not (tmp_path / "profiles").exists()


def test_profile_tool_list_has_no_removal_or_search() -> None:
    names = {tool["function"]["name"] for tool in PROFILE_TOOLS}

    assert names == {
        "pobierz_profil",
        "zapisz_pola_profilu",
        "dodaj_pozycje_profilu",
        "dodaj_doswiadczenie",
        "dodaj_wyksztalcenie",
        "dodaj_jezyk",
        "dodaj_projekt",
        "dodaj_certyfikat",
        "zatwierdz_profil",
    }


def _complete_draft(tmp_path: Path) -> None:
    _run(
        "zapisz_pola_profilu",
        tmp_path,
        '{"name": "Anna Nowak", "location": "Kraków", '
        '"target_roles": ["Data Analyst"], "skills": ["SQL"]}',
    )


def test_items_are_appended_without_duplicates(tmp_path) -> None:
    _complete_draft(tmp_path)

    first = _run(
        "dodaj_pozycje_profilu",
        tmp_path,
        '{"section": "skills", "items": ["Excel", "sql", " Power BI "]}',
    )
    assert first["status"] == "ok"
    assert first["draft"]["skills"] == ["SQL", "Excel", "Power BI"]

    second = _run(
        "dodaj_pozycje_profilu",
        tmp_path,
        '{"section": "skills", "items": ["Tableau"]}',
    )
    assert second["draft"]["skills"] == ["SQL", "Excel", "Power BI", "Tableau"]


def test_items_accept_a_long_list_up_to_the_limit(tmp_path) -> None:
    _complete_draft(tmp_path)
    for batch in range(5):
        items = [f"skill {batch}-{i}" for i in range(20)]
        result = _run(
            "dodaj_pozycje_profilu",
            tmp_path,
            json.dumps({"section": "skills", "items": items}),
        )
        assert result["status"] == "ok"

    assert len(result["draft"]["skills"]) == 101  # the first skill plus 100 more


def test_too_many_items_in_one_call_is_rejected(tmp_path) -> None:
    items = [f"s{i}" for i in range(21)]
    result = _run(
        "dodaj_pozycje_profilu",
        tmp_path,
        json.dumps({"section": "skills", "items": items}),
    )
    assert result["status"] == "validation_error"


def test_unknown_section_is_rejected(tmp_path) -> None:
    result = _run(
        "dodaj_pozycje_profilu", tmp_path, '{"section": "hobby", "items": ["x"]}'
    )
    assert result["status"] == "validation_error"


def test_experience_entry_is_appended_and_checked(tmp_path) -> None:
    _complete_draft(tmp_path)

    ok = _run(
        "dodaj_doswiadczenie",
        tmp_path,
        json.dumps(
            {
                "company": "Acme",
                "position": "Analityk",
                "start": "2021-03",
                "end": "obecnie",
                "description": ["Raporty w SQL", "Dashboardy w Power BI"],
            }
        ),
    )
    assert ok["status"] == "ok"
    assert ok["draft"]["experience"][0]["company"] == "Acme"
    assert ok["draft"]["experience"][0]["end"] == "obecnie"

    bad_date = _run(
        "dodaj_doswiadczenie",
        tmp_path,
        '{"company": "Acme", "position": "Analityk", "start": "marzec 2021"}',
    )
    assert bad_date["status"] == "validation_error"
    assert (
        len(json.loads((tmp_path / "draft.json").read_text("utf-8"))["experience"]) == 1
    )


def test_education_and_contact_are_saved_and_checked(tmp_path) -> None:
    _complete_draft(tmp_path)

    education = _run(
        "dodaj_wyksztalcenie",
        tmp_path,
        '{"institution": "UEP", "degree": "Informatyka (lic.)", "end": "2020"}',
    )
    assert education["status"] == "ok"

    contact = _run(
        "zapisz_pola_profilu",
        tmp_path,
        '{"email": "anna@example.com", "phone": "+48 600 123 456"}',
    )
    assert contact["status"] == "ok"

    bad_email = _run("zapisz_pola_profilu", tmp_path, '{"email": "anna.example"}')
    assert bad_email["status"] == "validation_error"
    bad_phone = _run("zapisz_pola_profilu", tmp_path, '{"phone": "abc"}')
    assert bad_phone["status"] == "validation_error"


def test_full_profile_is_saved_with_sections(tmp_path) -> None:
    _complete_draft(tmp_path)
    _run(
        "dodaj_doswiadczenie",
        tmp_path,
        '{"company": "Acme", "position": "Analityk", "start": "2021"}',
    )
    _run("zapisz_pola_profilu", tmp_path, '{"email": "anna@example.com"}')

    result = _run("zatwierdz_profil", tmp_path, confirm=lambda: True)

    assert result["status"] == "saved"
    saved = json.loads((tmp_path / "profiles" / "anna-nowak.json").read_text("utf-8"))
    assert saved["email"] == "anna@example.com"
    assert saved["experience"][0]["position"] == "Analityk"


def test_remove_item_by_position(tmp_path) -> None:
    _complete_draft(tmp_path)
    from agent_project.profile_tools import remove_item

    result = remove_item("skills", 0, tmp_path / "draft.json")

    assert result["status"] == "removed"
    assert result["missing_fields"] == ["skills"]
    assert remove_item("skills", 5, tmp_path / "draft.json")["status"] == "not_found"
    assert (
        remove_item("name", 0, tmp_path / "draft.json")["status"] == "validation_error"
    )


def test_language_has_a_level_and_readding_it_changes_the_level(tmp_path) -> None:
    _complete_draft(tmp_path)

    first = _run("dodaj_jezyk", tmp_path, '{"name": "angielski", "level": "B1"}')
    assert first["draft"]["languages"] == [{"name": "angielski", "level": "B1"}]

    _run("dodaj_jezyk", tmp_path, '{"name": "niemiecki"}')
    again = _run("dodaj_jezyk", tmp_path, '{"name": "Angielski", "level": "B2"}')
    assert again["draft"]["languages"] == [
        {"name": "niemiecki"},
        {"name": "Angielski", "level": "B2"},
    ]

    bad = _run("dodaj_jezyk", tmp_path, '{"name": "angielski", "level": "dobry"}')
    assert bad["status"] == "validation_error"


def test_old_profile_with_language_names_still_loads() -> None:
    from agent_project.models import Profile

    profile = Profile.model_validate(
        {"name": "Jan", "location": "Warszawa", "languages": ["polski", "angielski"]}
    )

    assert [language.name for language in profile.languages] == ["polski", "angielski"]
    assert profile.languages[0].level is None


def test_job_preferences_use_the_values_of_offers(tmp_path) -> None:
    result = _run(
        "zapisz_pola_profilu",
        tmp_path,
        json.dumps(
            {
                "work_modes": ["remote", "hybrid"],
                "employment_types": ["FULL_TIME", "CONTRACTOR"],
                "salary_expectation": {"min": 9000},
                "driving_licenses": ["B"],
                "github": "https://github.com/anna-nowak",
                "summary": "Analityczka danych z dwuletnim doświadczeniem w SQL.",
            }
        ),
    )
    assert result["status"] == "ok"
    assert result["draft"]["salary_expectation"] == {
        "min": 9000,
        "currency": "PLN",
        "unit": "MONTH",
    }

    for bad in (
        '{"work_modes": ["z domu"]}',
        '{"employment_types": ["B2B"]}',
        '{"salary_expectation": {"min": -5}}',
        '{"salary_expectation": {"min": 9000, "unit": "WEEK"}}',
        '{"driving_licenses": ["Z"]}',
        '{"github": "anna-nowak"}',
    ):
        assert _run("zapisz_pola_profilu", tmp_path, bad)["status"] == (
            "validation_error"
        ), bad
    saved = json.loads((tmp_path / "draft.json").read_text("utf-8"))
    assert saved["work_modes"] == ["remote", "hybrid"]


def test_project_and_certificate_are_saved_and_checked(tmp_path) -> None:
    _complete_draft(tmp_path)

    project = _run(
        "dodaj_projekt",
        tmp_path,
        json.dumps(
            {
                "name": "Analiza cen mieszkań",
                "description": ["Model regresji z błędem 8%"],
                "technologies": ["Python", "pandas"],
                "url": "https://github.com/anna-nowak/ceny",
            }
        ),
    )
    assert project["status"] == "ok"
    certificate = _run(
        "dodaj_certyfikat",
        tmp_path,
        '{"name": "AWS Cloud Practitioner", "issuer": "AWS", "year": "2024"}',
    )
    assert certificate["draft"]["certificates"][0]["issuer"] == "AWS"

    assert (
        _run("dodaj_projekt", tmp_path, '{"name": "X", "url": "github"}')["status"]
        == "validation_error"
    )
    assert (
        _run("dodaj_certyfikat", tmp_path, '{"name": "X", "year": "w zeszłym roku"}')[
            "status"
        ]
        == "validation_error"
    )

    result = _run("zatwierdz_profil", tmp_path, confirm=lambda: True)
    saved = json.loads((tmp_path / "profiles" / "anna-nowak.json").read_text("utf-8"))
    assert result["status"] == "saved"
    assert saved["projects"][0]["technologies"] == ["Python", "pandas"]
    assert "salary_expectation" not in saved  # unset fields are not written as null
