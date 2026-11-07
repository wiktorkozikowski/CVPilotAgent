import asyncio

from agent_project import mcp_server


def test_server_exposes_two_safe_tools():
    tools = asyncio.run(mcp_server.server.list_tools())

    assert {tool.name for tool in tools} == {"szukaj_ofert", "dodaj_do_listy"}


def test_add_tool_uses_shortlist_file(tmp_path, monkeypatch):
    offers_path = tmp_path / "job_offers.json"
    offers_path.write_text(
        '[{"id": "a", "title": "Dev", "company": "Firma", "location": "Warszawa", '
        '"required_skills": ["Python"], "source": "test", "url": "https://x"}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_server, "OFFERS_PATH", offers_path)
    monkeypatch.setattr(mcp_server, "SHORTLIST_PATH", tmp_path / "shortlist.json")

    first = mcp_server.dodaj_do_listy("a")
    second = mcp_server.dodaj_do_listy("a")

    assert first["status"] == "added"
    assert second["status"] == "already_exists"
