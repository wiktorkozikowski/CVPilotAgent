import json
from pathlib import Path

from agent_project.agent import SYSTEM_PROMPT, run_agent
from agent_project.matching import search_offers
from agent_project.models import load_offers, load_profile

DATA = Path(__file__).resolve().parent.parent / "data"


def reply(name: str | None = None, arguments: dict | None = None, text=None) -> dict:
    calls = []
    if name is not None:
        calls.append(
            {"id": f"call_{name}", "name": name, "arguments": json.dumps(arguments)}
        )
    return {"text": text, "tool_calls": calls, "usage": {}, "latency_ms": 0}


class ObservationDrivenClient:
    """Picks each next call from the last tool result, as a model would."""

    def __init__(self) -> None:
        self.candidates: list[str] = []
        self.attempts = 0

    def complete(self, messages, tools) -> dict:
        del tools
        last = messages[-1]
        if last["role"] != "tool":
            return reply("szukaj_ofert", {"limit": 3})

        result = json.loads(last["content"])
        if "offers" in result:
            self.candidates = [item["offer"]["id"] for item in result["offers"]]
        elif result["status"] == "added":
            return reply(text=f"Dodałem ofertę {result['offer_id']} do listy.")
        elif result["status"] != "already_exists":
            return reply(text="Nie udało się dodać oferty.")

        next_offer = self.candidates[self.attempts]
        self.attempts += 1
        return reply("dodaj_do_listy", {"offer_id": next_offer})


def test_agent_skips_offer_already_on_shortlist(tmp_path) -> None:
    profile = load_profile(DATA / "profile.json")
    offers = load_offers(DATA / "job_offers.json")
    top_id = search_offers(profile, offers, limit=3)[0].offer.id
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(json.dumps([top_id]), encoding="utf-8")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Dodaj najlepszą ofertę do mojej listy."},
    ]

    answer = run_agent(
        ObservationDrivenClient(),
        messages,
        profile,
        offers,
        tmp_path / "trace.jsonl",
        shortlist_path=shortlist_path,
    )

    tool_results = [
        json.loads(m["content"]) for m in messages if m["role"] == "tool"
    ]
    statuses = [result.get("status") for result in tool_results]
    assert statuses == ["ok", "already_exists", "added"]

    second_id = tool_results[2]["offer_id"]
    assert second_id != top_id
    assert answer == f"Dodałem ofertę {second_id} do listy."
    assert json.loads(shortlist_path.read_text(encoding="utf-8")) == [top_id, second_id]


class RemovalClient:
    """Asks to remove an offer and tries to confirm it inside the arguments."""

    def complete(self, messages, tools) -> dict:
        del tools
        last = messages[-1]
        if last["role"] != "tool":
            return reply(
                "usun_z_listy", {"offer_id": "offer-001", "confirmed": True}
            )
        result = json.loads(last["content"])
        assert result["status"] == "confirmation_required"
        return reply(
            text="Nie usunąłem oferty offer-001, bo wymaga potwierdzenia użytkownika."
        )


def test_removal_needs_user_confirmation_not_model_arguments(tmp_path) -> None:
    profile = load_profile(DATA / "profile.json")
    offers = load_offers(DATA / "job_offers.json")
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(json.dumps(["offer-001"]), encoding="utf-8")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Usuń ofertę offer-001 z listy."},
    ]

    answer = run_agent(
        RemovalClient(),
        messages,
        profile,
        offers,
        tmp_path / "trace.jsonl",
        shortlist_path=shortlist_path,
    )

    assert answer.startswith("Nie usunąłem")
    assert json.loads(shortlist_path.read_text(encoding="utf-8")) == ["offer-001"]
