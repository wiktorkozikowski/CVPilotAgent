import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_project.matching import search_offers
from agent_project.models import Offer, Profile
from agent_project.shortlist import add_to_shortlist, remove_from_shortlist
from agent_project.tracing import append_trace

MAX_RUN_COST_USD = 0.10

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "szukaj_ofert",
            "description": (
                "Znajduje oferty pasujące do umiejętności użytkownika i zwraca je "
                "od najlepiej dopasowanych. Użyj, gdy użytkownik chce zobaczyć "
                "oferty pracy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Ile ofert zwrócić, od 1 do 10. Domyślnie 10.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dodaj_do_listy",
            "description": (
                "Dodaje ofertę do listy wybranych ofert. Podaj id oferty z wyników "
                "szukaj_ofert. Zwraca status: added, already_exists albo not_found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "offer_id": {
                        "type": "string",
                        "description": "Id oferty z wyników szukaj_ofert.",
                    }
                },
                "required": ["offer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "usun_z_listy",
            "description": (
                "Usuwa ofertę z listy wybranych. Działa tylko po potwierdzeniu "
                "przez użytkownika. Zwraca status: removed, not_found albo "
                "confirmation_required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "offer_id": {
                        "type": "string",
                        "description": "Id oferty z listy wybranych.",
                    }
                },
                "required": ["offer_id"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "Jesteś asystentem, który pomaga znaleźć pracę. "
    "Twoim celem jest pokazanie użytkownikowi ofert pasujących do jego "
    "umiejętności i, na jego prośbę, dodanie wybranych ofert do listy.\n"
    "Używaj narzędzia szukaj_ofert. Do listy dodawaj tylko oferty z wyników "
    "szukaj_ofert, używając ich id. Nie wymyślaj ofert, firm, linków ani id. "
    "Opieraj się tylko na wynikach narzędzi.\n"
    "Jeśli dodanie zwróci already_exists albo not_found, nie powtarzaj tego "
    "samego wywołania. Wybierz inną ofertę albo poinformuj użytkownika.\n"
    "Usunięcie z listy wymaga zgody użytkownika. Jeśli usun_z_listy zwróci "
    "confirmation_required, napisz, że oferta nie została usunięta.\n"
    "Gdy wyniki są gotowe, podsumuj je krótko w kilku punktach i zakończ."
)


def _deny(offer_id: str) -> bool:
    return False


def run_tool(
    call: dict,
    profile: Profile,
    offers: list[Offer],
    shortlist_path: Path,
    confirm: Callable[[str], bool] = _deny,
) -> str:
    arguments = json.loads(call["arguments"] or "{}")

    if call["name"] == "szukaj_ofert":
        try:
            matches = search_offers(profile, offers, limit=arguments.get("limit", 10))
        except ValueError as error:
            return json.dumps(
                {"status": "error", "message": str(error)}, ensure_ascii=False
            )
        return json.dumps(
            {
                "status": "ok",
                "offers": [match.model_dump(mode="json") for match in matches],
            },
            ensure_ascii=False,
        )

    if call["name"] == "dodaj_do_listy":
        offer_id = arguments.get("offer_id")
        if not isinstance(offer_id, str) or not offer_id:
            return "Błąd: offer_id musi być niepustym tekstem."
        result = add_to_shortlist(offer_id, offers, shortlist_path)
        return result.model_dump_json()

    if call["name"] == "usun_z_listy":
        offer_id = arguments.get("offer_id")
        if not isinstance(offer_id, str) or not offer_id:
            return "Błąd: offer_id musi być niepustym tekstem."
        # Confirmation comes from the application (e.g. a user prompt), never from
        # the model's arguments, so a "confirmed" field from the model is ignored.
        confirmed = confirm(offer_id)
        result = remove_from_shortlist(offer_id, shortlist_path, confirmed=confirmed)
        return result.model_dump_json()

    return f"Nieznane narzędzie: {call['name']}"


def run_agent(
    client,
    messages,
    profile,
    offers,
    trace_path: Path,
    max_steps: int = 12,
    shortlist_path: Path = Path("data/shortlist.json"),
    confirm: Callable[[str], bool] = _deny,
    max_cost_usd: float = MAX_RUN_COST_USD,
) -> str:
    spent_usd = 0.0
    for step in range(1, max_steps + 1):
        reply = client.complete(messages, TOOLS)
        spent_usd += reply["usage"].get("cost") or 0.0
        append_trace(
            trace_path,
            "model_decision",
            {
                "step": step,
                "provider": reply.get("provider"),
                "model": reply.get("model"),
                "text": reply["text"],
                "tool_calls": reply["tool_calls"],
                "usage": reply["usage"],
                "latency_ms": reply["latency_ms"],
                "run_cost_usd": spent_usd,
            },
        )

        if spent_usd > max_cost_usd:
            append_trace(
                trace_path,
                "cost_limit",
                {"step": step, "spent_usd": spent_usd, "limit_usd": max_cost_usd},
            )
            raise RuntimeError(
                f"Przekroczono limit kosztu uruchomienia: {spent_usd:.4f} USD "
                f"z {max_cost_usd:.2f} USD"
            )

        if not reply["tool_calls"]:
            return reply["text"]

        call = reply["tool_calls"][0]
        messages.append(
            {
                "role": "assistant",
                "content": reply["text"] or "",
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                ],
            }
        )

        result = run_tool(call, profile, offers, shortlist_path, confirm)
        append_trace(
            trace_path,
            "tool_result",
            {"step": step, "tool": call["name"], "result": result},
        )
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    raise RuntimeError("Osiągnięto limit kroków")
