import json
from pathlib import Path
from typing import Any

from agent_project.cv_tools import Offer, Profile, search_offers
from agent_project.tracing import append_trace

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
    }
]

SYSTEM_PROMPT = (
    "Jesteś asystentem, który pomaga znaleźć pracę. "
    "Twoim celem jest pokazanie użytkownikowi ofert pasujących do jego "
    "umiejętności.\n"
    "Używaj narzędzia szukaj_ofert. Nie wymyślaj ofert, firm ani linków. "
    "Opieraj się tylko na wynikach narzędzia.\n"
    "Gdy wyniki są gotowe, podsumuj je krótko w kilku punktach i zakończ."
)

def run_tool(call: dict, profile: Profile, offers: list[Offer]) -> str:
    if call["name"] != "szukaj_ofert":
        return f"Nieznane narzędzie: {call['name']}"

    arguments = json.loads(call["arguments"] or "{}")
    matches = search_offers(profile, offers, limit=arguments.get("limit", 10))
    if not matches:
        return "Brak pasujących ofert."

    lines = []
    for match in matches:
        offer = match.offer
        skills = ", ".join(match.matched_skills)
        lines.append(
            f"- {offer.title}, {offer.company}, {offer.location}: "
            f"pasuje {skills} ({offer.url})"
        )
    return "\n".join(lines)

def run_agent(
    client, messages, profile, offers, trace_path: Path, max_steps: int = 12
) -> str:
    for step in range(1, max_steps + 1):
        reply = client.complete(messages, TOOLS)
        append_trace(
            trace_path,
            "model_decision",
            {
                "step": step,
                "text": reply["text"],
                "tool_calls": reply["tool_calls"],
                "usage": reply["usage"],
                "latency_ms": reply["latency_ms"],
            },
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

        result = run_tool(call, profile, offers)
        append_trace(
            trace_path,
            "tool_result",
            {"step": step, "tool": call["name"], "result": result},
        )
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    raise RuntimeError("Osiągnięto limit kroków")

        