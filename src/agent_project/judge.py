"""The model rates each candidate offer against the profile, a few at a time.

A small model picking from a long list ranks by titles. Here it sees at most
BATCH_SIZE offers per call, each with its requirements in full, and rates every
one from 0 to 10 by a fixed scale. The ranking (offer_server.ranking) chooses
the candidates; this decides which of them really fit.
"""

import json
from pathlib import Path
from typing import Any

from agent_project.llm import ModelClient
from agent_project.models import OfferMatch, Profile
from agent_project.offer_db import OfferDB, years_of_experience
from agent_project.tracing import append_trace
from offer_server.ranking import required_years

BATCH_SIZE = 8
# The most of an offer's requirements the model reads.
REQUIREMENTS_CHARS = 1200
# Offers rated at least this are shown.
MIN_FIT = 7
MAX_REASON_CHARS = 300

RATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ocen_oferty",
        "description": "Zapisuje ocenę dopasowania każdej oferty do kandydata.",
        "parameters": {
            "type": "object",
            "properties": {
                "oceny": {
                    "type": "array",
                    "maxItems": BATCH_SIZE,
                    "items": {
                        "type": "object",
                        "properties": {
                            "oferta": {
                                "type": "string",
                                "description": "Numer oferty, np. o1.",
                            },
                            "ocena": {"type": "integer", "minimum": 0, "maximum": 10},
                            "powod": {
                                "type": "string",
                                "description": (
                                    "Jedno krótkie zdanie: co pasuje i czego brakuje."
                                ),
                            },
                        },
                        "required": ["oferta", "ocena", "powod"],
                    },
                }
            },
            "required": ["oceny"],
        },
    },
}

RATE_PROMPT = (
    "Jesteś rekruterem. Oceniasz, czy kandydat realnie pasuje do każdej "
    "z ofert pracy. Oceniaj wymagania, nie temat: oferta o podobnej "
    "dziedzinie, której wymagań kandydat nie spełnia, dostaje niską ocenę.\n"
    "Skala:\n"
    "9-10: stanowisko odpowiada temu, czego szuka kandydat, ma prawie "
    "wszystkie kluczowe wymagania, poziom i doświadczenie pasują.\n"
    "7-8: stanowisko pasuje, ma większość kluczowych wymagań, braki są "
    "drobne lub łatwe do nadrobienia.\n"
    "4-6: pokrewne stanowisko albo brakuje mu kilku kluczowych wymagań.\n"
    "0-3: inny zawód, brakuje większości wymagań albo oferta wymaga wyraźnie "
    "więcej doświadczenia (lat, seniority) niż ma kandydat.\n"
    "Bądź surowy: ocena 7 lub więcej oznacza, że warto aplikować. Nie "
    "zakładaj umiejętności, których kandydat nie wymienił. Oceń każdą ofertę "
    "z listy jednym wywołaniem ocen_oferty."
)


def profile_for_model(profile: Profile) -> str:
    """What the rater needs to know about the candidate; no contact data."""
    lines = [
        f"Szukane stanowiska: {', '.join(profile.target_roles) or 'nie podano'}",
        f"Poziom: {', '.join(profile.seniority) or 'nie podano'}",
    ]
    years = years_of_experience(profile)
    if years is not None:
        lines.append(f"Doświadczenie zawodowe: ok. {years:g} lat")
    lines.append(f"Umiejętności: {', '.join(profile.skills) or 'nie podano'}")
    if profile.languages:
        spoken = [
            f"{lang.name} {lang.level or ''}".strip() for lang in profile.languages
        ]
        lines.append(f"Języki: {', '.join(spoken)}")
    for job in profile.experience:
        tasks = "; ".join(job.description)
        period = f"{job.start}-{job.end or 'obecnie'}"
        lines.append(f"Praca: {job.position} ({period}) {tasks}")
    for school in profile.education:
        lines.append(f"Wykształcenie: {school.degree}, {school.institution}")
    modes = ", ".join(profile.work_modes) or "dowolny"
    lines.append(f"Miasto: {profile.location or 'nie podano'}; tryb pracy: {modes}")
    if profile.summary:
        lines.append(f"O sobie: {profile.summary}")
    return "\n".join(lines)


def _offer_for_model(number: int, match: OfferMatch, db: OfferDB) -> str:
    doc = db.get(match.offer.id)
    offer = match.offer
    place = offer.location or "brak miasta"
    if offer.remote:
        place += ", zdalnie"
    lines = [f"[o{number}] {offer.title} | {offer.company} | {place}"]
    if doc is not None and doc.level:
        lines.append(f"Poziom w ogłoszeniu: {doc.level}")
    years = required_years(doc) if doc is not None else None
    if years is not None:
        lines.append(f"Wymagane doświadczenie: {years}+ lat")
    text = (doc.requirements or doc.duties or doc.description) if doc else ""
    text = text or offer.requirements or ""
    if len(text) > REQUIREMENTS_CHARS:
        text = text[:REQUIREMENTS_CHARS].rsplit(" ", 1)[0] + " …"
    lines.append(f"Wymagania: {text or 'brak opisu'}")
    return "\n".join(lines)


def _parse(reply: dict[str, Any], batch: list[OfferMatch]) -> dict[str, dict]:
    """Ratings by offer id; anything malformed or about another offer is dropped."""
    numbers = {f"o{i}": match.offer.id for i, match in enumerate(batch, start=1)}
    rated: dict[str, dict] = {}
    for call in reply["tool_calls"]:
        if call["name"] != "ocen_oferty":
            continue
        try:
            items = json.loads(call["arguments"] or "{}").get("oceny")
        except (json.JSONDecodeError, AttributeError):
            continue
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            offer_id = numbers.get(str(item.get("oferta", "")).strip().strip("[]"))
            score, reason = item.get("ocena"), item.get("powod")
            if offer_id is None or isinstance(score, bool):
                continue
            if not isinstance(score, int) or not 0 <= score <= 10:
                continue
            reason = reason if isinstance(reason, str) else ""
            rated[offer_id] = {"fit": score, "reason": reason[:MAX_REASON_CHARS]}
    return rated


def rate_offers(
    client: ModelClient,
    profile: Profile,
    matches: list[OfferMatch],
    db: OfferDB,
    trace_path: Path,
    max_cost_usd: float,
) -> dict[str, dict]:
    """Rate every candidate, BATCH_SIZE per call. Returns {offer id: {fit,
    reason}}; an offer the model left out of its answer is asked about once
    more, then counts as unrated. Stops with RuntimeError past `max_cost_usd`."""
    rated: dict[str, dict] = {}
    spent = 0.0
    step = 0
    pending = [matches[i : i + BATCH_SIZE] for i in range(0, len(matches), BATCH_SIZE)]
    retried: set[str] = set()
    while pending:
        batch = pending.pop(0)
        step += 1
        offers = "\n\n".join(
            _offer_for_model(i, match, db) for i, match in enumerate(batch, start=1)
        )
        candidate = profile_for_model(profile)
        messages = [
            {"role": "system", "content": RATE_PROMPT},
            {"role": "user", "content": f"KANDYDAT\n{candidate}\n\nOFERTY\n{offers}"},
        ]
        reply = client.complete(messages, [RATE_TOOL])
        spent += reply.get("usage", {}).get("cost") or 0.0
        append_trace(
            trace_path,
            "model_decision",
            {
                "step": step,
                "provider": reply.get("provider"),
                "model": reply.get("model"),
                "text": reply.get("text"),
                "tool_calls": reply["tool_calls"],
                "usage": reply.get("usage", {}),
                "latency_ms": reply.get("latency_ms"),
                "run_cost_usd": spent,
            },
        )
        found = _parse(reply, batch)
        rated.update(found)
        append_trace(
            trace_path,
            "tool_result",
            {
                "step": step,
                "tool": "ocen_oferty",
                "result": json.dumps(
                    {"rated": len(found), "of": len(batch)}, ensure_ascii=False
                ),
            },
        )
        if spent > max_cost_usd:
            append_trace(
                trace_path,
                "cost_limit",
                {"step": step, "spent_usd": spent, "limit_usd": max_cost_usd},
            )
            raise RuntimeError(
                f"Przekroczono limit kosztu uruchomienia: {spent:.6g} USD "
                f"z {max_cost_usd:.6g} USD"
            )
        missing = [
            m for m in batch if m.offer.id not in found and m.offer.id not in retried
        ]
        if missing:
            retried.update(m.offer.id for m in missing)
            pending.append(missing)
    return rated


def best_fits(
    matches: list[OfferMatch], rated: dict[str, dict], limit: int
) -> list[tuple[OfferMatch, dict]]:
    """Offers rated at least MIN_FIT, best rating first, then ranking order."""
    order = {match.offer.id: position for position, match in enumerate(matches)}
    good = [
        (match, rated[match.offer.id])
        for match in matches
        if rated.get(match.offer.id, {}).get("fit", -1) >= MIN_FIT
    ]
    good.sort(key=lambda pair: (-pair[1]["fit"], order[pair[0].offer.id]))
    return good[:limit]


def summary(matches: list[OfferMatch], rated: dict[str, dict], shown: int) -> str:
    """A short answer for the user under the table."""
    if shown:
        return (
            f"Oceniłem {len(rated)} z {len(matches)} kandydatów z rankingu. "
            f"Pokazuję {shown} z oceną co najmniej {MIN_FIT}/10."
        )
    if not rated:
        return "Nie udało się ocenić kandydatów. Spróbuj ponownie."
    best = max(rated.values(), key=lambda item: item["fit"])
    return (
        f"Nie znalazłem ofert z oceną co najmniej {MIN_FIT}/10 wśród "
        f"{len(matches)} kandydatów. Najlepsza miała {best['fit']}/10: "
        f"{best['reason']}"
    )
