import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_project.matching import search_offers
from agent_project.models import Offer, OfferMatch, Profile
from agent_project.offer_db import LEVELS, MAX_QUERY_CHARS, OfferDB
from agent_project.shortlist import add_to_shortlist, remove_from_shortlist
from agent_project.tracing import append_trace

MAX_RUN_COST_USD = 0.10
# The model sees this much of each offer's requirements; the rest stays for CVs.
REQUIREMENTS_FOR_MODEL = 400
MAX_PICKS = 30
MAX_REASON_CHARS = 300

# Finds offers for a profile: the offer database, or the local matching.
SearchFn = Callable[[Profile, int], list[OfferMatch]]

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "szukaj_ofert",
            "description": (
                "Znajduje oferty pasujące do umiejętności użytkownika i zwraca je "
                "od najlepiej dopasowanych. Pole conflicts wymienia, co w ofercie "
                "nie zgadza się z preferencjami użytkownika (poziom, tryb pracy, "
                "forma zatrudnienia, wynagrodzenie). Użyj, gdy użytkownik chce "
                "zobaczyć oferty pracy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Ile ofert zwrócić, od 1 do 30. Domyślnie 10.",
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

# Tools for browsing the offer database; the model gets them only when the app
# has one (OfferDB). None of them changes the database.
DB_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "przeszukaj_oferty",
            "description": (
                "Przeszukuje całą bazę ofert po znaczeniu Twojego zapytania (np. "
                "„analityk danych SQL Power BI”, „obsługa klienta po angielsku”). "
                "Użyj, by znaleźć oferty, których szukaj_ofert nie pokazał: inne "
                "nazwy stanowisk, pokrewne role, konkretne umiejętności. Zwraca "
                "krótkie opisy ofert z polem similarity (0-1, wyżej = bliżej "
                "zapytania). Oferty bez poziomu w tytule przechodzą filtr poziom."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zapytanie": {
                        "type": "string",
                        "description": f"Czego szukać, do {MAX_QUERY_CHARS} znaków.",
                    },
                    "miasto": {
                        "type": "string",
                        "description": "Tylko oferty z tego miasta.",
                    },
                    "zdalnie": {
                        "type": "boolean",
                        "description": "true: tylko zdalna; false: tylko z biurem.",
                    },
                    "poziom": {"type": "string", "enum": list(LEVELS)},
                    "forma_zatrudnienia": {
                        "type": "string",
                        "enum": ["FULL_TIME", "PART_TIME", "CONTRACTOR", "INTERN"],
                    },
                    "pomin_widziane": {
                        "type": "boolean",
                        "description": (
                            "true: pomija oferty, które już widziałeś w tej "
                            "rozmowie. Domyślnie true."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Ile ofert zwrócić, od 1 do 30. Domyślnie 10.",
                    },
                },
                "required": ["zapytanie"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pokaz_oferte",
            "description": (
                "Zwraca jedną ofertę w całości: wymagania, obowiązki, link, "
                "wynagrodzenie i termin ważności. Użyj, gdy krótki opis nie "
                "wystarcza do oceny."
            ),
            "parameters": {
                "type": "object",
                "properties": {"offer_id": {"type": "string"}},
                "required": ["offer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "porownaj_z_profilem",
            "description": (
                "Porównuje jedną ofertę z profilem użytkownika: które jego "
                "umiejętności oferta wymienia, których nie, czy tytuł pasuje do "
                "szukanych stanowisk i co jest niezgodne z preferencjami."
            ),
            "parameters": {
                "type": "object",
                "properties": {"offer_id": {"type": "string"}},
                "required": ["offer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wybierz_oferty",
            "description": (
                "Zapisuje Twój wybór: oferty, które polecasz użytkownikowi, od "
                "najlepszej, każdą z krótkim powodem. Wywołaj raz, na końcu, po "
                "przejrzeniu ofert; kolejne wywołanie zastępuje poprzedni wybór. "
                "Zwraca status ok, listę wybranych id (chosen) i id, których nie "
                "ma w bazie (unknown)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "oferty": {
                        "type": "array",
                        "maxItems": MAX_PICKS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "offer_id": {"type": "string"},
                                "powod": {
                                    "type": "string",
                                    "description": (
                                        "Co pasuje i czego brakuje, jedno lub "
                                        "dwa zdania."
                                    ),
                                },
                            },
                            "required": ["offer_id", "powod"],
                        },
                    }
                },
                "required": ["oferty"],
            },
        },
    },
]
DB_TOOL_NAMES = {tool["function"]["name"] for tool in DB_TOOLS}

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
    "Jeśli oferta ma niepuste conflicts, napisz przy niej, co nie pasuje do "
    "preferencji użytkownika.\n"
    "Gdy wyniki są gotowe, podsumuj je krótko w kilku punktach i zakończ."
)

# The prompt when the model can browse the offer database: it looks around and
# makes the choice itself, so it must not stop after the first ranking.
DB_SYSTEM_PROMPT = (
    "Jesteś asystentem, który pomaga znaleźć pracę. Masz dostęp do całej bazy "
    "ofert i sam wybierasz oferty najlepiej pasujące do profilu użytkownika.\n"
    "Pracuj w krokach:\n"
    "1. szukaj_ofert: ranking dla całego profilu, punkt wyjścia.\n"
    "2. przeszukaj_oferty, 2-3 razy: własne zapytania z innymi nazwami "
    "stanowisk, pokrewnymi rolami i najważniejszymi umiejętnościami z profilu; "
    "dodaj filtry zgodne z preferencjami (miasto, praca zdalna, poziom).\n"
    "3. porownaj_z_profilem albo pokaz_oferte dla 1-3 ofert, co do których "
    "masz wątpliwości. Nie oceniaj oferty tylko po tytule.\n"
    "4. wybierz_oferty: oferty, które naprawdę polecasz, od najlepszej, "
    "najwyżej tyle, ile prosi użytkownik (jeśli dobrych jest mniej, wybierz "
    "mniej). W powodzie napisz, co pasuje i czego brakuje.\n"
    "5. Krótkie podsumowanie wyboru, bez wypisywania wszystkich ofert, bo "
    "tabela je pokazuje.\n"
    "Nie kończ przed wywołaniem wybierz_oferty. Masz najwyżej 12 kroków. "
    "Używaj tylko id z wyników narzędzi; nie wymyślaj ofert, firm ani linków. "
    "Do listy wybranych (dodaj_do_listy) dodawaj tylko na prośbę użytkownika. "
    "Usunięcie z listy wymaga zgody użytkownika: jeśli usun_z_listy zwróci "
    "confirmation_required, napisz, że oferta nie została usunięta."
)
# Sent once by the app when the model wants to finish without a choice.
PICK_REMINDER = (
    "Nie zapisałeś jeszcze wyboru. Wywołaj teraz wybierz_oferty z ofertami, "
    "które polecasz (z powodem przy każdej), a potem krótko podsumuj."
)


def system_prompt(db: OfferDB | None) -> str:
    return SYSTEM_PROMPT if db is None else DB_SYSTEM_PROMPT


def tools_for(db: OfferDB | None) -> list[dict[str, Any]]:
    return TOOLS if db is None else [*TOOLS, *DB_TOOLS]


def _deny(offer_id: str) -> bool:
    return False


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _match_for_model(match: OfferMatch) -> dict[str, Any]:
    data = match.model_dump(mode="json", exclude_none=True)
    requirements = data["offer"].get("requirements")
    if requirements and len(requirements) > REQUIREMENTS_FOR_MODEL:
        data["offer"]["requirements"] = requirements[:REQUIREMENTS_FOR_MODEL] + " …"
    return data


def _remember(offers: list[Offer], db: OfferDB | None, offer_id: str) -> None:
    """Make an offer from the database known to the shortlist functions."""
    if db is None or any(offer.id == offer_id for offer in offers):
        return
    offer = db.offer(offer_id)
    if offer is not None:
        offers.append(offer)


def _status(status: str, **payload: Any) -> str:
    return json.dumps({"status": status, **payload}, ensure_ascii=False)


def _browse(arguments: dict[str, Any], offers: list[Offer], db: OfferDB) -> str:
    query = arguments.get("zapytanie")
    limit = arguments.get("limit", 10)
    city = arguments.get("miasto")
    remote = arguments.get("zdalnie")
    skip_seen = arguments.get("pomin_widziane", True)
    if not isinstance(query, str):
        return _error("zapytanie musi być tekstem.")
    if isinstance(limit, bool) or not isinstance(limit, int):
        return _error("limit musi być liczbą całkowitą od 1 do 30.")
    if city is not None and not isinstance(city, str):
        return _error("miasto musi być tekstem.")
    if remote is not None and not isinstance(remote, bool):
        return _error("zdalnie musi być true albo false.")
    if not isinstance(skip_seen, bool):
        return _error("pomin_widziane musi być true albo false.")
    try:
        found = db.browse(
            query,
            limit=limit,
            city=city or None,
            remote=remote,
            level=arguments.get("poziom"),
            employment_type=arguments.get("forma_zatrudnienia"),
            exclude={offer.id for offer in offers} if skip_seen else None,
        )
    except ValueError as error:
        return _error(str(error))
    for item in found:
        _remember(offers, db, item["id"])
    return _status("ok", offers=found)


def _pick(
    arguments: dict[str, Any],
    offers: list[Offer],
    db: OfferDB,
    picks: list[dict[str, str]] | None,
) -> str:
    items = arguments.get("oferty")
    if not isinstance(items, list) or len(items) > MAX_PICKS:
        return _error(f"oferty musi być listą do {MAX_PICKS} pozycji.")
    chosen: list[dict[str, str]] = []
    unknown: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            return _error("Każda pozycja to obiekt z offer_id i powod.")
        offer_id, reason = item.get("offer_id"), item.get("powod")
        if not isinstance(offer_id, str) or not isinstance(reason, str):
            return _error("offer_id i powod muszą być tekstem.")
        if db.get(offer_id) is None:
            unknown.append(offer_id)
        elif all(pick["offer_id"] != offer_id for pick in chosen):
            chosen.append({"offer_id": offer_id, "reason": reason[:MAX_REASON_CHARS]})
            _remember(offers, db, offer_id)
    if picks is not None:
        picks[:] = chosen
    return _status("ok", chosen=[pick["offer_id"] for pick in chosen], unknown=unknown)


def _run_db_tool(
    name: str,
    arguments: dict[str, Any],
    profile: Profile | None,
    offers: list[Offer],
    db: OfferDB,
    picks: list[dict[str, str]] | None,
) -> str:
    if name == "przeszukaj_oferty":
        return _browse(arguments, offers, db)
    if name == "wybierz_oferty":
        return _pick(arguments, offers, db, picks)

    offer_id = arguments.get("offer_id")
    if not isinstance(offer_id, str) or not offer_id:
        return _error("offer_id musi być niepustym tekstem.")
    if name == "pokaz_oferte":
        result = db.details(offer_id)
    elif profile is None:
        return _status("no_profile", message="Profil nie jest zapisany.")
    else:
        result = db.compare(profile, offer_id)
    if result is None:
        return _status("not_found", offer_id=offer_id)
    _remember(offers, db, offer_id)
    return _status("ok", offer=result)


def run_tool(
    call: dict,
    profile: Profile,
    offers: list[Offer],
    shortlist_path: Path,
    confirm: Callable[[str], bool] = _deny,
    search: SearchFn | None = None,
    db: OfferDB | None = None,
    picks: list[dict[str, str]] | None = None,
) -> str:
    """Check one model request in Python, then run the matching function.

    Offers the model sees (from `search` or `db`) are appended to `offers`, so a
    later dodaj_do_listy can find them. wybierz_oferty fills `picks`.
    """
    try:
        arguments = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError:
        return _error("Argumenty narzędzia nie są poprawnym JSON-em.")
    if not isinstance(arguments, dict):
        return _error("Argumenty narzędzia muszą być obiektem JSON.")

    if call["name"] == "szukaj_ofert":
        limit = arguments.get("limit", 10)
        if isinstance(limit, bool) or not isinstance(limit, int):
            return _error("limit musi być liczbą całkowitą od 1 do 30.")
        try:
            if search is not None:
                matches = search(profile, limit)
            elif db is not None:
                matches = db.match_profile(profile, limit)
            else:
                matches = search_offers(profile, offers, limit=limit)
        except ValueError as error:
            return _error(str(error))
        known = {offer.id for offer in offers}
        offers.extend(m.offer for m in matches if m.offer.id not in known)
        return json.dumps(
            {"status": "ok", "offers": [_match_for_model(m) for m in matches]},
            ensure_ascii=False,
        )

    if db is not None and call["name"] in DB_TOOL_NAMES:
        return _run_db_tool(call["name"], arguments, profile, offers, db, picks)

    if call["name"] == "dodaj_do_listy":
        offer_id = arguments.get("offer_id")
        if not isinstance(offer_id, str) or not offer_id:
            return _error("offer_id musi być niepustym tekstem.")
        _remember(offers, db, offer_id)
        result = add_to_shortlist(offer_id, offers, shortlist_path)
        return result.model_dump_json()

    if call["name"] == "usun_z_listy":
        offer_id = arguments.get("offer_id")
        if not isinstance(offer_id, str) or not offer_id:
            return _error("offer_id musi być niepustym tekstem.")
        # Confirmation comes from the application (e.g. a user prompt), never from
        # the model's arguments, so a "confirmed" field from the model is ignored.
        confirmed = confirm(offer_id)
        result = remove_from_shortlist(offer_id, shortlist_path, confirmed=confirmed)
        return result.model_dump_json()

    return _error(f"Nieznane narzędzie: {call['name']}")


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
    tools: list[dict[str, Any]] | None = None,
    dispatch: Callable[[dict], str] | None = None,
    search: SearchFn | None = None,
    db: OfferDB | None = None,
    picks: list[dict[str, str]] | None = None,
) -> str:
    """Run the model loop. `tools` and `dispatch` let other flows reuse this loop.

    With `db` the model also gets the tools that browse the offer database.
    With `picks` as well, a model that wants to finish before wybierz_oferty is
    reminded once (the reminder costs one of the `max_steps`).
    """
    active_tools = tools_for(db) if tools is None else tools
    remind = db is not None and picks is not None and tools is None
    picked = False
    spent_usd = 0.0
    for step in range(1, max_steps + 1):
        reply = client.complete(messages, active_tools)
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
                f"Przekroczono limit kosztu uruchomienia: {spent_usd:.6g} USD "
                f"z {max_cost_usd:.6g} USD"
            )

        if not reply["tool_calls"]:
            if remind and not picked and step < max_steps:
                remind = False
                messages.append({"role": "assistant", "content": reply["text"] or ""})
                messages.append({"role": "user", "content": PICK_REMINDER})
                append_trace(
                    trace_path, "reminder", {"step": step, "text": PICK_REMINDER}
                )
                continue
            return reply["text"]

        call = reply["tool_calls"][0]
        picked = picked or call["name"] == "wybierz_oferty"
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

        if dispatch is None:
            result = run_tool(
                call,
                profile,
                offers,
                shortlist_path,
                confirm,
                search=search,
                db=db,
                picks=picks,
            )
        else:
            result = dispatch(call)
        append_trace(
            trace_path,
            "tool_result",
            {"step": step, "tool": call["name"], "result": result},
        )
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    raise RuntimeError("Osiągnięto limit kroków")
