"""The CVPilot web app: build a profile in a chat, let the agent search the
offer database for it, keep a shortlist. One process holds the page, the agent
loop and the vector index of offers; the model is called through OpenRouter.

Run with `uv run python -m agent_project.web` and open http://127.0.0.1:8000.
Every page and API call needs the login of the one account (agent_project.auth).
Only the standard library is used for the server; the pages are HTML files.

Settings in the environment (.env):
    CVPILOT_HOST, CVPILOT_PORT   where to listen (default 127.0.0.1:8000; on a
                                 server keep 127.0.0.1 behind an HTTPS proxy)
    CVPILOT_ACCOUNT_FILE         the account (default var/account.json)
    OFFERS_INDEX_DIR             the offer index (default var/index)
"""

import json
import os
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from agent_project.agent import (
    DB_TOOLS,
    MAX_RUN_COST_USD,
    run_agent,
    run_tool,
    system_prompt,
)
from agent_project.auth import COOKIE_NAME, Account, Auth, LockedOut, account_path
from agent_project.judge import best_fits, rate_offers, summary
from agent_project.llm import ModelClient, OpenRouterClient
from agent_project.matching import search_offers
from agent_project.models import Offer, OfferMatch, Profile, load_offers
from agent_project.offer_db import OfferDB
from agent_project.profile_tools import (
    CHAT_DB_PROMPT,
    CHAT_SEARCH_PROMPT,
    PROFILE_SYSTEM_PROMPT,
    add_items,
    finalize_profile,
    get_draft,
    load_profile_into_draft,
    remove_item,
    replace_item,
    run_profile_agent,
    save_entry,
    update_draft,
)
from agent_project.profiles import list_profiles, load_profile_by_id
from agent_project.shortlist import (
    add_to_shortlist,
    load_seen_offers,
    load_shortlist,
    remember_offers,
    remove_from_shortlist,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 64 * 1024
MAX_CHAT_CHARS = 2000
SEARCH_LIMIT = 10  # offers shown when the user does not say how many
MAX_SEARCH_LIMIT = 30
# Candidates the ranking picks; the model rates each one (agent_project.judge).
CANDIDATE_POOL = 40
# The chat may browse the database too, but not pick (that is the search button).
CHAT_DB_TOOLS = [t for t in DB_TOOLS if t["function"]["name"] != "wybierz_oferty"]
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Cache-Control": "no-store",
}
PUBLIC_PATHS = {"/login", "/api/login"}
SAVED_MESSAGE = (
    "Zapisałem profil. Mogę teraz wyszukać pasujące oferty: napisz, ile ofert "
    "mam znaleźć (od 1 do {limit}), albo kliknij „Szukaj ofert”."
)
LOADED_MESSAGE = (
    "Wczytałem profil: {name}. Możesz go zmienić w rozmowie (zmiany zapiszesz "
    "przyciskiem „Zapisz zmiany w profilu”) albo wyszukać oferty: napisz, ile "
    "ofert mam znaleźć (od 1 do {limit}), albo kliknij „Szukaj ofert”."
)


class _Config:
    def __init__(
        self,
        data_dir: Path,
        client_factory: Callable[[], ModelClient],
        offer_db: OfferDB | None = None,
        auth: Auth | None = None,
        secure_cookie: bool = True,
    ) -> None:
        self.data_dir = data_dir
        # Set: offers come from the offer database (the vector index) and the
        # model can browse it. None: from job_offers.json (tests and demos).
        self.offer_db = offer_db
        # None only in tests: main() never starts the app without an account.
        self.auth = auth
        self.secure_cookie = secure_cookie
        # Offers the chat model has seen, so it can put them on the shortlist.
        self.chat_offers: list[Offer] = []
        self.seen_offers_path = data_dir / "offers_seen.json"
        self.profiles_dir = data_dir / "profiles"
        self.draft_path = data_dir / "profile_draft.json"
        self.client_factory = client_factory
        # One agent run at a time: the model client and trace files are not shared.
        self.search_lock = threading.Lock()
        # The profile chat: model messages (for the model) and the visible transcript.
        self.chat_lock = threading.Lock()
        self.chat_messages: list[dict[str, Any]] = []
        self.chat_view: list[dict[str, str]] = []
        # The profile the chat saved: later saves overwrite it, searches use it.
        self.chat_profile_id: str | None = None
        # The chat's latest offer search, so a page reload still shows it.
        self.chat_search: dict[str, Any] | None = None

    def reset_chat(self) -> None:
        self.chat_messages.clear()
        self.chat_view.clear()
        self.chat_profile_id = None
        self.chat_search = None
        self.chat_offers = []
        self.draft_path.unlink(missing_ok=True)

    def chat_system_prompt(self) -> str:
        prompt = PROFILE_SYSTEM_PROMPT + CHAT_SEARCH_PROMPT
        return prompt if self.offer_db is None else prompt + CHAT_DB_PROMPT

    def add_assistant_note(self, text: str) -> None:
        """A message from the app, shown in the chat and known to the model."""
        if not self.chat_messages:
            self.chat_messages.append(
                {"role": "system", "content": self.chat_system_prompt()}
            )
        self.chat_messages.append({"role": "assistant", "content": text})
        self.chat_view.append({"role": "assistant", "text": text})

    def chat_state(self) -> dict[str, Any]:
        return {
            "profile_id": self.chat_profile_id,
            "search": self.chat_search,
            **get_draft(self.draft_path),
        }


def _parse_limit(value: Any) -> int:
    """The number of offers asked for; raises ValueError with a message to show."""
    if value is None:
        return SEARCH_LIMIT
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Liczba ofert musi być liczbą całkowitą od 1 do {MAX_SEARCH_LIMIT}."
        )
    if not 1 <= value <= MAX_SEARCH_LIMIT:
        raise ValueError(f"Liczba ofert musi być od 1 do {MAX_SEARCH_LIMIT}.")
    return value


def _find_matches(config: "_Config", profile: Profile, limit: int) -> list[OfferMatch]:
    """Offers for a profile: from the offer database if set, else job_offers.json."""
    if config.offer_db is None:
        offers = load_offers(config.data_dir / "job_offers.json")
        return search_offers(profile, offers, limit=limit)
    matches = config.offer_db.match_profile(profile, limit)
    remember_offers(config.seen_offers_path, [m.offer for m in matches])
    return matches


def _known_offers(config: "_Config", offer_id: str | None = None) -> list[Offer]:
    """Offers a shortlist may hold: the local ones, those the agent showed
    earlier and, for `offer_id`, that offer from the database."""
    offers = load_offers(config.data_dir / "job_offers.json")
    local = {offer.id for offer in offers}
    offers += [
        o for o in load_seen_offers(config.seen_offers_path) if o.id not in local
    ]
    if offer_id and config.offer_db is not None:
        if all(offer.id != offer_id for offer in offers):
            found = config.offer_db.offer(offer_id)
            if found is not None:
                remember_offers(config.seen_offers_path, [found])
                offers.append(found)
    return offers


def _shortlist_path(config: "_Config", profile_id: Any) -> Path:
    """Each profile has its own shortlist file; unknown ids raise KeyError."""
    if not isinstance(profile_id, str):
        raise KeyError(profile_id)
    load_profile_by_id(config.profiles_dir, profile_id)
    return config.data_dir / "shortlists" / f"{profile_id}.json"


def _trace_name(trace_path: Path, config: _Config) -> str:
    return str(trace_path.relative_to(config.data_dir.parent))


def _trace_steps(trace_path: Path) -> list[dict[str, Any]]:
    """Turn the JSONL trace of one run into steps the UI can show."""
    if not trace_path.exists():
        return []
    steps: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        payload = record["payload"]
        if record["event"] == "model_decision":
            steps.append(
                {
                    "kind": "decyzja",
                    "step": payload["step"],
                    "provider": payload.get("provider"),
                    "text": payload.get("text"),
                    "tools": [
                        {"name": call["name"], "arguments": call["arguments"]}
                        for call in payload["tool_calls"]
                    ],
                    "cost_usd": payload["usage"].get("cost"),
                    "latency_ms": payload.get("latency_ms"),
                    "run_cost_usd": payload.get("run_cost_usd"),
                }
            )
        elif record["event"] == "tool_result":
            steps.append(
                {
                    "kind": "wynik",
                    "step": payload["step"],
                    "tool": payload["tool"],
                    "result": payload["result"],
                }
            )
        elif record["event"] == "cost_limit":
            steps.append({"kind": "limit", **payload})
    return steps


def _match_to_dict(match) -> dict[str, Any]:
    return {
        "id": match.offer.id,
        "title": match.offer.title,
        "company": match.offer.company,
        "location": match.offer.location,
        "url": match.offer.url,
        "matched_skills": match.matched_skills,
        "requirements": match.offer.requirements,
        "conflicts": match.conflicts,
        "score": round(match.score, 2),
    }


def _make_handler(config: _Config) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # keep the console quiet; traces hold the run details

        def _send(
            self,
            status: int,
            body: bytes,
            content_type: str,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in {**SECURITY_HEADERS, **(headers or {})}.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(
            self,
            status: int,
            payload: dict[str, Any],
            headers: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8", headers)

        def _send_page(self, name: str) -> None:
            body = (STATIC_DIR / name).read_bytes()
            self._send(200, body, "text/html; charset=utf-8")

        def _redirect(self, location: str) -> None:
            self._send(303, b"", "text/plain", {"Location": location})

        # --- login ----------------------------------------------------------

        def _session_token(self) -> str | None:
            try:
                cookie = SimpleCookie(self.headers.get("Cookie", ""))
            except CookieError:
                return None
            morsel = cookie.get(COOKIE_NAME)
            return morsel.value if morsel else None

        def _logged_in(self) -> bool:
            return config.auth is None or config.auth.valid(self._session_token())

        def _cookie(self, value: str, max_age: int) -> str:
            parts = [
                f"{COOKIE_NAME}={value}",
                "Path=/",
                "HttpOnly",
                "SameSite=Strict",
                f"Max-Age={max_age}",
            ]
            if config.secure_cookie:
                parts.append("Secure")
            return "; ".join(parts)

        def _same_origin(self) -> bool:
            """Another site's page cannot post here: the browser sends Origin
            with cross-site POSTs, and it must name this host."""
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            return urlparse(origin).netloc == self.headers.get("Host", "")

        def _login(self, data: dict[str, Any]) -> None:
            username, password = data.get("username"), data.get("password")
            if not isinstance(username, str) or not isinstance(password, str):
                self._send_json(400, {"error": "Podaj login i hasło."})
                return
            if config.auth is None:
                self._send_json(200, {"status": "ok"})
                return
            try:
                token = config.auth.login(username, password)
            except LockedOut as error:
                self._send_json(429, {"error": str(error)})
                return
            if token is None:
                self._send_json(401, {"error": "Zły login albo hasło."})
                return
            cookie = self._cookie(token, config.auth.session_seconds)
            self._send_json(200, {"status": "ok"}, {"Set-Cookie": cookie})

        def _logout(self) -> None:
            if config.auth is not None:
                config.auth.logout(self._session_token())
            self._send_json(
                200, {"status": "logged_out"}, {"Set-Cookie": self._cookie("", 0)}
            )

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ValueError("Za duże żądanie.")
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("Oczekiwano obiektu JSON.")
            return data

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/login":
                if config.auth is not None and self._logged_in():
                    self._redirect("/")
                else:
                    self._send_page("login.html")
                return
            if not self._logged_in():
                if self.path.startswith("/api/"):
                    self._send_json(401, {"error": "Zaloguj się."})
                else:
                    self._redirect("/login")
                return
            if self.path == "/":
                self._send_page("index.html")
                return
            if self.path == "/api/status":
                self._send_json(
                    200,
                    {
                        "offers": None
                        if config.offer_db is None
                        else config.offer_db.stats(),
                        "login": config.auth is not None,
                    },
                )
                return
            if self.path.startswith("/api/shortlist?") or self.path == "/api/shortlist":
                query = parse_qs(urlparse(self.path).query)
                self._shortlist(query.get("profile_id", [""])[0])
                return
            if self.path == "/api/chat":
                with config.chat_lock:
                    view = list(config.chat_view)
                    state = config.chat_state()
                self._send_json(200, {"messages": view, **state})
                return
            if self.path == "/api/profiles":
                profiles = list_profiles(config.profiles_dir)
                self._send_json(
                    200,
                    {
                        "profiles": [
                            {"id": pid, **profile.model_dump()}
                            for pid, profile in profiles
                        ]
                    },
                )
                return
            self._send_json(404, {"error": "Nie znaleziono."})

        def do_POST(self) -> None:  # noqa: N802
            # A plain HTML form from another site cannot send JSON, and the
            # SameSite=Strict cookie stays home; Origin is checked as well.
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("application/json") or (
                not self._same_origin()
            ):
                self._send_json(403, {"error": "Niedozwolone żądanie."})
                return
            if self.path not in PUBLIC_PATHS and not self._logged_in():
                self._send_json(401, {"error": "Zaloguj się."})
                return
            try:
                data = self._read_json()
            except (ValueError, json.JSONDecodeError):
                self._send_json(400, {"error": "Nieprawidłowe dane JSON."})
                return

            if self.path == "/api/login":
                self._login(data)
            elif self.path == "/api/logout":
                self._logout()
            elif self.path == "/api/chat/message":
                self._chat_message(data)
            elif self.path == "/api/chat/confirm":
                self._chat_confirm()
            elif self.path == "/api/chat/reset":
                with config.chat_lock:
                    config.reset_chat()
                self._send_json(200, {"status": "reset"})
            elif self.path == "/api/chat/load":
                self._chat_load(data)
            elif self.path == "/api/profile/remove":
                self._profile_remove(data)
            elif self.path == "/api/profile/edit":
                self._profile_edit(data)
            elif self.path == "/api/search":
                self._search(data)
            elif self.path in ("/api/shortlist/add", "/api/shortlist/remove"):
                self._shortlist_change(self.path, data)
            else:
                self._send_json(404, {"error": "Nie znaleziono."})

        def _profile_edit(self, data: dict[str, Any]) -> None:
            # The page's own edit forms: change the draft directly, no model involved.
            # The saved profile changes only after "Zatwierdź" / "Zapisz zmiany".
            action = data.get("action")
            section = data.get("section")
            with config.chat_lock:
                if action == "fields":
                    fields = data.get("fields")
                    if not isinstance(fields, dict):
                        result = {"status": "validation_error", "message": "Brak pól."}
                    else:
                        result = update_draft(fields, config.draft_path)
                elif action == "add_items":
                    result = add_items(section, data.get("items"), config.draft_path)
                elif action == "replace_item":
                    result = replace_item(
                        section, data.get("index"), data.get("value"), config.draft_path
                    )
                elif action == "entry":
                    result = save_entry(
                        section, data.get("entry"), config.draft_path, data.get("index")
                    )
                else:
                    result = {
                        "status": "validation_error",
                        "message": "Nieznana operacja.",
                    }
            if result["status"] == "ok":
                self._send_json(200, result)
            elif result["status"] == "not_found":
                self._send_json(404, {"error": "Nie ma takiej pozycji."})
            else:
                self._send_json(400, {"error": result.get("message", "Błąd.")})

        def _profile_remove(self, data: dict[str, Any]) -> None:
            # The page's ✕ button: removes one item from the draft, no model involved.
            section = data.get("section")
            index = data.get("index")
            with config.chat_lock:
                result = remove_item(section, index, config.draft_path)
            if result["status"] == "removed":
                self._send_json(200, result)
            elif result["status"] == "not_found":
                self._send_json(404, {"error": "Nie ma takiej pozycji."})
            else:
                self._send_json(400, {"error": result.get("message", "Błąd.")})

        def _chat_message(self, data: dict[str, Any]) -> None:
            text = data.get("message")
            if not isinstance(text, str) or not text.strip():
                self._send_json(400, {"error": "Napisz wiadomość."})
                return
            if len(text) > MAX_CHAT_CHARS:
                self._send_json(
                    400,
                    {"error": f"Wiadomość może mieć najwyżej {MAX_CHAT_CHARS} znaków."},
                )
                return
            text = text.strip()

            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            trace_path = config.data_dir.parent / "traces" / f"chat_{stamp}.jsonl"
            # Filled when the model searches during this turn.
            found: dict[str, Any] = {}

            def chat_search(arguments: dict[str, Any]) -> dict[str, Any]:
                profile_id = config.chat_profile_id
                if profile_id is None:
                    return {
                        "status": "no_profile",
                        "message": (
                            "Profil nie jest zapisany: kliknij „Zatwierdź profil”."
                        ),
                    }
                try:
                    limit = _parse_limit(arguments.get("limit"))
                    profile = load_profile_by_id(config.profiles_dir, profile_id)
                    matches = _find_matches(config, profile, limit)
                except ValueError as error:
                    return {"status": "error", "message": str(error)}
                except KeyError:
                    return {
                        "status": "no_profile",
                        "message": "Nie ma zapisanego profilu.",
                    }
                found["search"] = {
                    "profile_id": profile_id,
                    "matches": [_match_to_dict(m) for m in matches],
                }
                return {
                    "status": "ok",
                    "offers": [
                        {
                            "title": m.offer.title,
                            "company": m.offer.company,
                            "score": round(m.score, 2),
                            "conflicts": m.conflicts,
                        }
                        for m in matches
                    ],
                }

            def chat_browse(call: dict) -> str:
                # The database tools of the search agent, run for the chat's
                # saved profile (porownaj_z_profilem needs one).
                profile = None
                if config.chat_profile_id is not None:
                    try:
                        profile = load_profile_by_id(
                            config.profiles_dir, config.chat_profile_id
                        )
                    except KeyError:
                        profile = None
                return run_tool(
                    call,
                    profile,
                    config.chat_offers,
                    # Not used: CHAT_DB_TOOLS has no shortlist tools.
                    config.data_dir / "shortlists" / "_chat.json",
                    db=config.offer_db,
                )

            with config.chat_lock:
                if not config.chat_messages:
                    config.chat_messages.append(
                        {"role": "system", "content": config.chat_system_prompt()}
                    )
                # Remember the lengths so a failed run leaves no half-finished turn.
                messages_before = len(config.chat_messages)
                view_before = len(config.chat_view)
                config.chat_messages.append({"role": "user", "content": text})
                config.chat_view.append({"role": "user", "text": text})
                try:
                    answer = run_profile_agent(
                        config.client_factory(),
                        config.chat_messages,
                        trace_path,
                        draft_path=config.draft_path,
                        profiles_dir=config.profiles_dir,
                        search=chat_search,
                        extra_tools=None if config.offer_db is None else CHAT_DB_TOOLS,
                        extra_dispatch=None if config.offer_db is None else chat_browse,
                    )
                except ValueError as error:
                    del config.chat_messages[messages_before:]
                    del config.chat_view[view_before:]
                    self._send_json(500, {"error": str(error)})
                    return
                except RuntimeError as error:
                    del config.chat_messages[messages_before:]
                    del config.chat_view[view_before:]
                    self._send_json(502, {"error": str(error)})
                    return
                except Exception:
                    del config.chat_messages[messages_before:]
                    del config.chat_view[view_before:]
                    self._send_json(
                        500, {"error": "Asystent nie zakończył odpowiedzi."}
                    )
                    return
                config.chat_view.append({"role": "assistant", "text": answer or ""})
                if found:
                    config.chat_search = found["search"]
                state = config.chat_state()

            # "search" is this turn's search only, so the page redraws it once.
            self._send_json(
                200, {"answer": answer or "", **state, "search": found.get("search")}
            )

        def _chat_confirm(self) -> None:
            # The button click is the user's confirmation; the model cannot press it.
            # The chat goes on: the draft stays, so later edits update this profile.
            with config.chat_lock:
                result = finalize_profile(
                    config.draft_path,
                    config.profiles_dir,
                    confirmed=True,
                    profile_id=config.chat_profile_id,
                    keep_draft=True,
                )
                if result["status"] == "saved":
                    config.chat_profile_id = result["profile_id"]
                    note = SAVED_MESSAGE.format(limit=MAX_SEARCH_LIMIT)
                    config.add_assistant_note(note)
                    result = {**result, "message": note}
            if result["status"] == "saved":
                self._send_json(200, result)
            else:
                missing = ", ".join(result.get("missing_fields", []))
                self._send_json(
                    400,
                    {"error": f"Brakuje jeszcze: {missing}.", **result},
                )

        def _chat_load(self, data: dict[str, Any]) -> None:
            # Starts a new chat around a saved profile: edits and searches use it.
            profile_id = data.get("profile_id")
            if not isinstance(profile_id, str):
                self._send_json(400, {"error": "Wybierz profil."})
                return
            try:
                profile = load_profile_by_id(config.profiles_dir, profile_id)
            except KeyError:
                self._send_json(404, {"error": "Nie ma takiego profilu."})
                return
            with config.chat_lock:
                config.reset_chat()
                load_profile_into_draft(profile, config.draft_path)
                config.chat_profile_id = profile_id
                config.add_assistant_note(
                    LOADED_MESSAGE.format(name=profile.name, limit=MAX_SEARCH_LIMIT)
                )
                view = list(config.chat_view)
                state = config.chat_state()
            self._send_json(200, {"messages": view, **state})

        def _search(self, data: dict[str, Any]) -> None:
            profile_id = data.get("profile_id")
            if not isinstance(profile_id, str):
                self._send_json(400, {"error": "Wybierz profil."})
                return
            try:
                limit = _parse_limit(data.get("limit"))
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            try:
                profile = load_profile_by_id(config.profiles_dir, profile_id)
            except KeyError:
                self._send_json(404, {"error": "Nie ma takiego profilu."})
                return

            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            trace_path = (
                config.data_dir.parent / "traces" / f"ui_{profile_id}_{stamp}.jsonl"
            )
            if config.offer_db is not None:
                self._rate_search(profile_id, profile, limit, trace_path)
                return

            # Without the database: the agent searches job_offers.json.
            matches = _find_matches(config, profile, limit)
            offers = load_offers(config.data_dir / "job_offers.json")

            def search(_profile, limit: int, found=matches):
                return found[:limit]

            messages = [
                {"role": "system", "content": system_prompt(None)},
                {
                    "role": "user",
                    "content": (
                        f"Pokaż {limit} ofert pasujących do profilu: {profile.name}."
                    ),
                },
            ]
            try:
                with config.search_lock:
                    answer = run_agent(
                        config.client_factory(),
                        messages,
                        profile,
                        offers,
                        trace_path,
                        shortlist_path=_shortlist_path(config, profile_id),
                        search=search,
                        max_cost_usd=MAX_RUN_COST_USD,
                    )
            except ValueError as error:
                # Missing key or model name: safe to show, it names the setting.
                self._send_json(500, {"error": str(error)})
                return
            except RuntimeError as error:
                self._send_run_error(error, trace_path)
                return
            except Exception:
                self._send_json(500, {"error": "Agent nie zakończył pracy."})
                return
            self._send_json(
                200,
                {
                    "profile": {"id": profile_id, **profile.model_dump()},
                    "answer": answer,
                    "picked": False,
                    "matches": [_match_to_dict(m) for m in matches[:limit]],
                    "trace": _trace_name(trace_path, config),
                    "steps": _trace_steps(trace_path),
                },
            )

        def _rate_search(
            self, profile_id: str, profile: Profile, limit: int, trace_path: Path
        ) -> None:
            """The ranking picks CANDIDATE_POOL offers, the model rates each one
            against the profile (agent_project.judge) and the `limit` best are
            shown, with a score from 0 to 1."""
            try:
                matches = config.offer_db.match_profile(profile, CANDIDATE_POOL)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            try:
                with config.search_lock:
                    rated = rate_offers(
                        config.client_factory(),
                        profile,
                        matches,
                        config.offer_db,
                        trace_path,
                        max_cost_usd=MAX_RUN_COST_USD,
                    )
            except ValueError as error:
                self._send_json(500, {"error": str(error)})
                return
            except RuntimeError as error:
                self._send_run_error(error, trace_path)
                return
            except Exception:
                self._send_json(500, {"error": "Ocena ofert nie powiodła się."})
                return
            rows = [
                {**_match_to_dict(match), **rating}
                for match, rating in best_fits(matches, rated, limit)
            ]
            # The offers shown stay readable on the shortlist later.
            shown = [config.offer_db.offer(row["id"]) for row in rows]
            remember_offers(config.seen_offers_path, [o for o in shown if o])
            self._send_json(
                200,
                {
                    "profile": {"id": profile_id, **profile.model_dump()},
                    "answer": summary(matches, rated, len(rows)),
                    # True: the model rated the candidates.
                    "picked": True,
                    "candidates": len(matches),
                    "matches": rows,
                    "trace": _trace_name(trace_path, config),
                    "steps": _trace_steps(trace_path),
                },
            )

        def _send_run_error(self, error: RuntimeError, trace_path: Path) -> None:
            self._send_json(
                502,
                {
                    "error": str(error),
                    "trace": _trace_name(trace_path, config),
                    "steps": _trace_steps(trace_path),
                },
            )

        def _shortlist(self, profile_id: str) -> None:
            try:
                shortlist_path = _shortlist_path(config, profile_id)
            except KeyError:
                self._send_json(404, {"error": "Nie ma takiego profilu."})
                return
            offers = _known_offers(config)
            ids = load_shortlist(shortlist_path)
            by_id = {offer.id: offer for offer in offers}
            self._send_json(
                200,
                {
                    "offers": [
                        {
                            "id": oid,
                            "title": by_id[oid].title,
                            "company": by_id[oid].company,
                        }
                        for oid in ids
                        if oid in by_id
                    ]
                },
            )

        def _shortlist_change(self, path: str, data: dict[str, Any]) -> None:
            offer_id = data.get("offer_id")
            if not isinstance(offer_id, str) or not offer_id:
                self._send_json(400, {"error": "Podaj id oferty."})
                return
            try:
                shortlist_path = _shortlist_path(config, data.get("profile_id"))
            except KeyError:
                self._send_json(404, {"error": "Nie ma takiego profilu."})
                return
            if path == "/api/shortlist/add":
                result = add_to_shortlist(
                    offer_id, _known_offers(config, offer_id), shortlist_path
                )
            else:
                # The browser sends confirmed=true only after the user clicked
                # "Usuń" and confirmed in the dialog; the model never calls this.
                result = remove_from_shortlist(
                    offer_id, shortlist_path, confirmed=data.get("confirmed") is True
                )
            self._send_json(200, result.model_dump())

    return Handler


def make_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    data_dir: Path = DATA_DIR,
    client_factory: Callable[[], ModelClient] = OpenRouterClient,
    offer_db: OfferDB | None = None,
    auth: Auth | None = None,
    secure_cookie: bool = True,
) -> ThreadingHTTPServer:
    """`auth=None` turns the login off; only tests do that."""
    config = _Config(data_dir, client_factory, offer_db, auth, secure_cookie)
    return ThreadingHTTPServer((host, port), _make_handler(config))


def main() -> None:
    load_dotenv()
    path = account_path()
    if not path.exists():
        sys.exit(f"Brak konta ({path}). Załóż je: uv run python -m agent_project.auth")
    auth = Auth(Account.load(path))
    try:
        offer_db = OfferDB.open()
    except FileNotFoundError:
        sys.exit(
            "Brak indeksu ofert. Zbuduj go: uv run python -m offer_server.update "
            "--seed data/offers_sample_justjoin.json data/offers_sample_rocketjobs.json"
        )
    host = os.getenv("CVPILOT_HOST", "127.0.0.1")
    port = int(os.getenv("CVPILOT_PORT", "8000"))
    # Browsers keep Secure cookies on http://127.0.0.1 too; 0 only for old ones.
    secure = os.getenv("CVPILOT_SECURE_COOKIE", "1") != "0"
    server = make_server(host, port, offer_db=offer_db, auth=auth, secure_cookie=secure)
    stats = offer_db.stats()
    print(f"Baza ofert: {stats['offers']} ofert, indeks z {stats['built_at']}.")
    print(f"CVPilot: http://{host}:{port} (konto {auth.account.username}).")
    print("Zatrzymanie: Ctrl+C.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
