"""Local web UI: build a profile in a chat, or search offers for a saved profile.

Run with `uv run python -m agent_project.web` and open http://127.0.0.1:8000.
Only the standard library is used for the server; the page is one HTML file.
"""

import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from agent_project.agent import SYSTEM_PROMPT, run_agent
from agent_project.llm import ModelClient, OpenRouterClient
from agent_project.matching import search_offers
from agent_project.models import Offer, load_offers
from agent_project.offers_client import (
    OffersServerClient,
    OffersServerError,
    load_seen_offers,
    remember_offers,
)
from agent_project.profile_tools import (
    PROFILE_SYSTEM_PROMPT,
    finalize_profile,
    get_draft,
    remove_item,
    run_profile_agent,
)
from agent_project.profiles import list_profiles, load_profile_by_id
from agent_project.shortlist import (
    add_to_shortlist,
    load_shortlist,
    remove_from_shortlist,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 64 * 1024
MAX_CHAT_CHARS = 2000
SEARCH_LIMIT = 10


class _Config:
    def __init__(
        self,
        data_dir: Path,
        client_factory: Callable[[], ModelClient],
        offers_client: OffersServerClient | None = None,
    ) -> None:
        self.data_dir = data_dir
        # Set: offers come from the offer server. None: from job_offers.json.
        self.offers_client = offers_client
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

    def reset_chat(self) -> None:
        self.chat_messages.clear()
        self.chat_view.clear()
        self.draft_path.unlink(missing_ok=True)


def _known_offers(config: "_Config") -> list[Offer]:
    """Offers a shortlist may hold: the local ones and those the server returned."""
    offers = load_offers(config.data_dir / "job_offers.json")
    local = {offer.id for offer in offers}
    offers += [
        o for o in load_seen_offers(config.seen_offers_path) if o.id not in local
    ]
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

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
            if self.path == "/":
                body = (STATIC_DIR / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/api/shortlist?") or self.path == "/api/shortlist":
                query = parse_qs(urlparse(self.path).query)
                self._shortlist(query.get("profile_id", [""])[0])
                return
            if self.path == "/api/chat":
                with config.chat_lock:
                    view = list(config.chat_view)
                self._send_json(200, {"messages": view, **get_draft(config.draft_path)})
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
            try:
                data = self._read_json()
            except (ValueError, json.JSONDecodeError):
                self._send_json(400, {"error": "Nieprawidłowe dane JSON."})
                return

            if self.path == "/api/chat/message":
                self._chat_message(data)
            elif self.path == "/api/chat/confirm":
                self._chat_confirm()
            elif self.path == "/api/chat/reset":
                with config.chat_lock:
                    config.reset_chat()
                self._send_json(200, {"status": "reset"})
            elif self.path == "/api/profile/remove":
                self._profile_remove(data)
            elif self.path == "/api/search":
                self._search(data)
            elif self.path in ("/api/shortlist/add", "/api/shortlist/remove"):
                self._shortlist_change(self.path, data)
            else:
                self._send_json(404, {"error": "Nie znaleziono."})

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
            with config.chat_lock:
                if not config.chat_messages:
                    config.chat_messages.append(
                        {"role": "system", "content": PROFILE_SYSTEM_PROMPT}
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
                draft = get_draft(config.draft_path)

            self._send_json(200, {"answer": answer or "", **draft})

        def _chat_confirm(self) -> None:
            # The button click is the user's confirmation; the model cannot press it.
            with config.chat_lock:
                result = finalize_profile(
                    config.draft_path, config.profiles_dir, confirmed=True
                )
                if result["status"] == "saved":
                    config.chat_messages.clear()
                    config.chat_view.clear()
            if result["status"] == "saved":
                self._send_json(200, result)
            else:
                missing = ", ".join(result.get("missing_fields", []))
                self._send_json(
                    400,
                    {"error": f"Brakuje jeszcze: {missing}.", **result},
                )

        def _search(self, data: dict[str, Any]) -> None:
            profile_id = data.get("profile_id")
            if not isinstance(profile_id, str):
                self._send_json(400, {"error": "Wybierz profil."})
                return
            try:
                profile = load_profile_by_id(config.profiles_dir, profile_id)
            except KeyError:
                self._send_json(404, {"error": "Nie ma takiego profilu."})
                return

            search = None
            if config.offers_client is None:
                offers = load_offers(config.data_dir / "job_offers.json")
                matches = search_offers(profile, offers, limit=SEARCH_LIMIT)
            else:
                # One call to the server: the table below and the agent's tool
                # both use these candidates.
                try:
                    matches = config.offers_client.search(profile, SEARCH_LIMIT)
                except OffersServerError as error:
                    self._send_json(502, {"error": str(error)})
                    return
                remember_offers(config.seen_offers_path, [m.offer for m in matches])
                offers = _known_offers(config)

                def search(_profile, limit: int, found=matches):
                    return found[:limit]

            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            trace_path = (
                config.data_dir.parent / "traces" / f"ui_{profile_id}_{stamp}.jsonl"
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Pokaż oferty pasujące do profilu: {profile.name}.",
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
                    )
            except ValueError as error:
                # Missing key or model name: safe to show, it names the setting.
                self._send_json(500, {"error": str(error)})
                return
            except RuntimeError as error:
                self._send_json(
                    502,
                    {
                        "error": str(error),
                        "trace": _trace_name(trace_path, config),
                        "steps": _trace_steps(trace_path),
                    },
                )
                return
            except Exception:
                self._send_json(500, {"error": "Agent nie zakończył pracy."})
                return

            self._send_json(
                200,
                {
                    "profile": {"id": profile_id, **profile.model_dump()},
                    "answer": answer,
                    "matches": [_match_to_dict(m) for m in matches],
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
                    offer_id, _known_offers(config), shortlist_path
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
    offers_client: OffersServerClient | None = None,
) -> ThreadingHTTPServer:
    config = _Config(data_dir, client_factory, offers_client)
    return ThreadingHTTPServer((host, port), _make_handler(config))


def main() -> None:
    offers_client = OffersServerClient.from_env()
    server = make_server(offers_client=offers_client)
    source = offers_client.url if offers_client else "data/job_offers.json"
    print(f"Oferty z: {source}")
    print("Otwórz http://127.0.0.1:8000 w przeglądarce. Zatrzymaj: Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
