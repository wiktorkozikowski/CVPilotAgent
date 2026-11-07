"""Local web UI: add a profile, or search offers for a saved profile with the agent.

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

from pydantic import ValidationError

from agent_project.agent import SYSTEM_PROMPT, run_agent
from agent_project.llm import ModelClient, OpenRouterClient
from agent_project.matching import search_offers
from agent_project.models import Profile, load_offers
from agent_project.profiles import list_profiles, load_profile_by_id, save_profile

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 64 * 1024
SEARCH_LIMIT = 10


class _Config:
    def __init__(
        self,
        data_dir: Path,
        client_factory: Callable[[], ModelClient],
    ) -> None:
        self.data_dir = data_dir
        self.profiles_dir = data_dir / "profiles"
        self.client_factory = client_factory
        # One agent run at a time: the model client and trace files are not shared.
        self.search_lock = threading.Lock()


def _match_to_dict(match) -> dict[str, Any]:
    return {
        "id": match.offer.id,
        "title": match.offer.title,
        "company": match.offer.company,
        "location": match.offer.location,
        "url": match.offer.url,
        "matched_skills": match.matched_skills,
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

            if self.path == "/api/profiles":
                self._create_profile(data)
            elif self.path == "/api/search":
                self._search(data)
            else:
                self._send_json(404, {"error": "Nie znaleziono."})

        def _create_profile(self, data: dict[str, Any]) -> None:
            try:
                profile = Profile.model_validate(data)
            except ValidationError as error:
                self._send_json(
                    400,
                    {
                        "error": "Popraw dane profilu.",
                        "details": error.errors(include_url=False, include_input=False),
                    },
                )
                return
            profile_id = save_profile(config.profiles_dir, profile)
            self._send_json(201, {"id": profile_id, **profile.model_dump()})

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

            offers = load_offers(config.data_dir / "job_offers.json")
            matches = search_offers(profile, offers, limit=SEARCH_LIMIT)

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
                        shortlist_path=config.data_dir / "shortlist.json",
                    )
            except ValueError as error:
                # Missing key or model name: safe to show, it names the setting.
                self._send_json(500, {"error": str(error)})
                return
            except RuntimeError as error:
                self._send_json(502, {"error": str(error)})
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
                    "trace": str(trace_path.relative_to(config.data_dir.parent)),
                },
            )

    return Handler


def make_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    data_dir: Path = DATA_DIR,
    client_factory: Callable[[], ModelClient] = OpenRouterClient,
) -> ThreadingHTTPServer:
    config = _Config(data_dir, client_factory)
    return ThreadingHTTPServer((host, port), _make_handler(config))


def main() -> None:
    server = make_server()
    print("Otwórz http://127.0.0.1:8000 w przeglądarce. Zatrzymaj: Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
