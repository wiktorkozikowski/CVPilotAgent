"""HTTP API of the offer server (standard library only).

    GET  /health   -> {"status": "ok", "offers": ..., "model": ..., "built_at": ...}
    POST /search   {"profile": {...}, "limit": 10, "query": "..."}
                   -> {"status": "ok", "candidates": [...]}

Settings come from the environment:
    OFFER_SERVER_HOST   (default 127.0.0.1)
    OFFER_SERVER_PORT   (default 8100)
    OFFER_SERVER_INDEX  (default ./index)
    OFFER_SERVER_TOKEN  required when the host is not local; clients send it as
                        "Authorization: Bearer <token>"

Profiles are never written to disk or to the log.
"""

import hmac
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from offer_server.embedder import Embedder, FastEmbedder
from offer_server.index import OfferIndex, load_index
from offer_server.ranking import MAX_LIMIT, SearchProfile, search

MAX_BODY_BYTES = 64 * 1024
MAX_QUERY_CHARS = 200
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
MIN_TOKEN_CHARS = 16
PLACEHOLDER_TOKEN = "change-me"  # the value in deploy/offer-server.env.example


def _make_handler(
    index: OfferIndex, embedder: Embedder, token: str | None
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Only the request line, never the body with the profile.
            sys.stderr.write(f"{self.address_string()} {format % args}\n")

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if token is None:
                return True
            sent = self.headers.get("Authorization", "")
            return hmac.compare_digest(sent.encode(), f"Bearer {token}".encode())

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send(404, {"status": "error", "message": "Nie znaleziono."})
                return
            self._send(
                200,
                {
                    "status": "ok",
                    "offers": len(index.offers),
                    "model": index.meta.get("model"),
                    "built_at": index.meta.get("built_at"),
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/search":
                self._send(404, {"status": "error", "message": "Nie znaleziono."})
                return
            if not self._authorized():
                self._send(401, {"status": "error", "message": "Brak dostępu."})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 < length <= MAX_BODY_BYTES:
                self._send(
                    400, {"status": "error", "message": "Złe lub za duże żądanie."}
                )
                return
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send(400, {"status": "error", "message": "To nie jest JSON."})
                return
            if not isinstance(data, dict):
                self._send(400, {"status": "error", "message": "Oczekiwano obiektu."})
                return

            limit = data.get("limit", 10)
            query = data.get("query")
            if isinstance(limit, bool) or not isinstance(limit, int):
                limit = -1
            if not 1 <= limit <= MAX_LIMIT:
                message = f"limit musi być liczbą od 1 do {MAX_LIMIT}."
                self._send(400, {"status": "error", "message": message})
                return
            if query is not None and (
                not isinstance(query, str) or len(query) > MAX_QUERY_CHARS
            ):
                message = f"query to tekst do {MAX_QUERY_CHARS} znaków."
                self._send(400, {"status": "error", "message": message})
                return
            try:
                profile = SearchProfile.model_validate(data.get("profile"))
            except ValidationError as error:
                first = error.errors()[0]
                where = ".".join(str(part) for part in first["loc"])
                message = f"Niepoprawny profil ({where}): {first['msg']}"
                self._send(400, {"status": "error", "message": message})
                return
            try:
                found = search(index, embedder, profile, limit, query or None)
            except ValueError as error:
                self._send(400, {"status": "error", "message": str(error)})
                return
            self._send(
                200,
                {
                    "status": "ok",
                    "model": index.meta.get("model"),
                    "candidates": [c.model_dump() for c in found],
                },
            )

    return Handler


def make_server(
    index: OfferIndex,
    embedder: Embedder,
    host: str = "127.0.0.1",
    port: int = 8100,
    token: str | None = None,
) -> ThreadingHTTPServer:
    if host not in LOCAL_HOSTS and (
        not token or len(token) < MIN_TOKEN_CHARS or token == PLACEHOLDER_TOKEN
    ):
        raise ValueError(
            "Serwer dostępny z sieci wymaga OFFER_SERVER_TOKEN: własnego, "
            f"co najmniej {MIN_TOKEN_CHARS} znaków (nagłówek Bearer)."
        )
    if index.meta.get("model") != embedder.name:
        raise ValueError(
            f"Indeks zbudowano modelem {index.meta.get('model')}, a serwer używa "
            f"{embedder.name}. Przebuduj indeks albo zmień OFFER_SERVER_MODEL."
        )
    return ThreadingHTTPServer((host, port), _make_handler(index, embedder, token))


def main() -> None:
    host = os.getenv("OFFER_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("OFFER_SERVER_PORT", "8100"))
    index_dir = Path(os.getenv("OFFER_SERVER_INDEX", "index"))
    token = os.getenv("OFFER_SERVER_TOKEN") or None

    index = load_index(index_dir)
    embedder = FastEmbedder(index.meta.get("model"))
    server = make_server(index, embedder, host, port, token)
    print(
        f"Serwer ofert: http://{host}:{port}  ({len(index.offers)} ofert, "
        f"model {embedder.name}). Zatrzymanie: Ctrl+C.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
