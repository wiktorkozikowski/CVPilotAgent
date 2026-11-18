"""The index the web app searches, swapped for a newer file when one appears.

offer_server.update writes a new index file every hour. `IndexHolder.get()`
checks the file at most every RELOAD_CHECK_SECONDS and loads a newer one, so
the app needs no restart. A broken or mismatched new file is logged and the old
index keeps serving.
"""

import sys
import threading
import time
from pathlib import Path

from offer_server.embedder import Embedder
from offer_server.index import OfferIndex, index_path, load_index

RELOAD_CHECK_SECONDS = 30


class IndexHolder:
    def __init__(
        self, index: OfferIndex, embedder: Embedder, folder: Path | None = None
    ) -> None:
        if index.meta.get("model") != embedder.name:
            raise ValueError(
                f"Indeks zbudowano modelem {index.meta.get('model')}, a aplikacja "
                f"używa {embedder.name}. Przebuduj indeks albo zmień OFFERS_MODEL."
            )
        self.index = index
        self.embedder = embedder
        self.folder = folder
        self._stamp = self._file_stamp()
        self._checked = time.monotonic()
        self._lock = threading.Lock()

    def _file_stamp(self) -> float | None:
        if self.folder is None:
            return None
        try:
            return index_path(self.folder).stat().st_mtime
        except OSError:
            return None

    def get(self) -> OfferIndex:
        if self.folder is None:
            return self.index
        now = time.monotonic()
        if now - self._checked < RELOAD_CHECK_SECONDS:
            return self.index
        with self._lock:
            if now - self._checked < RELOAD_CHECK_SECONDS:
                return self.index
            self._checked = now
            stamp = self._file_stamp()
            if stamp is not None and stamp != self._stamp:
                self._stamp = stamp
                self.reload()
        return self.index

    def reload(self) -> None:
        try:
            fresh = load_index(self.folder)
        except (OSError, ValueError, KeyError) as error:
            sys.stderr.write(f"Nowego indeksu nie da się wczytać: {error}\n")
            return
        if fresh.meta.get("model") != self.embedder.name:
            sys.stderr.write(
                f"Nowy indeks ma model {fresh.meta.get('model')}, aplikacja używa "
                f"{self.embedder.name}; zostaje stary indeks.\n"
            )
            return
        self.index = fresh
        sys.stderr.write(
            f"Wczytano nowy indeks: {len(fresh.offers)} ofert "
            f"({fresh.meta.get('built_at')}).\n"
        )
