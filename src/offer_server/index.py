"""The offer index on disk: cleaned offers plus one vector per text piece.

Everything sits in one file, `index.npz`, written to a temporary name and then
renamed. A running server therefore never reads a half-written index.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from offer_server.documents import OfferDoc, chunks
from offer_server.embedder import Embedder

INDEX_FILE = "index.npz"


@dataclass
class OfferIndex:
    offers: list[OfferDoc]
    vectors: np.ndarray  # one row per piece, unit length
    owners: np.ndarray  # for each row, the position of its offer in `offers`
    meta: dict
    # For each offer, a hash of the texts that were embedded; equal hash means
    # the vectors can be reused when the index is rebuilt.
    fingerprints: list[str] = field(default_factory=list)

    def offer_scores(self, query: np.ndarray) -> np.ndarray:
        """For each offer, the similarity of its piece closest to `query`."""
        best = np.full(len(self.offers), -1.0, dtype=np.float32)
        if self.offers:
            np.maximum.at(best, self.owners, self.vectors @ query)
        return best

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """The `k` offers whose best piece is closest to `query`, best first."""
        best = self.offer_scores(query)
        order = np.argsort(-best)[:k]
        return [(int(i), float(best[i])) for i in order]


def fingerprint(pieces: list[str]) -> str:
    return hashlib.sha256("\x1f".join(pieces).encode("utf-8")).hexdigest()


def build_index(
    offers: list[OfferDoc], embedder: Embedder, previous: OfferIndex | None = None
) -> OfferIndex:
    """Embed the offers. Vectors of offers whose text has not changed since
    `previous` (built with the same model) are copied instead of recomputed."""
    reusable: dict[str, np.ndarray] = {}
    if previous is not None and previous.meta.get("model") == embedder.name:
        for position, print_ in enumerate(previous.fingerprints):
            reusable[print_] = previous.vectors[previous.owners == position]

    rows: list[np.ndarray | None] = []  # per offer: reused vectors or None
    owners: list[int] = []
    prints: list[str] = []
    to_embed: list[str] = []
    for position, offer in enumerate(offers):
        pieces = chunks(offer)
        print_ = fingerprint(pieces)
        prints.append(print_)
        owners.extend([position] * len(pieces))
        if print_ in reusable:
            rows.append(reusable[print_])
        else:
            rows.append(None)
            to_embed.extend(pieces)

    fresh = embedder.embed(to_embed) if to_embed else None
    blocks = []
    taken = 0
    for position, offer in enumerate(offers):
        if rows[position] is not None:
            blocks.append(rows[position])
        else:
            count = len(chunks(offer))
            blocks.append(fresh[taken : taken + count])
            taken += count
    vectors = np.vstack(blocks).astype(np.float32) if blocks else np.zeros((0, 1))
    meta = {
        "model": embedder.name,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "offers": len(offers),
        "pieces": len(owners),
        "embedded_pieces": len(to_embed),
    }
    return OfferIndex(offers, vectors, np.array(owners, dtype=np.int64), meta, prints)


def _json_bytes(value) -> np.ndarray:
    return np.frombuffer(
        json.dumps(value, ensure_ascii=False).encode("utf-8"), dtype=np.uint8
    )


def save_index(index: OfferIndex, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / INDEX_FILE
    temporary = folder / (INDEX_FILE + ".tmp")
    with temporary.open("wb") as file:
        np.savez(
            file,
            vectors=index.vectors,
            owners=index.owners,
            offers=_json_bytes([offer.model_dump() for offer in index.offers]),
            fingerprints=_json_bytes(index.fingerprints),
            meta=_json_bytes(index.meta),
        )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, target)
    return target


def index_path(folder: Path) -> Path:
    return folder / INDEX_FILE


def load_index(folder: Path) -> OfferIndex:
    with np.load(index_path(folder), allow_pickle=False) as data:

        def decoded(name: str):
            return json.loads(data[name].tobytes().decode("utf-8"))

        return OfferIndex(
            offers=[OfferDoc.model_validate(item) for item in decoded("offers")],
            vectors=data["vectors"],
            owners=data["owners"],
            meta=decoded("meta"),
            fingerprints=decoded("fingerprints"),
        )
