"""The offer index on disk: cleaned offers plus one vector per text piece."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from offer_server.documents import OfferDoc, chunks
from offer_server.embedder import Embedder

OFFERS_FILE = "offers.json"
VECTORS_FILE = "vectors.npy"
OWNERS_FILE = "owners.npy"
META_FILE = "meta.json"


@dataclass
class OfferIndex:
    offers: list[OfferDoc]
    vectors: np.ndarray  # one row per piece, unit length
    owners: np.ndarray  # for each row, the position of its offer in `offers`
    meta: dict

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


def build_index(offers: list[OfferDoc], embedder: Embedder) -> OfferIndex:
    texts: list[str] = []
    owners: list[int] = []
    for position, offer in enumerate(offers):
        for text in chunks(offer):
            texts.append(text)
            owners.append(position)
    vectors = embedder.embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)
    meta = {
        "model": embedder.name,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "offers": len(offers),
        "pieces": len(texts),
    }
    return OfferIndex(offers, vectors, np.array(owners, dtype=np.int64), meta)


def save_index(index: OfferIndex, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / OFFERS_FILE).write_text(
        json.dumps([offer.model_dump() for offer in index.offers], ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(folder / VECTORS_FILE, index.vectors)
    np.save(folder / OWNERS_FILE, index.owners)
    (folder / META_FILE).write_text(
        json.dumps(index.meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(folder: Path) -> OfferIndex:
    offers = [
        OfferDoc.model_validate(item)
        for item in json.loads((folder / OFFERS_FILE).read_text(encoding="utf-8"))
    ]
    return OfferIndex(
        offers=offers,
        vectors=np.load(folder / VECTORS_FILE),
        owners=np.load(folder / OWNERS_FILE),
        meta=json.loads((folder / META_FILE).read_text(encoding="utf-8")),
    )
