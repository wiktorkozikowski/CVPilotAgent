"""Turns texts into unit-length vectors.

`FastEmbedder` runs a local ONNX model (no data leaves the server). The model is
downloaded once into the cache folder. `HashingEmbedder` needs no model and is
used in tests.
"""

import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from offer_server.paths import model_cache_dir
from offer_server.text import plain

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """One row per text, each of length 1."""
        ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


class FastEmbedder:
    def __init__(self, model: str | None = None, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding  # heavy import, only when used

        self.name = model or os.getenv("OFFERS_MODEL") or DEFAULT_MODEL
        cache = cache_dir or model_cache_dir()
        self._model = TextEmbedding(model_name=self.name, cache_dir=str(cache))

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed(texts, batch_size=32))))


class HashingEmbedder:
    """Bag of words hashed into a fixed vector: crude, but fast and offline."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in plain(text).split():
                word = word.strip(".,:;()!?\"'")
                if word:
                    digest = hashlib.md5(word.encode()).digest()
                    vectors[row, int.from_bytes(digest[:4], "little") % self.dim] += 1
        return _normalize(vectors)
