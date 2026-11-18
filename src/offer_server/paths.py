"""Where the offer store, the index and the embedding model live.

Defaults are folders in `var/` at the project root; the environment can move
them (e.g. to a bigger disk on the server).
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
VAR_DIR = ROOT_DIR / "var"


def _setting(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def index_dir() -> Path:
    return _setting("OFFERS_INDEX_DIR", VAR_DIR / "index")


def store_dir() -> Path:
    return _setting("OFFERS_STORE_DIR", VAR_DIR / "store")


def model_cache_dir() -> Path:
    return _setting("OFFERS_MODEL_CACHE", VAR_DIR / "models_cache")
