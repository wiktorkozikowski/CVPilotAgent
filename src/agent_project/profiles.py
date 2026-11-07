"""Stores several user profiles as separate JSON files in a folder."""

import json
import re
import unicodedata
from pathlib import Path

from agent_project.models import Profile

PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug[:64].strip("-") or "profil"


def list_profiles(profiles_dir: Path) -> list[tuple[str, Profile]]:
    """Return (id, profile) pairs sorted by id; an empty list if no folder yet."""
    if not profiles_dir.is_dir():
        return []
    result = []
    for path in sorted(profiles_dir.glob("*.json")):
        result.append((path.stem, Profile.model_validate_json(path.read_text("utf-8"))))
    return result


def load_profile_by_id(profiles_dir: Path, profile_id: str) -> Profile:
    """Load one profile; raises KeyError if the id is unknown or malformed."""
    if not PROFILE_ID_PATTERN.fullmatch(profile_id):
        raise KeyError(profile_id)
    path = profiles_dir / f"{profile_id}.json"
    if not path.is_file():
        raise KeyError(profile_id)
    return Profile.model_validate_json(path.read_text(encoding="utf-8"))


def save_profile(profiles_dir: Path, profile: Profile) -> str:
    """Save a new profile under a unique id derived from its name; returns the id."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    base = _slugify(profile.name)
    profile_id = base
    counter = 2
    while (profiles_dir / f"{profile_id}.json").exists():
        profile_id = f"{base}-{counter}"
        counter += 1
    path = profiles_dir / f"{profile_id}.json"
    path.write_text(
        json.dumps(profile.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile_id
