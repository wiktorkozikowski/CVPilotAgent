"""The user's shortlist of chosen offers, stored as a JSON list of offer ids."""

import json
from pathlib import Path

from agent_project.models import Offer, ShortlistResult


def load_shortlist(path: Path) -> list[str]:
    """Read the ids of offers already on the shortlist; empty if none saved yet."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def add_to_shortlist(offer_id: str, offers: list[Offer], path: Path) -> ShortlistResult:
    """Add an offer that exists in `offers` to the shortlist file, once."""
    if offer_id not in {offer.id for offer in offers}:
        return ShortlistResult(status="not_found", offer_id=offer_id)

    shortlist = load_shortlist(path)
    if offer_id in shortlist:
        return ShortlistResult(status="already_exists", offer_id=offer_id)

    shortlist.append(offer_id)
    _save_shortlist(path, shortlist)
    return ShortlistResult(status="added", offer_id=offer_id)


def remove_from_shortlist(
    offer_id: str, path: Path, confirmed: bool
) -> ShortlistResult:
    """Remove an offer from the shortlist, but only after the user confirmed.

    `confirmed` must come from the application, never from the model's arguments.
    `not_found` here means the offer is not on the shortlist.
    """
    if not confirmed:
        return ShortlistResult(status="confirmation_required", offer_id=offer_id)

    shortlist = load_shortlist(path)
    if offer_id not in shortlist:
        return ShortlistResult(status="not_found", offer_id=offer_id)

    shortlist.remove(offer_id)
    _save_shortlist(path, shortlist)
    return ShortlistResult(status="removed", offer_id=offer_id)


def _save_shortlist(path: Path, shortlist: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(shortlist, ensure_ascii=False, indent=2), encoding="utf-8"
    )
