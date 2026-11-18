"""Builds the offer index from raw offer files.

    uv run python -m offer_server.build_index data/offers_sample_*.json

Accepts offers_sample_*.json (a JSON list) and fetch_all output (*.jsonl).
The running web app loads the new index by itself.
"""

import argparse
from datetime import date
from pathlib import Path

from offer_server.documents import clean_offers, load_raw_offers
from offer_server.embedder import FastEmbedder
from offer_server.index import build_index, save_index
from offer_server.paths import index_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Buduje indeks ofert.")
    parser.add_argument("inputs", nargs="+", type=Path, help="pliki z ofertami")
    parser.add_argument("--out", type=Path, default=index_dir())
    parser.add_argument("--model", help="model fastembed (domyślnie z embedder.py)")
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=date.today(),
        help="oferty ważne krócej niż do tego dnia są pomijane (RRRR-MM-DD)",
    )
    args = parser.parse_args()

    offers, stats = clean_offers(load_raw_offers(args.inputs), args.today)
    print(
        f"Wczytano {stats['read']}, wygasłe {stats['expired']}, "
        f"duplikaty {stats['duplicates']}, zostało {stats['kept']} "
        f"(z sekcją wymagań: {stats['with_requirements']})."
    )
    embedder = FastEmbedder(args.model)
    print(f"Wektoryzacja modelem {embedder.name}…", flush=True)
    index = build_index(offers, embedder)
    index.meta["stats"] = stats
    save_index(index, args.out)
    print(f"Zapisano {index.meta['pieces']} wektorów w {args.out}.")


if __name__ == "__main__":
    main()
