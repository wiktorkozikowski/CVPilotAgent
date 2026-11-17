"""Pobieranie pełnego zbioru ofert z sitemap (JustJoin, RocketJobs).

Uruchamiaj wyłącznie po potwierdzeniu zgody na piśmie (zakres, cel, okres).

Bez flagi --i-have-written-consent skrypt działa w trybie podglądu: pobiera
tylko sitemapę i podaje liczbę adresów oraz szacowany czas.

Przykłady:
  python -m agent_project.fetch_all justjoin --out ~/raw_offers
  python -m agent_project.fetch_all justjoin --out ~/raw_offers --i-have-written-consent

Wynik: plik JSON Lines <out>/<serwis>_offers.jsonl, jeden rekord na linię.
Katalog --out trzymaj poza repozytorium (dane nie mogą trafić do git).
"""

import argparse
import json
import time
import urllib.error
from pathlib import Path

from agent_project.fetch_offers import SITES, fetch, offer_urls, parse_offer

# Limit uzgodniony z firmami: nie więcej niż 1 zapytanie na sekundę.
DELAY_SECONDS = 1.0
# Odpowiedzi, przy których przerywamy, zamiast dalej obciążać serwer.
STOP_STATUSES = {429, 500, 502, 503, 504}


def load_done(path: Path) -> set[str]:
    """Adresy ofert już zapisane; ucięta ostatnia linia po przerwaniu jest pomijana."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                done.add(json.loads(line)["url"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Pełne pobranie ofert z sitemap.")
    parser.add_argument("site", choices=sorted(SITES))
    parser.add_argument("--out", type=Path, required=True, help="katalog wyjściowy")
    parser.add_argument(
        "--i-have-written-consent",
        action="store_true",
        help="potwierdza, że zgoda na piśmie obejmuje ten zakres",
    )
    args = parser.parse_args()

    base = SITES[args.site]
    urls = offer_urls(base)
    hours = len(urls) * DELAY_SECONDS / 3600
    print(f"{args.site}: {len(urls)} adresów w sitemapie")
    print(f"Szacowany czas pobierania: ok. {hours:.1f} h (bez ponownych prób)")

    if not args.i_have_written_consent:
        print("Tryb podglądu: nic więcej nie pobrano.")
        print("Dodaj --i-have-written-consent, aby rozpocząć pobieranie.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.site}_offers.jsonl"
    done = load_done(out_path)
    todo = [url for url in urls if url not in done]
    print(f"Do pobrania: {len(todo)} (już zapisane: {len(done)})", flush=True)

    saved = skipped = 0
    last_start = 0.0
    with out_path.open("a", encoding="utf-8") as out:
        for number, url in enumerate(todo, 1):
            pause = DELAY_SECONDS - (time.monotonic() - last_start)
            if pause > 0:
                time.sleep(pause)
            last_start = time.monotonic()

            try:
                page = fetch(url, attempts=2)
            except urllib.error.HTTPError as error:
                if error.code in STOP_STATUSES:
                    print(
                        f"Zatrzymano: serwer zwrócił {error.code} dla {url}. "
                        "Postęp jest zapisany, uruchom ponownie później."
                    )
                    break
                skipped += 1  # np. 404 dla wygasłej oferty
                continue
            except (urllib.error.URLError, TimeoutError) as error:
                print(f"  pominięto {url}: {error}", flush=True)
                skipped += 1
                continue

            offer = parse_offer(url, page, args.site)
            if offer is None:
                skipped += 1  # strona bez bloku JobPosting
                continue

            out.write(json.dumps(offer, ensure_ascii=False) + "\n")
            out.flush()
            saved += 1
            if number % 100 == 0:
                print(
                    f"  {number}/{len(todo)} (zapisano {saved}, pominięto {skipped})",
                    flush=True,
                )

    print(f"Zakończono: zapisano {saved}, pominięto {skipped}. Plik: {out_path}")


if __name__ == "__main__":
    main()
