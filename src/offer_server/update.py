"""Hourly update of the offer database and the index.

    uv run python -m offer_server.update --seed data/offers_sample_*.json

One run:
 1. Reads the raw offer store (var/store/offers.jsonl, one offer per line). An empty
    store is first filled from the --seed files.
 2. Only with written consent of the job boards (OFFER_UPDATE_WRITTEN_CONSENT=yes
    or --i-have-written-consent): reads each site's sitemap, drops offers that
    left it and downloads new ones, one request per second, at most --max-new
    per run and never longer than --max-minutes, so runs do not overlap. What
    is left is downloaded in the next run.
 3. Rebuilds the index: expired offers and duplicates go away, vectors of
    unchanged offers are reused, only new texts are embedded.
 4. Writes the index in one atomic step; the running web app picks it up by
    itself (see holder.py), no restart needed.

Without consent steps 1, 3 and 4 still run, so expired offers disappear.
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
from collections.abc import Callable
from datetime import date
from pathlib import Path

from offer_server.documents import RawOffer, clean_offers, load_raw_offers
from offer_server.embedder import Embedder, FastEmbedder
from offer_server.index import build_index, index_path, load_index, save_index
from offer_server.paths import index_dir, store_dir
from offer_server.scrape import SITES, fetch, offer_urls, parse_offer

STORE_FILE = "offers.jsonl"
LOCK_FILE = "update.lock"
DELAY_SECONDS = 1.0  # agreed limit: at most one request per second
STOP_STATUSES = {403, 429, 500, 502, 503, 504}
STALE_LOCK_SECONDS = 3 * 3600
SAVE_EVERY = 100


def log(message: str) -> None:
    print(message, flush=True)


# --- raw offer store --------------------------------------------------------


def load_store(folder: Path) -> dict[str, dict]:
    """Raw offers by URL; a broken last line (an interrupted write) is skipped."""
    path = folder / STORE_FILE
    if not path.exists():
        return {}
    offers = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            offers[record["url"]] = record
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return offers


def save_store(folder: Path, offers: dict[str, dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (STORE_FILE + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in offers.values():
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, folder / STORE_FILE)


# --- lock, so two runs never write at the same time ------------------------


class AlreadyRunning(RuntimeError):
    pass


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age < STALE_LOCK_SECONDS:
                raise AlreadyRunning(f"Inna aktualizacja trwa ({self.path}).")
            self.path.unlink(missing_ok=True)  # left by a crashed run
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise AlreadyRunning(f"Inna aktualizacja trwa ({self.path}).") from error
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc) -> None:
        self.path.unlink(missing_ok=True)


# --- downloading ----------------------------------------------------------


def sync_site(
    site: str,
    store: dict[str, dict],
    budget: int,
    deadline: float,
    folder: Path,
    get: Callable[[str], str] = fetch,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, int]:
    """Bring one site's part of the store in line with its sitemap."""
    stats = {"removed": 0, "added": 0, "skipped": 0, "left": 0}
    try:
        urls = offer_urls(SITES[site], get)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        log(f"{site}: sitemapa niedostępna ({error}); nic nie zmieniono.")
        return stats
    if not urls:
        log(f"{site}: pusta sitemapa; nic nie usunięto na wszelki wypadek.")
        return stats

    active = set(urls)
    for url in [u for u, r in store.items() if r.get("source") == site]:
        if url not in active:
            del store[url]
            stats["removed"] += 1

    new = [url for url in urls if url not in store]
    random.Random().shuffle(new)  # no part of the sitemap waits forever
    # Offers saved before the level tag was read are fetched again, after the
    # new ones; a record that has the key (even None) is not fetched again.
    new += [u for u in urls if u in store and "experience_level" not in store[u]]
    last_start = 0.0
    for number, url in enumerate(new):
        if number >= budget or clock() >= deadline:
            stats["left"] = len(new) - number
            break
        pause = DELAY_SECONDS - (clock() - last_start)
        if pause > 0:
            sleep(pause)
        last_start = clock()
        try:
            page = get(url)
        except urllib.error.HTTPError as error:
            if error.code in STOP_STATUSES:
                log(
                    f"{site}: serwer zwrócił {error.code}, "
                    "przerywam do następnej godziny."
                )
                stats["left"] = len(new) - number
                break
            stats["skipped"] += 1  # e.g. 404 for an offer that just expired
            continue
        except (urllib.error.URLError, TimeoutError, OSError):
            stats["skipped"] += 1
            continue
        record = parse_offer(url, page, site)
        if record is None:
            stats["skipped"] += 1
            continue
        store[url] = record
        stats["added"] += 1
        if stats["added"] % SAVE_EVERY == 0:
            save_store(folder, store)
    return stats


# --- one whole run ----------------------------------------------------------


def run_update(
    store_dir: Path,
    index_dir: Path,
    embedder: Embedder,
    sites: list[str],
    consent: bool,
    seeds: list[Path],
    max_new: int,
    max_minutes: float,
    today: date,
    get: Callable[[str], str] = fetch,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    imports: list[Path] | None = None,
) -> dict:
    with RunLock(store_dir / LOCK_FILE):
        store = load_store(store_dir)
        if not store and seeds:
            for offer in load_raw_offers(seeds):
                store[offer.url] = offer.model_dump()
            log(f"Pusta baza: wczytano {len(store)} ofert z plików startowych.")
        if imports:
            before = len(store)
            loaded = load_raw_offers(imports)
            for offer in loaded:
                store[offer.url] = offer.model_dump()  # newer copy replaces older
            log(
                f"Import: {len(loaded)} ofert z {len(imports)} plików, "
                f"nowych w bazie: {len(store) - before}."
            )

        summary: dict = {"sites": {}}
        if consent:
            end = clock() + max_minutes * 60
            budget = max_new
            for number, site in enumerate(sites):
                # Each site gets an equal share of what is left of the pages and
                # the time, so a big site cannot starve the next one; a site that
                # needs less leaves the rest to the sites after it.
                sites_left = len(sites) - number
                share = budget // sites_left
                deadline = clock() + (end - clock()) / sites_left
                stats = sync_site(
                    site, store, share, deadline, store_dir, get, sleep, clock
                )
                budget -= stats["added"] + stats["skipped"]
                summary["sites"][site] = stats
                log(
                    f"{site}: +{stats['added']} nowych, -{stats['removed']} "
                    f"zniknęło z sitemapy, {stats['skipped']} pominiętych, "
                    f"{stats['left']} czeka na kolejny przebieg."
                )
        else:
            log(
                "Brak zgody na pobieranie (OFFER_UPDATE_WRITTEN_CONSENT): "
                "tylko odświeżam indeks z zapisanych ofert."
            )
        save_store(store_dir, store)

        raw = [RawOffer.model_validate(record) for record in store.values()]
        docs, clean_stats = clean_offers(raw, today)
        previous = None
        if index_path(index_dir).exists():
            try:
                previous = load_index(index_dir)
            except (OSError, ValueError, KeyError) as error:
                log(f"Nie udało się wczytać starego indeksu ({error}); buduję od zera.")
        index = build_index(docs, embedder, previous)
        index.meta["stats"] = clean_stats
        save_index(index, index_dir)
        summary.update(
            {
                "stored": len(store),
                "indexed": len(docs),
                "embedded_pieces": index.meta["embedded_pieces"],
                **{key: clean_stats[key] for key in ("expired", "duplicates")},
            }
        )
        log(
            f"Indeks: {len(docs)} ofert (w bazie {len(store)}, wygasłe "
            f"{clean_stats['expired']}, duplikaty {clean_stats['duplicates']}), "
            f"nowo zwektoryzowane fragmenty: {index.meta['embedded_pieces']}."
        )
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aktualizacja bazy ofert i indeksu.")
    parser.add_argument("--store", type=Path, default=store_dir())
    parser.add_argument("--index", type=Path, default=index_dir())
    parser.add_argument(
        "--sites", nargs="+", choices=sorted(SITES), default=sorted(SITES)
    )
    parser.add_argument(
        "--seed",
        nargs="*",
        type=Path,
        default=[],
        help="pliki startowe dla pustej bazy",
    )
    parser.add_argument(
        "--import",
        dest="imports",
        nargs="*",
        type=Path,
        default=[],
        help="dołącz oferty z plików (np. *_offers.jsonl z fetch_all.py)",
    )
    parser.add_argument(
        "--max-new", type=int, default=2500, help="najwięcej nowych ofert na przebieg"
    )
    parser.add_argument(
        "--max-minutes", type=float, default=50, help="limit czasu pobierania"
    )
    parser.add_argument(
        "--i-have-written-consent",
        action="store_true",
        help="zgoda serwisów na piśmie obejmuje regularne pobieranie",
    )
    args = parser.parse_args()

    consent = args.i_have_written_consent or (
        os.getenv("OFFER_UPDATE_WRITTEN_CONSENT", "").strip().lower() == "yes"
    )
    started = time.monotonic()
    try:
        run_update(
            store_dir=args.store,
            index_dir=args.index,
            embedder=FastEmbedder(),
            sites=args.sites,
            consent=consent,
            seeds=args.seed,
            max_new=args.max_new,
            max_minutes=args.max_minutes,
            today=date.today(),
            imports=args.imports,
        )
    except AlreadyRunning as error:
        log(str(error))
        sys.exit(0)
    log(f"Gotowe w {time.monotonic() - started:.0f} s.")


if __name__ == "__main__":
    main()
