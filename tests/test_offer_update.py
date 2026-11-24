import json
import os
import time
import urllib.error
from datetime import date
from itertools import pairwise

import pytest

from offer_server import holder as holder_module
from offer_server.embedder import HashingEmbedder
from offer_server.holder import IndexHolder
from offer_server.index import build_index, load_index, save_index
from offer_server.scrape import parse_offer
from offer_server.update import (
    LOCK_FILE,
    STALE_LOCK_SECONDS,
    AlreadyRunning,
    RunLock,
    load_store,
    run_update,
)

TODAY = date(2026, 11, 17)
BASE = "https://justjoin.it"


def page(title: str, valid: str = "2026-12-31", kinds="FULL_TIME") -> str:
    posting = {
        "@type": "JobPosting",
        "title": title,
        "hiringOrganization": {"name": "Firma"},
        "jobLocation": {"address": {"addressLocality": "Warszawa"}},
        "employmentType": kinds,
        "validThrough": valid,
        "datePosted": "2026-11-16",
        "description": "<p>Wymagania</p><ul><li>SQL</li><li>Python</li></ul>",
    }
    return (
        '<html><script type="application/ld+json">'
        + json.dumps(posting)
        + "</script></html>"
    )


class FakeBoard:
    """A job board: a sitemap index, one sitemap part and offer pages."""

    def __init__(self, offers: dict[str, str], fail: dict[str, int] | None = None):
        self.offers = offers  # slug -> title
        self.fail = fail or {}
        self.requested: list[str] = []

    def url(self, slug: str) -> str:
        return f"{BASE}/job-offer/{slug}"

    def get(self, url: str) -> str:
        self.requested.append(url)
        if url == f"{BASE}/sitemaps/active-jobs.xml":
            return f"<sitemapindex><loc>{BASE}/part1.xml</loc></sitemapindex>"
        if url == f"{BASE}/part1.xml":
            locs = "".join(f"<loc>{self.url(s)}</loc>" for s in self.offers)
            return f"<urlset>{locs}</urlset>"
        slug = url.rsplit("/", 1)[-1]
        if slug in self.fail:
            raise urllib.error.HTTPError(url, self.fail[slug], "x", {}, None)
        return page(self.offers[slug])


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.embedded = 0

    def embed(self, texts):
        self.embedded += len(texts)
        return super().embed(texts)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def run(tmp_path, board=None, consent=True, seeds=(), max_new=100, embedder=None):
    clock = FakeClock()
    return run_update(
        store_dir=tmp_path / "store",
        index_dir=tmp_path / "index",
        embedder=embedder or HashingEmbedder(),
        sites=["justjoin"],
        consent=consent,
        seeds=list(seeds),
        max_new=max_new,
        max_minutes=50,
        today=TODAY,
        get=board.get if board else None,
        sleep=clock.sleep,
        clock=clock,
    )


def seed_file(tmp_path):
    records = [
        {
            "id": "seed-analyst",
            "source": "justjoin",
            "url": f"{BASE}/job-offer/seed-analyst",
            "title": "Data Analyst",
            "company": "Firma",
            "location": "Warszawa",
            "valid_through": "2026-12-31",
            "description": "Wymagania\nSQL",
        },
        {
            "id": "seed-old",
            "source": "justjoin",
            "url": f"{BASE}/job-offer/seed-old",
            "title": "Stara oferta",
            "company": "Firma",
            "location": "Warszawa",
            "valid_through": "2026-11-01",
            "description": "Wymagania\nExcel",
        },
    ]
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_without_consent_only_seeds_and_drops_expired(tmp_path) -> None:
    summary = run(tmp_path, consent=False, seeds=[seed_file(tmp_path)])

    assert summary["stored"] == 2
    assert summary["indexed"] == 1
    assert summary["expired"] == 1
    assert summary["sites"] == {}
    assert [o.title for o in load_index(tmp_path / "index").offers] == ["Data Analyst"]


def test_sync_adds_new_offers_and_removes_those_gone_from_sitemap(tmp_path) -> None:
    board = FakeBoard({"seed-analyst": "Data Analyst", "new-dev": "Python Developer"})

    summary = run(tmp_path, board, seeds=[seed_file(tmp_path)])

    stats = summary["sites"]["justjoin"]
    assert stats == {"removed": 1, "added": 1, "skipped": 0, "left": 0}
    store = load_store(tmp_path / "store")
    assert set(store) == {board.url("seed-analyst"), board.url("new-dev")}
    # The offer already in the store is not downloaded again.
    assert board.url("seed-analyst") not in board.requested


def test_budget_leaves_the_rest_for_the_next_run(tmp_path) -> None:
    board = FakeBoard({f"o{i}": f"Oferta {i}" for i in range(5)})

    first = run(tmp_path, board, max_new=2)
    second = run(tmp_path, board, max_new=10)

    assert first["sites"]["justjoin"]["added"] == 2
    assert first["sites"]["justjoin"]["left"] == 3
    assert second["sites"]["justjoin"]["added"] == 3
    assert second["stored"] == 5


def test_a_big_site_does_not_use_up_the_budget_of_the_next(tmp_path) -> None:
    board = FakeBoard({f"o{i}": f"Oferta {i}" for i in range(10)})
    clock = FakeClock()
    rocket = "https://rocketjobs.pl"

    def get(url: str) -> str:
        # The second site lists offers of its own, at its own address.
        if url.startswith(rocket):
            return board.get(url.replace(rocket, BASE)).replace(BASE, rocket)
        return board.get(url)

    summary = run_update(
        store_dir=tmp_path / "store",
        index_dir=tmp_path / "index",
        embedder=HashingEmbedder(),
        sites=["justjoin", "rocketjobs"],
        consent=True,
        seeds=[],
        max_new=6,
        max_minutes=50,
        today=TODAY,
        get=get,
        sleep=clock.sleep,
        clock=clock,
    )

    # Each site has 10 new offers and may fetch half of the 6 pages.
    assert summary["sites"]["justjoin"]["added"] == 3
    assert summary["sites"]["rocketjobs"]["added"] == 3
    assert summary["stored"] == 6


def test_rate_limit_answer_stops_downloading(tmp_path) -> None:
    board = FakeBoard(
        {"a": "A", "b": "B", "c": "C"}, fail={"a": 429, "b": 429, "c": 429}
    )

    summary = run(tmp_path, board)

    stats = summary["sites"]["justjoin"]
    assert stats["added"] == 0
    assert stats["left"] == 3
    offer_requests = [u for u in board.requested if "/job-offer/" in u]
    assert len(offer_requests) == 1


def test_missing_offer_page_is_skipped(tmp_path) -> None:
    board = FakeBoard({"a": "A", "gone": "B"}, fail={"gone": 404})

    stats = run(tmp_path, board)["sites"]["justjoin"]

    assert (stats["added"], stats["skipped"]) == (1, 1)


def test_unchanged_offers_are_not_embedded_again(tmp_path) -> None:
    board = FakeBoard({"a": "Data Analyst", "b": "Java Developer"})
    embedder = CountingEmbedder()

    run(tmp_path, board, embedder=embedder)
    after_first = embedder.embedded
    board.offers["c"] = "Tester"
    summary = run(tmp_path, board, embedder=embedder)

    assert after_first > 0
    assert summary["embedded_pieces"] == embedder.embedded - after_first
    assert summary["embedded_pieces"] == 2  # only the new offer: title + one piece
    assert summary["indexed"] == 3


def test_requests_are_spaced_one_second_apart(tmp_path) -> None:
    board = FakeBoard({f"o{i}": f"Oferta {i}" for i in range(3)})
    clock = FakeClock()
    times = []
    original = board.get

    def timed_get(url):
        times.append(clock.now)
        return original(url)

    run_update(
        tmp_path / "store",
        tmp_path / "index",
        HashingEmbedder(),
        ["justjoin"],
        True,
        [],
        10,
        50,
        TODAY,
        get=timed_get,
        sleep=clock.sleep,
        clock=clock,
    )

    offer_times = times[2:]  # after the two sitemap requests
    assert all(b - a >= 1.0 for a, b in pairwise(offer_times))


def test_lock_blocks_a_second_run_but_not_a_stale_one(tmp_path) -> None:
    lock = tmp_path / LOCK_FILE
    with RunLock(lock):
        with pytest.raises(AlreadyRunning):
            with RunLock(lock):
                pass
    assert not lock.exists()

    lock.write_text("123")
    old = time.time() - STALE_LOCK_SECONDS - 10
    os.utime(lock, (old, old))
    with RunLock(lock):
        pass


def test_employment_type_given_as_text_becomes_a_list() -> None:
    record = parse_offer(
        f"{BASE}/job-offer/x", page("X", kinds="CONTRACTOR"), "justjoin"
    )

    assert record["employment_type"] == ["CONTRACTOR"]
    assert record["description"] == "Wymagania\nSQL\nPython"


def test_the_sites_level_tag_is_read_from_the_page_data() -> None:
    # As on justjoin.it: the level is in the page's script data, quotes escaped.
    data = '<script>self.__next_f.push("{\\"offer\\":{\\"slug\\":\\"x\\",'
    data += '\\"experienceLevel\\":{\\"label\\":\\"mid\\",\\"value\\":\\"mid\\"}}")'
    data += "</script>"

    tagged = parse_offer(f"{BASE}/job-offer/x", page("X") + data, "justjoin")
    untagged = parse_offer(f"{BASE}/job-offer/y", page("Y"), "justjoin")

    assert tagged["experience_level"] == "mid"
    assert untagged["experience_level"] is None


def test_offers_saved_without_a_level_are_fetched_again(tmp_path) -> None:
    board = FakeBoard({"old": "Data Analyst"})
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    old = parse_offer(board.url("old"), page("Data Analyst"), "justjoin")
    del old["experience_level"]  # saved before the level was read
    (store_dir / "offers.jsonl").write_text(json.dumps(old) + "\n", "utf-8")

    run(tmp_path, board)
    run(tmp_path, board)

    # Fetched once to get the level; the record has the key now, even as None.
    assert board.requested.count(board.url("old")) == 1
    assert "experience_level" in load_store(store_dir)[board.url("old")]


def test_running_app_picks_up_a_new_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(holder_module, "RELOAD_CHECK_SECONDS", 0)
    folder = tmp_path / "index"
    board = FakeBoard({"a": "Data Analyst"})
    run(tmp_path, board)
    holder = IndexHolder(load_index(folder), HashingEmbedder(), folder)
    assert len(holder.get().offers) == 1

    board.offers["b"] = "Java Developer"
    run(tmp_path, board)
    stamp = os.stat(folder / "index.npz").st_mtime + 5
    os.utime(folder / "index.npz", (stamp, stamp))  # coarse clocks on some disks

    assert len(holder.get().offers) == 2


def test_app_keeps_old_index_when_new_one_has_another_model(tmp_path, monkeypatch):
    monkeypatch.setattr(holder_module, "RELOAD_CHECK_SECONDS", 0)
    folder = tmp_path / "index"
    run(tmp_path, FakeBoard({"a": "Data Analyst"}))
    holder = IndexHolder(load_index(folder), HashingEmbedder(), folder)

    other = build_index(holder.index.offers * 2, HashingEmbedder(dim=64))
    save_index(other, folder)
    stamp = os.stat(folder / "index.npz").st_mtime + 5
    os.utime(folder / "index.npz", (stamp, stamp))

    assert len(holder.get().offers) == 1


def test_fetch_all_output_with_nulls_and_a_broken_line_is_imported(tmp_path) -> None:
    from offer_server.documents import load_raw_offers

    good = {
        "id": "a",
        "source": "rocketjobs",
        "url": "https://rocketjobs.pl/oferta-pracy/a",
        "title": "Magazynier",
        "company": None,
        "location": None,
        "remote": None,
        "employment_type": "FULL_TIME",
        "description": None,
    }
    no_url = {"id": "b", "source": "rocketjobs", "title": "X"}
    path = tmp_path / "rocketjobs_offers.jsonl"
    path.write_text(
        json.dumps(good) + "\n" + json.dumps(no_url) + "\n" + '{"id": "c", "sou',
        encoding="utf-8",
    )

    offers = load_raw_offers([path])

    assert len(offers) == 1
    assert offers[0].company == ""
    assert offers[0].employment_type == ["FULL_TIME"]


def test_import_merges_downloaded_files_into_an_existing_store(tmp_path) -> None:
    run(tmp_path, consent=False, seeds=[seed_file(tmp_path)])
    downloaded = tmp_path / "justjoin_offers.jsonl"
    downloaded.write_text(
        json.dumps(
            {
                "id": "imported",
                "source": "justjoin",
                "url": f"{BASE}/job-offer/imported",
                "title": "Python Developer",
                "company": "Firma",
                "location": "Kraków",
                "valid_through": "2026-12-31",
                "description": "Wymagania\nPython",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_update(
        tmp_path / "store",
        tmp_path / "index",
        HashingEmbedder(),
        ["justjoin"],
        False,
        [seed_file(tmp_path)],
        10,
        50,
        TODAY,
        imports=[downloaded],
    )

    assert summary["stored"] == 3
    assert summary["indexed"] == 2  # the expired seed offer is still left out


def test_a_folder_of_renamed_files_is_loaded(tmp_path) -> None:
    from offer_server.documents import load_raw_offers

    folder = tmp_path / "data"
    folder.mkdir()
    line = {
        "id": "a",
        "source": "justjoin",
        "url": f"{BASE}/job-offer/a",
        "title": "Tester",
        "company": "Firma",
    }
    # JSON Lines content under a .json name, as after renaming a fetch_all file.
    (folder / "justjoin.json").write_text(
        json.dumps(line) + "\n" + json.dumps({**line, "id": "b", "url": "u2"}) + "\n",
        encoding="utf-8",
    )
    (folder / "options.json").write_text('{"roles": ["X"]}', encoding="utf-8")
    (folder / "notes.txt").write_text("nie oferty", encoding="utf-8")

    offers = load_raw_offers([folder])

    assert [offer.id for offer in offers] == ["a", "b"]
