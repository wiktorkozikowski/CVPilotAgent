from datetime import date

import pytest

from agent_project.offer_db import OfferDB
from offer_server.documents import RawOffer, clean_offers
from offer_server.embedder import HashingEmbedder
from offer_server.holder import IndexHolder
from offer_server.index import build_index

TODAY = date(2026, 11, 12)


def raw_offer(offer_id: str, **fields) -> RawOffer:
    data = {
        "id": offer_id,
        "source": "justjoin",
        "url": f"https://justjoin.it/job-offer/{offer_id}",
        "title": "Data Analyst",
        "company": "Firma",
        "location": "Warszawa",
        "remote": False,
        "employment_type": ["FULL_TIME"],
        "valid_through": "2026-12-31T00:00:00Z",
        "description": "Wymagania\nSQL, Excel",
    }
    data.update(fields)
    return RawOffer.model_validate(data)


DB_OFFERS = [
    raw_offer(
        "remote-analyst",
        company="Zdalna Firma",
        location="",
        remote=True,
    ),
    raw_offer(
        "bi-krakow",
        title="Specjalista ds. raportowania",
        company="Raporty SA",
        location="Kraków",
        description="Wymagania\nPower BI, SQL, DAX",
    ),
    raw_offer(
        "java",
        title="Senior Java Developer",
        company="Kod",
        description="Must have\nJava, Spring, Kafka",
    ),
    raw_offer(
        "warehouse",
        title="Magazynier",
        company="Logistyka",
        location="Kraków",
        employment_type=["PART_TIME"],
        description="Wymagania\nwózek widłowy, UDT",
    ),
]


class CountingDB(OfferDB):
    """The real OfferDB with a crude offline embedder; counts profile searches."""

    def __init__(self, holder: IndexHolder) -> None:
        super().__init__(holder)
        self.limits: list[int] = []

    def match_profile(self, profile, limit=10, query=None):
        self.limits.append(limit)
        return super().match_profile(profile, limit, query)


def make_offer_db(raws: list[RawOffer] = DB_OFFERS) -> CountingDB:
    docs, _ = clean_offers(raws, TODAY)
    embedder = HashingEmbedder()
    return CountingDB(IndexHolder(build_index(docs, embedder), embedder))


@pytest.fixture
def offer_db() -> CountingDB:
    return make_offer_db()
