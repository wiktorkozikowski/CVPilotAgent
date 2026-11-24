import json

from conftest import make_offer_db, raw_offer

from agent_project.judge import BATCH_SIZE, profile_for_model, rate_offers
from agent_project.models import Profile

PROFILE = Profile.model_validate(
    {
        "name": "Anna Nowak",
        "email": "anna@example.com",
        "phone": "+48 600 100 200",
        "location": "Kraków",
        "target_roles": ["Data Analyst"],
        "skills": ["SQL", "Power BI"],
        "seniority": ["junior"],
    }
)


class RecordingClient:
    """Rates every offer it is shown with the next rating from `fits`."""

    def __init__(self, fits):
        self.fits = list(fits)
        self.prompts = []

    def complete(self, messages, tools=None):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        count = prompt.count("[o")
        ratings = [
            {"oferta": f"o{i}", "ocena": self.fits.pop(0), "powod": "x"}
            for i in range(1, count + 1)
        ]
        return {
            "text": None,
            "tool_calls": [
                {
                    "id": "c",
                    "name": "ocen_oferty",
                    "arguments": json.dumps({"oceny": ratings}),
                }
            ],
            "usage": {"cost": 0.001},
            "latency_ms": 0,
        }


def test_candidates_are_rated_in_batches_without_contact_data(tmp_path) -> None:
    offers = [
        raw_offer(f"a{i}", title=f"Data Analyst {i}", company=f"Firma {i}")
        for i in range(BATCH_SIZE + 2)
    ]
    db = make_offer_db(offers)
    matches = db.match_profile(PROFILE, len(offers))
    client = RecordingClient(range(len(offers)))

    rated = rate_offers(client, PROFILE, matches, db, tmp_path / "t.jsonl", 0.10)

    assert len(client.prompts) == 2  # 8 offers, then 2
    assert len(rated) == len(offers)
    for prompt in client.prompts:
        assert "anna@example.com" not in prompt
        assert "600 100 200" not in prompt
        assert "Umiejętności: SQL, Power BI" in prompt


def test_profile_for_model_names_the_level_and_skills() -> None:
    text = profile_for_model(PROFILE)

    assert "Poziom: junior" in text
    assert "Szukane stanowiska: Data Analyst" in text
    assert "Anna" not in text
