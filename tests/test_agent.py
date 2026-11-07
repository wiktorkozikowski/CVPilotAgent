import json
from pathlib import Path

from agent_project.agent import SYSTEM_PROMPT, run_agent
from agent_project.llm import MockClient
from agent_project.models import load_offers, load_profile

DATA = Path(__file__).resolve().parent.parent / "data"


def test_agent_finds_offers(tmp_path) -> None:
    profile = load_profile(DATA / "profile.json")
    offers = load_offers(DATA / "job_offers.json")
    client = MockClient(
        [
            {
                "text": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "szukaj_ofert",
                        "arguments": '{"limit": 2}',
                    }
                ],
                "usage": {},
                "latency_ms": 0,
            },
            {
                "text": "Znalazłem 2 oferty.",
                "tool_calls": [],
                "usage": {},
                "latency_ms": 0,
            },
        ]
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Pokaż oferty pasujące do mojego profilu."},
    ]
    trace_path = tmp_path / "trace.jsonl"

    answer = run_agent(client, messages, profile, offers, trace_path)

    assert answer == "Znalazłem 2 oferty."
    assert len(trace_path.read_text(encoding="utf-8").splitlines()) == 3
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    result = json.loads(tool_messages[0]["content"])
    assert result["status"] == "ok"
    assert len(result["offers"]) == 2
    assert result["offers"][0]["offer"]["title"] == "Junior Data Analyst"