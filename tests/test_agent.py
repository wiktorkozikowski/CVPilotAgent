import json
from pathlib import Path

import pytest

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


def test_agent_runs_every_tool_call_of_one_reply(tmp_path) -> None:
    profile = load_profile(DATA / "profile.json")
    offers = load_offers(DATA / "job_offers.json")
    client = MockClient(
        [
            {
                "text": None,
                "tool_calls": [
                    {"id": "a", "name": "szukaj_ofert", "arguments": '{"limit": 1}'},
                    {"id": "b", "name": "szukaj_ofert", "arguments": '{"limit": 2}'},
                ],
                "usage": {},
                "latency_ms": 0,
            },
            {"text": "Gotowe.", "tool_calls": [], "usage": {}, "latency_ms": 0},
        ]
    )
    messages = [{"role": "user", "content": "Pokaż oferty."}]

    answer = run_agent(client, messages, profile, offers, tmp_path / "trace.jsonl")

    assert answer == "Gotowe."
    # One assistant message with both calls, then one answer per call, in order.
    assistant = [m for m in messages if m["role"] == "assistant"][0]
    assert [c["id"] for c in assistant["tool_calls"]] == ["a", "b"]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]
    assert [len(json.loads(m["content"])["offers"]) for m in tool_messages] == [1, 2]


def test_agent_stops_when_run_cost_exceeds_limit(tmp_path) -> None:
    profile = load_profile(DATA / "profile.json")
    offers = load_offers(DATA / "job_offers.json")
    expensive = {
        "text": None,
        "tool_calls": [
            {"id": "call_1", "name": "szukaj_ofert", "arguments": '{"limit": 2}'}
        ],
        "usage": {"cost": 0.08},
        "latency_ms": 0,
        "provider": "TestProvider",
        "model": "test-model",
    }
    client = MockClient([dict(expensive), dict(expensive)])
    messages = [{"role": "user", "content": "Pokaż oferty."}]
    trace_path = tmp_path / "trace.jsonl"

    with pytest.raises(RuntimeError, match="limit kosztu"):
        run_agent(client, messages, profile, offers, trace_path, max_cost_usd=0.10)

    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    decisions = [r for r in records if r["event"] == "model_decision"]
    assert len(decisions) == 2
    assert decisions[0]["payload"]["provider"] == "TestProvider"
    assert decisions[0]["payload"]["model"] == "test-model"
    assert decisions[1]["payload"]["run_cost_usd"] > 0.10
    assert records[-1]["event"] == "cost_limit"
