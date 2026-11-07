"""Live demonstration runs with the real model; traces go to evidence/.

Run with `uv run python -m agent_project.demo_runs`. Each run uses a temporary
shortlist, so data/shortlist.json is never changed. Costs a few cents in total.
"""

import tempfile
from pathlib import Path

from agent_project.agent import SYSTEM_PROMPT, run_agent
from agent_project.llm import OpenRouterClient
from agent_project.models import load_offers, load_profile

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
EVIDENCE_DIR = ROOT_DIR / "evidence"

SEARCH_TEXT = "Pokaż oferty pasujące do mojego profilu."
SCENARIOS = [
    # (trace file, user message, extra run_agent arguments)
    ("live_success.jsonl", SEARCH_TEXT, {}),
    (
        "live_observation_changes_plan.jsonl",
        "Dodaj do listy ofertę offer-001. Potem spróbuj dodać tę samą ofertę "
        "jeszcze raz i, jeśli to potrzebne, dodaj inną pasującą ofertę.",
        {},
    ),
    # A deliberately tiny budget: the run must stop with a clear error.
    ("live_cost_limit_failure.jsonl", SEARCH_TEXT, {"max_cost_usd": 0.0000001}),
]


def main() -> None:
    profile = load_profile(DATA_DIR / "profile.json")
    offers = load_offers(DATA_DIR / "job_offers.json")
    EVIDENCE_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        shortlist_path = Path(tmp) / "shortlist.json"
        for file_name, user_text, extra in SCENARIOS:
            trace_path = EVIDENCE_DIR / file_name
            trace_path.unlink(missing_ok=True)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            try:
                answer = run_agent(
                    OpenRouterClient(),
                    messages,
                    profile,
                    offers,
                    trace_path,
                    shortlist_path=shortlist_path,
                    **extra,
                )
                outcome = f"ok: {answer[:80]!r}"
            except RuntimeError as error:
                outcome = f"zatrzymany: {error}"
            print(f"{file_name}: {outcome}")


if __name__ == "__main__":
    main()
