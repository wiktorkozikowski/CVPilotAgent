# My Agent Project

Student project for **Designing LLM/VLM Agents**.

## Setup

Create your own repository with **Use this template**, then clone your new
repository:

```bash
git clone <your-repository-url>
cd <your-repository-directory>
uv sync --locked
cp .env.example .env
uv run pytest
```

Add the individual OpenRouter key supplied by the instructor to `.env`. Never
commit this file or place the key in code, traces, issues, or reports.

## Coding Agent

If you already use Codex or Claude Code, nothing else is required. Otherwise,
install [Antigravity CLI](https://antigravity.google/docs/cli/install/) before
Meeting 1.

## Project Brief

- **Project title:** CVPilotAgent, agent tworzący spersonalizowane CV
- **Intended user:** osoba szukająca pracy, np. junior analityk lub developer
- **Problem and primary task:** użytkownik wpisuje swoje umiejętności w pliku
  JSON. Agent wyszukuje pasujące oferty pracy i wypisuje je posortowane wg
  dopasowania. Kolejne etapy (w planie): generowanie CV dopasowanego do ofert
  i wysłanie go po akceptacji użytkownika.
- **Why an agent is appropriate:** wybór następnego kroku zależy od wyników
  poprzednich, np. ile ofert znaleziono i których umiejętności brakuje.
- **Planned data:** `data/profile.json` (profil użytkownika) i
  `data/job_offers.json` (przykładowe oferty). Dane są fikcyjne. Prawdziwy
  profil nie trafia do repozytorium.
- **Tools:** `szukaj_ofert` (tylko odczyt), `dodaj_do_listy` (odwracalny zapis)
  i `usun_z_listy` (wymaga potwierdzenia użytkownika w aplikacji). Pierwsze dwa
  są też w serwerze MCP, usuwanie nie. Każde narzędzie waliduje argumenty w kodzie.
- **Status:** zrealizowane wyszukiwanie, lista wybranych, pętla z limitem kroków
  (12) i kosztu (0,10 USD na uruchomienie), interfejs w przeglądarce
  (`agent_project.web`) z przebiegiem agenta pod wynikami. Dane ofert są
  fikcyjne. Dalej: generowanie CV i wysyłka, dopiero po zgodzie użytkownika.
- **Evidence:** `evidence/live_*.jsonl` to przebiegi z prawdziwym modelem
  (`uv run python -m agent_project.demo_runs`), `flow_trace.jsonl` i
  `confirmation_trace.jsonl` to przebiegi z mockiem.
- **Planned visual input:** brak na tym etapie.
- **Success criteria:** dla przykładowego profilu oferty są uporządkowane
  według dopasowania umiejętności, a oferty bez wspólnych umiejętności nie
  pojawiają się na liście. Testy przechodzą bez połączenia z modelem.
- **Non-goals and safety boundaries:** agent nie wysyła CV ani nie
  odpowiada na oferty bez zgody użytkownika. Nie scrapuje serwisów z ofertami
  bez zgody ich regulaminu. Nie zapisuje danych osobowych w śladach.

Replace the empty fields above with your project definition and keep them
current as the project develops.

## Starter Code

The repository contains only:

- `src/agent_project/llm.py` — an OpenRouter client and a mock client. A live
  call returns the text, tool calls, finish reason, model, provider, latency,
  and token usage with cost. Requests time out after 60 seconds and replies are
  limited to 1,024 tokens. If a provider returns an empty reply instead of the
  model's answer, the client retries once on another provider.
  `image_part(path)` adds an image to a message;
- `src/agent_project/tracing.py` — a JSON Lines trace helper;
- `tests/test_setup.py` — setup checks;
- `data/` — approved project data;
- `traces/` — generated traces, ignored by Git; and
- `evidence/` — the traces and results you submit.

The agent loop, domain tools, state, retrieval, multimodal behavior, evaluation,
and safety controls are intentionally left for you to design and implement.

## Course Limits

- One model for all live work: `google/gemma-4-26b-a4b-it`.
- USD 2 per key for the whole course, enforced by OpenRouter.
- In your own code: at most 12 model steps and USD 0.10 per agent run.

## Working with the Course Model

- The model usually requests one tool call per turn. Tell it in the system
  prompt that independent calls may be requested together.
- For typed output, such as a plan or facts read from an image, offer one tool
  whose parameters are your Pydantic schema and validate its arguments. Keep
  `tool_choice` at its default, `"auto"`, and give lists a maximum length;
  free-form JSON output sometimes repeats itself until the token limit.
- Results vary between identical runs. Run your fixed evaluation set at least
  three times and report the worst run as well as the average.
- A typical agent run costs well under USD 0.01. Use the mock client in tests.
- Copy the traces you submit to `evidence/`. Check them first for keys,
  personal data, and absolute file paths.

## Commands

```bash
uv run pytest
uv run ruff check .

uv run python -c "from pathlib import Path; from agent_project.agent import SYSTEM_PROMPT, run_agent; from agent_project.models import load_offers, load_profile; from agent_project.llm import OpenRouterClient; data = Path('data'); messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': 'Pokaz oferty pasujace do mojego profilu.'}]; print(run_agent(OpenRouterClient(), messages, load_profile(data / 'profile.json'), load_offers(data / 'job_offers.json'), Path('traces/agent_run.jsonl')))"
```

### Agent w przeglądarce (UI)

1. Przygotuj projekt: `uv sync --locked`, a w `.env` wpisz `MODEL_NAME` i
   `OPENROUTER_API_KEY`. Klucz jest potrzebny tylko do wyszukiwania. Dodawanie
   profili i listy działają bez niego.
2. Uruchom serwer:

   ```bash
   uv run python -m agent_project.web
   ```

3. Otwórz http://127.0.0.1:8000. Serwer słucha tylko lokalnie. Zatrzymanie:
   Ctrl+C. Po zmianie w kodzie uruchom serwer ponownie. Zmiany w danych
   (`data/job_offers.json`, `data/options.json`) działają bez restartu.

Interfejs ma trzy zakładki:

- **Dodaj profil.** Wpisz imię, miasto, poziom (junior, mid, senior), stanowiska
  i umiejętności. Gotowe wartości zaznaczasz z list, a własne wpisujesz w polach
  „Inne”. Profil zapisuje się w `data/profiles/<id>.json`.
- **Szukaj dopasowań.** Wybierz profil i kliknij „Szukaj dopasowań”. Agent
  wywołuje model z OpenRouter, który wybiera narzędzia (`szukaj_ofert`, a w razie
  potrzeby `dodaj_do_listy`). Strona pokazuje odpowiedź modelu, tabelę ofert z
  punktacją, przebieg każdego kroku i ścieżkę do śladu w `traces/`. Przycisk
  „Dodaj do listy” zapisuje ofertę na liście profilu, dla którego szukałeś.
- **Lista wybranych.** Wybierz profil, żeby zobaczyć jego listę. Usunięcie oferty
  wymaga potwierdzenia w oknie przeglądarki. Listy są w
  `data/shortlists/<id>.json`.

Ograniczenia każdego uruchomienia: najwyżej 12 kroków modelu i 0,10 USD
(`MAX_RUN_COST_USD` w `agent.py`).

Oferty są w `data/job_offers.json`. To 24 fikcyjne oferty. Aby dodać własne,
dopisz obiekty z polami `id`, `title`, `company`, `location`, `required_skills`,
`source` i `url`. Stanowiska i umiejętności do formularza bierze się z
`data/options.json`.

Use the assignments published on the course website as the source of truth.

