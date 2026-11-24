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
- **Tools:** do odczytu bazy ofert `szukaj_ofert` (ranking dla profilu),
  `przeszukaj_oferty` (własne zapytanie i filtry), `pokaz_oferte`,
  `porownaj_z_profilem` i `wybierz_oferty` (oceny agenta i dodane przez
  niego oferty, z powodami);
  `dodaj_do_listy` (odwracalny zapis) i `usun_z_listy` (wymaga potwierdzenia
  użytkownika w aplikacji). `szukaj_ofert` i `dodaj_do_listy` są też w serwerze
  MCP, usuwanie nie. Każde narzędzie waliduje argumenty w kodzie.
- **Status:** zrealizowane wyszukiwanie, lista wybranych, pętla z limitem kroków
  (24) i kosztu (0,10 USD na uruchomienie), interfejs w przeglądarce
  (`agent_project.web`) z logowaniem na jedno konto i przebiegiem agenta pod
  wynikami. Cały projekt działa jako jedna aplikacja na serwerze, a agent sam
  przegląda wektorową bazę prawdziwych ofert. Dalej: generowanie CV i wysyłka,
  dopiero po zgodzie użytkownika.
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

### Aplikacja (jedna, na serwerze)

Cały projekt działa jako jedna aplikacja na jednej maszynie: strona
z logowaniem, pętla agenta, baza ofert z indeksem wektorowym i lokalny model
do wektoryzacji (fastembed). Model językowy to dalej
`google/gemma-4-26b-a4b-it` przez OpenRouter. Użytkownik korzysta z niej tylko
przez przeglądarkę, po zalogowaniu.

```
przeglądarka ──HTTPS──► Caddy ──► agent_project.web (127.0.0.1:8000)
                                   ├─ logowanie (jedno konto, var/account.json)
                                   ├─ agent ──► OpenRouter (gemma)
                                   │    └─ narzędzia: szukaj_ofert, przeszukaj_oferty,
                                   │       pokaz_oferte, porownaj_z_profilem, wybierz_oferty
                                   └─ baza ofert w pamięci (var/index, fastembed)
offer-update.timer (co godzinę) ──► offer_server.update ──► var/store, var/index
```

Wyszukiwanie łączy ranking z kodu z oceną modelu. Kod liczy ranking dla
całego profilu i tabela zawsze pokazuje jego czołówkę w całości. Agent
przegląda 3× większą pulę z rankingu, szuka własnymi zapytaniami z filtrami
i sprawdza wątpliwe oferty. Na końcu (`wybierz_oferty`) daje ofertom
z rankingu ocenę „polecam” albo „uwaga” z powodem i dopisuje pod rankingiem
dobre oferty, których ranking nie znalazł. Agent nie może usunąć oferty
z rankingu, więc wynik nie jest gorszy niż sam ranking.
Szczegóły bazy, wektoryzacji i aktualizacji: [docs/baza-ofert.md](docs/baza-ofert.md).

**Konto.** Jest jedno i nie da się zarejestrować przez stronę. Zakłada się je
(albo zmienia hasło) na maszynie z aplikacją:

```bash
uv run python -m agent_project.auth      # pyta o login i hasło (min. 10 znaków)
```

W `var/account.json` zapisuje się tylko skrót hasła (scrypt). Sesja trwa 12 h
i kończy się też przy restarcie aplikacji. Po 5 złych hasłach z rzędu logowanie
jest zablokowane na 15 minut. Ciasteczko sesji ma `HttpOnly`, `SameSite=Strict`
i `Secure`, a zapytania z innych stron są odrzucane.

**Uruchomienie lokalne** (Windows albo Linux):

```bash
uv sync --locked
cp .env.example .env                     # MODEL_NAME i OPENROUTER_API_KEY
uv run python -m offer_server.update --seed data/offers_sample_justjoin.json data/offers_sample_rocketjobs.json
uv run python -m agent_project.auth
uv run python -m agent_project.web       # albo: powershell -File start.ps1
```

Potem otwórz http://127.0.0.1:8000 i zaloguj się. Pierwsza aktualizacja
pobiera model (ok. 240 MB) do `var/models_cache/`.

**Wdrożenie na serwer (Linux, systemd).** Pliki w `deploy/` zakładają
repozytorium sklonowane do `/home/ubuntu/server/CVpilot` i użytkownika
`ubuntu`. Na innej maszynie zmień ścieżki i użytkownika.

```bash
cd ~/server/CVpilot && git pull && uv sync --locked --no-dev
sudo cp deploy/cvpilot.env.example /etc/cvpilot.env
sudo nano /etc/cvpilot.env && sudo chmod 600 /etc/cvpilot.env   # klucz OpenRouter
.venv/bin/python -m agent_project.auth                          # konto
sudo cp deploy/cvpilot.service deploy/offer-update.service deploy/offer-update.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start offer-update          # pierwsza baza i indeks; postęp: journalctl -u offer-update -f
sudo systemctl enable --now cvpilot offer-update.timer
journalctl -u cvpilot -f
```

Aplikacja słucha tylko na `127.0.0.1`. Do internetu wystawia ją Caddy
z HTTPS (`deploy/Caddyfile.example`: wpisz swoją domenę, otwórz porty 80
i 443). Bez HTTPS hasło szłoby przez sieć jawnie, dlatego portu 8000 nie
otwiera się w zaporze. Stary serwer ofert (`offer-server.service`, port 8100)
jest już niepotrzebny: `sudo systemctl disable --now offer-server`.

Interfejs:

- **Profil w rozmowie.** Opowiadasz o sobie zwykłym tekstem, a model pyta
  o brakujące pola. Profil zapisuje się dopiero po kliknięciu „Zatwierdź
  profil”; sam model nie może go zapisać. W rozmowie model może też przeglądać
  bazę ofert (np. „jakie są oferty z Power BI w Krakowie?”).
- **Szukaj ofert.** Tabela po prawej pokazuje ranking z kodu z oceną agenta
  przy ofertach („⚠” przy słabszych) i pod nim oferty oznaczone „znalezione
  przez agenta”, a pod tabelą przebieg każdego kroku i ścieżkę do śladu
  w `traces/`.
- **Lista wybranych.** Każdy profil ma własną listę
  (`data/shortlists/<id>.json`). Usunięcie oferty wymaga potwierdzenia
  w oknie przeglądarki.

Ograniczenia każdego uruchomienia: najwyżej 24 kroki modelu (`MAX_STEPS`)
i 0,10 USD (`MAX_RUN_COST_USD`, oba w `agent.py`). Kilka niezależnych narzędzi,
o które model prosi w jednej odpowiedzi, wykonuje się w jednym kroku. Uwaga:
24 kroki to więcej niż limit kursu (12).

Testy (`uv run pytest`) nie łączą się z modelem i nie pobierają modelu do
wektoryzacji: indeks budują z `HashingEmbedder`, a aplikację uruchamiają bez
logowania (`auth=None`), co `main()` wyklucza. Plik `data/job_offers.json`
(24 fikcyjne oferty) służy już tylko testom i serwerowi MCP bez indeksu.

Use the assignments published on the course website as the source of truth.
 