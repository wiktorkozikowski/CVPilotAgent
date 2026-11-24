# CVPilotAgent

Agent, który pomaga szukać pracy. Z rozmowy buduje profil kandydata,
przegląda bazę prawdziwych ofert i pokazuje te najlepiej dopasowane, z oceną
i powodem przy każdej. Projekt na kurs **Designing LLM/VLM Agents**.

- [Project Brief](#project-brief)
- [Jak to działa](#jak-to-działa)
- [Uruchomienie lokalne](#uruchomienie-lokalne)
- [Serwer](#serwer)
- [Konto i hasło](#konto-i-hasło)
- [Testy i polecenia](#testy-i-polecenia)
- [Zasady kursu](#zasady-kursu)

## Project Brief

- **Project title:** CVPilotAgent, agent tworzący spersonalizowane CV
- **Intended user:** osoba szukająca pracy, np. junior analityk lub developer
- **Problem and primary task:** użytkownik opisuje swoje umiejętności
  w rozmowie (profil zapisuje się jako JSON). Agent wyszukuje pasujące oferty
  pracy i pokazuje je posortowane wg dopasowania. Kolejne etapy (w planie):
  generowanie CV dopasowanego do ofert i wysłanie go po akceptacji
  użytkownika.
- **Why an agent is appropriate:** wybór następnego kroku zależy od wyników
  poprzednich, np. ile ofert znaleziono i których umiejętności brakuje.
- **Planned data:** `data/profile.json` (profil użytkownika) i
  `data/job_offers.json` (przykładowe oferty). Dane są fikcyjne. Prawdziwy
  profil nie trafia do repozytorium.
- **Tools:** do odczytu bazy ofert `szukaj_ofert` (ranking dla profilu),
  `przeszukaj_oferty` (własne zapytanie i filtry), `pokaz_oferte`,
  `porownaj_z_profilem` i `wybierz_oferty` (oferty wybrane przez agenta
  spośród 100 kandydatów z rankingu, z powodami); `dodaj_do_listy` (odwracalny zapis)
  i `usun_z_listy` (wymaga potwierdzenia użytkownika w aplikacji).
  `szukaj_ofert` i `dodaj_do_listy` są też w serwerze MCP, usuwanie nie.
  Każde narzędzie waliduje argumenty w kodzie.
- **Status:** zrealizowane wyszukiwanie, lista wybranych, pętla z limitem
  kroków (24) i kosztu (0,10 USD na uruchomienie), interfejs w przeglądarce
  (`agent_project.web`) z logowaniem na jedno konto i przebiegiem agenta pod
  wynikami. Cały projekt działa jako jedna aplikacja na serwerze, a agent
  przegląda wektorową bazę prawdziwych ofert. Dalej: generowanie CV
  i wysyłka, dopiero po zgodzie użytkownika.
- **Evidence:** `evidence/live_*.jsonl` to przebiegi z prawdziwym modelem
  (`uv run python -m agent_project.demo_runs`), `flow_trace.jsonl`
  i `confirmation_trace.jsonl` to przebiegi z mockiem.
- **Planned visual input:** brak na tym etapie.
- **Success criteria:** dla przykładowego profilu oferty są uporządkowane
  według dopasowania umiejętności, a oferty bez wspólnych umiejętności nie
  pojawiają się na liście. Testy przechodzą bez połączenia z modelem.
- **Non-goals and safety boundaries:** agent nie wysyła CV ani nie
  odpowiada na oferty bez zgody użytkownika. Nie scrapuje serwisów z ofertami
  bez zgody ich regulaminu. Nie zapisuje danych osobowych w śladach.

## Jak to działa

Cały projekt to jedna aplikacja na jednej maszynie: strona z logowaniem,
pętla agenta, baza ofert z indeksem wektorowym i lokalny model do
wektoryzacji (fastembed). Model językowy to `google/gemma-4-26b-a4b-it`
przez OpenRouter. Użytkownik korzysta z aplikacji tylko przez przeglądarkę,
po zalogowaniu.

```
przeglądarka ──HTTPS / tunel SSH──► agent_project.web (127.0.0.1:8000)
                                   ├─ logowanie (jedno konto, var/account.json)
                                   ├─ agent ──► OpenRouter (gemma)
                                   │    └─ narzędzia: szukaj_ofert, przeszukaj_oferty,
                                   │       pokaz_oferte, porownaj_z_profilem, wybierz_oferty
                                   └─ baza ofert w pamięci (var/index, fastembed)
offer-update.timer (co godzinę) ──► offer_server.update ──► var/store, var/index
```

**Wyszukiwanie** ma dwa etapy:

1. **Kandydaci z kodu.** Ranking liczy dopasowanie całej bazy do profilu
   (podobieństwo treści, stanowisko, umiejętności, lokalizacja) i wybiera
   100 najlepszych ofert. Oferty innego poziomu niż profil odpadają już tutaj.
2. **Wybór agenta.** Agent przegląda tych 100 kandydatów, sprawdza wątpliwe
   oferty (`porownaj_z_profilem`, `pokaz_oferte`) i w `wybierz_oferty` wybiera
   tylko naprawdę dobrze dopasowane, najwyżej tyle, ile chcesz, każdą
   z powodem.

Tabela pokazuje tylko wybór agenta. Wysoki wynik rankingu nie wystarcza:
oferta musi pasować stanowiskiem, kluczowymi wymaganiami, poziomem
i lokalizacją. Jeśli dobrych ofert jest mniej, agent pokazuje mniej, a jeśli
nie ma żadnej, pisze, że nie znalazł dopasowań i dlaczego. Szczegóły bazy,
wektoryzacji i aktualizacji ofert: [docs/baza-ofert.md](docs/baza-ofert.md).

**Interfejs:**

- **Profil w rozmowie.** Opowiadasz o sobie zwykłym tekstem, a model pyta
  o brakujące pola. Profil zapisuje się dopiero po kliknięciu „Zatwierdź
  profil”; sam model nie może go zapisać. W rozmowie model może też
  przeglądać bazę ofert (np. „jakie są oferty z Power BI w Krakowie?”).
- **Szukaj ofert.** Tabela pokazuje oferty wybrane przez agenta z powodem
  przy każdej („spoza rankingu” przy ofertach, które agent znalazł własnym
  zapytaniem). Nad tabelą jest liczba wybranych spośród kandydatów albo
  informacja, że dobrych dopasowań nie ma. Pod tabelą jest przebieg każdego
  kroku i ścieżka do śladu w `traces/`.
- **Lista wybranych.** Każdy profil ma własną listę
  (`data/shortlists/<id>.json`). Usunięcie oferty wymaga potwierdzenia
  w oknie przeglądarki.

**Ograniczenia każdego uruchomienia:** najwyżej 24 kroki modelu
(`MAX_STEPS`) i 0,10 USD (`MAX_RUN_COST_USD`, oba w `agent.py`). Kilka
niezależnych narzędzi, o które model prosi w jednej odpowiedzi, wykonuje się
w jednym kroku. Uwaga: 24 kroki to więcej niż limit kursu (12).

## Uruchomienie lokalne

Windows albo Linux:

```bash
uv sync --locked
cp .env.example .env                     # wpisz MODEL_NAME i OPENROUTER_API_KEY
uv run python -m offer_server.update --seed data/offers_sample_justjoin.json data/offers_sample_rocketjobs.json
uv run python -m agent_project.auth      # konto do logowania
uv run python -m agent_project.web
```

Na Windows wystarczy `powershell -ExecutionPolicy Bypass -File start.ps1`:
sam zbuduje indeks i poprosi o konto, jeśli ich brakuje. Potem otwórz
http://127.0.0.1:8000 i zaloguj się. Pierwsza aktualizacja pobiera model do
wektoryzacji (ok. 240 MB) do `var/models_cache/`.

Plik `.env` zawiera klucz OpenRouter: nie commituj go i nie wpisuj klucza do
kodu, śladów ani raportów.

## Serwer

Pliki w `deploy/` zakładają repozytorium sklonowane do
`/home/ubuntu/server/CVpilot` i użytkownika `ubuntu`. Na innej maszynie zmień
ścieżki i użytkownika.

### Pierwsza instalacja

```bash
cd ~/server/CVpilot && git pull && uv sync --locked --no-dev
sudo cp deploy/cvpilot.env.example /etc/cvpilot.env
sudo nano /etc/cvpilot.env && sudo chmod 600 /etc/cvpilot.env   # klucz OpenRouter
.venv/bin/python -m agent_project.auth                          # konto
sudo cp deploy/cvpilot.service deploy/offer-update.service deploy/offer-update.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start offer-update          # pierwsza baza i indeks; postęp: journalctl -u offer-update -f
sudo systemctl enable --now cvpilot offer-update.timer
journalctl -u cvpilot -f                   # logi aplikacji
```

Stary serwer ofert (`offer-server.service`, port 8100) jest już
niepotrzebny: `sudo systemctl disable --now offer-server`.

### Połączenie

Aplikacja słucha tylko na `127.0.0.1:8000`, więc adres
`http://ADRES_SERWERA:8000` nie działa. Portu 8000 nie otwiera się w zaporze:
bez szyfrowania hasło szłoby przez sieć jawnie. Są dwa sposoby, żeby się
połączyć.

**Tunel SSH (bez domeny).** Na swoim komputerze otwórz tunel i zostaw okno
otwarte:

```bash
ssh -L 8000:127.0.0.1:8000 ubuntu@ADRES_SERWERA
```

Potem wejdź w przeglądarce na http://127.0.0.1:8000. Tunel trzeba otworzyć
ponownie po zamknięciu okna, zerwaniu połączenia albo restarcie serwera;
restart samej aplikacji go nie przerywa.

**Domena z HTTPS.** Caddy wystawia aplikację do internetu i sam zdobywa
certyfikat. Skopiuj `deploy/Caddyfile.example` do `/etc/caddy/Caddyfile`,
wpisz swoją domenę (jej rekord DNS musi wskazywać na serwer), otwórz porty 80
i 443 i uruchom `sudo systemctl reload caddy`.

### Aktualizacja

```bash
cd ~/server/CVpilot && git pull && uv sync --locked --no-dev && sudo systemctl restart cvpilot
git log --oneline -1 && systemctl status cvpilot   # wersja i stan aplikacji
```

- `git pull` pobiera tylko gałąź, na której jest serwer
  (`git branch --show-current`). Nowa gałąź na GitHubie wymaga
  `git checkout <gałąź>` albo scalenia jej z tą śledzoną.
- `uv sync` doinstalowuje zależności zgodnie z `uv.lock` (`--no-dev` pomija
  narzędzia programistyczne).
- Zmienione na serwerze pliki danych (np. `data/profiles/`) zostają. Jeśli
  `git pull` zgłosi, że nadpisałby zmiany: `git stash`, `git pull`,
  `git stash pop`.
- Restart kończy sesje, więc po nim trzeba się zalogować ponownie.

## Konto i hasło

Konto jest jedno i nie da się zarejestrować przez stronę. Zakłada się je na
maszynie z aplikacją; ta sama komenda ustawia nowe hasło, gdy zapomnisz
starego (hasła nie da się odczytać):

```bash
uv run python -m agent_project.auth      # lokalnie
cd ~/server/CVpilot && .venv/bin/python -m agent_project.auth && sudo systemctl restart cvpilot   # na serwerze
```

Komenda pyta o login i hasło (min. 10 znaków). W `var/account.json`
zapisuje się tylko skrót hasła (scrypt). Sesja trwa 12 h i kończy się przy
restarcie aplikacji. Po 5 złych hasłach z rzędu logowanie jest zablokowane na
15 minut; restart aplikacji zdejmuje blokadę. Ciasteczko sesji ma
`HttpOnly`, `SameSite=Strict` i `Secure`, a zapytania z innych stron są
odrzucane.

## Testy i polecenia

```bash
uv run pytest
uv run ruff check .
uv run python -m agent_project.demo_runs   # przebiegi z prawdziwym modelem do evidence/
```

Testy nie łączą się z modelem i nie pobierają modelu do wektoryzacji: indeks
budują z `HashingEmbedder`, a aplikację uruchamiają bez logowania
(`auth=None`), czego `main()` nie dopuszcza. Plik `data/job_offers.json`
(24 fikcyjne oferty) służy już tylko testom i serwerowi MCP bez indeksu.

Najważniejsze pliki:

- `src/agent_project/agent.py`: pętla agenta, narzędzia, prompty i limity;
- `src/agent_project/web.py`: serwer strony, logowanie, wyszukiwanie;
- `src/agent_project/offer_db.py`: dostęp agenta do bazy ofert;
- `src/agent_project/llm.py`: klient OpenRouter i klient testowy (mock);
- `src/offer_server/`: pobieranie ofert, indeks wektorowy i ranking;
- `deploy/`: usługi systemd i konfiguracja Caddy;
- `traces/`: ślady uruchomień (poza Gitem), `evidence/`: ślady do oddania.

## Zasady kursu

- One model for all live work: `google/gemma-4-26b-a4b-it`.
- USD 2 per key for the whole course, enforced by OpenRouter.
- In your own code: at most 12 model steps and USD 0.10 per agent run.

Praca z modelem kursu:

- The model usually requests one tool call per turn. Tell it in the system
  prompt that independent calls may be requested together.
- For typed output, offer one tool whose parameters are your Pydantic schema
  and validate its arguments. Keep `tool_choice` at its default, `"auto"`,
  and give lists a maximum length.
- Results vary between identical runs. Run your fixed evaluation set at least
  three times and report the worst run as well as the average.
- A typical agent run costs well under USD 0.01. Use the mock client in tests.
- Copy the traces you submit to `evidence/`. Check them first for keys,
  personal data, and absolute file paths.

Use the assignments published on the course website as the source of truth.
