# Serwer ofert CVPilot

Osobny serwis, który trzyma oferty pracy i szuka pasujących do profilu.
Agent (`src/agent_project`) wysyła mu profil bez danych kontaktowych
i dostaje listę kandydatów. Na tej liście będą później budowane CV.

```
agent: szukaj_ofert ──HTTP POST /search {profil}──► serwer ofert
                                                   1. wektor profilu (lokalny model)
                                                   2. najbliższe oferty w indeksie
                                                      + oferty z tytułem = stanowisko
                                                   3. ocena: podobieństwo, tytuł,
                                                      umiejętności, lokalizacja,
                                                      kary za niezgodne preferencje
agent ◄──────── lista kandydatów (wynik, konflikty, fragment wymagań) ──┘
```

Wektoryzacja działa lokalnie (fastembed, model ONNX), więc profil nie trafia
do żadnej zewnętrznej usługi i nie kosztuje nic z limitu OpenRouter.

## Wymagania

- Python 3.12+ i [uv](https://docs.astral.sh/uv/)
- ok. 300 MB na model (pobierany raz do `models_cache/`) i kilka MB na indeks
- Linux, macOS albo Windows; GPU niepotrzebne

## Uruchomienie

```bash
cd server
uv sync --locked
# 1. Baza ofert i indeks (pierwszy raz z plików startowych)
uv run python -m offer_server.update --seed ../data/offers_sample_justjoin.json ../data/offers_sample_rocketjobs.json
# 2. Serwer
uv run python -m offer_server.app
```

Indeks pomija oferty po terminie ważności i duplikaty, ujednolica nazwy miast,
wycina z opisu sekcje wymagań i obowiązków, a potem wektoryzuje tytuł
i fragmenty wymagań. (`offer_server.build_index` robi to samo jednorazowo
z dowolnych plików, bez bazy w `store/`.)

## Aktualizacja co godzinę

`offer_server.update` to jeden przebieg aktualizacji. Na serwerze uruchamia go
co godzinę timer systemd (`deploy/offer-update.timer`). Przebieg:

1. Czyta bazę surowych ofert `store/offers.jsonl`. Pusta baza jest najpierw
   wypełniana z plików `--seed`.
2. **Tylko za pisemną zgodą serwisów** (`OFFER_UPDATE_WRITTEN_CONSENT=yes`):
   czyta sitemapy JustJoin i RocketJobs, usuwa oferty, których w nich już nie
   ma, i pobiera nowe. Jedno zapytanie na sekundę, najwyżej 2500 nowych ofert
   i 50 minut na przebieg (`--max-new`, `--max-minutes`). Przy odpowiedzi
   403, 429 lub 5xx przerywa do następnej godziny. Reszta czeka na kolejne
   przebiegi.
3. Przebudowuje indeks: wygasłe oferty i duplikaty odpadają, wektory
   niezmienionych ofert są kopiowane, wektoryzowane są tylko nowe teksty.
4. Zapisuje indeks jednym atomowym krokiem. Działający serwer sam go wczytuje
   (sprawdza plik co 30 s), bez restartu i bez przerwy w działaniu.

Bez zgody punkt 2 jest pomijany: przebieg tylko usuwa wygasłe oferty. Próbki
w `data/` są z 7 listopada, więc bez pobierania baza z czasem się wyczerpie.

Oferty pobrane wcześniej skryptem agenta `fetch_all.py` wczytuje się tak:

- **Pusta baza:** usługa `offer-update` ma `--seed
  /home/ubuntu/server/CVpilot/agent_project/data`, więc pierwszy przebieg sam
  wczyta wszystkie pliki z tego folderu.
- **Baza już coś ma:** dołącz je jednym przebiegiem z `--import`:

  ```bash
  cd ~/server/CVpilot/server
  set -a; . <(sudo cat /etc/cvpilot-offer-server.env); set +a   # plik ma prawa 600
  .venv/bin/python -m offer_server.update --import ~/server/CVpilot/agent_project/data
  ```

`--seed` i `--import` przyjmują pliki albo foldery (wtedy wszystkie `*.json`
i `*.jsonl` w środku). Format rozpoznają po treści, więc plik z `fetch_all.py`
po zmianie nazwy na `.json` też się wczyta. Uszkodzone linie są pomijane
i liczone, a pliki bez ofert (np. `options.json`) pomijane w całości.

Folder `agent_project/data/` jest w `.gitignore` repozytorium, żeby pobrane
oferty nie trafiły do gita.

Dwa przebiegi nigdy nie piszą naraz (plik `store/update.lock`; blokada starsza
niż 3 h jest uznawana za porzuconą).

Pierwsze pełne pobranie jest długie: sitemapy mają ok. 26 tys. ofert, przy
jednym zapytaniu na sekundę to ok. 7 h, więc rozłoży się na kilkanaście
przebiegów godzinnych. Potem przebieg pobiera tylko oferty dodane w ciągu
godziny. Wektoryzacja wszystkich 26 tys. ofert to kilkadziesiąt minut pracy
procesora; kolejne przebiegi liczą tylko nowe.

## Model i wektoryzacja

### Model

Serwer używa gotowego modelu
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
w wersji ONNX, uruchamianego przez bibliotekę
[fastembed](https://github.com/qdrant/fastembed):

- zamienia tekst na wektor 384 liczb i rozumie ok. 50 języków, w tym polski
  i angielski;
- czyta ok. 128 tokenów tekstu, a resztę ucina;
- działa na procesorze (CPU), bez GPU;
- przy pierwszym użyciu pobiera się raz (ok. 240 MB) do `models_cache/`, który
  nie trafia do gita;
- profil i oferty nie opuszczają serwera, a wektoryzacja nic nie kosztuje.

**Model nie jest douczany (fine-tuning).** Używamy go w opublikowanej postaci,
z niezmienionymi wagami. Został wcześniej wytrenowany na dużych zbiorach zdań
o podobnym znaczeniu, więc od razu umie umieszczać teksty o podobnej treści
blisko siebie. Douczenie na naszych danych wymagałoby par „profil ↔ oferta”
z oceną, czy pasują (np. z ofert dodawanych do listy wybranych); takich danych
jeszcze nie ma.

### Wektoryzacja ofert (przy każdej aktualizacji)

Kod: `chunks` w [`documents.py`](offer_server/documents.py), `build_index`
w [`index.py`](offer_server/index.py).

1. Z opisu oferty wycinane są sekcje „Wymagania” i „Obowiązki” (po nagłówkach
   typu „Nasze wymagania”, „Requirements”, „Zakres obowiązków”). Benefity
   i opis firmy są pomijane. Gdy sekcji nie ma, używany jest cały opis.
2. Oferta jest dzielona na fragmenty:
   - tytuł z firmą, np. `Data Analyst. Firma.`;
   - do 6 kawałków wymagań i obowiązków po ok. 70 słów, każdy zaczynający się
     od tytułu, żeby kawałek o „SQL” wciąż wiedział, do jakiego stanowiska
     należy.

   Dzielenie bierze się stąd, że model czyta tylko ok. 128 tokenów: cała
   oferta zostałaby ucięta po pierwszych zdaniach.
3. Każdy fragment dostaje swój wektor. Wszystkie wektory, oferty i ich opis
   trafiają do jednego pliku `index/index.npz`, który serwer trzyma w pamięci.
4. Przy kolejnej aktualizacji wektoryzowane są tylko nowe lub zmienione oferty
   (rozpoznawane po skrócie SHA-256 ich fragmentów), a resztę się kopiuje.

### Wektoryzacja profilu i porównanie (przy każdym zapytaniu)

Kod: `search` w [`ranking.py`](offer_server/ranking.py).

1. Profil z zapytania jest składany w jeden tekst. Najważniejsze części idą
   na początek, bo model czyta tylko początek: opcjonalne `query`,
   stanowiska, poziom, umiejętności, stanowiska z doświadczenia,
   podsumowanie. Przykład:
   `Data Analyst. junior. SQL, Excel, Power BI. Specjalista IT`
2. Ten tekst jest wektoryzowany **tym samym modelem** co oferty, bo tylko
   wektory z jednego modelu da się ze sobą porównać. Dlatego serwer nie
   wystartuje z indeksem zbudowanym innym modelem.
3. Wektor profilu jest porównywany z wektorami wszystkich fragmentów
   (podobieństwo cosinusowe). Oferta dostaje wynik swojego najlepiej
   pasującego fragmentu (`similarity`, zwykle 0,3–0,8).
4. Do oceny trafia 100 ofert z najwyższym podobieństwem oraz każda oferta,
   której tytuł zawiera co najmniej połowę słów docelowego stanowiska.
   Mały model potrafi przeoczyć oczywiste dopasowanie, a tak go nie zgubi.
5. Profil nie jest nigdzie zapisywany.

### Wynik końcowy

Podobieństwo wektorów to nie cały wynik. Sam mały model dawał magazynierowi
na pierwszym miejscu „Camunda Engineer” zamiast ofert „Magazynier”, więc
wspierają go proste sygnały tekstowe:

```
score = 0,6 × podobieństwo wektorów
      + 0,2 × zgodność tytułu ze stanowiskiem   (udział słów stanowiska w tytule)
      + 0,1 × umiejętności znalezione w ofercie (5 i więcej = pełna wartość)
      + 0,1 × lokalizacja                       (miasto użytkownika albo zdalnie)
score = score × 0,5 za każdą niezgodność z preferencjami
        (poziom, tryb pracy, forma zatrudnienia, za niskie wynagrodzenie)
```

Wagi są stałymi na początku `ranking.py` (`SIMILARITY_WEIGHT` i kolejne).
Czyste porównanie wektorów daje ustawienie pozostałych wag na 0.

### Zmiana modelu

Ustaw `OFFER_SERVER_MODEL` na inny model z listy fastembed. Najbliższa
aktualizacja zbuduje indeks od nowa, a serwer trzeba wtedy zrestartować, bo
zostaje przy modelu, z którym wystartował. Mocniejsze modele wielojęzyczne:

| Model | Rozmiar | Tokenów na tekst |
|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (obecny) | ok. 0,2 GB | 128 |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | ok. 1 GB | 512 |
| `google/embeddinggemma-300m` | ok. 1,2 GB | 2048 |
| `intfloat/multilingual-e5-large` | ok. 2,2 GB | 512 |

Większy model zwykle lepiej rozumie znaczenie, ale wektoryzuje wolniej
i potrzebuje więcej pamięci. Warto porównać wyniki na tych samych profilach
przed zmianą.

## Ustawienia (zmienne środowiskowe)

| Zmienna | Domyślnie | Znaczenie |
|---|---|---|
| `OFFER_SERVER_HOST` | `127.0.0.1` | Adres nasłuchu. Inny niż lokalny wymaga tokenu. |
| `OFFER_SERVER_PORT` | `8100` | Port. |
| `OFFER_SERVER_INDEX` | `index` | Folder z indeksem. |
| `OFFER_SERVER_TOKEN` | brak | Token; klient wysyła `Authorization: Bearer <token>`. |
| `OFFER_SERVER_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Model fastembed. Po zmianie przebuduj indeks. |
| `OFFER_SERVER_MODEL_CACHE` | `models_cache` | Gdzie trzymać pobrany model. |
| `OFFER_SERVER_STORE` | `store` | Folder z bazą surowych ofert (`offers.jsonl`). |
| `OFFER_UPDATE_WRITTEN_CONSENT` | `no` | `yes` pozwala aktualizacji pobierać oferty z serwisów. |

Serwer odmówi startu, gdy indeks zbudowano innym modelem niż ten, którego używa.

Po stronie agenta w `.env`:

```
OFFERS_SERVER_URL=http://adres-serwera:8100
OFFERS_SERVER_TOKEN=<ten sam token>
```

Bez `OFFERS_SERVER_URL` agent szuka lokalnie w `data/job_offers.json`.

## API

`GET /health` (bez tokenu):
`{"status": "ok", "offers": 572, "model": "...", "built_at": "..."}`

`POST /search`:

```json
{
  "profile": {
    "location": "Poznań",
    "target_roles": ["Data Analyst"],
    "skills": ["SQL", "Excel"],
    "seniority": ["junior"],
    "work_modes": ["hybrid", "remote"],
    "employment_types": ["FULL_TIME"],
    "salary_expectation": {"min": 7000, "currency": "PLN", "unit": "MONTH"},
    "summary": "…",
    "experience": [{"position": "Specjalista IT", "description": ["…"]}]
  },
  "limit": 10,
  "query": "opcjonalne doprecyzowanie, do 200 znaków"
}
```

Każde pole profilu jest opcjonalne, ale potrzebne jest stanowisko albo
umiejętność. Pola spoza listy (np. `name`, `email`, `phone`) są odrzucane
z błędem 400. To celowe: dane kontaktowe nie powinny opuszczać programu
użytkownika. `limit` to 1–20.

Odpowiedź: `{"status": "ok", "model": "...", "candidates": [...]}`, gdzie każdy
kandydat ma pola oferty (`id`, `title`, `company`, `location`, `remote`, `url`,
`employment_type`, `salary_*`) oraz `score`, `similarity`, `matched_skills`,
`conflicts` (np. `"poziom: senior"`) i `requirements` (fragment wymagań do
900 znaków, przydatny przy pisaniu CV).

Błędy: 400 (złe dane), 401 (brak lub zły token), 404 (zła ścieżka).
Serwer nie zapisuje profili ani nie loguje treści zapytań.

## Wdrożenie na VPS (Linux, systemd)

Pliki w `deploy/` są ustawione pod VPS, na którym projekt leży
w `/home/ubuntu/server/CVpilot`, działa jako użytkownik `ubuntu`, a pobrane
wcześniej oferty są w `/home/ubuntu/server/CVpilot/agent_project/data`.
Na innej maszynie zmień w trzech plikach `deploy/` użytkownika i ścieżki.

```bash
# 1. Najnowszy kod (folder server/ pojawia się po pull)
cd ~/server/CVpilot && git pull
ls ~/server/CVpilot/server                   # musi być pyproject.toml i offer_server/

# 2. Zależności (uv sam pobierze Pythona 3.12, jeśli go nie ma)
cd ~/server/CVpilot/server
uv sync --locked --no-dev                    # brak uv: curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Ustawienia
sudo cp deploy/offer-server.env.example /etc/cvpilot-offer-server.env
sudo nano /etc/cvpilot-offer-server.env      # własny token; zgoda tylko gdy ją masz
sudo chmod 600 /etc/cvpilot-offer-server.env

# 4. Usługi: serwer i aktualizacja co godzinę
sudo cp deploy/offer-server.service deploy/offer-update.service \
    deploy/offer-update.timer /etc/systemd/system/
sudo systemctl daemon-reload
# Pierwsza aktualizacja wczytuje oferty z agent_project/data, pobiera model
# (ok. 240 MB) i buduje indeks; przy dziesiątkach tysięcy ofert to nawet
# kilkadziesiąt minut. Postęp: journalctl -u offer-update -f
sudo systemctl start offer-update
sudo systemctl enable --now offer-server offer-update.timer
curl http://127.0.0.1:8100/health
systemctl list-timers offer-update.timer     # kiedy następny przebieg
journalctl -u offer-update -n 20             # wynik ostatniej aktualizacji
journalctl -u offer-server -f                # logi serwera
```

Jeśli port ma być widoczny z zewnątrz, otwórz go w zaporze (np. `sudo ufw allow 8100/tcp`),
ale lepiej wystawić go przez HTTPS (patrz niżej).

Token wygeneruj np. tak: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`.

Usługi działają jako `ubuntu`, czyli użytkownik z prawem do `sudo`. Ochrona
w plikach systemd (`ProtectSystem=strict`, `ProtectHome=read-only`,
zapis tylko w `server/`) ogranicza szkody, ale osobny użytkownik bez
uprawnień (np. `cvpilot`) byłby bezpieczniejszy, jeśli serwer ma być dostępny
z internetu.

Serwer mówi zwykłym HTTP. Gdy ma być dostępny z internetu, postaw go za
reverse proxy z HTTPS (np. Caddy albo nginx) albo udostępnij tylko w sieci
prywatnej (VPN, Tailscale). Bez HTTPS token i profil idą przez sieć jawnie.

## Testy

```bash
uv run pytest      # bez pobierania modelu: testy używają HashingEmbedder
uv run ruff check .
```

## Ograniczenia

- Mały model (ok. 128 tokenów na tekst) słabo rozumie długie opisy; obejścia
  i mocniejsze modele są opisane w sekcji „Model i wektoryzacja”.
- Zgodność tytułu działa na słowach: „Analityk danych” nie trafi w tytuł
  „Data Analyst” (pomaga tylko wektor).
- Oferty zdalne zagraniczne (np. Lizbona) nie są odsiewane.
