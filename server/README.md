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
# 1. Indeks z plików ofert (JSON z próbki albo JSON Lines z fetch_all.py)
uv run python -m offer_server.build_index ../data/offers_sample_justjoin.json ../data/offers_sample_rocketjobs.json --out index
# 2. Serwer
uv run python -m offer_server.app
```

`build_index` pomija oferty po terminie ważności (`--today RRRR-MM-DD`, domyślnie
dziś) i duplikaty, ujednolica nazwy miast, wycina z opisu sekcje wymagań
i obowiązków, a potem wektoryzuje tytuł i fragmenty wymagań. Po nowym pobraniu
ofert uruchom go ponownie i zrestartuj serwer.

## Ustawienia (zmienne środowiskowe)

| Zmienna | Domyślnie | Znaczenie |
|---|---|---|
| `OFFER_SERVER_HOST` | `127.0.0.1` | Adres nasłuchu. Inny niż lokalny wymaga tokenu. |
| `OFFER_SERVER_PORT` | `8100` | Port. |
| `OFFER_SERVER_INDEX` | `index` | Folder z indeksem. |
| `OFFER_SERVER_TOKEN` | brak | Token; klient wysyła `Authorization: Bearer <token>`. |
| `OFFER_SERVER_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Model fastembed. Po zmianie przebuduj indeks. |
| `OFFER_SERVER_MODEL_CACHE` | `models_cache` | Gdzie trzymać pobrany model. |

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

## Wdrożenie na innej maszynie (Linux, systemd)

```bash
# 1. Użytkownik bez uprawnień i kod
sudo useradd --system --create-home --shell /usr/sbin/nologin cvpilot
sudo git clone https://github.com/wiktorkozikowski/CVPilotAgent.git /opt/cvpilot
sudo chown -R cvpilot:cvpilot /opt/cvpilot

# 2. uv i zależności, jako użytkownik cvpilot (uv sam pobierze Pythona 3.12)
sudo -u cvpilot sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
cd /opt/cvpilot/server
sudo -u cvpilot ~cvpilot/.local/bin/uv sync --locked --no-dev

# 3. Indeks (za pierwszym razem pobiera model, ok. 240 MB)
sudo -u cvpilot .venv/bin/python -m offer_server.build_index \
    ../data/offers_sample_justjoin.json ../data/offers_sample_rocketjobs.json --out index

# 4. Usługa
sudo cp deploy/offer-server.service /etc/systemd/system/
sudo cp deploy/offer-server.env.example /etc/cvpilot-offer-server.env
sudo nano /etc/cvpilot-offer-server.env      # wpisz własny token
sudo chmod 600 /etc/cvpilot-offer-server.env
sudo systemctl daemon-reload
sudo systemctl enable --now offer-server
curl http://127.0.0.1:8100/health
journalctl -u offer-server -f                # logi, gdyby coś nie działało
```

Jeśli port ma być widoczny z zewnątrz, otwórz go w zaporze (np. `sudo ufw allow 8100/tcp`),
ale lepiej wystawić go przez HTTPS (patrz niżej).

Token wygeneruj np. tak: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`.

Serwer mówi zwykłym HTTP. Gdy ma być dostępny z internetu, postaw go za
reverse proxy z HTTPS (np. Caddy albo nginx) albo udostępnij tylko w sieci
prywatnej (VPN, Tailscale). Bez HTTPS token i profil idą przez sieć jawnie.

## Testy

```bash
uv run pytest      # bez pobierania modelu: testy używają HashingEmbedder
uv run ruff check .
```

## Ograniczenia

- Mały model (ok. 128 tokenów na tekst) słabo rozumie długie opisy. Dlatego
  oferta jest dzielona na fragmenty i liczy się najlepszy z nich, a wynik
  wspierają tytuł i umiejętności znalezione w tekście. Lepszy, ale cięższy
  model to np. `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  (ok. 1 GB) albo `google/embeddinggemma-300m` (ok. 1,2 GB).
- Zgodność tytułu działa na słowach: „Analityk danych” nie trafi w tytuł
  „Data Analyst” (pomaga tylko wektor).
- Oferty zdalne zagraniczne (np. Lizbona) nie są odsiewane.
