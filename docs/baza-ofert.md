# Baza ofert i wyszukiwanie

Oferty, ich wektory i model do wektoryzacji leżą na tej samej maszynie co
aplikacja, w folderze `var/` (poza gitem). Aplikacja (`agent_project.web`)
wczytuje indeks do pamięci przy starcie i sama podmienia go na nowszy, gdy
godzinna aktualizacja zapisze nowy plik.

```
var/store/offers.jsonl   surowe oferty (offer_server.update)
var/index/index.npz      oczyszczone oferty + wektory fragmentów
var/models_cache/        model ONNX (fastembed), pobierany raz, ok. 240 MB
var/account.json         jedyne konto aplikacji (agent_project.auth)
```

Wektoryzacja działa lokalnie, więc profil nie trafia do żadnej zewnętrznej
usługi i nie kosztuje nic z limitu OpenRouter. Do OpenRouter idzie tylko
rozmowa z modelem językowym.

## Jak agent korzysta z bazy

Model nie dostaje gotowej listy. Ma narzędzia tylko do odczytu
(`src/agent_project/agent.py`, `offer_db.py`):

| Narzędzie | Co robi |
|---|---|
| `szukaj_ofert` | ranking dla całego profilu (opis niżej), punkt startowy |
| `przeszukaj_oferty` | wyszukiwanie po znaczeniu zapytania, które model pisze sam, z filtrami: miasto, zdalnie, poziom, forma zatrudnienia; pomija oferty już widziane |
| `pokaz_oferte` | jedna oferta w całości (wymagania, obowiązki, link, termin) |
| `porownaj_z_profilem` | które umiejętności z profilu oferta wymienia, których nie, zgodność tytułu, niezgodności z preferencjami |
| `wybierz_oferty` | wybór modelu z powodem przy każdej ofercie; to on trafia do tabeli |

Każde narzędzie sprawdza argumenty w kodzie. Model nie pisze zapytań do bazy
(np. SQL), nie widzi danych kontaktowych z profilu i niczego w bazie nie
zmienia. Przebieg ma limit 12 kroków i 0,10 USD (limity kursu), więc model nie sprawdza
każdej oferty osobno.

## Pierwsze uruchomienie

```bash
uv sync --locked
# Baza i indeks z plików startowych (pobiera model przy pierwszym razie)
uv run python -m offer_server.update --seed data/offers_sample_justjoin.json data/offers_sample_rocketjobs.json
```

Indeks pomija oferty po terminie ważności i duplikaty, ujednolica nazwy miast,
wycina z opisu sekcje wymagań i obowiązków, a potem wektoryzuje tytuł
i fragmenty wymagań. (`offer_server.build_index` robi to samo jednorazowo
z dowolnych plików, bez bazy w `var/store/`.)

## Aktualizacja co godzinę

`offer_server.update` to jeden przebieg aktualizacji. Na serwerze uruchamia go
co godzinę timer systemd (`deploy/offer-update.timer`). Przebieg:

1. Czyta bazę surowych ofert `var/store/offers.jsonl`. Pusta baza jest najpierw
   wypełniana z plików `--seed`.
2. **Tylko za pisemną zgodą serwisów** (`OFFER_UPDATE_WRITTEN_CONSENT=yes`):
   czyta sitemapy JustJoin i RocketJobs, usuwa oferty, których w nich już nie
   ma, i pobiera nowe. Jedno zapytanie na sekundę, najwyżej 2500 nowych ofert
   i 50 minut na przebieg (`--max-new`, `--max-minutes`). Przy odpowiedzi
   403, 429 lub 5xx przerywa do następnej godziny. Reszta czeka na kolejne
   przebiegi.
3. Przebudowuje indeks: wygasłe oferty i duplikaty odpadają, wektory
   niezmienionych ofert są kopiowane, wektoryzowane są tylko nowe teksty.
4. Zapisuje indeks jednym atomowym krokiem. Działająca aplikacja sama go wczytuje
   (sprawdza plik co 30 s), bez restartu i bez przerwy w działaniu.

Bez zgody punkt 2 jest pomijany: przebieg tylko usuwa wygasłe oferty. Próbki
w `data/` są z 7 listopada, więc bez pobierania baza z czasem się wyczerpie.

Oferty pobrane wcześniej skryptem `fetch_all.py` wczytuje się tak:

- **Pusta baza:** usługa `offer-update` ma `--seed .../data`, więc pierwszy
  przebieg sam wczyta wszystkie pliki z ofertami z folderu `data/`.
- **Baza już coś ma:** dołącz je jednym przebiegiem z `--import`:

  ```bash
  cd ~/server/CVpilot
  .venv/bin/python -m offer_server.update --import <folder z plikami z fetch_all.py>
  ```

`--seed` i `--import` przyjmują pliki albo foldery (wtedy wszystkie `*.json`
i `*.jsonl` w środku). Format rozpoznają po treści. Uszkodzone linie są
pomijane i liczone, a pliki bez ofert (np. `options.json`, profile) pomijane
w całości.

Dwa przebiegi nigdy nie piszą naraz (plik `var/store/update.lock`; blokada starsza
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
- przy pierwszym użyciu pobiera się raz (ok. 240 MB) do `var/models_cache/`, który
  nie trafia do gita;
- profil i oferty nie opuszczają serwera, a wektoryzacja nic nie kosztuje.

**Model nie jest douczany (fine-tuning).** Używamy go w opublikowanej postaci,
z niezmienionymi wagami. Został wcześniej wytrenowany na dużych zbiorach zdań
o podobnym znaczeniu, więc od razu umie umieszczać teksty o podobnej treści
blisko siebie. Douczenie na naszych danych wymagałoby par „profil ↔ oferta”
z oceną, czy pasują (np. z ofert dodawanych do listy wybranych); takich danych
jeszcze nie ma.

### Wektoryzacja ofert (przy każdej aktualizacji)

Kod: `chunks` w [`documents.py`](../src/offer_server/documents.py), `build_index`
w [`index.py`](../src/offer_server/index.py).

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
   trafiają do jednego pliku `var/index/index.npz`, który aplikacja trzyma w pamięci.
4. Przy kolejnej aktualizacji wektoryzowane są tylko nowe lub zmienione oferty
   (rozpoznawane po skrócie SHA-256 ich fragmentów), a resztę się kopiuje.

### Wektoryzacja profilu i porównanie (przy każdym zapytaniu)

Kod: `search` w [`ranking.py`](../src/offer_server/ranking.py).

1. Profil jest składany w jeden tekst. Najważniejsze części idą
   na początek, bo model czyta tylko początek: opcjonalne `query`,
   stanowiska, poziom, umiejętności, stanowiska z doświadczenia,
   podsumowanie. Przykład:
   `Data Analyst. junior. SQL, Excel, Power BI. Specjalista IT`
2. Ten tekst jest wektoryzowany **tym samym modelem** co oferty, bo tylko
   wektory z jednego modelu da się ze sobą porównać. Dlatego aplikacja nie
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

Ustaw `OFFERS_MODEL` na inny model z listy fastembed. Najbliższa
aktualizacja zbuduje indeks od nowa, a aplikację trzeba wtedy zrestartować, bo
zostaje przy modelu, z którym wystartowała. Mocniejsze modele wielojęzyczne:

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
| `OFFERS_INDEX_DIR` | `var/index` | Folder z indeksem. |
| `OFFERS_STORE_DIR` | `var/store` | Folder z bazą surowych ofert (`offers.jsonl`). |
| `OFFERS_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Model fastembed. Po zmianie przebuduj indeks. |
| `OFFERS_MODEL_CACHE` | `var/models_cache` | Gdzie trzymać pobrany model. |
| `OFFER_UPDATE_WRITTEN_CONSENT` | `no` | `yes` pozwala aktualizacji pobierać oferty z serwisów. |

Aplikacja odmówi startu, gdy indeks zbudowano innym modelem niż ten, którego
używa.

## Ograniczenia

- Mały model (ok. 128 tokenów na tekst) słabo rozumie długie opisy; obejścia
  i mocniejsze modele są opisane w sekcji „Model i wektoryzacja”.
- Zgodność tytułu działa na słowach: „Analityk danych” nie trafi w tytuł
  „Data Analyst” (pomaga tylko wektor).
- Oferty zdalne zagraniczne (np. Lizbona) nie są odsiewane.
