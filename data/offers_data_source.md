# Źródło i format danych: offers_sample_*.json

Notatka dla innego modelu LLM, który ma pracować z tymi plikami. Opisuje, skąd
pochodzą dane, jak zostały pobrane, jaki mają format i czego w nich nie ma.

## Pliki

| Plik | Serwis | Liczba ofert | Źródłowa sitemapa (stan z 2026-11-07) |
|---|---|---|---|
| `offers_sample_justjoin.json` | https://justjoin.it | 300 | 11 161 adresów |
| `offers_sample_rocketjobs.json` | https://rocketjobs.pl | 297 | 15 045 adresów |

Każdy plik to lista JSON (UTF-8) rekordów o tym samym schemacie (patrz niżej).
To **losowa próbka**, nie cała baza. Losowanie: `random.Random(42).sample(...)` na
liście adresów z sitemapy, więc jest powtarzalne. RocketJobs ma 297 zamiast 300,
bo 3 strony nie zawierały bloku `JobPosting` (prawdopodobnie wygasłe oferty).
Dane pobrano 2026-11-07.

## Jak dane zostały pobrane

Kod: `src/agent_project/fetch_offers.py`
(`python -m agent_project.fetch_offers justjoin|rocketjobs [liczba]`).
Tylko biblioteka standardowa Pythona.

1. Pobranie sitemapy `https://<serwis>/sitemaps/active-jobs.xml`. To indeks, który
   przekierowuje na `public.justjoin.com` i wskazuje pliki `active-jobs/part*.xml`
   z adresami aktywnych ofert (JustJoin: `/job-offer/<slug>`,
   RocketJobs: `/oferta-pracy/<slug>`).
2. Wylosowanie próbki adresów.
3. Pobranie każdej strony oferty (HTTP GET, po jednej naraz, 1 s przerwy,
   User-Agent `CVPilotAgent-student-project`, do 4 prób przy błędach sieci).
4. Wyciągnięcie z HTML bloku `<script type="application/ld+json">` typu
   `JobPosting` (schema.org), który serwis sam osadza na stronie, i spłaszczenie
   go do rekordu. Pole `description` jest HTML-em zamienionym na czysty tekst
   (jedna linia = jeden akapit lub punkt listy).

Zasady przyjęte przy pobieraniu: `robots.txt` obu serwisów blokuje `/api/`
i adresy list z filtrami (`/oferty-pracy/*,*`), więc z nich nie korzystano.
Użyto tylko sitemap i stron pojedynczych ofert, które nie są zablokowane.
Regulaminów serwisów nie analizowano.

## Schemat rekordu

| Pole | Typ | Znaczenie |
|---|---|---|
| `id` | string | Ostatni segment adresu oferty (slug). Unikalny w obrębie serwisu. |
| `source` | string | `justjoin` lub `rocketjobs`. |
| `url` | string | Pełny adres oferty. |
| `title` | string | Tytuł stanowiska. |
| `company` | string | Nazwa firmy (`hiringOrganization.name`). |
| `location` | string | Miasto z `jobLocation.address.addressLocality`. Tylko pierwsze miejsce, jeśli jest ich kilka. |
| `remote` | bool | `true`, gdy `jobLocationType == "TELECOMMUTE"`. Oferta zdalna nadal może mieć wpisane miasto. |
| `employment_type` | list[string] \| null | Np. `FULL_TIME`, `CONTRACTOR`, `PART_TIME`, `INTERN`. |
| `salary_min`, `salary_max` | number \| null | Widełki; `null`, gdy oferta ich nie podaje. |
| `salary_currency` | string \| null | Głównie `PLN`, sporadycznie `EUR`, `USD`. |
| `salary_unit` | string \| null | `HOUR`, `DAY`, `MONTH` lub `YEAR`. Widełki z różnych ofert nie są porównywalne bez przeliczenia. |
| `date_posted` | string (ISO 8601) | Data z `datePosted`. |
| `valid_through` | string (ISO 8601) | Data ważności oferty. |
| `description` | string | Pełny opis oferty jako zwykły tekst. |

## Czego w danych nie ma

- **Brak osobnego pola z wymaganymi umiejętnościami** (`required_skills`).
  Wymagania są tylko w tekście `description`. Sekcję o nazwie typu
  "Requirements", "Must have", "Wymagania", "Twój profil" znaleziono
  w około 221 z 300 ofert JustJoin i 196 z 297 ofert RocketJobs (przybliżone
  wyszukiwanie po nagłówkach, więc faktyczna liczba może być wyższa).
  Wyciągnięcie listy technologii, stażu i poziomu wymaga przetworzenia tekstu.
- Brak kategorii i poziomu (junior/mid/senior) jako pól. Kategoria jest zwykle
  zapisana w końcówce slugu w `url`, poziom często w `title`.
- Brak informacji o formie pracy poza flagą `remote` (nie ma trybu hybrydowego).

## Znane cechy i pułapki

- Widełki podaje tylko część ofert: 102 z 300 (JustJoin) i 77 z 297 (RocketJobs).
- `location` bywa niejednolite, np. "Kraków" i "Krakow" jako osobne wartości.
- JustJoin to głównie oferty IT. RocketJobs zawiera też oferty spoza IT
  (np. pakowacz, farmaceuta), więc nie każda pasuje do profilu technicznego.
- `date_posted` w próbce jest niemal zawsze z ostatniego tygodnia, więc to
  prawdopodobnie data odświeżenia ogłoszenia, a nie pierwszej publikacji
  (wniosek z rozkładu dat, nie potwierdzony przez serwis). Nie nadaje się do
  ustalania, które oferty są naprawdę nowe.
- Ta sama oferta może występować pod kilkoma adresami (np. różne kategorie),
  więc przy większym zbiorze potrzebne jest usuwanie duplikatów.
- Opisy są w różnych językach (polski i angielski) i mają różną strukturę.

## Różnica względem `job_offers.json`

`job_offers.json` to ręcznie napisane oferty testowe w formacie modelu `Offer`
(z polem `required_skills`). Pliki `offers_sample_*.json` to prawdziwe oferty
w innym schemacie (bez `required_skills`, za to z widełkami, datami
i pełnym opisem). Nie są ze sobą zgodne bez konwersji.
