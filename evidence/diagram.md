# Co model może zażądać, a co kontroluje aplikacja

```mermaid
flowchart LR
    subgraph MODEL["Model (LLM)"]
        M1["Prośba: szukaj_ofert(limit)"]
        M2["Prośba: dodaj_do_listy(offer_id)"]
        M3["Prośba: usun_z_listy(offer_id)"]
        M4["Odpowiedź tekstowa dla użytkownika"]
    end

    subgraph APP["Aplikacja w Pythonie (kontrola)"]
        C1["Walidacja limit: 1-10"]
        C2["Walidacja offer_id: niepusty tekst"]
        C3["Sprawdzenie, czy oferta istnieje"]
        C4["Sprawdzenie duplikatu na liście"]
        C5["Potwierdzenie użytkownika (callback)"]
        D1[("profile.json, job_offers.json\n(tylko odczyt)")]
        D2[("shortlist.json\n(zapis)")]
    end

    M1 --> C1 --> D1
    M2 --> C2 --> C3 --> C4 --> D2
    M3 --> C2 --> C5 --> D2
    D1 -- "wynik: status ok + oferty" --> M4
    D2 -- "wynik: added / already_exists / not_found / removed / confirmation_required" --> M4
```

## Czego model nie kontroluje

- Potwierdzenia usunięcia: argument `confirmed` podany przez model jest ignorowany. Potwierdzenie pochodzi tylko z callbacku aplikacji (`confirm`).
- Ścieżek do plików i danych: model podaje tylko `limit` i `offer_id`.
- Wag dopasowania: są stałe w `matching.py`.
- Zewnętrzny klient MCP dostaje tylko `szukaj_ofert`, `dodaj_do_listy`, `pobierz_profil` i `zapisz_pola_profilu`, bez usuwania i bez zatwierdzania profilu.

## Budowanie profilu w rozmowie

```mermaid
flowchart LR
    subgraph MODEL2["Model (LLM)"]
        P1["Prośba: zapisz_pola_profilu(name, location, ...)"]
        P2["Prośba: pobierz_profil()"]
        P3["Prośba: zatwierdz_profil()"]
        P4["Pytanie do użytkownika albo podsumowanie"]
    end

    subgraph APP2["Aplikacja w Pythonie (kontrola)"]
        Q1["Walidacja pól: ProfileDraft (typy, długości, nieznane pola)"]
        Q2["Brak zapisu, jeśli którekolwiek pole jest błędne"]
        Q3["Lista brakujących pól z kodu, nie z tekstu modelu"]
        Q4["Zapis tylko przy kompletnym profilu"]
        Q5["Potwierdzenie użytkownika (callback confirm)"]
        DR[("profile_draft.json\n(robocze, zapis)")]
        PR[("profiles/<id>.json\n(zapis końcowy)")]
    end

    P1 --> Q1 --> Q2 --> DR
    P2 --> Q3
    P3 --> Q4 --> Q5 --> PR
    DR -- "validation_error / ok + missing_fields" --> P4
    PR -- "saved / incomplete / confirmation_required" --> P4
```

- Model nie zatwierdza profilu argumentami. Zapis wymaga `confirm()` z aplikacji, a argument `confirmed` od modelu jest ignorowany.
