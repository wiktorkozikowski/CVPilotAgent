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
- Zewnętrzny klient MCP dostaje tylko `szukaj_ofert` i `dodaj_do_listy`, bez usuwania.
