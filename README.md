# tera-monitor-web
Lekki panel www (Flask) do monitorowania TERA Classic: zlicza graczy online, mierzy ping do serwera, wyswietla wykresy, prowadzi statystyki uptime/downtime i potrafi wysylac powiadomienia (np. Home Assistant).

## Co robi aplikacja
- Co `SAMPLE_SECONDS` sekund pobiera liczbe graczy z `https://tera-europe-classic.de/stats.php` oraz mierzy ping do hosta 103.249.70.12 (ICMP lub TCP fallback).
- Dane zapisywane sa w SQLite (`data/monitor.db`) i udostepniane jako wykresy Matplotlib (`/plot_players/<minuty>`, `/plot_ping/<minuty>`) oraz sparklines (`/spark/...`).
- Widok glowny (`/`) pokazuje status ONLINE/OFFLINE, rekord graczy, min/avg/max pingu, czasy uptime/downtime, ostatnie zdarzenia i mini-wykresy.
- Eksport/import: `admin/import_csv`, `export/players.csv`, `export/ping.csv`, `export/db`, `report.csv` (SLA dzien/tydzien/miesiac).
- Powiadomienia: integracja z Home Assistant (push), alert wysokiego pingu (`PING_ALERT_MS`, cooldown `PING_ALERT_COOLDOWN`) i zmiany statusu serwera.
- Dwa jezyki UI (PL/EN) wybierane parametrem `?lang=...`.

## Struktura repo
- `app.py` — glowna aplikacja Flask, watek zbierajacy probki, logika SLA, wykresy, API JSON/CSV.
- `templates/` — widoki Jinja (`index.html`, `layout.html`, przyklad modularny).
- `static/` — style (w tym motyw TERA) i JS (aktualizacja danych, predykcja).
- `data/` — przykladowa baza `monitor.db`, logi CSV, ustawienia (`config.json`, `notify_state.json`, `records.json`); katalog tworzony automatycznie.
- `requirements.txt` — zaleznosci Pythona.
- `Dockerfile`, `docker-compose.yml` — obraz z Python 3.11 slim, port 9999, healthcheck, montaz wolumenu roboczego.

## Konfiguracja (zmienne srodowiskowe)
- `APP_TZ` (domyslnie `Europe/Warsaw`) — strefa czasowa wykresow/raportow.
- `SAMPLE_SECONDS` (dom. 5) — czestotliwosc pomiarow ping/players.
- `GLITCH_SECONDS` (dom. 15) — scalanie krotkich fluktuacji statusu ONLINE/OFFLINE.
- Powiadomienia HA: `HA_BASE_URL`, `HA_TOKEN`, `HA_NOTIFY_SERVICES` (lista svc, np. `notify.mobile_app,notify.desktop`), `HA_VERIFY_SSL` (`1`/`0`), `ENABLE_STATUS_NOTIFY`, `ENABLE_PING_NOTIFY`, `PING_ALERT_MS`, `PING_ALERT_COOLDOWN`.

## Uruchomienie lokalne (Python)
1) `python -m venv .venv`  
2) `.\.venv\Scripts\Activate.ps1` (Windows)  
3) `pip install -r requirements.txt`  
4) `python app.py` (domyslnie port `9999`)

## Uruchomienie w kontenerze
- Build: `docker build -t tera-monitor-web .`
- Run: `docker run -p 9999:9999 -v ${PWD}:/app --env-file .env --name tera-monitor-web tera-monitor-web`
- Docker Compose: `docker-compose up --build`

## Dodatkowe notatki
- W katalogu `data/` znajduja sie przykladowe dane; w produkcji mozesz podpiac wlasny wolumen, aby zachowac baze i konfiguracje.
