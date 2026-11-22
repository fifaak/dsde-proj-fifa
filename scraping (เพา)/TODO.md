# TODO — scraping (เพา)

**Criterion:** Required task, graded separately from DE. Needs **≥3 sources** via simple scraping/APIs,
**or 1 source** that needs Playwright/Selenium/Crawlee. More relevant sources can earn extra points.
**Target date:** Sun 29 Nov 2026, so the pipeline, analysis and dashboard can use the data.

## Status

- [x] Source 1 — OpenStreetMap Nominatim: camera landmark geocoding (`geocode_cameras.py` → `data_out/cameras_geo.csv`)

## Remaining

- [ ] **Source 2 — hourly weather, 25–31 Aug 2026, Bangkok** (e.g. Open-Meteo historical API)
      → `data_out/weather_hourly.csv` with `timestamp, rain_mm, temp_c, cloud_cover`
- [ ] **Source 3 — official truck-ban rules/zone** (Metropolitan Police traffic division / BMA announcement):
      roads, times, vehicle classes → replace `config/ban_rules.json` and `config/ban_zone.geojson`
      (or a road list). Today's zone is only an approximation from the Ratchadaphisek ring road.
- [ ] **Source 4 (Playwright) — BMA Traffic camera list with real coordinates.** The site did not load from the
      dev machine, probably because it needs a Thai IP. On its own this source meets the requirement.
      → overwrite `data_out/cameras_geo.csv`, same columns: `camera_id, lat, lon, method`
- [ ] Optional: public-holiday calendar, live traffic (Longdo / iTIC) for comparison
- [ ] For each source: a script in this folder, the raw response saved, and a note in `README.md`
      (URL, date scraped, license, rate limit)
- [ ] Tell เก้อ when files land; keep the column names above so `rules.py` and the dashboard keep working
