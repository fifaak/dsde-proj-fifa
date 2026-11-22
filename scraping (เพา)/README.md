# scraping (เพา)/ — web scraping (required task)

Course rule: **≥3 sources** with simple scraping (เพา)/APIs, or **1 source** that needs Playwright/Selenium/Crawlee.
This part does not count toward Data Engineering.

| Source | Status | Script | Output |
|---|---|---|---|
| OpenStreetMap Nominatim — camera landmark geocoding | done | `geocode_cameras.py` | `data_out/cameras_geo.csv` |
| Ban zone from geocoded ring-road junctions (approximate) | done | `build_ban_zone.py` | `config/ban_zone.geojson` |
| Hourly rain / weather, 25–31 Aug 2026 (e.g. Open-Meteo) | TODO | | `data_out/weather_hourly.csv` (`timestamp, rain_mm, temp_c`) |
| Official truck-ban rules/zone (police traffic division / BMA announcement) | TODO | | replace `config/ban_rules.json`, `config/ban_zone.geojson` |
| BMA Traffic camera list with coordinates (Playwright; needs a Thai IP) | TODO | | replace `data_out/cameras_geo.csv` (same columns) |

Keep output columns as listed so `pipeline (เก้อ)/rules.py` and the dashboard keep working.
Nominatim policy: ≤1 request/s, results cached in `data_out/geocode_cache.json`.
