# TODO — pipeline (เก้อ)

**Criterion:** Component 1, Data Engineering: cleansing and preparation; running end to end earns extra points.
Scraping does **not** count here.
**Target date:** integration done by Tue 1 Dec 2026.

## Status

- [x] Cleaning + quality flags (`extract_features.py`, `data_out/images_clean.csv`)
- [x] Vehicle detection, YOLO11s CoreML (`detect.py`)
- [x] Candidates + truck classifier (`classify_trucks.py embed`, `classify_vit.py`)
- [x] Ban rules, static-object filter, aggregation (`rules.py`)
- [x] One-command end-to-end run: `make all` (`Makefile`) and `make flow` (Prefect, `flow.py`); finished stages are
      skipped, every stage logs row counts
- [x] Cleaning as a script (`clean.py`), same output as the EDA notebook §8 (byte-identical `images_clean.csv`)
- [x] Join scraped data (`join_scraped.py` → `context_camera_hour.csv`): ban zone per camera from
      `config/ban_zone.geojson`, weather per camera-hour (nearest point, timezone, hour label); `rules.py` adds
      `lat, lon, zone_edge_m, rain_mm, temp_c, cloud_cover, rain_class`
- [x] Data-quality checks between stages (`validate.py`): schema, nulls, keys, time range, row counts; non-zero exit
- [x] Portable paths (`portable_paths.py`, `make portable`): `data/cctv_images/<camera_id>/<file>`
- [x] Data access: GitHub Release `models-v1` + `models/download.sh` (`make models`, `make data`)
- [x] Setup doc (`SETUP.md`), pinned environments (`requirements.lock`, `requirements-coreml.lock`), CoreML export
      (`export_coreml.py`), non-Mac fallback (`BACKEND=torch`)
- [x] Pipeline steps for the architecture diagram (`docs (ฟีฟ่า)/architecture.md`)
- [x] Scraped data from เพา joined: course camera coordinates + nearest road (`external_data/data/cameras.csv`),
      rain per camera-hour (`rain_hourly.csv`), temperature/cloud (`data_out/weather_hourly.csv`), official 10-wheel
      area rule + 10:00–15:00 ring-road exemption (`truck_rules.csv` → `official_rules.py` → `config/ban_rules.json`)

## Remaining

- [ ] Official source for the 6-wheel area windows (not in `truck_rules.csv`; `rules.truck6` keeps 06–09, 16–20)
- [ ] Confirm `ring_roads` in `config/ban_rules.json` with the สนข. report (section 2.3.1.13)
