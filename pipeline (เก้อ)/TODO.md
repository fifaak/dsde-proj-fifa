# TODO — pipeline (เก้อ)

**Criterion:** Component 1, Data Engineering: cleansing and preparation; running end to end earns extra points.
Scraping does **not** count here.
**Target date:** integration done by Tue 1 Dec 2026.

## Status

- [x] Cleaning + quality flags (`extract_features.py`, `data_out/images_clean.csv`)
- [x] Vehicle detection, YOLO11s CoreML (`detect.py`)
- [x] Candidates + truck classifier (`classify_trucks.py embed`, `classify_vit.py`)
- [x] Ban rules, static-object filter, aggregation (`rules.py`)

## Remaining

- [ ] **One-command end-to-end run** (`Makefile` or a Prefect/Airflow flow): scraping → clean → detect →
      classify → rules → dashboard data. Each step skips work already done and logs row counts.
- [ ] **Join scraped data**: weather → `agg_camera_hour.csv` (rain per camera-hour); official rules/zone →
      `rules.py`; real camera coordinates → `cameras_geo.csv`
- [ ] **Data-quality checks between stages**: schema, row counts, nulls, timestamp range; fail loudly
- [ ] **Portable paths**: parquet/csv files store absolute image paths from the dev Mac. Store paths relative
      to a configurable `DATA_DIR` so teammates (and the dashboard gallery) can find the images.
- [ ] **Data access**: put the raw image zip, Thai dataset and model weights on Google Drive + a download script
- [ ] **Setup doc**: the two environments (`requirements.txt`, `requirements-coreml.txt`), CoreML export steps,
      the non-Mac fallback (`detect.py --backend torch`, `classify_trucks.py embed --backend torch`)
- [ ] Give ฟีฟ่า the pipeline steps for the architecture diagram (`docs (ฟีฟ่า)/architecture.md`)
