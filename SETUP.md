# Setup

| | |
|---|---|
| OS | macOS on Apple Silicon (CoreML, `BACKEND=coreml`) · Linux or Windows WSL2 (PyTorch, `BACKEND=torch`) |
| Tools | `git`, `make`, `curl`, `unzip` (macOS: `xcode-select --install`), [uv](https://docs.astral.sh/uv/) (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| Disk | ~6 GB (images 1.6 GB, models 0.6 GB, environments) |

## 1. Clone

```bash
git clone https://github.com/fifaak/dsde-proj-fifa.git
cd dsde-proj-fifa
make touch
```

## 2. Environments

| Env | Python | Packages | Runs |
|---|---|---|---|
| `.venv` | 3.14 | `requirements.lock` (pinned from `requirements.txt`) | every step except the two CoreML steps, Prefect, dashboard, notebooks |
| `.venv312` | 3.12 | `pipeline (เก้อ)/requirements-coreml.lock` (pinned from `requirements-coreml.txt`) | `detect.py` and `classify_trucks.py embed` with `--backend coreml` (Apple Silicon only) |

```bash
make setup
```

Same without make:

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv venv --python 3.12 .venv312
uv pip install --python .venv312/bin/python -r "pipeline (เก้อ)/requirements-coreml.lock"
```

Same without uv:

```bash
python3.14 -m venv .venv && .venv/bin/python -m pip install -r requirements.lock
python3.12 -m venv .venv312 && .venv312/bin/python -m pip install -r "pipeline (เก้อ)/requirements-coreml.lock"
```

Check:

```bash
.venv/bin/python -c "import torch, pandas, shapely, prefect; print(torch.__version__, pandas.__version__, prefect.__version__)"
.venv312/bin/python -c "import coremltools, torch, open_clip; print(coremltools.__version__, torch.__version__)"
```

## 3. Models and images

```bash
make models
make data
```

| Command | Result |
|---|---|
| `make models` | `models/truck_clipvitb.pt`, `models/yolo11s.pt`, `models/yolo11s_nonms.mlpackage`, `models/clip_ViT-B-16.mlpackage` (GitHub Release `models-v1`, SHA-256 checked) |
| `make data` | `data/cctv_images/<camera_id>/*.jpg` (61,319 images) |
| `make coreml` | rebuilds both `.mlpackage` files from `yolo11s.pt` and open_clip `ViT-B-16 / datacomp_xl_s13b_b90k` |
| `make portable` | rewrites image paths in `data_out/*` and `modeling (พัท)/labels/labels.csv` to `data/cctv_images/<camera_id>/<file>` |

## 4. Run

```bash
make all
make dashboard
```

| Command | Result |
|---|---|
| `make all` | scraped inputs (if missing) → extract → cleanse → detect → candidates → classify → join → rules → validate; finished stages are skipped |
| `make flow` | the same stages as a Prefect flow, with a data-quality table artifact |
| `make validate` | schema, nulls, keys, time range and row-count checks on every file in `data_out/` and `config/` |
| `make <stage>` | `extract`, `cleanse`, `scraped`, `scrape`, `detect`, `candidates`, `classify`, `join`, `rules`, `clip` |
| `make all FORCE="join rules"` | re-runs the listed stages and everything after them |
| `make scrape` | re-runs `SCRAPERS` in `scraping (เพา)/` |
| `make clip` | `data_out/truck_violations.mp4` |
| `make evidence` | `insights (ม่อน)/evidence/` |

| Variable | Default |
|---|---|
| `BACKEND` | `coreml` on Apple Silicon, `torch` elsewhere |
| `PY` / `PY_ML` | `.venv/bin/python` / `.venv312/bin/python` |
| `DATA` | `data/cctv_images` |
| `SCRAPERS` | `geocode_cameras.py build_ban_zone.py` |
| `WEATHER_TZ` | `Asia/Bangkok` (zone of naive timestamps in `weather_hourly.csv`) |
| `WEATHER_LABEL` | `end` (`rain_mm` at 07:00 = rain 06:00–07:00), or `start` |
| `CCTV_INFO` / `RAIN` / `TRUCK_RULES` | `scraping (เพา)/external_data/data/cameras.csv` / `rain_hourly.csv` / `truck_rules.csv` |
| `RAIN_LABEL` | `start` (`hour_start` 14:00 = rain 14:00–15:00) |

Prefect UI:

```bash
make prefect-ui
PREFECT_API_URL=http://127.0.0.1:4200/api make flow
```

## 5. Stages

| Stage | Script | Env | Output |
|---|---|---|---|
| extract | `pipeline (เก้อ)/extract_features.py` | `.venv` | `data_out/features.csv` |
| cleanse | `pipeline (เก้อ)/clean.py` | `.venv` | `data_out/images_clean.csv`, `data_out/cameras.csv` |
| scraped | `scraping (เพา)/geocode_cameras.py`, `build_ban_zone.py` | `.venv` | `data_out/cameras_geo.csv`, `config/ban_zone.geojson` |
| detect | `pipeline (เก้อ)/detect.py` | `.venv312` (coreml) / `.venv` (torch) | `data_out/detections.parquet` |
| candidates | `pipeline (เก้อ)/classify_trucks.py embed` | `.venv312` (coreml) / `.venv` (torch) | `data_out/candidates.parquet`, `data_out/candidates_emb.npy` |
| classify | `pipeline (เก้อ)/classify_vit.py` | `.venv` | `data_out/vehicles.parquet` |
| join | `pipeline (เก้อ)/official_rules.py`, `pipeline (เก้อ)/join_scraped.py` | `.venv` | `config/ban_rules.json` (truck10 + ring-road exemption), `data_out/context_camera_hour.csv`, coordinates, roads and `in_zone` in `data_out/cameras_geo.csv` |
| rules | `pipeline (เก้อ)/rules.py` | `.venv` | `data_out/heavy_sightings.parquet`, `data_out/agg_camera_hour.csv` |
| validate | `pipeline (เก้อ)/validate.py` | `.venv` | pass / fail per file |

## 6. Scraped inputs

| File | Format |
|---|---|
| `data_out/cameras_geo.csv` | `camera_id, lat, lon, method`, one row per camera |
| `config/ban_zone.geojson` | FeatureCollection; Polygon / MultiPolygon zone (first feature a Polygon for the clip); LineString roads allowed, width from `properties.buffer_m` (default 60 m) |
| `config/ban_rules.json` | `rules.truck6`, `rules.truck10`: `[["HH:MM", "HH:MM"], ...]`; `public_holidays`: `["YYYY-MM-DD", ...]` |
| `data_out/weather_hourly.csv` | `timestamp, rain_mm, temp_c, cloud_cover`; optional `lat, lon` (several points) or `camera_id`; Open-Meteo CSV export also accepted |
| `scraping (เพา)/external_data/data/cameras.csv` | `camera_id, latitude, longitude, road_name, road_class, road_distance_m, …`; replaces geocoded coordinates |
| `scraping (เพา)/external_data/data/rain_hourly.csv` | `camera_id, hour_start, rain_mm, grid_latitude, grid_longitude`; rain per camera, overrides city rain |
| `scraping (เพา)/external_data/data/truck_rules.csv` | `rule_id, scope_type, vehicle_type, days_of_week, start_time, end_time, exceptions, …`; area rule → `rules.truck10`, `truck10_ring_exempt` |

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `make all` starts detect/classify right after cloning | `make touch` |
| `No such file or directory: 'data/cctv_images/…'` | `make data` |
| `FileNotFoundError: '/private/tmp/…'` | `make portable` |
| `models/… not found` | `make models` |
| coremltools / CoreML errors | run on Apple Silicon with `.venv312`, or `make all BACKEND=torch` |
| `FAIL` from `validate.py` | the named file breaks its contract; fix the producing stage, then `make all FORCE=<stage>` |
| Windows | WSL2 (Ubuntu), then the Linux steps |
