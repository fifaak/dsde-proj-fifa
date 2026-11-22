# Bangkok CCTV — Truck Ban Monitor

Turns 61,319 raw BMA CCTV snapshots (100 cameras, 25–31 Aug 2026, 06:00–21:00, ~10 min interval) into
heavy-truck sightings, checks them against the inner-Bangkok truck-ban windows, and shows the result in a dashboard.

## Pipeline

| Stage | Script | Model / method | Output | Time (M4 Pro) |
|---|---|---|---|---|
| 1. Clean + EDA | `extract_features.py`, `eda_cctv.ipynb` | filename parsing, image stats | `data_out/images_clean.csv` (60,782 valid) | 20 s |
| 2. Detect vehicles | `pipeline/detect.py` | YOLO11s → CoreML fp16 on Neural Engine | `data_out/detections.parquet` (526,748 boxes) | 3.5 min |
| 3. Classify size | `pipeline/classify_trucks.py` (candidates + zero-shot), `pipeline/classify_vit.py` (final class) | CLIP ViT-B/16 (DataComp-XL) fine-tuned end-to-end on RTX 3090 (`pipeline/train_gpu.py`) with Thai-vehicles crops degraded to CCTV quality + 450 hand labels | `data_out/vehicles.parquet` | 8.5 min embed + inference |
| 4. Ban rules + aggregate | `pipeline/rules.py`, `config/ban_rules.json` | static-object filter, weekday/ban windows | `data_out/heavy_sightings.parquet`, `data_out/agg_camera_hour.csv` | 5 s |
| 5. Dashboard | `dashboard/app.py` | Streamlit + Plotly | — | — |
| 6. Camera map + ban zone | `pipeline/geocode_cameras.py`, `pipeline/build_ban_zone.py` | Nominatim geocoding of camera landmarks; zone ≈ inside Ratchadaphisek ring road (+400 m) — **approximate, not the official map** | `data_out/cameras_geo.csv`, `config/ban_zone.geojson` | ~3 min |
| 7. Violation clip | `viz/make_timelapse.py` | matplotlib + ffmpeg; only weekday 6-wheel ban windows, only trucks inside the zone | `data_out/truck_violations.mp4` (1080p, 40 s) | 1 min |

Ban flags apply only to cameras inside the zone (94/100); run `build_ban_zone.py` before `rules.py`.

```bash
# envs: .venv (py3.14: pandas, open_clip, sklearn, streamlit) and .venv312 (py3.12, CoreML: pipeline/requirements-coreml.txt)
python extract_features.py data/cctv_images data_out/features.csv            # then run eda_cctv.ipynb
.venv312/bin/python pipeline/detect.py data_out/images_clean.csv data_out/detections.parquet
.venv312/bin/python pipeline/classify_trucks.py embed data_out/detections.parquet data_out/candidates
python pipeline/classify_trucks.py train data_out/candidates labels/labels.csv models/truck_probe.joblib
python pipeline/classify_trucks.py predict data_out/candidates models/truck_probe.joblib data_out/vehicles.parquet
python pipeline/rules.py
streamlit run dashboard/app.py
```

## Model quality (be honest about it)

- YOLO11s CoreML output matches Ultralytics' own CoreML runner on 98% of boxes (≤2 px).
- Heavy-truck (6+ wheels) classifier, same 90-label holdout for every model (heavy precision / recall):
  zero-shot CLIP 0.39 / – · YOLOE zero-shot 0.45–0.50 · CLIP linear probe 0.64 / 0.55 (p≥0.6) ·
  EfficientNet-B0 0.60 / 0.52 · ConvNeXt-V2-T 0.58 / 0.62 · DINOv2 ViT-B 0.64 / 0.55 ·
  **CLIP ViT-B/16 fine-tuned 0.65 / 0.69 (p≥0.6) ← used**. Small holdout (29 heavy) → differences of ±0.05 are noise.
  Labels in `labels/labels.csv`; Thai data: Roboflow "Thailand-vehicles-2024-01-16" (CC BY 4.0).
- 6- vs 10-wheel cannot be separated reliably at 352×288 → all heavy trucks use the 6-wheel ban windows.
- Main confusions: passenger vans/minibuses, box-body pickups (4-wheel), partially visible buses.
- Use the numbers to compare hours and cameras, **not** as evidence against individual vehicles (no plates readable).

## First results (fine-tuned CLIP ViT-B classifier, ban zone applied)

- 13,177 heavy-truck sightings after removing 503 static boxes (parked trucks, on-screen labels);
  **3,014** fall inside the ban zone during weekday 6-wheel ban windows.
- Weekday heavy-truck rate: 0.15–0.23 per frame at 06–08, peak ~0.37 at 11:00, 0.22–0.26 at 16–17, low after 19:00
  (partly lower night-time recall).
- In-zone cameras, ratio of (ban hours / 10–14h) weekday vs weekend = **0.94 (95% bootstrap CI 0.82–1.08)** →
  no clear extra drop in ban hours beyond the normal daily rhythm. Only 2 weekend days, so weak evidence either way.
- Cameras with most in-zone ban-window sightings: 11, 75, 68, 26, 32, 188 (camera 11 still mixes in passenger vans).
- Old CLIP-probe results are kept in `data_out/vehicles_clipprobe.parquet` for comparison.

## Next

- Camera coordinates (scrape BMA camera list) → `data_out/cameras_geo.csv` enables the map.
- More labels / fine-tune a small CNN on crops to raise precision.
- Join weather (rain), holidays, road type for the scraping requirement.
