# Bangkok CCTV — Truck Ban Monitor

Turns 61,319 raw BMA CCTV snapshots (100 cameras, 25–31 Aug 2026, 06:00–21:00, ~10 min interval) into
heavy-truck sightings, checks them against the inner-Bangkok truck-ban windows, and shows the result in a dashboard.

## Repository layout (one folder per role)

| Folder | Role / owner | Contents |
|---|---|---|
| `scraping (เพา)/` | Web scraping — เพา | geocoding, ban zone, (TODO) weather, official rules, BMA camera list — see `scraping (เพา)/README.md` |
| `pipeline (เก้อ)/` | Data engineering — เก้อ | clean → detect → classify → ban rules |
| `config/` | Data engineering | `ban_rules.json`, `ban_zone.geojson` |
| `modeling (พัท)/` | Data science — พัท | hand labels, training (`train_gpu.py` on RTX 3090), model evaluation |
| `analysis (พัท)/` | Data science — พัท | EDA notebook (`eda_cctv.ipynb` / `.html`) |
| `dashboard (เต้)/` | BI — เต้ | Streamlit app, violation time-lapse clip |
| `insights (ม่อน)/` | Insights — ม่อน | findings + limitations write-up |
| `docs (ฟีฟ่า)/` | PM — ฟีฟ่า | architecture diagram + data contracts (`docs (ฟีฟ่า)/architecture.md`), slides, video |
| `data_out/` | shared | stage outputs = contracts between folders (see `docs (ฟีฟ่า)/architecture.md`) |

Owners are listed in `.github/CODEOWNERS` (fill in GitHub usernames). Work on a branch and open a PR.

**Remaining work:** each role folder has a `TODO.md` checklist mapped to the course criteria; timeline and
submission checklist are in `docs (ฟีฟ่า)/TODO.md` (deadline Sun 6 Dec 2026).

## Pipeline

| Stage | Script | Model / method | Output | Time (M4 Pro) |
|---|---|---|---|---|
| 1. Clean + EDA | `pipeline (เก้อ)/extract_features.py`, `analysis (พัท)/eda_cctv.ipynb` | filename parsing, image stats | `data_out/images_clean.csv` (60,782 valid) | 20 s |
| 2. Detect vehicles | `pipeline (เก้อ)/detect.py` | YOLO11s → CoreML fp16 on Neural Engine | `data_out/detections.parquet` (526,748 boxes) | 3.5 min |
| 3a. Candidates | `pipeline (เก้อ)/classify_trucks.py embed` | truck/bus/large-car boxes + CLIP zero-shot scores | `data_out/candidates.parquet` | 8.5 min |
| 3b. Classify size | `pipeline (เก้อ)/classify_vit.py` | CLIP ViT-B/16 (DataComp-XL) fine-tuned end-to-end on RTX 3090 (`modeling (พัท)/train_gpu.py`) with Thai-vehicles crops degraded to CCTV quality + 450 hand labels | `data_out/vehicles.parquet` | 12 min |
| 4. Camera map + ban zone | `scraping (เพา)/geocode_cameras.py`, `scraping (เพา)/build_ban_zone.py` | Nominatim geocoding of camera landmarks; zone ≈ inside Ratchadaphisek ring road (+400 m) — **approximate, not the official map** | `data_out/cameras_geo.csv`, `config/ban_zone.geojson` | ~3 min |
| 5. Ban rules + aggregate | `pipeline (เก้อ)/rules.py`, `config/ban_rules.json` | static-object filter, weekday/ban windows, in-zone only | `data_out/heavy_sightings.parquet`, `data_out/agg_camera_hour.csv` | 5 s |
| 6. Dashboard | `dashboard (เต้)/app.py` | Streamlit + Plotly | — | — |
| 7. Violation clip | `dashboard (เต้)/make_timelapse.py` | matplotlib + ffmpeg; only weekday 6-wheel ban windows, only trucks inside the zone | `data_out/truck_violations.mp4` (1080p, 40 s) | 1 min |

Ban flags apply only to cameras inside the zone (94/100); run `scraping (เพา)/build_ban_zone.py` before `pipeline (เก้อ)/rules.py`.
Run everything from the repo root.

```bash
# envs: .venv (requirements.txt) and .venv312 (Python 3.12, CoreML: pipeline (เก้อ)/requirements-coreml.txt)
python "pipeline (เก้อ)/extract_features.py" data/cctv_images data_out/features.csv      # then run "analysis (พัท)/eda_cctv.ipynb"
.venv312/bin/python "pipeline (เก้อ)/detect.py" data_out/images_clean.csv data_out/detections.parquet
.venv312/bin/python "pipeline (เก้อ)/classify_trucks.py" embed data_out/detections.parquet data_out/candidates
python "pipeline (เก้อ)/classify_vit.py" data_out/candidates.parquet models/truck_clipvitb.pt data_out/vehicles.parquet
python "scraping (เพา)/geocode_cameras.py" && python "scraping (เพา)/build_ban_zone.py"
python "pipeline (เก้อ)/rules.py"
streamlit run "dashboard (เต้)/app.py"
python "dashboard (เต้)/make_timelapse.py" --fps 6
```

**Model weights** are in the GitHub Release
[`models-v1`](https://github.com/fifaak/dsde-proj-fifa/releases/tag/models-v1) (git caps files at 100 MB):
`truck_clipvitb.pt` (fine-tuned CLIP ViT-B, 343 MB), `yolo11s.pt`, and the CoreML packages `yolo11s_nonms.mlpackage`
and `clip_ViT-B-16.mlpackage`. Fetch them all with:

```bash
bash models/download.sh
```

The same release holds the raw CCTV images (`cctv_images.zip`, 61,319 images, 1.6 GB). Fetch them into
`data/cctv_images/` along with the weights:

```bash
bash models/download.sh --data
```

It also holds `candidates_emb.npy` (CLIP embeddings of the candidate crops, 117 MB). Not published: the Thai-vehicles
dataset zip (get it from Roboflow, Thailand-vehicles-2024).

## Model quality (be honest about it)

- YOLO11s CoreML output matches Ultralytics' own CoreML runner on 98% of boxes (≤2 px).
- Heavy-truck (6+ wheels) classifier, same 90-label holdout for every model (heavy precision / recall):
  zero-shot CLIP 0.39 / – · YOLOE zero-shot 0.45–0.50 · CLIP linear probe 0.64 / 0.55 (p≥0.6) ·
  EfficientNet-B0 0.60 / 0.52 · ConvNeXt-V2-T 0.58 / 0.62 · DINOv2 ViT-B 0.64 / 0.55 ·
  **CLIP ViT-B/16 fine-tuned 0.65 / 0.69 (p≥0.6) ← used**. Small holdout (29 heavy) → differences of ±0.05 are noise.
  Labels in `modeling (พัท)/labels/labels.csv`; Thai data: Roboflow "Thailand-vehicles-2024-01-16" (CC BY 4.0).
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

## Next

- `scraping (เพา)/`: weather, official ban rules/zone, BMA camera list (required: ≥3 sources or 1 with Playwright).
- `pipeline (เก้อ)/`: one-command end-to-end run (Makefile / Prefect) and join of scraped data.
- `insights (ม่อน)/` + `docs (ฟีฟ่า)/`: findings, slides, 15-min video, submission by Sun 6 Dec 2026.
