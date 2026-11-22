# Architecture

How the four required components and web scraping connect. Boxes are folders; arrows are files in `data_out/`
(the contracts between stages — keep their columns stable).

```mermaid
flowchart LR
    subgraph S[scraping/ · web scraping]
        S1[geocode_cameras.py<br/>OSM Nominatim]
        S2[build_ban_zone.py<br/>ring-road zone]
        S3[TODO: weather API]
        S4[TODO: official ban rules]
        S5[TODO: BMA camera list<br/>Playwright]
    end
    subgraph P[pipeline/ · data engineering]
        P1[extract_features.py<br/>clean + quality flags]
        P2[detect.py<br/>YOLO11s CoreML]
        P3[classify_trucks.py<br/>candidates + CLIP]
        P4[classify_vit.py<br/>fine-tuned ViT-B]
        P5[rules.py<br/>ban windows + zone]
    end
    subgraph M[modeling/ + analysis/ · data science]
        M1[labels/ + train_gpu.py<br/>RTX 3090]
        M2[EDA + statistics]
    end
    subgraph D[dashboard/ · BI]
        D1[app.py Streamlit]
        D2[make_timelapse.py clip]
    end
    I[insights/ · findings]

    RAW[(CCTV images<br/>61,319 jpg)] --> P1 -->|images_clean.csv| P2 -->|detections.parquet| P3 -->|candidates.parquet| P4 -->|vehicles.parquet| P5
    M1 -->|truck_clipvitb.pt| P4
    S1 -->|cameras_geo.csv| S2 -->|ban_zone.geojson| P5
    S1 -->|cameras_geo.csv| D1
    P5 -->|heavy_sightings.parquet<br/>agg_camera_hour.csv| D1 & D2 & M2
    S3 & S4 & S5 -.-> P5
    M2 --> I
    D1 --> I
```

## Data contracts (`data_out/`)

| File | Producer | Consumers | Key columns |
|---|---|---|---|
| `images_clean.csv` | `pipeline/extract_features.py` + EDA | detect, rules, clip | `path, camera_id, timestamp, very_dark, invalid…` |
| `detections.parquet` | `pipeline/detect.py` | classify | `path, camera_id, timestamp, cls, conf, x1..y2` |
| `candidates.parquet` | `pipeline/classify_trucks.py embed` | classify_vit, modeling | detection cols + `zs_*` |
| `vehicles.parquet` | `pipeline/classify_vit.py` | rules | + `p_light, p_bus, p_heavy, p_other, vclass, heavy_type` |
| `cameras_geo.csv` | `scraping/geocode_cameras.py`, `build_ban_zone.py` | rules, dashboard, clip | `camera_id, lat, lon, method, in_zone` |
| `heavy_sightings.parquet` | `pipeline/rules.py` | dashboard, clip, analysis | + `in_zone, in_ban6, in_ban10, violation` |
| `agg_camera_hour.csv` | `pipeline/rules.py` | dashboard, analysis | `camera_id, date, hour, weekday, in_zone, frames, heavy, ban6` |
