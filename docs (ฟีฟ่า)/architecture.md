# Architecture

How the four required components and web scraping connect. Boxes are folders; arrows are files in `data_out/`
(the contracts between stages — keep their columns stable).

```mermaid
flowchart LR
    subgraph S[scraping (เพา)/ · web scraping]
        S1[geocode_cameras.py<br/>OSM Nominatim]
        S2[build_ban_zone.py<br/>ring-road zone]
        S3[TODO: weather API]
        S4[TODO: official ban rules]
        S5[TODO: BMA camera list<br/>Playwright]
    end
    subgraph P[pipeline (เก้อ)/ · data engineering · make all / Prefect flow]
        P0[extract_features.py<br/>image features]
        P1[clean.py<br/>cleaning rules + quality flags]
        P2[detect.py<br/>YOLO11s CoreML]
        P3[classify_trucks.py<br/>candidates + CLIP]
        P4[classify_vit.py<br/>fine-tuned ViT-B]
        P6[join_scraped.py<br/>zone + weather per camera-hour]
        P5[rules.py<br/>ban windows + zone]
        V[validate.py<br/>checks between stages]
    end
    subgraph M[modeling (พัท)/ + analysis (พัท)/ · data science]
        M1[labels/ + train_gpu.py<br/>RTX 3090]
        M2[EDA + statistics]
    end
    subgraph D[dashboard (เต้)/ · BI]
        D1[app.py Streamlit]
        D2[make_timelapse.py clip]
    end
    I[insights (ม่อน)/ · findings]

    RAW[(CCTV images<br/>61,319 jpg)] --> P0 -->|features.csv| P1 -->|images_clean.csv| P2 -->|detections.parquet| P3 -->|candidates.parquet| P4 -->|vehicles.parquet| P5
    M1 -->|truck_clipvitb.pt| P4
    S1 -->|cameras_geo.csv| S2 -->|ban_zone.geojson| P6
    P1 -->|images_clean.csv| P6 -->|context_camera_hour.csv| P5
    S1 -->|cameras_geo.csv| D1
    P5 -->|heavy_sightings.parquet<br/>agg_camera_hour.csv| D1 & D2 & M2
    S3 & S4 & S5 -.-> P6
    S4 -.->|ban_rules.json| P5
    V -.- P1 & P2 & P3 & P4 & P6 & P5
    M2 --> I
    D1 --> I
```

## Data contracts (`data_out/`)

| File | Producer | Consumers | Key columns |
|---|---|---|---|
| `features.csv` | `pipeline (เก้อ)/extract_features.py` | clean | `path, camera_id, timestamp, width, height, brightness, sharpness, ahash, parse_ok, read_ok…` |
| `images_clean.csv` | `pipeline (เก้อ)/clean.py` (rules of the EDA notebook §8) | detect, join, rules, clip | `path` (`data/cctv_images/<camera_id>/<file>`), `camera_id, timestamp, very_dark, invalid…` |
| `detections.parquet` | `pipeline (เก้อ)/detect.py` | classify | `path, camera_id, timestamp, cls, conf, x1..y2` |
| `candidates.parquet` | `pipeline (เก้อ)/classify_trucks.py embed` | classify_vit, modeling | detection cols + `zs_*` |
| `vehicles.parquet` | `pipeline (เก้อ)/classify_vit.py` | rules | + `p_light, p_bus, p_heavy, p_other, vclass, heavy_type` |
| `cameras_geo.csv` | `scraping (เพา)/geocode_cameras.py` (`in_zone`: `pipeline (เก้อ)/join_scraped.py`) | join, dashboard, clip | `camera_id, lat, lon, method, in_zone` |
| `weather_hourly.csv` | `scraping (เพา)/` (TODO) | join | `timestamp, rain_mm, temp_c, cloud_cover` (+ optional `lat, lon` or `camera_id`) |
| `context_camera_hour.csv` | `pipeline (เก้อ)/join_scraped.py` | rules | `camera_id, date, hour, lat, lon, geo_method, in_zone, zone_edge_m, weather_km, rain_mm, temp_c, cloud_cover, rain_class` |
| `heavy_sightings.parquet` | `pipeline (เก้อ)/rules.py` | dashboard, clip, analysis | + `in_zone, in_ban6, in_ban10, violation`, + `lat, lon, zone_edge_m, rain_mm, temp_c, cloud_cover, rain_class` |
| `agg_camera_hour.csv` | `pipeline (เก้อ)/rules.py` | dashboard, analysis | `camera_id, date, hour, weekday, in_zone, frames, heavy, ban6`, + same context columns |
