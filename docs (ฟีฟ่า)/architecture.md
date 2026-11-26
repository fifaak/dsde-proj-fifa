# Architecture

How the four required components and web scraping connect. Boxes are folders; arrows are files in `data_out/`
(the contracts between stages — keep their columns stable).

```mermaid
flowchart LR
    subgraph S[scraping (เพา)/ · web scraping]
        S1[geocode_cameras.py<br/>OSM Nominatim]
        S2[build_ban_zone.py<br/>ring-road zone]
        S3[weather_hourly.csv + rain_hourly.csv<br/>Open-Meteo ECMWF]
        S4[truck_rules.csv<br/>สนข. + ราชกิจจานุเบกษา]
        S5[external_data/cameras.csv<br/>course coordinates + OSM roads]
    end
    subgraph P[pipeline (เก้อ)/ · data engineering · make all / Prefect flow]
        P0[extract_features.py<br/>image features]
        P1[clean.py<br/>cleaning rules + quality flags]
        P2[detect.py<br/>YOLO11s CoreML]
        P3[classify_trucks.py<br/>candidates + CLIP]
        P4[classify_vit.py<br/>fine-tuned ViT-B]
        P7[official_rules.py<br/>truck_rules.csv → ban_rules.json]
        P6[join_scraped.py<br/>coordinates, zone, roads, weather per camera-hour]
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
    S3 -->|rain, temperature, cloud| P6
    S5 -->|lat, lon, road_name| P6
    S4 --> P7 -->|ban_rules.json| P5
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
| `cameras_geo.csv` | `scraping (เพา)/geocode_cameras.py`, coordinates/roads/`in_zone` from `pipeline (เก้อ)/join_scraped.py` | join, dashboard, clip | `camera_id, lat, lon, method, in_zone, road_name, road_class, road_distance_m, geo_offset_m, osm_lat, osm_lon` |
| `weather_hourly.csv` | `scraping (เพา)/` | join | `timestamp, rain_mm, temp_c, cloud_cover` |
| `external_data/data/cameras.csv` | `scraping (เพา)/external_data/collect.py` | join | `camera_id, latitude, longitude, road_name, road_class, road_distance_m` |
| `external_data/data/rain_hourly.csv` | `scraping (เพา)/external_data/collect.py` | join | `camera_id, hour_start, rain_mm, grid_latitude, grid_longitude` |
| `external_data/data/truck_rules.csv` | `scraping (เพา)/external_data/collect.py` | official_rules | `rule_id, scope_type, vehicle_type, days_of_week, start_time, end_time, exceptions` |
| `config/ban_rules.json` | `pipeline (เก้อ)/official_rules.py` | rules, dashboard, clip | `rules.truck6, rules.truck10, truck10_ring_exempt, ring_roads, public_holidays` |
| `context_camera_hour.csv` | `pipeline (เก้อ)/join_scraped.py` | rules | `camera_id, date, hour, lat, lon, geo_method, road_name, road_class, in_zone, zone_edge_m, weather_km, rain_source, rain_mm, temp_c, cloud_cover, rain_class` |
| `heavy_sightings.parquet` | `pipeline (เก้อ)/rules.py` | dashboard, clip, analysis | + `in_zone, in_ban6, in_ban10, violation`, + `lat, lon, road_name, zone_edge_m, rain_mm, temp_c, cloud_cover, rain_class, ring_road` |
| `agg_camera_hour.csv` | `pipeline (เก้อ)/rules.py` | dashboard, analysis | `camera_id, date, hour, weekday, in_zone, frames, heavy, ban6`, + same context columns, `ban10` |
