import argparse
import json
import re
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, shape
from shapely.ops import unary_union

OUT = Path("data_out")
TZ = "Asia/Bangkok"
KEY = ["camera_id", "date", "hour"]
WX = ["rain_mm", "temp_c", "cloud_cover"]
COLS = KEY + ["lat", "lon", "geo_method", "in_zone", "zone_edge_m", "weather_km"] + WX + ["rain_class"]
ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date_time", "valid_time", "time_local"],
    "rain_mm": ["rain_mm", "precipitation_mm", "precipitation", "rain", "prcp", "rainfall"],
    "temp_c": ["temp_c", "temperature_2m", "temperature", "temp", "air_temperature"],
    "cloud_cover": ["cloud_cover", "cloudcover", "cloud_cover_pct", "clouds"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "longitude"],
    "camera_id": ["camera_id"],
}
RAIN_BINS = [-np.inf, 0.1, 2.5, 7.6, np.inf]
RAIN_LABELS = ["none", "light", "moderate", "heavy"]
LAT0, LON0 = 13.75, 100.5
KX, KY = 111320.0 * np.cos(np.radians(LAT0)), 110574.0


def to_m(xy):
    return np.c_[(xy[:, 0] - LON0) * KX, (xy[:, 1] - LAT0) * KY]


def km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def load_zone(path):
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = gj.get("features") or [gj]
    areas, roads = [], []
    for f in feats:
        geom = shape(f.get("geometry", f))
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            areas.append(geom)
        elif geom.geom_type in ("LineString", "MultiLineString"):
            roads.append(shapely.transform(geom, to_m).buffer(float((f.get("properties") or {}).get("buffer_m", 60))))
    if not areas and not roads:
        raise SystemExit(f"{path}: no Polygon, MultiPolygon, LineString or MultiLineString features")
    return (unary_union(areas) if areas else None), (unary_union(roads) if roads else None)


def tag_cameras(geo, zone_path):
    areas, roads = load_zone(zone_path)
    areas_m = shapely.transform(areas, to_m) if areas is not None else None
    inside, edge = [], []
    for lon, lat in zip(geo.lon, geo.lat):
        if pd.isna(lon) or pd.isna(lat):
            inside.append(False)
            edge.append(np.nan)
            continue
        pm = Point((lon - LON0) * KX, (lat - LAT0) * KY)
        hit = (areas is not None and areas.contains(Point(lon, lat))) or (roads is not None and roads.contains(pm))
        inside.append(bool(hit))
        edge.append(float(round(areas_m.boundary.distance(pm))) if areas_m is not None else np.nan)
    return np.array(inside, dtype=bool), np.array(edge, dtype=float)


def norm(name):
    return re.sub(r"\s*\(.*\)\s*$", "", str(name)).strip().lower().replace(" ", "_")


def read_weather(path, naive_tz=TZ, label="end"):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    head = next((i for i, line in enumerate(lines)
                 if any(norm(c) in ALIASES["timestamp"] for c in line.split(","))), None)
    if head is None:
        raise SystemExit(f"{path}: no time column, expected one of {ALIASES['timestamp']}")
    meta = dict(zip(map(norm, lines[0].split(",")), lines[1].split(","))) if head >= 2 else {}
    w = pd.read_csv(path, skiprows=head, encoding="utf-8-sig")
    w.columns = [norm(c) for c in w.columns]
    w = w.rename(columns={next(a for a in alts if a in w.columns): canon
                          for canon, alts in ALIASES.items() if any(a in w.columns for a in alts)})
    for k in ("lat", "lon"):
        if k not in w.columns:
            v = next((meta[a] for a in ALIASES[k] if meta.get(a, "").strip()), None)
            if v is not None:
                w[k] = float(v)
    raw = w.timestamp.astype(str).str.strip()
    if raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$").any():
        ts = pd.to_datetime(raw, utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    else:
        ts = pd.to_datetime(raw)
        off = meta.get("utc_offset_seconds", "").strip()
        src = timezone(timedelta(seconds=int(float(off)))) if off else naive_tz
        if src != TZ:
            ts = ts.dt.tz_localize(src).dt.tz_convert(TZ).dt.tz_localize(None)
    w["ts"] = ts
    group = ["camera_id"] if "camera_id" in w.columns else (["lat", "lon"] if {"lat", "lon"} <= set(w.columns) else [])
    w = w.dropna(subset=["ts"]).drop_duplicates(group + ["ts"]).sort_values(group + ["ts"])
    step = (w.groupby(group).ts.diff() if group else w.ts.diff()).median()
    step = step if pd.notna(step) and step > pd.Timedelta(0) else pd.Timedelta(hours=1)
    start = (w.ts - step if label == "end" else w.ts).dt.floor("h")
    w["date"], w["hour"] = start.dt.strftime("%Y-%m-%d"), start.dt.hour
    for c in WX:
        w[c] = pd.to_numeric(w[c], errors="coerce") if c in w.columns else np.nan
    g = w.groupby(group + ["date", "hour"])
    return pd.concat([g.rain_mm.sum(min_count=1), g.temp_c.mean(), g.cloud_cover.mean()], axis=1).reset_index(), group


def attach_weather(cams, w, group):
    if group == ["camera_id"]:
        return cams.assign(weather_km=0.0), w, group
    if group == ["lat", "lon"]:
        pts = w[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
        d = km(cams.lat.to_numpy(float)[:, None], cams.lon.to_numpy(float)[:, None],
               pts.lat.to_numpy(float)[None], pts.lon.to_numpy(float)[None])
        j = np.where(np.isnan(d), np.inf, d).argmin(1)
        cams = cams.assign(wx_lat=pts.lat.to_numpy()[j], wx_lon=pts.lon.to_numpy()[j],
                           weather_km=d[np.arange(len(cams)), j].round(2))
        return cams, w.rename(columns={"lat": "wx_lat", "lon": "wx_lon"}), ["wx_lat", "wx_lon"]
    return cams.assign(weather_km=np.nan), w, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(OUT / "images_clean.csv"))
    ap.add_argument("--geo", default=str(OUT / "cameras_geo.csv"))
    ap.add_argument("--zone", default="config/ban_zone.geojson")
    ap.add_argument("--weather", default=str(OUT / "weather_hourly.csv"))
    ap.add_argument("--weather-tz", default=TZ)
    ap.add_argument("--label", choices=["end", "start"], default="end")
    ap.add_argument("--out", default=str(OUT / "context_camera_hour.csv"))
    a = ap.parse_args()

    imgs = pd.read_csv(a.images, usecols=["camera_id", "timestamp"], parse_dates=["timestamp"])
    grid = (imgs.assign(date=imgs.timestamp.dt.strftime("%Y-%m-%d"), hour=imgs.timestamp.dt.hour)[KEY]
            .drop_duplicates().sort_values(KEY).reset_index(drop=True))

    geo_raw = pd.read_csv(a.geo)
    missing = sorted(set(grid.camera_id) - set(geo_raw.camera_id))
    if missing:
        raise SystemExit(f"{a.geo}: no row for cameras {missing}")
    geo = geo_raw.drop_duplicates("camera_id").reset_index(drop=True)
    in_zone, edge = tag_cameras(geo, a.zone)
    cams = pd.DataFrame({"camera_id": geo.camera_id, "lat": geo.lat, "lon": geo.lon,
                         "geo_method": geo["method"] if "method" in geo.columns else "unknown",
                         "in_zone": in_zone, "zone_edge_m": edge})

    zone_map = dict(zip(cams.camera_id, cams.in_zone))
    new_flag = geo_raw.camera_id.map(zone_map).astype(bool)
    if "in_zone" not in geo_raw.columns or not geo_raw.in_zone.astype(bool).equals(new_flag):
        geo_raw["in_zone"] = new_flag
        geo_raw.to_csv(a.geo, index=False)
        print(f"{a.geo}: in_zone updated from {a.zone}")

    wpath = Path(a.weather)
    if wpath.exists():
        w, group = read_weather(wpath, a.weather_tz, a.label)
        cams, w, keys = attach_weather(cams, w, group)
        ctx = grid.merge(cams, on="camera_id", how="left").merge(w[keys + ["date", "hour"] + WX],
                                                                  on=keys + ["date", "hour"], how="left")
        dist = f", camera to nearest point <= {cams.weather_km.max():.1f} km" if cams.weather_km.notna().any() else ""
        print(f"weather: {wpath} | {len(w[keys].drop_duplicates()) if keys else 1} point(s){dist} | "
              f"camera-hours with rain_mm {ctx.rain_mm.notna().mean():.1%} | "
              f"rainy camera-hours {int((ctx.rain_mm >= 0.1).sum()):,}")
    else:
        ctx = grid.merge(cams.assign(weather_km=np.nan), on="camera_id", how="left")
        for c in WX:
            ctx[c] = np.nan
        print(f"weather: {wpath} not found, {', '.join(WX)} left empty")
    ctx["rain_class"] = pd.cut(ctx.rain_mm, RAIN_BINS, right=False, labels=RAIN_LABELS)
    ctx[COLS].to_csv(a.out, index=False)
    print(f"zone: {a.zone} | {int(cams.in_zone.sum())} of {len(cams)} cameras inside, outside "
          f"{cams.loc[~cams.in_zone, 'camera_id'].tolist()}, no coordinates "
          f"{cams.loc[cams.lat.isna() | cams.lon.isna(), 'camera_id'].tolist()}")
    print(f"{a.out}: {len(ctx):,} camera-hours, {ctx.camera_id.nunique()} cameras, {ctx.date.nunique()} days")


if __name__ == "__main__":
    main()
