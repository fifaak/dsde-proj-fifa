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
EXTERNAL = Path("scraping (เพา)/external_data/data")
TZ = "Asia/Bangkok"
KEY = ["camera_id", "date", "hour"]
WX = ["rain_mm", "temp_c", "cloud_cover"]
COLS = KEY + ["lat", "lon", "geo_method", "road_name", "road_class", "in_zone", "zone_edge_m", "weather_km",
              "rain_source"] + WX + ["rain_class"]
GEO_COLS = ["camera_id", "lat", "lon", "method", "in_zone", "road_name", "road_class", "road_distance_m",
            "geo_offset_m", "osm_lat", "osm_lon", "query", "osm_name"]
ROAD_COLS = ["road_name", "road_class", "road_distance_m"]
ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date_time", "valid_time", "time_local", "hour_start"],
    "rain_mm": ["rain_mm", "precipitation_mm", "precipitation", "rain", "prcp", "rainfall"],
    "temp_c": ["temp_c", "temperature_2m", "temperature", "temp", "air_temperature"],
    "cloud_cover": ["cloud_cover", "cloudcover", "cloud_cover_pct", "clouds"],
    "lat": ["lat", "latitude", "grid_latitude"],
    "lon": ["lon", "lng", "longitude", "grid_longitude"],
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


def norm(name):
    return re.sub(r"\s*\(.*\)\s*$", "", str(name)).strip().lower().replace(" ", "_")


def canonical(df, names=None):
    df.columns = [norm(c) for c in df.columns]
    items = [(k, v) for k, v in ALIASES.items() if names is None or k in names]
    return df.rename(columns={next(a for a in alts if a in df.columns): canon
                              for canon, alts in items if any(a in df.columns for a in alts)})


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


def read_cameras(path):
    c = canonical(pd.read_csv(path, encoding="utf-8-sig"), ["camera_id", "lat", "lon"])
    keep = ["camera_id", "lat", "lon"] + [x for x in ROAD_COLS if x in c.columns]
    return c[keep].dropna(subset=["lat", "lon"]).drop_duplicates("camera_id")


def read_weather(path, naive_tz=TZ, label="end"):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    head = next((i for i, line in enumerate(lines)
                 if any(norm(c) in ALIASES["timestamp"] for c in line.split(","))), None)
    if head is None:
        raise SystemExit(f"{path}: no time column, expected one of {ALIASES['timestamp']}")
    meta = dict(zip(map(norm, lines[0].split(",")), lines[1].split(","))) if head >= 2 else {}
    w = canonical(pd.read_csv(path, skiprows=head, encoding="utf-8-sig"))
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
    points = (w.groupby("camera_id")[["lat", "lon"]].first().reset_index()
              if group == ["camera_id"] and {"lat", "lon"} <= set(w.columns) else None)
    step = (w.groupby(group).ts.diff() if group else w.ts.diff()).median()
    step = step if pd.notna(step) and step > pd.Timedelta(0) else pd.Timedelta(hours=1)
    start = (w.ts - step if label == "end" else w.ts).dt.floor("h")
    w["date"], w["hour"] = start.dt.strftime("%Y-%m-%d"), start.dt.hour
    for c in WX:
        w[c] = pd.to_numeric(w[c], errors="coerce") if c in w.columns else np.nan
    g = w.groupby(group + ["date", "hour"])
    agg = pd.concat([g.rain_mm.sum(min_count=1), g.temp_c.mean(), g.cloud_cover.mean()], axis=1).reset_index()
    return agg, group, points


def attach_weather(cams, w, group, points=None):
    if group == ["camera_id"]:
        if points is None:
            return cams.assign(weather_km=np.nan), w, group
        p = cams[["camera_id", "lat", "lon"]].merge(points, on="camera_id", how="left", suffixes=("", "_wx"))
        return cams.assign(weather_km=km(p.lat, p.lon, p.lat_wx, p.lon_wx).round(2).to_numpy()), w, group
    if group == ["lat", "lon"]:
        pts = w[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
        d = km(cams.lat.to_numpy(float)[:, None], cams.lon.to_numpy(float)[:, None],
               pts.lat.to_numpy(float)[None], pts.lon.to_numpy(float)[None])
        j = np.where(np.isnan(d), np.inf, d).argmin(1)
        cams = cams.assign(wx_lat=pts.lat.to_numpy()[j], wx_lon=pts.lon.to_numpy()[j],
                           weather_km=d[np.arange(len(cams)), j].round(2))
        return cams, w.rename(columns={"lat": "wx_lat", "lon": "wx_lon"}), ["wx_lat", "wx_lon"]
    return cams.assign(weather_km=np.nan), w, []


def weather_frame(path, tz, label, cams, grid):
    w, group, points = read_weather(path, tz, label)
    n = len(points[["lat", "lon"]].drop_duplicates()) if points is not None else None
    cams, w, keys = attach_weather(cams, w, group, points)
    extra = [k for k in keys if k != "camera_id"]
    f = grid.merge(cams[["camera_id", "weather_km"] + extra], on="camera_id", how="left")
    f = f.merge(w[keys + ["date", "hour"] + WX], on=keys + ["date", "hour"], how="left")
    return f[KEY + ["weather_km"] + WX], n or (len(w[keys].drop_duplicates()) if keys else 1)


def update_geo(geo_raw, info, zone_path):
    geo = geo_raw.copy()
    if info is not None:
        if "osm_lat" not in geo.columns:
            geo["osm_lat"], geo["osm_lon"] = geo.lat, geo.lon
        geo = geo.drop(columns=[c for c in ROAD_COLS if c in geo.columns])
        geo = geo.merge(info.rename(columns={"lat": "info_lat", "lon": "info_lon"}), on="camera_id", how="left")
        has = geo.info_lat.notna()
        geo.loc[has, "lat"] = geo.loc[has, "info_lat"]
        geo.loc[has, "lon"] = geo.loc[has, "info_lon"]
        geo.loc[has, "method"] = "cctv_info"
        geo["geo_offset_m"] = (km(geo.lat, geo.lon, geo.osm_lat, geo.osm_lon) * 1000).round()
        geo = geo.drop(columns=["info_lat", "info_lon"])
    in_zone, edge = tag_cameras(geo, zone_path)
    geo["in_zone"] = in_zone
    order = [c for c in GEO_COLS if c in geo.columns] + [c for c in geo.columns if c not in GEO_COLS]
    return geo[order], edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(OUT / "images_clean.csv"))
    ap.add_argument("--geo", default=str(OUT / "cameras_geo.csv"))
    ap.add_argument("--cameras", default=str(EXTERNAL / "cameras.csv"))
    ap.add_argument("--zone", default="config/ban_zone.geojson")
    ap.add_argument("--weather", default=str(OUT / "weather_hourly.csv"))
    ap.add_argument("--weather-tz", default=TZ)
    ap.add_argument("--label", choices=["end", "start"], default="end")
    ap.add_argument("--rain", default=str(EXTERNAL / "rain_hourly.csv"))
    ap.add_argument("--rain-label", choices=["end", "start"], default="start")
    ap.add_argument("--out", default=str(OUT / "context_camera_hour.csv"))
    a = ap.parse_args()

    imgs = pd.read_csv(a.images, usecols=["camera_id", "timestamp"], parse_dates=["timestamp"])
    grid = (imgs.assign(date=imgs.timestamp.dt.strftime("%Y-%m-%d"), hour=imgs.timestamp.dt.hour)[KEY]
            .drop_duplicates().sort_values(KEY).reset_index(drop=True))

    geo_raw = pd.read_csv(a.geo).drop_duplicates("camera_id").reset_index(drop=True)
    missing = sorted(set(grid.camera_id) - set(geo_raw.camera_id))
    if missing:
        raise SystemExit(f"{a.geo}: no row for cameras {missing}")
    info = read_cameras(a.cameras) if Path(a.cameras).exists() else None
    geo, edge = update_geo(geo_raw, info, a.zone)
    if not geo.equals(geo_raw):
        geo.to_csv(a.geo, index=False)
        print(f"{a.geo}: updated (coordinates {'from ' + a.cameras if info is not None else 'kept'}, in_zone from {a.zone})")
    if info is not None:
        moved = geo.loc[geo.method.eq("cctv_info"), "geo_offset_m"]
        print(f"cameras: {a.cameras} | {int(geo.method.eq('cctv_info').sum())} of {len(geo)} with course coordinates "
              f"and road names | shift from geocoded point median {moved.median():.0f} m, max {moved.max():.0f} m")
    else:
        print(f"cameras: {a.cameras} not found, geocoded coordinates kept")

    cams = pd.DataFrame({"camera_id": geo.camera_id, "lat": geo.lat, "lon": geo.lon,
                         "geo_method": geo["method"] if "method" in geo.columns else "unknown",
                         "road_name": geo["road_name"] if "road_name" in geo.columns else None,
                         "road_class": geo["road_class"] if "road_class" in geo.columns else None,
                         "in_zone": geo.in_zone, "zone_edge_m": edge})
    ctx = grid.merge(cams, on="camera_id", how="left")
    for c in WX + ["weather_km"]:
        ctx[c] = np.nan
    ctx["rain_source"] = pd.Series([None] * len(ctx), dtype=object)
    for name, path, label in (("city", a.weather, a.label), ("camera", a.rain, a.rain_label)):
        if not Path(path).exists():
            print(f"weather ({name}): {path} not found")
            continue
        f, points = weather_frame(path, a.weather_tz, label, cams, grid)
        f = ctx[KEY].merge(f, on=KEY, how="left")
        for c in ("temp_c", "cloud_cover"):
            ctx[c] = f[c].where(f[c].notna(), ctx[c]).to_numpy()
        hit = f.rain_mm.notna().to_numpy()
        ctx.loc[hit, "rain_mm"] = f.rain_mm.to_numpy()[hit]
        ctx.loc[hit, "weather_km"] = f.weather_km.to_numpy()[hit]
        ctx.loc[hit, "rain_source"] = name
        dist = f", camera to point <= {f.weather_km.max():.1f} km" if f.weather_km.notna().any() else ""
        print(f"weather ({name}): {path} | {points} point(s){dist} | label {label} | camera-hours with rain "
              f"{hit.mean():.1%}")
    ctx["rain_class"] = pd.cut(ctx.rain_mm, RAIN_BINS, right=False, labels=RAIN_LABELS)
    ctx[COLS].to_csv(a.out, index=False)
    print(f"rain: {int((ctx.rain_mm >= 0.1).sum()):,} rainy camera-hours, source "
          f"{ctx.rain_source.value_counts().to_dict()}")
    print(f"zone: {a.zone} | {int(cams.in_zone.sum())} of {len(cams)} cameras inside, outside "
          f"{cams.loc[~cams.in_zone, 'camera_id'].tolist()}, no coordinates "
          f"{cams.loc[cams.lat.isna() | cams.lon.isna(), 'camera_id'].tolist()}")
    print(f"{a.out}: {len(ctx):,} camera-hours, {ctx.camera_id.nunique()} cameras, {ctx.date.nunique()} days")


if __name__ == "__main__":
    main()
