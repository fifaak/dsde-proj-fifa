import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).parent))
from join_scraped import COLS as CONTEXT_COLS  # noqa: E402
from join_scraped import EXTERNAL, canonical, load_zone, read_weather  # noqa: E402

OUT = Path("data_out")
BKK = (13.49, 100.32, 13.96, 100.94)
T0, T1 = pd.Timestamp("2026-08-25"), pd.Timestamp("2026-09-01")
HOURS = (5, 22)
CAPTURE_HOURS = range(6, 22)
DET_CLS = {"car", "motorcycle", "bus", "truck"}
ZS_CLASS = {"light", "bus", "truck6", "truck10", "other"}
VCLASS = {"light", "bus", "heavy", "other"}
RAIN_CLASS = {"none", "light", "moderate", "heavy"}
PERIODS = {"morning_peak", "midday", "evening_peak", "night"}
DET = ["path", "camera_id", "timestamp", "cls", "conf", "x1", "y1", "x2", "y2", "img_w", "img_h"]
ZS = ["zs_light", "zs_bus", "zs_truck6", "zs_truck10", "zs_other"]
PROBS = ["p_light", "p_bus", "p_heavy", "p_other"]
FEATURES = ["path", "camera_id", "camera_name", "location", "timestamp", "width", "height", "file_size", "brightness",
            "contrast", "saturation", "sharpness", "dark_frac", "bright_frac", "ahash", "parse_ok", "read_ok"]
CLEAN = ["path", "camera_id", "camera_name", "location", "timestamp", "date", "hour", "period", "width", "height",
         "file_size", "brightness", "contrast", "saturation", "sharpness", "very_dark", "blurry", "stale_repeat",
         "system_outage", "invalid"]
SIGHTINGS = DET + PROBS + ["vclass", "heavy_type", "date", "hour", "minute_of_day", "weekday", "in_zone", "in_ban6",
                           "in_ban10", "ban_hours6", "violation", "camera_name", "location"]
AGG = ["camera_id", "date", "hour", "weekday", "in_zone", "frames", "night", "heavy", "heavy_frames", "heavy10", "ban6",
       "heavy_per_frame", "camera_name", "location"]
EXTRA = ["lat", "lon", "road_name", "zone_edge_m", "rain_mm", "temp_c", "cloud_cover", "rain_class", "ring_road"]
_cache = {}


class Check:
    def __init__(self, path):
        self.path = Path(path)
        self.errors, self.warnings = [], []
        self.rows = self.cams = self.t_min = self.t_max = None

    def need(self, ok, msg):
        if not ok:
            self.errors.append(msg)
        return bool(ok)

    def warn(self, ok, msg):
        if not ok:
            self.warnings.append(msg)
        return bool(ok)

    @property
    def status(self):
        return "FAIL" if self.errors else "WARN" if self.warnings else "OK"

    def span(self):
        if self.t_min is None or pd.isna(self.t_min):
            return ""
        return f"{self.t_min:%Y-%m-%d %H:%M} -> {self.t_max:%Y-%m-%d %H:%M}"

    def line(self):
        info = [f"{self.rows:,} rows" if self.rows is not None else "", f"{self.cams} cameras" if self.cams else "",
                self.span()]
        return f"{self.status:<5} {str(self.path):<34} " + " | ".join([i for i in info if i] + self.errors + self.warnings)

    def row(self):
        return {"file": str(self.path), "status": self.status, "rows": int(self.rows or 0),
                "cameras": int(self.cams or 0), "time_range": self.span(),
                "issues": "; ".join(self.errors + self.warnings)}


def columns(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pq.read_schema(path).names
    return list(pd.read_csv(path, nrows=0).columns)


def load(path, cols=None):
    path = Path(path)
    have = columns(path)
    use = tuple(c for c in cols if c in have) if cols else None
    key = (str(path.resolve()), path.stat().st_mtime_ns, use)
    if key not in _cache:
        if path.suffix == ".parquet":
            _cache[key] = pd.read_parquet(path, columns=list(use) if use else None)
        else:
            _cache[key] = pd.read_csv(path, usecols=list(use) if use else None, low_memory=False,
                                      dtype={"ahash": str, "md5": str})
    return _cache[key]


def sibling(c, name):
    p = c.path.parent / name
    return p if p.exists() and p.resolve() != c.path.resolve() else None


def table(c, df, required, notnull=(), unique=(), time=None, ranges=None, cats=None):
    missing = [x for x in required if x not in df.columns]
    c.need(not missing, f"missing columns {missing}")
    c.rows = len(df)
    c.need(len(df) > 0, "no rows")
    if "camera_id" in df.columns:
        c.cams = int(df.camera_id.nunique())
    for col in notnull:
        if col in df.columns:
            n = int(df[col].isna().sum())
            c.need(n == 0, f"{n:,} nulls in {col}")
    for key in unique:
        if set(key) <= set(df.columns):
            n = int(df.duplicated(list(key)).sum())
            c.need(n == 0, f"{n:,} duplicate rows on {'+'.join(key)}")
    if time and time in df.columns and len(df):
        ts = pd.to_datetime(df[time], errors="coerce")
        c.need(ts.notna().all(), f"{int(ts.isna().sum()):,} unparseable {time}")
        c.t_min, c.t_max = ts.min(), ts.max()
        c.need(c.t_min >= T0 and c.t_max < T1, f"{time} outside {T0:%Y-%m-%d}..{T1:%Y-%m-%d}")
        off = ~ts.dt.hour.between(*HOURS)
        c.need(not off.any(), f"{int(off.sum()):,} rows outside {HOURS[0]:02d}-{HOURS[1]:02d}h")
    for col, (lo, hi) in (ranges or {}).items():
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            bad = int(((v < lo) | (v > hi)).sum())
            c.need(bad == 0, f"{bad:,} values of {col} outside [{lo}, {hi}]")
    for col, allowed in (cats or {}).items():
        if col in df.columns:
            extra = sorted(set(df[col].dropna().astype(str).unique()) - set(allowed))
            c.need(not extra, f"unexpected {col} values {extra[:5]}")


def hour_span(c, df):
    ts = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit="h")
    c.t_min, c.t_max = ts.min(), ts.max()


def image_paths(c, paths):
    s = paths.dropna().astype(str)
    absolute = s.str.match(r"^(/|[A-Za-z]:[\\/])")
    c.warn(not absolute.any(), f"{int(absolute.sum()):,} absolute image paths (make portable)")
    sample = s.sample(min(50, len(s)), random_state=0) if len(s) else s
    roots = {Path(p).parents[1] for p in sample if len(Path(p).parents) > 1}
    if any(r.is_dir() for r in roots):
        miss = sum(not Path(p).exists() for p in sample)
        c.warn(miss == 0, f"{miss}/{len(sample)} sampled image files not found")


def subset(c, child, parent_path, col="path"):
    parent = load(parent_path, [col])
    n = int((~child[col].isin(set(parent[col]))).sum())
    c.need(n == 0, f"{n:,} rows whose {col} is not in {Path(parent_path).name}")


def check_features(c):
    df = load(c.path, FEATURES)
    parsed = df.parse_ok.fillna(False).astype(bool) if "parse_ok" in df.columns else pd.Series(True, index=df.index)
    table(c, df[parsed], FEATURES, notnull=["path"], unique=[("path",)], time="timestamp",
          ranges={"dark_frac": (0, 1), "bright_frac": (0, 1), "brightness": (0, 255)})
    c.rows = len(df)
    c.warn(parsed.all(), f"{int((~parsed).sum())} files with unparsed names (dropped by clean.py)")
    if "read_ok" in df.columns:
        unread = ~df.read_ok.fillna(False).astype(bool)
        c.warn(not unread.any(), f"{int(unread.sum())} unreadable images (dropped by clean.py)")


def check_images_clean(c):
    df = load(c.path, CLEAN)
    table(c, df, CLEAN, notnull=["path", "camera_id", "timestamp", "hour"], unique=[("path",), ("camera_id", "timestamp")],
          time="timestamp", cats={"period": PERIODS})
    if "invalid" in df.columns:
        c.need(not df.invalid.astype(bool).any(), "invalid frames kept")
    if "path" in df.columns:
        image_paths(c, df.path)
        if f := sibling(c, "features.csv"):
            c.need(len(df) <= len(load(f, ["path"])), "more rows than features.csv")
            subset(c, df, f)


def check_cameras(c):
    df = load(c.path)
    table(c, df, ["camera_id", "camera_name", "location", "is_intersection", "direction"], notnull=["camera_id"],
          unique=[("camera_id",)])


def check_detections(c):
    df = load(c.path, DET)
    table(c, df, DET, notnull=DET, time="timestamp", ranges={"conf": (0, 1)}, cats={"cls": DET_CLS})
    if set(DET) <= set(df.columns):
        c.need(((df.x2 >= df.x1) & (df.y2 >= df.y1)).all(), "boxes with x2 < x1 or y2 < y1")
        c.need(((df.x1 >= 0) & (df.y1 >= 0) & (df.x2 <= df.img_w + 0.5) & (df.y2 <= df.img_h + 0.5)).all(),
               "boxes outside the image")
        if f := sibling(c, "images_clean.csv"):
            subset(c, df, f)


def check_candidates(c):
    df = load(c.path, DET + ZS + ["zs_class"])
    table(c, df, DET + ZS + ["zs_class"], notnull=["path", "camera_id", "timestamp"] + ZS, time="timestamp",
          ranges={z: (0, 1) for z in ZS}, cats={"cls": DET_CLS, "zs_class": ZS_CLASS})
    if set(ZS) <= set(df.columns):
        c.need(np.allclose(df[ZS].sum(axis=1), 1, atol=1e-3), "zero-shot probabilities do not sum to 1")
    if f := sibling(c, "detections.parquet"):
        c.need(len(df) <= pq.read_metadata(f).num_rows, "more candidates than detections")
    emb = c.path.with_name(c.path.stem + "_emb.npy")
    if emb.exists():
        n = np.load(emb, mmap_mode="r").shape[0]
        c.need(n == len(df), f"{emb.name} has {n:,} rows, candidates has {len(df):,}")


def check_vehicles(c):
    cols = DET + PROBS + ["vclass", "heavy_type"]
    df = load(c.path, cols)
    table(c, df, cols, notnull=["path", "camera_id", "timestamp", "vclass"] + PROBS, time="timestamp",
          ranges={p: (0, 1) for p in PROBS}, cats={"vclass": VCLASS, "heavy_type": {"", "truck6", "truck10"}})
    if set(PROBS + ["vclass", "heavy_type"]) <= set(df.columns):
        c.need(np.allclose(df[PROBS].sum(axis=1), 1, atol=1e-3), "class probabilities do not sum to 1")
        c.need(((df.vclass == "heavy") == (df.heavy_type.fillna("") != "")).all(), "heavy_type does not match vclass")
    if f := sibling(c, "candidates.parquet"):
        n = pq.read_metadata(f).num_rows
        c.need(len(df) == n, f"{len(df):,} rows but candidates.parquet has {n:,}")


def check_geo(c):
    df = load(c.path)
    table(c, df, ["camera_id", "lat", "lon", "method"], notnull=["camera_id"], unique=[("camera_id",)],
          ranges={"lat": (BKK[0], BKK[2]), "lon": (BKK[1], BKK[3])})
    if {"lat", "lon"} <= set(df.columns):
        c.warn(df.lat.notna().all() and df.lon.notna().all(),
               f"{int((df.lat.isna() | df.lon.isna()).sum())} cameras without coordinates")
    c.warn("in_zone" in df.columns, "no in_zone column (make join adds it)")
    ref = sibling(c, "cameras.csv") or sibling(c, "images_clean.csv")
    if ref and "camera_id" in df.columns:
        miss = sorted(set(load(ref, ["camera_id"]).camera_id) - set(df.camera_id))
        c.need(not miss, f"no coordinates row for cameras {miss[:10]}")


def check_rules(c):
    try:
        cfg = json.loads(c.path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        c.need(False, f"not valid JSON: {e}")
        return
    rules = cfg.get("rules") or {}
    c.rows = 0
    for k in ("truck6", "truck10"):
        wins = rules.get(k)
        if not c.need(isinstance(wins, list) and wins, f"rules.{k} missing"):
            continue
        c.rows += len(wins)
        for w in wins:
            ok = (isinstance(w, list) and len(w) == 2 and all(re.fullmatch(r"\d{2}:\d{2}", str(x)) for x in w)
                  and w[0] < w[1])
            c.need(ok, f"rules.{k}: bad window {w}")
    for w in cfg.get("truck10_ring_exempt", []):
        ok = isinstance(w, list) and len(w) == 2 and all(re.fullmatch(r"\d{2}:\d{2}", str(x)) for x in w) and w[0] < w[1]
        c.need(ok, f"truck10_ring_exempt: bad window {w}")
    ring = cfg.get("ring_roads", [])
    c.need(isinstance(ring, list) and all(isinstance(x, str) and x for x in ring), "ring_roads must be a list of names")
    hol = cfg.get("public_holidays")
    if c.need(isinstance(hol, list), "public_holidays must be a list"):
        bad = [h for h in hol if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(h))]
        c.need(not bad, f"public_holidays not YYYY-MM-DD: {bad}")


def check_zone(c):
    try:
        gj = json.loads(c.path.read_text(encoding="utf-8"))
        load_zone(c.path)
    except (Exception, SystemExit) as e:
        c.need(False, f"cannot read zone: {e}")
        return
    feats = gj.get("features") or [gj]
    c.rows = len(feats)
    for f in feats:
        g = shape(f.get("geometry", f))
        x0, y0, x1, y1 = g.bounds
        c.need(BKK[1] <= x0 and x1 <= BKK[3] and BKK[0] <= y0 and y1 <= BKK[2], f"{g.geom_type} outside Bangkok")
        c.need(g.is_valid and not g.is_empty, f"invalid or empty {g.geom_type}")
    first = shape(feats[0].get("geometry", feats[0])).geom_type
    c.warn(first == "Polygon", f"first feature is {first}, make_timelapse.py draws the first Polygon only")


def check_weather(c, label="end"):
    try:
        w, group, _ = read_weather(c.path, label=label)
    except (Exception, SystemExit) as e:
        c.need(False, f"cannot read weather: {e}")
        return None
    c.rows = len(w)
    c.need(len(w) > 0, "no rows")
    hour_span(c, w)
    c.need((w.rain_mm.dropna() >= 0).all(), "negative rain_mm")
    c.need(w.temp_c.dropna().between(10, 45).all(), "temp_c outside 10-45")
    c.need(w.cloud_cover.dropna().between(0, 100).all(), "cloud_cover outside 0-100")
    c.warn(w.rain_mm.notna().any(), "no rain_mm values")
    days = pd.date_range(T0, T1, inclusive="left").strftime("%Y-%m-%d")
    have = set(zip(w.date, w.hour))
    miss = [(d, h) for d in days for h in CAPTURE_HOURS if (d, h) not in have]
    c.warn(not miss, f"{len(miss)} of {len(days) * len(CAPTURE_HOURS)} capture hours without weather, first {miss[:3]}")
    return w, group


def reference_cameras():
    for p in (OUT / "cameras.csv", OUT / "images_clean.csv"):
        if p.exists():
            return set(load(p, ["camera_id"]).camera_id)
    return set()


def check_camera_rain(c):
    got = check_weather(c, label="start")
    if got is None:
        return
    w, group = got
    if not c.need(group == ["camera_id"], "no camera_id column"):
        return
    c.cams = int(w.camera_id.nunique())
    miss = sorted(reference_cameras() - set(w.camera_id))
    c.need(not miss, f"no rain rows for cameras {miss[:10]}")
    days = pd.date_range(T0, T1, inclusive="left").strftime("%Y-%m-%d")
    have = w[w.rain_mm.notna()].groupby("camera_id").apply(lambda d: len(set(zip(d.date, d.hour))), include_groups=False)
    short = have[have < len(days) * len(CAPTURE_HOURS)]
    c.warn(short.empty, f"{len(short)} cameras missing some capture hours")


def read_external(c, names, **kw):
    df = canonical(pd.read_csv(c.path, encoding="utf-8-sig", **kw), names)
    c.rows = len(df)
    return df


def check_cctv_info(c):
    df = read_external(c, ["camera_id", "lat", "lon"])
    df["camera_id"] = pd.to_numeric(df.camera_id, errors="coerce")
    table(c, df, ["camera_id", "lat", "lon", "road_name"], notnull=["camera_id", "lat", "lon"], unique=[("camera_id",)],
          ranges={"lat": (BKK[0], BKK[2]), "lon": (BKK[1], BKK[3])})
    miss = sorted(reference_cameras() - set(df.camera_id)) if "camera_id" in df.columns else []
    c.need(not miss, f"no coordinates for cameras {miss[:10]}")


def check_truck_rules(c):
    df = read_external(c, [], dtype=str).fillna("")
    need = ["rule_id", "scope_type", "vehicle_type", "days_of_week", "start_time", "end_time"]
    missing = [x for x in need if x not in df.columns]
    if not c.need(not missing, f"missing columns {missing}"):
        return
    c.need(df.rule_id.is_unique, "duplicate rule_id")
    hm = r"\d{1,2}[:.]\d{2}"
    bad = df[~(df.start_time.str.fullmatch(hm) & df.end_time.str.fullmatch(hm))]
    c.need(bad.empty, f"bad times in {bad.rule_id.tolist()[:5]}")
    area10 = df[df.scope_type.eq("area") & df.vehicle_type.str.contains("10 ล้อ")]
    c.warn(not area10.empty, "no area rule for 10-wheel trucks, config/ban_rules.json keeps its windows")


def check_context(c):
    df = load(c.path, CONTEXT_COLS)
    table(c, df, CONTEXT_COLS, notnull=["camera_id", "date", "hour", "in_zone"], unique=[("camera_id", "date", "hour")],
          ranges={"lat": (BKK[0], BKK[2]), "lon": (BKK[1], BKK[3]), "hour": (0, 23), "rain_mm": (0, 500),
                  "temp_c": (10, 45), "cloud_cover": (0, 100)}, cats={"rain_class": RAIN_CLASS})
    if {"date", "hour"} <= set(df.columns) and len(df):
        hour_span(c, df)
    if "rain_mm" in df.columns:
        c.warn(df.rain_mm.notna().all(), f"{int(df.rain_mm.isna().sum()):,} camera-hours without weather")
    if (f := sibling(c, "images_clean.csv")) and {"camera_id", "date", "hour"} <= set(df.columns):
        imgs = load(f, ["camera_id", "timestamp"])
        ts = pd.to_datetime(imgs.timestamp)
        want = set(zip(imgs.camera_id, ts.dt.strftime("%Y-%m-%d"), ts.dt.hour))
        have = set(zip(df.camera_id, df.date.astype(str), df.hour))
        c.need(want <= have, f"{len(want - have):,} camera-hours of images_clean.csv missing")
        c.need(have <= want, f"{len(have - want):,} camera-hours not in images_clean.csv")


def check_sightings(c):
    df = load(c.path, SIGHTINGS + EXTRA)
    table(c, df, SIGHTINGS, notnull=["path", "camera_id", "timestamp", "in_zone", "in_ban6", "violation"],
          time="timestamp", cats={"vclass": {"heavy"}, "heavy_type": {"truck6", "truck10"}})
    if {"in_ban6", "weekday", "in_zone", "violation", "in_ban10"} <= set(df.columns):
        c.need(not (df.in_ban6 & ~(df.weekday & df.in_zone)).any(), "in_ban6 set outside weekday in-zone frames")
        c.need(not (df.violation & ~(df.in_ban6 | df.in_ban10)).any(), "violation outside ban windows")
        c.need(not (df.in_ban10 & ~(df.weekday & df.in_zone)).any(), "in_ban10 set outside weekday in-zone frames")
    c.warn(set(EXTRA) <= set(df.columns), "no scraped context columns (run make join rules)")
    if f := sibling(c, "vehicles.parquet"):
        n = int((load(f, ["vclass"]).vclass == "heavy").sum())
        c.need(len(df) <= n, f"{len(df):,} sightings but only {n:,} heavy boxes in vehicles.parquet")


def check_agg(c):
    df = load(c.path, AGG + EXTRA + ["ban10"])
    table(c, df, AGG, notnull=["camera_id", "date", "hour", "frames", "heavy"], unique=[("camera_id", "date", "hour")],
          ranges={"frames": (1, 60), "night": (0, 1), "heavy": (0, 10000)})
    if len(df) and {"date", "hour"} <= set(df.columns):
        hour_span(c, df)
    if {"heavy", "frames", "heavy_per_frame"} <= set(df.columns):
        c.need(np.allclose(df.heavy_per_frame, df.heavy / df.frames), "heavy_per_frame != heavy / frames")
    c.warn(set(EXTRA + ["ban10"]) <= set(df.columns), "no scraped context columns (run make join rules)")
    if {"ban10", "weekday", "in_zone"} <= set(df.columns):
        c.need(not (df.ban10 & ~(df.weekday & df.in_zone)).any(), "ban10 set outside weekday in-zone hours")
    if (f := sibling(c, "images_clean.csv")) and "frames" in df.columns:
        n = len(load(f, ["path"]))
        c.need(int(df.frames.sum()) == n, f"frames sum {int(df.frames.sum()):,} != images_clean rows {n:,}")
    if (f := sibling(c, "heavy_sightings.parquet")) and "heavy" in df.columns:
        n = pq.read_metadata(f).num_rows
        c.need(int(df.heavy.sum()) == n, f"heavy sum {int(df.heavy.sum()):,} != heavy_sightings rows {n:,}")


CHECKS = {
    "features.csv": check_features,
    "images_clean.csv": check_images_clean,
    "cameras.csv": check_cameras,
    "detections.parquet": check_detections,
    "candidates.parquet": check_candidates,
    "vehicles.parquet": check_vehicles,
    "cameras_geo.csv": check_geo,
    "weather_hourly.csv": check_weather,
    "ban_rules.json": check_rules,
    "ban_zone.geojson": check_zone,
    "context_camera_hour.csv": check_context,
    "heavy_sightings.parquet": check_sightings,
    "agg_camera_hour.csv": check_agg,
}
EXTERNAL_CHECKS = {
    "external_data/data/cameras.csv": check_cctv_info,
    "external_data/data/rain_hourly.csv": check_camera_rain,
    "external_data/data/truck_rules.csv": check_truck_rules,
}
ALL = [OUT / "features.csv", OUT / "images_clean.csv", OUT / "cameras.csv", OUT / "detections.parquet",
       OUT / "candidates.parquet", OUT / "vehicles.parquet", OUT / "cameras_geo.csv", OUT / "weather_hourly.csv",
       EXTERNAL / "cameras.csv", EXTERNAL / "rain_hourly.csv", EXTERNAL / "truck_rules.csv",
       Path("config/ban_rules.json"), Path("config/ban_zone.geojson"), OUT / "context_camera_hour.csv",
       OUT / "heavy_sightings.parquet", OUT / "agg_camera_hour.csv"]
OPTIONAL = {"features.csv", "cameras.csv", "weather_hourly.csv", "rain_hourly.csv", "truck_rules.csv"}


def run(paths):
    results = []
    for p in map(Path, paths):
        fn = next((f for k, f in EXTERNAL_CHECKS.items() if p.as_posix().endswith(k)), None) or CHECKS.get(p.name)
        if fn is None:
            continue
        c = Check(p)
        if p.exists():
            try:
                fn(c)
            except Exception as e:
                c.need(False, f"{type(e).__name__}: {e}")
        else:
            (c.warn if p.name in OPTIONAL else c.need)(False, "missing")
        results.append(c)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    results = run(ALL if a.all or not a.files else a.files)
    for c in results:
        print(c.line(), flush=True)
    failed = [c for c in results if c.errors]
    if failed:
        print(f"data quality check failed: {', '.join(c.path.name for c in failed)}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
