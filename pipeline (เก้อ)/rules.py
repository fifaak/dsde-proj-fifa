"""Stage 4: truck-ban rule engine + aggregation.

Inputs
  data_out/images_clean.csv   one row per valid image (denominator: frames observed)
  data_out/vehicles.parquet   classified candidate boxes (vclass, heavy_type)
  config/ban_rules.json       ban windows

Steps
  1. Static-object filter: a "heavy" box that sits in the same place (16 px grid) in >= 3% of a camera's frames is a
     parked truck / painted sign / billboard, not traffic -> dropped.
  2. Day type: weekday vs weekend/holiday.  Ban flags: in_ban6 (6-wheel windows), in_ban10 (10-wheel windows).
  3. Aggregate per camera x date x hour: frames, heavy sightings, heavy per frame.

Outputs
  data_out/heavy_sightings.parquet   one row per heavy-truck sighting with ban flags
  data_out/agg_camera_hour.csv       camera x date x hour table (all frames, zeros included)

Usage: python "pipeline (เก้อ)/rules.py"
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data_out")
GRID, STATIC_SHARE = 16, 0.03


def to_min(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def in_windows(minute: pd.Series, windows) -> pd.Series:
    m = pd.Series(False, index=minute.index)
    for a, b in windows:
        m |= (minute >= to_min(a)) & (minute < to_min(b))
    return m


def main():
    cfg = json.loads(Path("config/ban_rules.json").read_text())
    imgs = pd.read_csv(OUT / "images_clean.csv", parse_dates=["timestamp"])
    veh = pd.read_parquet(OUT / "vehicles.parquet")
    heavy = veh[veh.vclass == "heavy"].copy()

    # 1. static objects
    frames_per_cam = imgs.groupby("camera_id").size()
    key = [heavy.camera_id] + [(heavy[c] / GRID).round().astype(int) for c in ("x1", "y1", "x2", "y2")]
    heavy["loc_frames"] = heavy.groupby(key).path.transform("nunique")
    heavy["static"] = heavy.loc_frames >= STATIC_SHARE * heavy.camera_id.map(frames_per_cam)
    print(f"heavy boxes {len(heavy)}, static removed {heavy.static.sum()} "
          f"({heavy[heavy.static].camera_id.nunique()} cameras)")
    heavy = heavy[~heavy.static].drop(columns=["static", "loc_frames"])

    # 2. day type + ban flags (ban applies only to cameras inside the ban zone, see scraping (เพา)/build_ban_zone.py)
    holidays = set(cfg["public_holidays"])
    geo_path = OUT / "cameras_geo.csv"
    zone = (pd.read_csv(geo_path).set_index("camera_id").in_zone if geo_path.exists() and
            "in_zone" in pd.read_csv(geo_path, nrows=1).columns else None)
    for df in (imgs, heavy):
        ts = pd.to_datetime(df.timestamp)
        df["date"] = ts.dt.date.astype(str)
        df["hour"] = ts.dt.hour
        df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
        df["weekday"] = ((ts.dt.dayofweek < 5) & ~df.date.isin(holidays)).to_numpy()
        df["in_zone"] = df.camera_id.map(zone).fillna(True).astype(bool) if zone is not None else True
        df["in_ban6"] = df.weekday & df.in_zone & in_windows(df.minute_of_day, cfg["rules"]["truck6"])
        df["in_ban10"] = df.weekday & df.in_zone & in_windows(df.minute_of_day, cfg["rules"]["truck10"])
        # same clock windows regardless of weekday: used to compare weekday vs weekend at matching hours
        df["ban_hours6"] = in_windows(df.minute_of_day, cfg["rules"]["truck6"])
    heavy["violation"] = np.where(heavy.heavy_type == "truck10", heavy.in_ban10, heavy.in_ban6)
    heavy = heavy.merge(imgs[["path", "camera_name", "location"]], on="path", how="left")
    heavy.to_parquet(OUT / "heavy_sightings.parquet")

    # 3. aggregate (frames are the denominator: one frame every ~10 min per camera)
    g = ["camera_id", "date", "hour", "weekday", "in_zone"]
    frames = imgs.groupby(g).agg(frames=("path", "size"), night=("very_dark", "mean"))
    h = heavy.groupby(g).agg(heavy=("path", "size"), heavy_frames=("path", "nunique"),
                             heavy10=("heavy_type", lambda s: (s == "truck10").sum()))
    agg = frames.join(h).fillna({"heavy": 0, "heavy_frames": 0, "heavy10": 0}).reset_index()
    agg["ban6"] = agg.weekday & agg.in_zone & in_windows(agg.hour * 60, cfg["rules"]["truck6"])
    agg["heavy_per_frame"] = agg.heavy / agg.frames
    agg = agg.merge(imgs.drop_duplicates("camera_id")[["camera_id", "camera_name", "location"]], on="camera_id")
    agg.to_csv(OUT / "agg_camera_hour.csv", index=False)

    wk = agg[agg.weekday]
    print(f"sightings kept {len(heavy)}  in 6-wheel ban windows (weekday) {heavy.in_ban6.sum()}")
    print("heavy per frame, weekday:", wk.groupby("hour").apply(lambda d: d.heavy.sum() / d.frames.sum()).round(3).to_dict())


if __name__ == "__main__":
    main()
