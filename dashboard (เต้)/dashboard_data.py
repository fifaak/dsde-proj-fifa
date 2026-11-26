"""Inputs and filter calculations shared by the dashboard views."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_out"
RESULTS = Path(os.environ.get("DSDE_DASHBOARD_RESULTS_DIR", DATA)).expanduser().resolve()
EVIDENCE = ROOT / "insights (ม่อน)" / "evidence"
HOUR = ["camera_id", "date", "hour"]
MODEL_THRESHOLD = 0.60


def minute_of_day(hhmm):
    hours, minutes = map(int, hhmm.split(":"))
    return 60 * hours + minutes


@st.cache_data(ttl=60)
def load_inputs(results_dir=DATA):
    hourly = pd.read_csv(results_dir / "agg_camera_hour.csv", dtype={"date": str})
    sightings = pd.read_parquet(results_dir / "heavy_sightings.parquet")
    sightings["timestamp"] = pd.to_datetime(sightings.timestamp)
    frames = pd.read_csv(
        DATA / "images_clean.csv", usecols=["camera_id", "timestamp"], parse_dates=["timestamp"]
    )
    frames["date"] = frames.timestamp.dt.strftime("%Y-%m-%d")
    frames["hour"] = frames.timestamp.dt.hour
    rules = json.loads((ROOT / "config/ban_rules.json").read_text(encoding="utf-8"))
    zone_path = ROOT / "config/ban_zone.geojson"
    zone = json.loads(zone_path.read_text(encoding="utf-8")) if zone_path.is_file() else None
    geo_path = results_dir / "cameras_geo.csv"
    geo = pd.read_csv(geo_path).dropna(subset=["lat", "lon"]) if geo_path.is_file() else None
    evidence_path = EVIDENCE / "index.csv"
    evidence = pd.read_csv(evidence_path, parse_dates=["timestamp"]) if evidence_path.is_file() else None
    weather_path = results_dir / "weather_hourly.csv"
    weather = pd.read_csv(weather_path) if weather_path.is_file() else None
    return hourly, sightings, frames, rules, zone, geo, evidence, weather


def filter_data(hourly, sightings, frames, dates, camera_ids, day_type, zone_membership,
                minimum_score, windows):
    selected = hourly.loc[hourly.date.isin(dates) & hourly.camera_id.isin(camera_ids)]
    if day_type is not None:
        selected = selected.loc[selected.weekday.eq(day_type)]
    if zone_membership is not None:
        selected = selected.loc[selected.in_zone.eq(zone_membership)]

    keys = selected[HOUR]
    # Frames stay in the denominator even when the confidence slider removes every detection.
    seen = sightings.merge(keys, on=HOUR, how="inner")
    if minimum_score > MODEL_THRESHOLD:
        seen = seen.loc[seen.p_heavy.ge(minimum_score)]
    observed = frames.merge(selected[HOUR + ["weekday", "in_zone"]], on=HOUR, how="inner")
    observed["slot"] = observed.timestamp.dt.floor("10min")
    minutes = observed.timestamp.dt.hour * 60 + observed.timestamp.dt.minute
    in_window = pd.Series(False, index=observed.index)
    for start, end in windows:
        in_window |= minutes.between(minute_of_day(start), minute_of_day(end), inclusive="left")
    observed["restricted"] = observed.weekday & observed.in_zone & in_window

    counts = seen.groupby(HOUR).agg(sightings=("timestamp", "size"), restricted_sightings=("in_ban6", "sum"))
    restricted_frames = observed.groupby(HOUR).restricted.sum().rename("restricted_frames")
    selected = selected.join(counts, on=HOUR).join(restricted_frames, on=HOUR)
    columns = ["sightings", "restricted_sightings", "restricted_frames"]
    selected[columns] = selected[columns].fillna(0).astype(int)
    return selected, seen, observed


def camera_summary(frames, sightings, cameras):
    observed = frames.groupby("camera_id").size().rename("frames")
    detected = sightings.groupby("camera_id").size().rename("sightings")
    ranking = observed.to_frame().join(detected).reset_index().merge(cameras, on="camera_id")
    ranking["sightings"] = ranking.sightings.fillna(0).astype(int)
    ranking["per_frame"] = ranking.sightings / ranking.frames
    return ranking.sort_values(["sightings", "camera_id"], ascending=[False, True])


def rain_by_hour(hourly, weather):
    if "rain_mm" in hourly and hourly.rain_mm.notna().any():
        daily = hourly.groupby(["date", "hour"], as_index=False).rain_mm.mean()
    elif weather is not None and {"timestamp", "rain_mm"}.issubset(weather.columns):
        timestamps = pd.to_datetime(weather.timestamp, errors="coerce", format="mixed")
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_convert("Asia/Bangkok")
        daily = weather.assign(date=timestamps.dt.strftime("%Y-%m-%d"), hour=timestamps.dt.hour)
        daily = daily.groupby(["date", "hour"], as_index=False).rain_mm.mean()
        daily = daily.merge(hourly[["date", "hour"]].drop_duplicates(), on=["date", "hour"])
    else:
        return pd.Series(dtype=float)
    return daily.groupby("hour").rain_mm.mean().dropna()


def timeline_by_slot(frames, sightings):
    observed = frames.groupby("slot").size().rename("frames")
    detected = sightings.groupby(sightings.timestamp.dt.floor("10min")).size().rename("sightings")
    timeline = observed.to_frame().join(detected)
    timeline["sightings"] = timeline.sightings.fillna(0).astype(int)
    return timeline
