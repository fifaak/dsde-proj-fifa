"""Violation clip: heavy trucks (6+ wheels) seen INSIDE the ban zone DURING weekday 6-wheel ban windows.

Only ban windows are shown (06:00-09:00 and 16:00-20:00 on weekdays); allowed hours and weekends are skipped.
Each frame = one 10-min slot.
  map    ban zone outline; red dots = violations at a camera, size = violations accumulated in this ban window
         (bright = seen in this slot, faded = earlier in the window), so hotspots build up over the window
  right  big counter for the current window + top-5 cameras in this window
  bottom timeline of all ban windows of the week, bar = violations per slot, cursor = now
Slots where fewer than 50 cameras reported (system outage 27 Aug morning) are skipped, and so are windows left
with fewer than 6 slots.

Usage:  python viz/make_timelapse.py [--fps 6] [--out data_out/truck_violations.mp4] [--png-only 5 30 ...]
Basemap: Esri World Dark Gray Canvas (© Esri, HERE, Garmin, © OpenStreetMap contributors), cached on first run.
"""
import argparse
import json
from pathlib import Path

import contextily as cx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon as MplPolygon

matplotlib.rcParams["font.family"] = ["Sukhumvit Set", "DejaVu Sans"]
OUT = Path("data_out")
THAI_DAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
BG, FG, MUTED, RED = "#0e1117", "#e6e6e6", "#8a8f98", "#ff3b30"
TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
TILE_CREDIT = "Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors"
MIN_CAMS = 50
R = 6378137.0


def merc(lon, lat):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return np.radians(lon) * R, np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R


def to_min(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def load():
    cfg = json.loads(Path("config/ban_rules.json").read_text())
    ring = np.array(json.loads(Path("config/ban_zone.geojson").read_text())["features"][0]["geometry"]["coordinates"][0])
    imgs = pd.read_csv(OUT / "images_clean.csv", parse_dates=["timestamp"])
    sight = pd.read_parquet(OUT / "heavy_sightings.parquet")
    geo = pd.read_csv(OUT / "cameras_geo.csv")
    geo["k"] = geo.groupby(["lat", "lon"]).cumcount()            # spread co-located cameras on a ~120 m ring
    geo["n"] = geo.groupby(["lat", "lon"]).camera_id.transform("size")
    geo["x"], geo["y"] = merc(geo.lon, geo.lat)
    ang = 2 * np.pi * geo.k / geo.n
    geo["x"] += np.where(geo.n > 1, 120 * np.cos(ang), 0)
    geo["y"] += np.where(geo.n > 1, 120 * np.sin(ang), 0)
    geo = geo.set_index("camera_id")
    names = imgs.drop_duplicates("camera_id").set_index("camera_id").camera_name.str.replace("-", " ")

    windows = cfg["rules"]["truck6"]
    imgs["slot"] = imgs.timestamp.dt.floor("10min")
    reporting = imgs.groupby("slot").camera_id.nunique()
    slots = []
    for ts in sorted(imgs.slot.unique()):
        ts = pd.Timestamp(ts)
        m = ts.hour * 60 + ts.minute
        win = next((w for w in windows if to_min(w[0]) <= m < to_min(w[1])), None)
        weekday = ts.dayofweek < 5 and ts.strftime("%Y-%m-%d") not in cfg["public_holidays"]
        if win and weekday and reporting.get(ts, 0) >= MIN_CAMS:
            slots.append((ts, f"{ts.date()} {win[0]}", win))
    slots = pd.DataFrame(slots, columns=["slot", "window_id", "window"])
    slots = slots[slots.groupby("window_id").slot.transform("size") >= 6].reset_index(drop=True)  # drop outage stubs

    v = sight[sight.in_ban6].copy()                               # in zone + weekday + 6-wheel window
    v["slot"] = pd.to_datetime(v.timestamp).dt.floor("10min")
    per_slot_cam = v.groupby(["slot", "camera_id"]).size()
    per_slot = v.groupby("slot").size().reindex(slots.slot, fill_value=0).values
    return ring, slots, per_slot_cam, per_slot, geo, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--out", default=str(OUT / "truck_violations.mp4"))
    ap.add_argument("--png-only", nargs="*", type=int)
    a = ap.parse_args()
    ring, slots, per_slot_cam, per_slot, geo, names = load()

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    MAP = [0.01, 0.19, 0.60, 0.70]
    ax = fig.add_axes(MAP)
    tl = fig.add_axes([0.04, 0.05, 0.93, 0.09], facecolor=BG)

    # ---- map: zoom to the ban zone
    zx, zy = merc(ring[:, 0], ring[:, 1])
    x0, x1, y0, y1 = zx.min() - 800, zx.max() + 800, zy.min() - 600, zy.max() + 900
    aspect = (MAP[2] * 19.2) / (MAP[3] * 10.8)
    w, h = x1 - x0, y1 - y0
    w, h = max(w, h * aspect), max(h, w / aspect)
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    ext = (xc - w / 2, xc + w / 2, yc - h / 2, yc + h / 2)
    cache = OUT / "basemap_esri_darkgray_zone.npz"
    if cache.exists():
        z = np.load(cache)
        img, bext = z["img"], tuple(z["ext"])
    else:
        img, bext = cx.bounds2img(ext[0], ext[2], ext[1], ext[3], zoom=14, source=TILES)
        np.savez_compressed(cache, img=img, ext=np.array(bext))
    ax.imshow(img, extent=bext, interpolation="bilinear", zorder=0)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_axis_off()
    ax.add_patch(MplPolygon(np.c_[zx, zy], closed=True, fc=RED, alpha=0.07, lw=0, zorder=1))
    ax.add_patch(MplPolygon(np.c_[zx, zy], closed=True, fill=False, ec=RED, lw=2, ls="--", alpha=0.8, zorder=2))
    ax.text(0.01, 0.985, "เส้นประ = เขตห้ามรถบรรทุก (ประมาณ)", transform=ax.transAxes, color=RED, fontsize=13, va="top")
    ax.text(0.99, 0.008, TILE_CREDIT, transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)
    zc = geo[geo.in_zone.astype(bool)]
    ax.scatter(zc.x, zc.y, s=10, c="#5a6270", edgecolors="none", zorder=3)           # cameras (context only)
    acc = ax.scatter([], [], zorder=4, edgecolors="none")
    now = ax.scatter([], [], zorder=5, edgecolors="white", linewidths=1.5)

    # ---- header
    fig.text(0.03, 0.955, "รถบรรทุกวิ่งในเขตห้าม ระหว่างเวลาห้าม", color=FG, fontsize=30, weight="bold", va="center")
    fig.text(0.03, 0.912, "เฉพาะวันธรรมดา ช่วงห้าม 6 ล้อขึ้นไป (06:00–09:00 และ 16:00–20:00) · 1 เฟรม = 10 นาที",
             color=MUTED, fontsize=15, va="center")

    # ---- right panel
    t_time = fig.text(0.64, 0.85, "", color=FG, fontsize=28, weight="bold", va="center")
    t_win = fig.text(0.64, 0.80, "", color=RED, fontsize=18, va="center")
    t_big = fig.text(0.64, 0.70, "", color=RED, fontsize=64, weight="bold", va="center")
    fig.text(0.64, 0.63, "คันที่เห็นในเขตห้าม ในช่วงห้ามนี้", color=FG, fontsize=16, va="center")
    fig.text(0.64, 0.56, "กล้องที่เจอมากสุดในช่วงนี้", color=MUTED, fontsize=15, va="center")
    t_top = [fig.text(0.64, 0.515 - k * 0.045, "", color=FG, fontsize=15, va="center") for k in range(5)]
    fig.text(0.64, 0.225, "● ขนาดจุด = จำนวนคันสะสมในช่วงห้ามนี้ · จุดสว่าง = เพิ่งเห็นในรอบ 10 นาทีนี้", color=MUTED,
             fontsize=12)
    fig.text(0.64, 0.195, "ตัวแยกรถบรรทุก precision ≈ 0.65 — เป็นค่าประมาณ ใช้ดูจุดและเวลาที่เกิดบ่อย ไม่ใช่หลักฐานรายคัน",
             color=MUTED, fontsize=11)

    # ---- timeline: only ban windows, concatenated
    n = len(slots)
    tl.bar(np.arange(n), per_slot, width=0.85, color=RED, lw=0)
    for i in range(n):
        if i == 0 or slots.window_id[i] != slots.window_id[i - 1]:
            if i:
                tl.axvline(i - 0.5, color="#3a3f47", lw=1.5)
            ts = slots.slot[i]
            tl.text(i + 0.5, 1.06, f"{THAI_DAYS[ts.dayofweek][:3]} {ts.day} · {'เช้า' if ts.hour < 12 else 'เย็น'}",
                    transform=tl.get_xaxis_transform(), color=MUTED, fontsize=10)
    tl.set_xlim(-0.5, n - 0.5)
    tl.set_ylim(0, per_slot.max() * 1.1)
    tl.set_xticks([])
    tl.tick_params(colors=MUTED, labelsize=9, length=0)
    for sp in tl.spines.values():
        sp.set_visible(False)
    tl.set_ylabel("คัน/10 นาที", color=MUTED, fontsize=10)
    cursor = tl.axvline(0, color="white", lw=2.5)

    def draw(i):
        ts, wid, win = slots.slot[i], slots.window_id[i], slots.window[i]
        in_win = slots.slot[(slots.window_id == wid) & (slots.slot <= ts)]
        c = per_slot_cam[per_slot_cam.index.get_level_values(0).isin(in_win)].groupby(level=1).sum()
        c = c[c.index.isin(geo.index)]
        cur = per_slot_cam.get(ts, pd.Series(dtype=int))
        cur = cur[cur.index.isin(geo.index)]
        acc.set_offsets(np.c_[geo.loc[c.index, "x"], geo.loc[c.index, "y"]] if len(c) else np.empty((0, 2)))
        acc.set_sizes(40 + 55 * c.values)
        acc.set_facecolors([(1, 0.23, 0.19, 0.35)] * len(c))
        now.set_offsets(np.c_[geo.loc[cur.index, "x"], geo.loc[cur.index, "y"]] if len(cur) else np.empty((0, 2)))
        now.set_sizes(60 + 55 * c.reindex(cur.index).values if len(cur) else [])
        now.set_facecolors([RED] * len(cur))

        t_time.set_text(f"{THAI_DAYS[ts.dayofweek]} {ts.day} ส.ค.   {ts:%H:%M} น.")
        t_win.set_text(f"ช่วงห้าม{'เช้า' if ts.hour < 12 else 'เย็น'} {win[0]}–{win[1]}")
        t_big.set_text(f"{int(c.sum())}")
        top = c.sort_values(ascending=False).head(5)
        for k, t in enumerate(t_top):
            if k < len(top):
                cid = top.index[k]
                t.set_text(f"{int(top.iloc[k]):>3} คัน   กล้อง {cid} · {names.get(cid, '')[:30]}")
            else:
                t.set_text("")
        cursor.set_xdata([i, i])

    if a.png_only is not None:
        for i in a.png_only:
            draw(i)
            fig.savefig(OUT / f"violation_frame_{i:03d}.png", facecolor=BG)
        print("saved frames", a.png_only)
        return
    writer = FFMpegWriter(fps=a.fps, codec="libx264", bitrate=6000, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, a.out, dpi=100):
        for i in range(n):
            draw(i)
            writer.grab_frame(facecolor=BG)
            if i == n - 1 or slots.window_id[i + 1] != slots.window_id[i]:
                for _ in range(a.fps):                               # hold the end of each window for 1 s
                    writer.grab_frame(facecolor=BG)
    print(f"wrote {a.out}: {n} slots, ~{(n + a.fps * slots.window_id.nunique()) / a.fps:.0f} s at {a.fps} fps")


if __name__ == "__main__":
    main()
