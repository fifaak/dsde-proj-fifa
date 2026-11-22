"""Bangkok truck-ban monitoring dashboard (Streamlit).

Run from the project root:  streamlit run dashboard/app.py
Reads data_out/agg_camera_hour.csv, data_out/heavy_sightings.parquet, config/ban_rules.json,
and (optional) data_out/cameras_geo.csv with columns camera_id, lat, lon for the map.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_out"
st.set_page_config(page_title="Bangkok Truck Ban Monitor", layout="wide")


@st.cache_data
def load():
    agg = pd.read_csv(DATA / "agg_camera_hour.csv")
    sight = pd.read_parquet(DATA / "heavy_sightings.parquet")
    cfg = json.loads((ROOT / "config/ban_rules.json").read_text())
    geo = pd.read_csv(DATA / "cameras_geo.csv") if (DATA / "cameras_geo.csv").exists() else None
    return agg, sight, cfg, geo


agg, sight, cfg, geo = load()
cams = agg.drop_duplicates("camera_id")[["camera_id", "camera_name", "location"]]
cams["label"] = cams.camera_id.astype(str) + " · " + cams.camera_name

# ---------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters")
dates = sorted(agg.date.unique())
sel_dates = st.sidebar.multiselect("Dates", dates, default=dates)
sel_cams = st.sidebar.multiselect("Cameras (empty = all)", cams.label.tolist())
cam_ids = cams[cams.label.isin(sel_cams)].camera_id.tolist() or cams.camera_id.tolist()
a = agg[agg.date.isin(sel_dates) & agg.camera_id.isin(cam_ids)]
s = sight[sight.date.isin(sel_dates) & sight.camera_id.isin(cam_ids)]

st.title("🚛 Bangkok Truck Ban Monitor")
st.caption("Heavy trucks (6+ wheels) detected in BMA CCTV snapshots, 25–31 Aug 2026, 06:00–21:00, one frame per camera "
           "every ~10 min. YOLO11s (CoreML) → fine-tuned CLIP ViT-B/16 crop classifier. "
           "Heavy-truck classifier: precision ≈ 0.65, recall ≈ 0.69 (holdout of 90 hand labels) — "
           "numbers are estimates for comparing hours/cameras, not evidence against individual vehicles.")

# ---------------------------------------------------------------- KPIs
wk = a[a.weekday]
ban_frames = wk[wk.ban6].frames.sum()
k = st.columns(5)
k[0].metric("Frames analysed", f"{a.frames.sum():,}")
k[1].metric("Cameras", a.camera_id.nunique())
k[2].metric("Heavy-truck sightings", f"{int(a.heavy.sum()):,}")
k[3].metric("Sightings in ban windows (weekday)", f"{int(s.in_ban6.sum()):,}")
k[4].metric("Heavy per frame in ban windows", f"{wk[wk.ban6].heavy.sum() / max(ban_frames, 1):.3f}")

# ---------------------------------------------------------------- hourly profile
st.subheader("When are heavy trucks on the road?")
rate = (a.groupby(["weekday", "hour"]).apply(lambda d: d.heavy.sum() / d.frames.sum(), include_groups=False)
        .rename("heavy_per_frame").reset_index())
rate["day type"] = rate.weekday.map({True: "Weekday (ban applies)", False: "Weekend (no ban)"})
fig = px.line(rate, x="hour", y="heavy_per_frame", color="day type", markers=True,
              labels={"heavy_per_frame": "heavy trucks per frame", "hour": "hour of day"})
for lo, hi in cfg["rules"]["truck6"]:
    fig.add_vrect(x0=int(lo[:2]) - 0.5, x1=int(hi[:2]) - 0.5, fillcolor="red", opacity=0.08, line_width=0,
                  annotation_text="6-wheel ban", annotation_position="top left")
fig.update_layout(height=380, xaxis=dict(dtick=1))
st.plotly_chart(fig, use_container_width=True)
st.caption("Shaded = 6-wheel ban windows (weekdays). After ~18:30 detection is weaker (dark, glare), "
           "so the evening drop is partly a measurement effect.")

c1, c2 = st.columns([3, 2])

# ---------------------------------------------------------------- heatmap
with c1:
    st.subheader("Camera × hour (weekday heavy per frame)")
    top = wk.groupby("camera_id").heavy.sum().nlargest(25).index
    hm = wk[wk.camera_id.isin(top)].groupby(["camera_id", "hour"]).apply(
        lambda d: d.heavy.sum() / d.frames.sum(), include_groups=False).unstack()
    hm.index = [cams.set_index("camera_id").label[i][:40] for i in hm.index]
    st.plotly_chart(px.imshow(hm, aspect="auto", color_continuous_scale="OrRd",
                              labels=dict(color="per frame")).update_layout(height=620), use_container_width=True)

# ---------------------------------------------------------------- ranking
with c2:
    st.subheader("Cameras with most sightings in ban windows")
    rk = (wk[wk.ban6].groupby("camera_id").agg(ban_frames=("frames", "sum"), sightings=("heavy", "sum"))
          .assign(per_frame=lambda d: (d.sightings / d.ban_frames).round(3)).reset_index()
          .merge(cams, on="camera_id").sort_values("sightings", ascending=False))
    st.dataframe(rk[["camera_id", "camera_name", "location", "sightings", "ban_frames", "per_frame"]],
                 hide_index=True, height=620)

# ---------------------------------------------------------------- map (optional)
if geo is not None:
    st.subheader("Map: heavy trucks during weekday ban windows")
    metric = st.radio("Colour by", ["per_frame", "sightings"], horizontal=True,
                      format_func={"per_frame": "sightings per frame", "sightings": "total sightings"}.get)
    m = rk.merge(geo, on="camera_id")
    # cameras sharing one geocoded point (same junction, different approach) are spread on a ~60 m ring
    m["k"] = m.groupby(["lat", "lon"]).cumcount()
    m["n"] = m.groupby(["lat", "lon"]).camera_id.transform("size")
    ang = 2 * np.pi * m.k / m.n
    m["lat_p"] = m.lat + np.where(m.n > 1, 0.00055 * np.sin(ang), 0)
    m["lon_p"] = m.lon + np.where(m.n > 1, 0.00055 * np.cos(ang), 0)
    m["size"] = 6 + m.sightings
    fig = px.scatter_map(m, lat="lat_p", lon="lon_p", size="size", size_max=28, color=metric,
                         color_continuous_scale="OrRd", hover_name="camera_name",
                         hover_data={"camera_id": True, "location": True, "sightings": True, "per_frame": ":.3f",
                                     "method": True, "lat_p": False, "lon_p": False, "size": False},
                         zoom=12.3, center={"lat": 13.758, "lon": 100.535}, height=680, map_style="carto-positron")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Camera positions are geocoded from the landmark in each camera name (OpenStreetMap Nominatim) — "
               "landmark-level accuracy; `*_approx` = nearby road/landmark. Replace data_out/cameras_geo.csv with "
               "official BMA coordinates when available.")
else:
    st.info("Add data_out/cameras_geo.csv (camera_id, lat, lon) — e.g. from the scraped BMA camera list — to show a map.")

# ---------------------------------------------------------------- gallery
st.subheader("Check the evidence: sightings during ban windows")
g1, g2 = st.columns([1, 3])
cam_pick = g1.selectbox("Camera", rk.camera_id.astype(str) + " · " + rk.camera_name)
n = g1.slider("Samples", 4, 24, 12)
pick = s[(s.camera_id == int(cam_pick.split(" · ")[0])) & s.in_ban6]
pick = pick.sample(min(n, len(pick)), random_state=0) if len(pick) else pick
cols = g2.columns(6)
for i, r in enumerate(pick.itertuples()):
    try:
        im = Image.open(r.path).convert("RGB")
        pw, ph = (r.x2 - r.x1) * 0.3, (r.y2 - r.y1) * 0.3
        crop = im.crop((max(0, r.x1 - pw), max(0, r.y1 - ph), min(r.img_w, r.x2 + pw), min(r.img_h, r.y2 + ph)))
        cols[i % 6].image(crop, caption=f"{str(r.timestamp)[5:16]} · p={r.p_heavy:.2f}", use_container_width=True)
    except FileNotFoundError:
        cols[i % 6].write("image missing")

with st.expander("Method & caveats"):
    st.markdown(f"""
- **Ban rules** (`config/ban_rules.json`): 6-wheel+ banned {cfg['rules']['truck6']}, 10-wheel+ {cfg['rules']['truck10']},
  weekdays except public holidays. Verify against the official announcement; some roads differ.
- **Sampling**: one snapshot every ~10 min → counts are *sightings per frame*, not traffic volume.
- **Classifier**: 6 vs 10 wheels cannot be told apart reliably at 352×288, so all heavy trucks use the 6-wheel windows.
  Common errors: passenger vans/minibuses and pickups with box bodies counted as heavy; buses partly visible.
- **Static objects** (parked trucks, on-screen labels, billboards) removed when the same box repeats in ≥3% of a camera's frames.
- **Data gaps**: system outage 27 Aug 06:00–08:50; night frames (after 18:30) have lower recall.
""")
