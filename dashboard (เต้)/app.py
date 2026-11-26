"""Bangkok truck-ban dashboard. Run from the repository root with Streamlit."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

from dashboard_data import (
    DATA,
    EVIDENCE,
    MODEL_THRESHOLD,
    RESULTS,
    ROOT,
    camera_summary,
    filter_data,
    load_inputs,
    minute_of_day,
    rain_by_hour,
    timeline_by_slot,
)

ACCENT = "#ff6347"
BLUE = "#77bae9"
TEXT = "#e5ebf3"
MUTED = "#a6b1bf"
GRID = "#303b48"
ALL_SLOTS = "ทุกช่วงที่เลือก"
ALL_CAMERAS = "ทุกกล้อง"


def chart(fig, height, **layout):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        margin=dict(l=0, r=0, t=12, b=0),
        height=height,
        **layout,
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def show_chart(fig, key, description):
    st.plotly_chart(fig, key=key, theme=None, alt=description, config={"displayModeBar": False})


def filters(hourly, cameras):
    date_options = sorted(hourly.date.unique())
    labels = cameras.set_index("camera_id").label.to_dict()
    with st.sidebar:
        st.header("ตัวกรอง")
        dates = st.multiselect("วันที่", date_options, placeholder="ทุกวันที่มีข้อมูล") or date_options
        ids = st.multiselect(
            "กล้อง", labels, format_func=labels.get, placeholder=ALL_CAMERAS
        ) or list(labels)
        day = st.radio("ประเภทวัน", ["ทุกวัน", "วันทำงาน", "วันหยุด"])
        zone = st.radio("พื้นที่", ["ทุกพื้นที่", "ในเขตห้าม", "นอกเขตห้าม"])
        score = st.slider(
            "ความมั่นใจขั้นต่ำ", MODEL_THRESHOLD, 1.0, MODEL_THRESHOLD, 0.01,
            help="ผลที่ส่งออกจากโมเดลเริ่มที่ 0.60 ตัวกรองนี้เปลี่ยนจำนวนครั้งที่พบ โดยใช้จำนวนภาพเดิม",
        )
    return dates, ids, {"วันทำงาน": True, "วันหยุด": False}.get(day), {
        "ในเขตห้าม": True, "นอกเขตห้าม": False,
    }.get(zone), score


def metrics(hourly):
    frames = int(hourly.frames.sum())
    restricted_frames = int(hourly.restricted_frames.sum())
    restricted = int(hourly.restricted_sightings.sum())
    for col, (label, value) in zip(st.columns(5), [
        ("ภาพที่วิเคราะห์", f"{frames:,}"),
        ("กล้อง", f"{hourly.camera_id.nunique():,}"),
        ("พบรถหนัก (ครั้ง)", f"{hourly.sightings.sum():,}"),
        ("พบในช่วงห้าม (ครั้ง)", f"{restricted:,}"),
        ("พบต่อภาพในช่วงห้าม", f"{restricted / restricted_frames:.3f}" if restricted_frames else "—"),
    ]):
        col.metric(label, value, border=True)


def map_figure(ranking, geo, zone, color_by):
    fig = go.Figure()
    points = ranking.merge(geo, on="camera_id")
    without = points.loc[points.sightings.eq(0)]
    if not without.empty:
        fig.add_trace(go.Scattermap(
            lat=without.lat, lon=without.lon, text=without.label,
            marker=dict(size=6, color=MUTED, opacity=0.65),
            hovertemplate="%{text}<br>พบ 0 ครั้ง<extra></extra>",
        ))
    with_sightings = points.loc[points.sightings.gt(0)]
    if not with_sightings.empty:
        fig.add_trace(go.Scattermap(
            lat=with_sightings.lat, lon=with_sightings.lon, text=with_sightings.label,
            customdata=with_sightings[["sightings", "frames", "per_frame"]].to_numpy(),
            marker=dict(
                size=8 + 22 * (with_sightings.sightings / with_sightings.sightings.max()).pow(0.5),
                color=with_sightings[color_by], colorscale=["#ffb38a", "#ff6347", "#ed392f"],
                opacity=0.85,
            ),
            hovertemplate=(
                "%{text}<br>พบ %{customdata[0]:,.0f} ครั้ง / %{customdata[1]:,.0f} ภาพ"
                "<br>%{customdata[2]:.3f} ครั้งต่อภาพ<extra></extra>"
            ),
        ))
    layers = []
    if zone:
        source = dict(sourcetype="geojson", source=zone, below="traces", color=ACCENT)
        layers = [
            dict(source, type="fill", opacity=0.12),
            dict(source, type="line", opacity=0.85, line=dict(width=2, dash=[3, 2])),
        ]
    return chart(fig, 438, showlegend=False, map=dict(
        style="carto-darkmatter",
        center=dict(lat=13.758, lon=100.535),
        zoom=10.85,
        layers=layers,
    ))


def timeline_figure(timeline, selected_slot):
    labels = timeline.index.strftime("%d/%m %H:%M")
    days = timeline.index.strftime("%d/%m")
    indices = list(range(len(timeline)))
    ticks = [i for i, label in enumerate(days) if i == 0 or label != days[i - 1]]
    selected = timeline.index.get_loc(selected_slot) if selected_slot != ALL_SLOTS else None
    fig = go.Figure(go.Bar(
        x=indices, y=timeline.sightings.to_numpy(), text=labels,
        customdata=timeline.frames.to_numpy(),
        marker_color=[ACCENT if selected is None or i == selected else "#a9493c" for i in indices],
        hovertemplate="%{text}<br>พบ %{y} ครั้ง / %{customdata} ภาพ<extra></extra>",
    ))
    if selected is not None:
        fig.add_vline(x=selected, line_color=TEXT, line_width=2)
    fig.update_xaxes(tickvals=ticks, ticktext=days[ticks], showgrid=False, range=[-1, len(indices)])
    fig.update_yaxes(title="ครั้ง / 10 นาที", rangemode="tozero")
    return chart(fig, 180, bargap=0.18)


def map_view(frames, sightings, cameras, geo, zone):
    st.subheader("แผนที่การพบรถบรรทุก")
    scope_col, time_col, color_col = st.columns([1, 2, 1])
    scopes = ["ช่วงห้าม", "ทุกเวลาที่เลือก"] if frames.restricted.any() else ["ทุกเวลาที่เลือก"]
    scope = scope_col.selectbox("ช่วงข้อมูลบนแผนที่", scopes)
    if scope == "ช่วงห้าม":
        frames = frames.loc[frames.restricted]
        sightings = sightings.loc[sightings.in_ban6]

    timeline = timeline_by_slot(frames, sightings)
    selected = time_col.select_slider(
        "เวลาในแผนที่",
        options=[ALL_SLOTS, *timeline.index],
        format_func=lambda value: value if value == ALL_SLOTS else value.strftime("%d/%m · %H:%M น."),
        disabled=timeline.empty,
    )
    color_by = color_col.selectbox(
        "สีของจุด", ["sightings", "per_frame"],
        format_func={"sightings": "จำนวนครั้งที่พบ", "per_frame": "ครั้งที่พบต่อภาพ"}.get,
    )
    if selected != ALL_SLOTS:
        frames = frames.loc[frames.slot.eq(selected)]
        sightings = sightings.loc[sightings.timestamp.dt.floor("10min").eq(selected)]
    ranking = camera_summary(frames, sightings, cameras)

    map_col, summary_col = st.columns([2.35, 1])
    with map_col:
        if geo is not None and not ranking.empty:
            show_chart(map_figure(ranking, geo, zone, color_by), "camera-map", "พิกัดกล้องและขอบเขตห้ามรถบรรทุก")
        else:
            st.info("ไม่มีพิกัดกล้องสำหรับช่วงที่เลือก")
    with summary_col.container(border=True):
        st.metric("พบรถหนักในช่วงที่เลือก", f"{len(sightings):,} ครั้ง")
        st.caption(f"จาก {len(frames):,} ภาพ · {ranking.camera_id.nunique():,} กล้อง")
        st.markdown("**กล้องที่พบมากที่สุด**")
        top = ranking.loc[ranking.sightings.gt(0)].head(5)
        if top.empty:
            st.caption("ไม่พบรถหนักตามตัวกรองนี้")
        else:
            st.markdown("\n\n".join(
                f"**{row.sightings:,} ครั้ง** · {row.label}" for row in top.itertuples(index=False)
            ))

    if geo is not None:
        approximate = zone and any(f.get("properties", {}).get("approximate") for f in zone["features"])
        if zone:
            st.caption(
                "เส้นประแสดงขอบเขตประมาณการที่ใช้คำนวณ · ขนาดจุดแสดงจำนวนครั้งที่พบ"
                if approximate else "เส้นประแสดงขอบเขตห้าม · ขนาดจุดแสดงจำนวนครั้งที่พบ"
            )
        if "osm_name" in geo:
            st.caption("พิกัดกล้องชุดนี้อ้างอิงจุดสังเกตจาก OpenStreetMap")
    if not timeline.empty:
        st.subheader("ไทม์ไลน์ทุก 10 นาที")
        show_chart(timeline_figure(timeline, selected), "timeline", "จำนวนครั้งที่พบรถหนักในแต่ละ 10 นาที")


def analysis_view(hourly, cameras, weather, windows):
    st.subheader("พบรถบรรทุกหนักช่วงไหน")
    profile = hourly.groupby(["weekday", "hour"], as_index=False)[["sightings", "frames"]].sum()
    rain = rain_by_hour(hourly, weather)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not rain.empty:
        fig.add_trace(go.Bar(x=rain.index, y=rain.to_numpy(), name="ฝนเฉลี่ย (มม.)", marker_color="#3f647e",
                             opacity=0.55), secondary_y=True)
        fig.update_yaxes(title_text="ฝนเฉลี่ย (มม.)", showgrid=False, rangemode="tozero", secondary_y=True)
    for weekday, series in profile.groupby("weekday", sort=False):
        fig.add_trace(go.Scatter(
            x=series.hour, y=(series.sightings / series.frames).to_numpy(), mode="lines+markers",
            name="วันทำงาน" if weekday else "วันหยุด", line=dict(color=ACCENT if weekday else BLUE, width=2.5),
            hovertemplate="%{x}:00 น. · %{y:.3f} ครั้งต่อภาพ",
        ), secondary_y=False)
    if hourly.restricted_frames.gt(0).any():
        for start, end in windows:
            fig.add_vrect(x0=minute_of_day(start) / 60 - 0.5, x1=minute_of_day(end) / 60 - 0.5,
                          fillcolor=ACCENT, opacity=0.12, line_width=0, layer="below")
        st.caption("แถบส้มคือช่วงเวลาห้ามสำหรับกล้องในเขตในวันทำงาน")
    fig.update_xaxes(title="ชั่วโมง (เวลาไทย)", dtick=1)
    fig.update_yaxes(title_text="ครั้งที่พบต่อภาพ", secondary_y=False, rangemode="tozero")
    show_chart(chart(fig, 325, legend=dict(orientation="h", y=1.1)), "hourly-chart", "อัตราการพบรถหนักรายชั่วโมง")
    if not rain.empty:
        st.caption("ฝนเฉลี่ยคำนวณจากวันที่เลือกที่มีข้อมูลของชั่วโมงนั้น")
    else:
        st.caption("ยังไม่มีข้อมูลฝนสำหรับช่วงที่เลือก")

    heat_col, ranking_col = st.columns([1.15, 1])
    with heat_col:
        st.subheader("กล้อง × ชั่วโมง")
        camera_hours = hourly.groupby(["camera_id", "hour"])[["sightings", "frames"]].sum()
        busiest = camera_hours.groupby("camera_id").sightings.sum().nlargest(15).index
        rates = (camera_hours.sightings / camera_hours.frames).unstack().reindex(busiest)
        heatmap = go.Figure(go.Heatmap(
            x=rates.columns, y=[str(camera_id) for camera_id in rates.index], z=rates.to_numpy(),
            colorscale=["#1b2633", "#a7493a", ACCENT], colorbar=dict(title="ครั้ง/ภาพ"),
            hovertemplate="กล้อง %{y} · %{x}:00 น.<br>%{z:.3f} ครั้งต่อภาพ<extra></extra>",
        ))
        heatmap.update_xaxes(dtick=1)
        heatmap.update_yaxes(type="category", autorange="reversed")
        show_chart(chart(heatmap, 390), "camera-hour", "อัตราการพบรถตามกล้องและชั่วโมง")
    with ranking_col:
        st.subheader("อันดับกล้องในช่วงห้าม")
        top = hourly.groupby("camera_id", as_index=False)[["restricted_frames", "restricted_sightings"]].sum()
        top = top.loc[top.restricted_frames.gt(0)].merge(cameras, on="camera_id")
        if top.empty:
            st.info("ไม่มีภาพในเขตและช่วงห้ามสำหรับตัวกรองนี้")
        else:
            top["ครั้งต่อภาพ"] = (top.restricted_sightings / top.restricted_frames).round(3)
            top = top.sort_values(["restricted_sightings", "camera_id"], ascending=[False, True])
            st.dataframe(top[["label", "restricted_sightings", "restricted_frames", "ครั้งต่อภาพ"]].rename(columns={
                "label": "กล้อง", "restricted_sightings": "พบ (ครั้ง)", "restricted_frames": "ภาพ",
            }), hide_index=True, height=390)


def local_crop(row):
    filename = str(row.path).replace("\\", "/").rsplit("/", 1)[-1]
    path = ROOT / "data" / "cctv_images" / str(row.camera_id) / filename
    if not path.is_file():
        path = Path(row.path)
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            pad_x, pad_y = 0.3 * (row.x2 - row.x1), 0.3 * (row.y2 - row.y1)
            return image.crop((max(0, row.x1 - pad_x), max(0, row.y1 - pad_y),
                               min(image.width, row.x2 + pad_x), min(image.height, row.y2 + pad_y)))
    except (OSError, ValueError):
        return None


def gallery_view(sightings, evidence, cameras):
    st.subheader("ภาพตัวอย่างจากกล้องในช่วงห้าม")
    candidates = sightings.loc[sightings.in_ban6]
    if candidates.empty:
        st.info("ไม่มีการพบรถหนักในช่วงห้ามตามตัวกรองนี้")
        return

    labels = cameras.set_index("camera_id").label.to_dict()
    camera_col, limit_col = st.columns(2)
    camera = camera_col.selectbox(
        "กล้องในแกลเลอรี", [ALL_CAMERAS, *sorted(candidates.camera_id.unique())],
        format_func=lambda value: labels.get(value, value),
    )
    limit = limit_col.slider("จำนวนภาพ", 3, 12, 6)
    if camera != ALL_CAMERAS:
        candidates = candidates.loc[candidates.camera_id.eq(camera)]
    images = []
    shown = set()
    if evidence is not None:
        keys = ["camera_id", "timestamp", "box_xyxy"]
        boxes = candidates[["camera_id", "timestamp", "p_heavy"]].copy()
        boxes["box_xyxy"] = [
            "{:.0f},{:.0f},{:.0f},{:.0f}".format(*box)
            for box in candidates[["x1", "y1", "x2", "y2"]].itertuples(index=False, name=None)
        ]
        available = evidence.drop(columns="p_heavy").merge(boxes, on=keys)
        available = available.drop_duplicates(["camera_id", "timestamp"])
        for row in available.sort_values("p_heavy", ascending=False).itertuples():
            path = EVIDENCE / row.file
            if path.is_file():
                images.append((path, row.camera_id, row.timestamp, row.p_heavy))
                shown.add((row.camera_id, row.timestamp))
                if len(images) == limit:
                    break

    if len(images) < limit and (ROOT / "data" / "cctv_images").is_dir():
        for row in candidates.sort_values("p_heavy", ascending=False).itertuples():
            key = (row.camera_id, row.timestamp)
            if key not in shown and (image := local_crop(row)) is not None:
                images.append((image, row.camera_id, row.timestamp, row.p_heavy))
                shown.add(key)
                if len(images) == limit:
                    break

    if not images:
        st.info("มีรายการที่โมเดลตรวจพบ แต่ยังไม่มีภาพตัวอย่างพร้อมแสดงสำหรับตัวกรองนี้")
        return
    st.caption("ภาพตัวอย่างจากผลของโมเดล ใช้ประกอบการตรวจสอบการจำแนก")
    for i in range(0, len(images), 3):
        for col, (image, camera_id, timestamp, score) in zip(st.columns(3), images[i:i + 3]):
            col.image(image, width="stretch", caption=f"กล้อง {camera_id} · {timestamp:%d/%m %H:%M} · p={score:.4f}")


def main():
    st.set_page_config(page_title="Bangkok Truck Ban Monitor", layout="wide")
    st.html("""
        <style>
        [data-testid="stAppViewContainer"] .block-container {padding-top: 3.1rem; max-width: 1550px;}
        [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
        </style>
    """)
    hourly, sightings, frames, rules, zone, geo, evidence, weather = load_inputs(RESULTS)
    cameras = hourly.drop_duplicates("camera_id")[["camera_id", "camera_name", "location"]].copy()
    cameras["label"] = cameras.camera_id.astype(str) + " · " + cameras.camera_name.str.replace("-", " ")
    windows = rules["rules"]["truck6"]
    dates, camera_ids, day, area, score = filters(hourly, cameras)
    hourly, sightings, frames = filter_data(
        hourly, sightings, frames, dates, camera_ids, day, area, score, windows
    )

    st.title("รถบรรทุกหนักในกรุงเทพฯ")
    st.caption("กล้องจราจร กทม. · 25–31 ส.ค. 2026 · หนึ่งภาพต่อกล้องประมาณทุก 10 นาที")
    if RESULTS != DATA:
        st.info(
            "พรีวิวข้อมูลเพา · ฝนจาก Open-Meteo และพิกัดจากชุดกล้องของวิชา "
            "· ยอดช่วงห้ามยังคำนวณจากขอบเขตและเกณฑ์ประมาณการเดิม"
        )
    metrics(hourly)
    if hourly.empty:
        st.info("ไม่มีภาพตามตัวกรองที่เลือก ลองเลือกวันที่ กล้อง หรือพื้นที่อื่น")
        return

    map_tab, analysis_tab, gallery_tab = st.tabs(["แผนที่และไทม์ไลน์", "กราฟและอันดับกล้อง", "ภาพจากกล้อง"])
    with map_tab:
        map_view(frames, sightings, cameras, geo, zone)
    with analysis_tab:
        analysis_view(hourly, cameras, weather, windows)
    with gallery_tab:
        gallery_view(sightings, evidence, cameras)
    with st.expander("วิธีนับและข้อจำกัด"):
        window_text = ", ".join(f"{start}–{end}" for start, end in windows)
        st.markdown(f"""
- ช่วงเวลาห้ามที่ใช้วิเคราะห์รถหนักคือ {window_text} น. ในวันทำงาน สำหรับกล้องในเขตห้าม
- ตัวเลขนับครั้งที่พบในภาพจาก YOLO และ CLIP ViT-B/16 รถคันเดิมอาจปรากฏหลายภาพ
- โมเดลมี precision ประมาณ 0.65 และ recall 0.69 จากชุดทดสอบ 90 ตัวอย่าง ภาพมีความละเอียดต่ำจึงแยก 6 ล้อกับ 10 ล้อได้จำกัด
- เช้าวันที่ 27 ส.ค. ข้อมูลขาดช่วง และหลังประมาณ 18:30 น. ตรวจจับยากขึ้นเพราะแสงน้อย
""")


main()
