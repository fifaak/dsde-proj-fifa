"""Evidence pack: CCTV frames showing heavy trucks inside the ban zone during weekday 6-wheel ban windows.

Source: data_out/heavy_sightings.parquet, rows with in_ban6 == True (inside zone + weekday + 06-09 / 16-20).
Only confident, clearly visible sightings are used (p_heavy >= 0.95, box height >= 30 px), up to PER_CAM per camera
for the top cameras, spread over different days/windows. Each frame is upscaled 2x with the truck boxed in red and a
caption (camera, place, time, ban window, model confidence).

Needs the raw CCTV images locally (paths in the parquet). Run from the repo root:
    python "insights (ม่อน)/make_evidence.py"
Output: insights (ม่อน)/evidence/*.jpg, index.csv, contact_sheet.jpg

These are model detections (precision ~0.65 overall, higher at p >= 0.95), not verified violations: a human should
check each image before using it as an example.
"""
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "evidence"
TOP_CAMS, PER_CAM, SCALE = 10, 5, 2
THAI_DAYS = ["จ.", "อ.", "พ.", "พฤ.", "ศ.", "ส.", "อา."]
FONT = "/System/Library/Fonts/SukhumvitSet.ttc"   # any Thai TrueType font works


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def main():
    h = pd.read_parquet("data_out/heavy_sightings.parquet")
    v = h[h.in_ban6 & (h.p_heavy >= 0.95) & ((h.y2 - h.y1) >= 30)].copy()
    v["ts"] = pd.to_datetime(v.timestamp)
    v["window"] = v.ts.dt.strftime("%Y-%m-%d") + v.ts.dt.hour.map(lambda x: " เช้า" if x < 12 else " เย็น")
    top = h[h.in_ban6].camera_id.value_counts().head(TOP_CAMS).index
    picks = []
    for cam in top:                                   # one per ban window first, so examples span days
        c = v[v.camera_id == cam].sort_values("p_heavy", ascending=False)
        c = c.drop_duplicates("path").drop_duplicates("window")
        picks.append(c.head(PER_CAM))
    sel = pd.concat(picks).sort_values(["camera_id", "ts"]).reset_index(drop=True)

    OUT.mkdir(exist_ok=True)
    f_big, f_small = font(20), font(16)
    rows, thumbs = [], []
    for r in sel.itertuples():
        im = Image.open(r.path).convert("RGB")
        im = im.resize((im.width * SCALE, im.height * SCALE), Image.BICUBIC)
        canvas = Image.new("RGB", (im.width, im.height + 62), (14, 17, 23))
        canvas.paste(im, (0, 62))
        d = ImageDraw.Draw(canvas)
        # every heavy truck in the same frame during the ban window gets a box
        for b in h[(h.path == r.path) & h.in_ban6].itertuples():
            d.rectangle([b.x1 * SCALE, 62 + b.y1 * SCALE, b.x2 * SCALE, 62 + b.y2 * SCALE], outline=(255, 59, 48),
                        width=3)
        win = "06:00–09:00" if r.ts.hour < 12 else "16:00–20:00"
        d.text((8, 4), f"กล้อง {r.camera_id} · {r.camera_name.replace('-', ' ')[:34]}", fill=(230, 230, 230), font=f_big)
        d.text((8, 34), f"{THAI_DAYS[r.ts.dayofweek]} {r.ts.day} ส.ค. 2026  {r.ts:%H:%M} น. · ช่วงห้าม {win} · "
                        f"p={r.p_heavy:.2f}", fill=(255, 59, 48), font=f_small)
        name = f"cam{r.camera_id}_{r.ts:%Y%m%d_%H%M}.jpg"
        canvas.save(OUT / name, quality=88)
        thumbs.append(canvas.resize((canvas.width // 2, canvas.height // 2)))
        rows.append({"file": name, "camera_id": r.camera_id, "camera_name": r.camera_name, "location": r.location,
                     "timestamp": r.timestamp, "ban_window": win, "p_heavy": round(r.p_heavy, 3),
                     "box_xyxy": f"{r.x1:.0f},{r.y1:.0f},{r.x2:.0f},{r.y2:.0f}", "source_image": Path(r.path).name})
    pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)

    cols = 5
    tw, th = thumbs[0].size
    sheet = Image.new("RGB", (cols * tw, -(-len(thumbs) // cols) * th), (14, 17, 23))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(OUT / "contact_sheet.jpg", quality=85)
    print(f"{len(rows)} evidence frames from {sel.camera_id.nunique()} cameras -> {OUT}")


if __name__ == "__main__":
    main()
