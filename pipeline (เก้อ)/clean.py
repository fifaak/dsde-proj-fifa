import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTAGES = [("2026-08-27 06:00", "2026-08-27 09:00")]
PERIOD_BINS = [0, 9, 16, 19, 24]
PERIOD_LABELS = ["morning_peak", "midday", "evening_peak", "night"]
KEEP = ["path", "camera_id", "camera_name", "location", "timestamp", "date", "hour", "period", "width", "height",
        "file_size", "brightness", "contrast", "saturation", "sharpness", "very_dark", "blurry", "stale_repeat",
        "system_outage", "invalid"]


def main():
    feat_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data_out/features.csv")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_out")
    raw = pd.read_csv(feat_csv, parse_dates=["timestamp"], dtype={"ahash": str, "md5": str})
    ok = raw.parse_ok.fillna(False).astype(bool) & raw.read_ok.fillna(False).astype(bool)
    df = (raw[ok].astype({"camera_id": int, "width": int, "height": int, "file_size": int})
          .sort_values(["camera_id", "timestamp"]).reset_index(drop=True))
    df["date"] = df.timestamp.dt.date
    df["hour"] = df.timestamp.dt.hour

    cams = df.drop_duplicates("camera_id")[["camera_id", "camera_name", "location"]].copy()
    cams["is_intersection"] = cams.camera_name.str.contains("แยก")
    cams["direction"] = np.select([cams.location.str.contains("รถมาจาก"), cams.location.str.contains("มุ่งหน้า")],
                                  ["from (inbound)", "towards (outbound)"], "other")
    cams.to_csv(out / "cameras.csv", index=False)

    placeholder = ((df.width == 400) & (df.height == 266)) | (df.bright_frac > 0.9)
    test_pattern = (df.width == 246) & (df.height == 205)
    df["very_dark"] = df.dark_frac > 0.6
    df["blurry"] = (df.sharpness < 100) & (df.bright_frac <= 0.9)
    df["stale_repeat"] = df.ahash.eq(df.groupby("camera_id").ahash.shift())
    df["system_outage"] = False
    for start, end in OUTAGES:
        df["system_outage"] |= (df.timestamp >= start) & (df.timestamp < end)
    df["invalid"] = placeholder | test_pattern
    df["period"] = pd.cut(df.hour, PERIOD_BINS, right=False, labels=PERIOD_LABELS)

    clean = df.loc[~df.invalid, KEEP]
    clean.to_csv(out / "images_clean.csv", index=False)
    print(f"images_clean.csv: {len(clean):,} of {len(raw):,} images, {clean.camera_id.nunique()} cameras | dropped "
          f"placeholder {int(placeholder.sum())}, test pattern {int(test_pattern.sum())}, unreadable {int((~ok).sum())}"
          f" | flagged very_dark {int(clean.very_dark.sum())}, blurry {int(clean.blurry.sum())}, "
          f"stale_repeat {int(clean.stale_repeat.sum())}, system_outage {int(clean.system_outage.sum())}")
    print(f"cameras.csv: {len(cams)} cameras")


if __name__ == "__main__":
    main()
