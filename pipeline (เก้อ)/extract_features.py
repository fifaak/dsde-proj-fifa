"""Build a per-image metadata + quality feature table from the raw CCTV image folder.

Usage: python "pipeline (เก้อ)/extract_features.py" <cctv_images_dir> <out_csv>

Filename pattern: <camera_id>_<camera_name>_<location>_<YYYYMMDD>_<HHMMSS>.jpg
Features are computed on a downscaled grayscale/HSV copy for speed.
"""
import hashlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

NAME_RE = re.compile(r"^(?P<cam>\d+)_(?P<name>.*)_(?P<date>\d{8})_(?P<time>\d{6})\.jpe?g$", re.I)


def parse_name(fname: str) -> dict:
    m = NAME_RE.match(fname)
    if not m:
        return {"parse_ok": False}
    name = m["name"]
    # name = "<code-or-name>_<location>"; split on the last underscore
    cam_name, _, location = name.rpartition("_")
    return {
        "parse_ok": True,
        "camera_id": int(m["cam"]),
        "camera_name": cam_name or name,
        "location": location,
        "timestamp": pd.to_datetime(m["date"] + m["time"], format="%Y%m%d%H%M%S"),
    }


def image_features(path: Path) -> dict:
    out = {"file_size": path.stat().st_size}
    try:
        raw = path.read_bytes()
        out["md5"] = hashlib.md5(raw).hexdigest()
        with Image.open(path) as im:
            out["width"], out["height"] = im.size
            out["mode"] = im.mode
            im.draft("RGB", (320, 240))  # fast JPEG downscale
            rgb = im.convert("RGB").resize((160, 120))
        a = np.asarray(rgb, dtype=np.float32)
        gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
        lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
               + gray[1:-1, :-2] + gray[1:-1, 2:])
        out.update(
            brightness=float(gray.mean()),
            contrast=float(gray.std()),
            saturation=float(hsv[..., 1].mean()),
            sharpness=float(lap.var()),
            dark_frac=float((gray < 30).mean()),
            bright_frac=float((gray > 240).mean()),
            r_mean=float(a[..., 0].mean()),
            g_mean=float(a[..., 1].mean()),
            b_mean=float(a[..., 2].mean()),
            ahash=_ahash(gray),
            read_ok=True,
        )
    except Exception as e:  # corrupted / truncated file
        out.update(read_ok=False, error=str(e)[:200])
    return out


def _ahash(gray: np.ndarray) -> str:
    small = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize((8, 8)), dtype=np.float32)
    bits = (small > small.mean()).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def process(path_str: str) -> dict:
    p = Path(path_str)
    row = {"path": str(p), "folder": p.parent.name, "filename": p.name}
    row.update(parse_name(p.name))
    row.update(image_features(p))
    return row


def main():
    root, out_csv = Path(sys.argv[1]), sys.argv[2]
    files = sorted(str(p) for p in root.rglob("*") if p.is_file())
    print(f"{len(files)} files")
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(process, files, chunksize=256))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {df.shape}")


if __name__ == "__main__":
    main()
