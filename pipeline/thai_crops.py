"""Turn the Roboflow "Thailand-vehicles-2024-01-16" dataset (CC BY 4.0, YOLO format, high-res Bangkok traffic
cameras) into CCTV-like training crops + CLIP embeddings for the truck classifier.

Domain gap: source images are 720p-6K, our CCTV frames are 352x288 (vehicle crops ~20-70 px). Every box is cropped
with the same 15% context as classify_trucks.crop_square, then degraded: resized so its longer side is a random
24-70 px, JPEG-compressed at quality 20-50, and (like our crops) padded to square and upscaled to 224.
Each box gets N_AUG degraded variants.

Class mapping (dataset -> probe class / heavy subtype):
  light_truck (pickup), van_mpv, passenger_car, light_bus (songthaew pickup) -> light
  medium_bus, heavy_bus -> bus;  motorcycle -> other
  medium_truck -> heavy / truck6;  heavy_truck, semi_trailer, full_trailer -> heavy / truck10

Usage (CoreML env): python pipeline/thai_crops.py <dataset_dir> data_out/thai
  -> data_out/thai.parquet (one row per crop: image, group, y4, heavy_type, aug) + data_out/thai_emb.npy
"""
import io
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import classify_trucks as ct  # noqa: E402

NAMES = ['full_trailer', 'heavy_bus', 'heavy_truck', 'light_bus', 'light_truck', 'medium_bus', 'medium_truck',
         'motorcycle', 'passenger_car', 'semi_trailer', 'van_mpv']
Y4 = {"light_truck": "light", "van_mpv": "light", "passenger_car": "light", "light_bus": "light",
      "medium_bus": "bus", "heavy_bus": "bus", "motorcycle": "other",
      "medium_truck": "heavy", "heavy_truck": "heavy", "semi_trailer": "heavy", "full_trailer": "heavy"}
HEAVY_TYPE = {"medium_truck": "truck6", "heavy_truck": "truck10", "semi_trailer": "truck10", "full_trailer": "truck10"}
N_AUG = 2


def degrade(crop: Image.Image, rng: random.Random) -> Image.Image:
    long_side = rng.randint(24, 70)
    k = long_side / max(crop.size)
    small = crop.resize((max(4, round(crop.width * k)), max(4, round(crop.height * k))), Image.BILINEAR)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=rng.randint(20, 50))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def to_tensor(img: Image.Image, size=224) -> np.ndarray:
    s = max(img.size)
    sq = Image.new("RGB", (s, s), tuple(int(v * 255) for v in ct.MEAN))
    sq.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    a = np.asarray(sq.resize((size, size), Image.BICUBIC), np.float32) / 255.0
    return ((a - ct.MEAN) / ct.STD).transpose(2, 0, 1)


def main():
    root, prefix = Path(sys.argv[1]), sys.argv[2]
    imgs = {p.stem: p for p in (root / "train/images").iterdir()}
    rows = []
    for f in sorted((root / "train/labels").glob("*.txt")):
        for line in f.read_text().split("\n"):
            if line.strip():
                c, x, y, w, h = map(float, line.split())
                rows.append((f.stem, NAMES[int(c)], x, y, w, h))
    boxes = pd.DataFrame(rows, columns=["image", "cls", "x", "y", "w", "h"])
    print(len(boxes), "boxes", boxes.cls.value_counts().to_dict(), flush=True)

    def crops_for_image(stem):
        im = Image.open(imgs[stem]).convert("RGB")
        im.thumbnail((1920, 1920))                               # 6K frames -> 1080p-ish first, saves time
        iw, ih = im.size
        rng = random.Random(hash(stem) & 0xFFFF)
        out = []
        for b in boxes[boxes.image == stem].itertuples():
            x1, y1, x2, y2 = (b.x - b.w / 2) * iw, (b.y - b.h / 2) * ih, (b.x + b.w / 2) * iw, (b.y + b.h / 2) * ih
            pw, ph = (x2 - x1) * 0.15, (y2 - y1) * 0.15
            crop = im.crop((int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(iw, x2 + pw)), int(min(ih, y2 + ph))))
            for k in range(N_AUG):
                out.append(((stem, b.cls, k), to_tensor(degrade(crop, rng))))
        return out

    enc = ct.Encoder("coreml")
    meta, embs = [], []
    stems = list(boxes.image.unique())
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(stems), 40):
            batch = [c for cs in pool.map(crops_for_image, stems[i:i + 40]) for c in cs]
            meta += [m for m, _ in batch]
            embs.append(enc(np.stack([t for _, t in batch])))
            print(f"{min(i + 40, len(stems))}/{len(stems)} images", flush=True)
    df = pd.DataFrame(meta, columns=["image", "cls", "aug"])
    df["y4"] = df.cls.map(Y4)
    df["heavy_type"] = df.cls.map(HEAVY_TYPE).fillna("")
    df.to_parquet(f"{prefix}.parquet")
    np.save(f"{prefix}_emb.npy", np.concatenate(embs))
    print(df.y4.value_counts().to_dict())


if __name__ == "__main__":
    main()
