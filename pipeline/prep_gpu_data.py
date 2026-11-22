"""Pre-render every training tensor for remote GPU training (the GPU box has only 4 CPU cores).

Thai crops get one freshly degraded variant per epoch (CCTV-like: 24-70 px, JPEG q20-50), padded to square and
resized to 128 px; our 450 labelled CCTV crops are rendered once. Everything is uint8 so the whole set fits in
GPU memory and training needs no CPU data loading.

Usage: python pipeline/prep_gpu_data.py <thai_dataset_dir> data_out/gpu [--epochs 8]
Output: thai_x.npy (E, N, 128, 128, 3) uint8, thai_y.npy, ours_x.npy (450, 128, 128, 3) uint8, ours_y.npy
"""
import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import train_cnn as tcnn  # noqa: E402

SIZE = 128


def square_u8(img: Image.Image) -> np.ndarray:
    s = max(img.size)
    sq = Image.new("RGB", (s, s), (124, 116, 104))
    sq.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    return np.asarray(sq.resize((SIZE, SIZE), Image.BICUBIC), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("thai_dir")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    tcnn.THAI_DIR = Path(a.thai_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    thai_crops, thai_y = tcnn.load_thai()
    ours_crops, ours_y = tcnn.load_ours()
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        X = np.stack([np.stack(list(pool.map(
            lambda i: square_u8(tcnn.tc.degrade(thai_crops[i], random.Random(e * 10**6 + i))), range(len(thai_crops)))))
            for e in range(a.epochs)])
        O = np.stack(list(pool.map(square_u8, ours_crops)))
    np.save(out / "thai_x.npy", X)
    np.save(out / "thai_y.npy", thai_y)
    np.save(out / "ours_x.npy", O)
    np.save(out / "ours_y.npy", ours_y)
    print("thai", X.shape, "ours", O.shape, f"{(X.nbytes + O.nbytes) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
