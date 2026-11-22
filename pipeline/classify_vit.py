"""Stage 3 (current): classify candidate boxes with the fine-tuned CLIP ViT-B/16 (DataComp-XL) crop classifier.

Model: vit_base_patch16_clip_224.datacompxl fine-tuned end-to-end on the RTX 3090 (pipeline/train_gpu.py) with
Thai-vehicles crops degraded to CCTV quality + our 450 hand labels. Holdout (90 of our labels, same split for all
models): heavy precision 0.65 / recall 0.69 at p_heavy >= 0.6, vs 0.64 / 0.55 for the old CLIP linear probe.

Preprocessing must match training: crop box + 15% context, pad to square (124,116,104), bicubic to 128 px, then
bilinear to 224 px and CLIP mean/std normalisation.
Decision: vclass = heavy if p_heavy >= HEAVY_TH, else the most likely of the other classes.
heavy_type (truck6 / truck10) still comes from the zero-shot CLIP scores in candidates.parquet (unvalidated).

Usage: python pipeline/classify_vit.py data_out/candidates.parquet models/truck_clipvitb.pt data_out/vehicles.parquet
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from prep_gpu_data import square_u8  # noqa: E402
from train_cnn import crop_box  # noqa: E402

ARCH, SIZE, HEAVY_TH = "vit_base_patch16_clip_224.datacompxl", 224, 0.6
CLASSES = ["light", "bus", "heavy", "other"]
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cand_path, weights, out = sys.argv[1:4]
    cand = pd.read_parquet(cand_path)
    m = timm.create_model(ARCH, pretrained=False, num_classes=len(CLASSES))
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    m = m.to(DEV).eval()
    mean = torch.tensor(m.pretrained_cfg["mean"], device=DEV).view(1, 3, 1, 1) * 255
    std = torch.tensor(m.pretrained_cfg["std"], device=DEV).view(1, 3, 1, 1) * 255

    order = cand.sort_values("path").index.to_numpy()
    rows = cand.loc[order]

    def render(chunk):                                       # decode each frame once, crop all its boxes
        ims = {p: Image.open(p).convert("RGB") for p in chunk.path.unique()}
        return np.stack([square_u8(crop_box(ims[r.path], r.x1, r.y1, r.x2, r.y2)) for r in chunk.itertuples()])

    BS = 256
    chunks = [rows.iloc[i:i + BS] for i in range(0, len(rows), BS)]
    probs, t0 = [], time.time()
    with ThreadPoolExecutor(os.cpu_count()) as pool, torch.no_grad():
        futs = [pool.submit(render, c) for c in chunks[:8]]
        for i in range(len(chunks)):
            x = torch.from_numpy(futs[i].result()).to(DEV).permute(0, 3, 1, 2).float()
            if i + 8 < len(chunks):
                futs.append(pool.submit(render, chunks[i + 8]))
            futs[i] = None
            x = (F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False) - mean) / std
            with torch.autocast(device_type=DEV if DEV != "cpu" else "cpu", dtype=torch.float16, enabled=DEV != "cpu"):
                probs.append(torch.softmax(m(x).float(), 1).cpu().numpy())
            if i % 50 == 0:
                done = (i + 1) * BS
                print(f"{done}/{len(rows)}  {done / (time.time() - t0):.0f} crops/s", flush=True)
    P = np.empty((len(cand), len(CLASSES)), np.float32)
    P[order] = np.concatenate(probs)
    for j, c in enumerate(CLASSES):
        cand[f"p_{c}"] = P[:, j]
    other = P.copy()
    other[:, CLASSES.index("heavy")] = -1
    cand["vclass"] = np.where(P[:, CLASSES.index("heavy")] >= HEAVY_TH, "heavy", np.array(CLASSES)[other.argmax(1)])
    cand["heavy_type"] = np.where(cand.vclass != "heavy", "",
                                  np.where(cand.zs_truck10 > cand.zs_truck6, "truck10", "truck6"))
    cand["classifier"] = f"{ARCH} finetuned, heavy if p>={HEAVY_TH}"
    cand.drop(columns=[c for c in cand.columns if c.startswith("zs_") and c != "zs_class"]).to_parquet(out)
    print(f"done in {(time.time() - t0) / 60:.1f} min:", cand.vclass.value_counts().to_dict())


if __name__ == "__main__":
    main()
