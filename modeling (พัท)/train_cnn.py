"""Fine-tune a small CNN (EfficientNet-B0, ImageNet weights) to classify candidate crops as light/bus/heavy/other.

Training data: Thai dataset crops (degraded on the fly to CCTV quality every epoch, see thai_crops.py) +
our hand-labelled CCTV crops (modeling (พัท)/labels/labels.csv, up-weighted). Evaluation: 5-fold CV over OUR labels
(Thai data always in train), so numbers are comparable with the CLIP probe (heavy P 0.66 / R 0.67).

Usage: python "modeling (พัท)/train_cnn.py" cv --thai-dir <thai_dataset_dir> [--folds 1] [--no-thai]   # evaluate
       python "modeling (พัท)/train_cnn.py" fit models/truck_cnn.pt --thai-dir <thai_dataset_dir>  # train on all, save
"""
import argparse
import os
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent))
import thai_crops as tc  # noqa: E402

CLASSES = ["light", "bus", "heavy", "other"]
SIZE, EPOCHS, BS, OURS_REPEAT = 128, 8, 64, 6
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def square(img: Image.Image) -> np.ndarray:
    s = max(img.size)
    sq = Image.new("RGB", (s, s), (124, 116, 104))
    sq.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    a = np.asarray(sq.resize((SIZE, SIZE), Image.BICUBIC), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


def crop_box(im, x1, y1, x2, y2, pad=0.15):
    w, h = x2 - x1, y2 - y1
    return im.crop((int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h)),
                    int(min(im.width, x2 + pad * w)), int(min(im.height, y2 + pad * h))))


def load_thai(root=Path(sys.argv[0]).parent):
    if THAI_DIR is None:
        raise SystemExit("pass --thai-dir <unzipped Thailand-vehicles-2024-01-16 YOLO dataset>")
    root = THAI_DIR
    cache = Path("data_out/thai_crops_cache.pkl")                      # decoding 2000 4K-6K JPEGs is the slow
    if cache.exists():                                                  # part; do it once, in parallel
        crops, ys = pickle.loads(cache.read_bytes())
        return [Image.fromarray(c) for c in crops], ys
    imgs = {p.stem: p for p in (root / "train/images").iterdir()}

    def one(f):
        lines = [l for l in f.read_text().split("\n") if l.strip()]
        if not lines:
            return []
        im = Image.open(imgs[f.stem])
        im.draft("RGB", (1920, 1920))                                   # JPEG DCT downscale while decoding
        im = im.convert("RGB")
        im.thumbnail((1920, 1920))
        out = []
        for line in lines:
            c, x, y, w, h = map(float, line.split())
            iw, ih = im.size
            cr = crop_box(im, (x - w / 2) * iw, (y - h / 2) * ih, (x + w / 2) * iw, (y + h / 2) * ih)
            cr.thumbnail((160, 160))                                    # keep RAM small; degraded later anyway
            out.append((cr, CLASSES.index(tc.Y4[tc.NAMES[int(c)]])))
        return out

    with ThreadPoolExecutor(os.cpu_count()) as pool:
        pairs = [p for ps in pool.map(one, sorted((root / "train/labels").glob("*.txt"))) for p in ps]
    crops, ys = [p[0] for p in pairs], np.array([p[1] for p in pairs])
    cache.write_bytes(pickle.dumps(([np.asarray(c) for c in crops], ys)))
    return crops, ys


def load_ours():
    lab = pd.read_csv(Path(__file__).resolve().parent / "labels" / "labels.csv")
    crops = [crop_box(Image.open(r.path).convert("RGB"), r.x1, r.y1, r.x2, r.y2) for r in lab.itertuples()]
    return crops, np.array([CLASSES.index(v) for v in lab.y4])


def make_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    return m.to(DEV, memory_format=torch.channels_last)


def train(thai, ours, epochs=EPOCHS, seed=0):
    """Speed-ups for Apple Silicon: batches are built on a CPU thread pool and prefetched while the GPU trains,
    channels_last memory format, bfloat16 autocast on MPS (no grad scaler needed)."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    m = make_model()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    items = [("t", i) for i in range(len(thai[0]))] + [("o", i) for i in range(len(ours[0]))] * OURS_REPEAT
    ys = np.array([thai[1][i] if s == "t" else ours[1][i] for s, i in items])
    w = torch.tensor(1.0 / np.bincount(ys, minlength=len(CLASSES)).clip(1), dtype=torch.float32, device=DEV)
    loss_fn = nn.CrossEntropyLoss(weight=w / w.sum() * len(CLASSES))
    nb = len(items) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * nb)

    def sample(arg):
        (s, i), sd = arg
        r = random.Random(sd)                                           # per-sample RNG: thread-safe, reproducible
        img = tc.degrade(thai[0][i], r) if s == "t" else ours[0][i]
        if r.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return square(img), (thai[1][i] if s == "t" else ours[1][i])

    def batch(ep, b, order):
        args = [(it, seed * 10**8 + ep * 10**6 + b * BS + j) for j, it in enumerate(order[b * BS:(b + 1) * BS])]
        out = list(pool.map(sample, args))
        x = torch.from_numpy(np.stack([o[0] for o in out])).contiguous(memory_format=torch.channels_last)
        return x, torch.tensor([o[1] for o in out])

    m.train()
    use_amp = DEV == "mps"
    with ThreadPoolExecutor(os.cpu_count()) as pool, ThreadPoolExecutor(1) as pre:
        for ep in range(epochs):
            order = items[:]
            rng.shuffle(order)
            nxt = pre.submit(batch, ep, 0, order)
            for b in range(nb):
                x, y = nxt.result()
                if b + 1 < nb:
                    nxt = pre.submit(batch, ep, b + 1, order)           # prepare next batch during this step
                x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
                with torch.autocast(device_type="mps", dtype=torch.bfloat16, enabled=use_amp):
                    loss = loss_fn(m(x).float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
    return m


@torch.no_grad()
def predict(m, crops):
    m.eval()
    out = []
    for i in range(0, len(crops), 256):
        x = torch.from_numpy(np.stack([square(c) for c in crops[i:i + 256]])).to(DEV)
        out.append(torch.softmax(m(x), 1).cpu().numpy())
    return np.concatenate(out)


THAI_DIR = None


def main():
    global THAI_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "fit"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--no-thai", action="store_true")
    ap.add_argument("--folds", type=int, default=5, help="run only the first N of the 5 folds (1 = single 80/20 holdout)")
    ap.add_argument("--thai-dir")
    a = ap.parse_args()
    THAI_DIR = Path(a.thai_dir) if a.thai_dir else None
    ours = load_ours()
    thai = ([], np.array([], int)) if a.no_thai else load_thai()
    print(f"thai crops {len(thai[0])}, our labels {len(ours[0])}, device {DEV}", flush=True)
    if a.cmd == "fit":
        m = train(thai, ours)
        torch.save(m.state_dict(), a.out)
        print("saved", a.out)
        return
    P = np.full((len(ours[1]), len(CLASSES)), np.nan)
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ours[1], ours[1])):
        if k >= a.folds:
            break
        t0 = time.time()
        m = train(thai, ([ours[0][i] for i in tr], ours[1][tr]), seed=k)
        P[te] = predict(m, [ours[0][i] for i in te])
        print(f"fold {k + 1}/{a.folds} done, training {time.time() - t0:.0f} s", flush=True)
    done = ~np.isnan(P[:, 0])
    y, P = ours[1][done], P[done]
    p = P.argmax(1)
    print(f"evaluated on {done.sum()} held-out labels")
    h = CLASSES.index("heavy")
    for th in (None, 0.5, 0.6, 0.7):
        ph = (p == h) if th is None else (P[:, h] >= th)
        tp = (ph & (y == h)).sum()
        print(f"{'argmax' if th is None else f'p>={th}':8s} heavy P {tp / max(ph.sum(), 1):.2f}  R {tp / (y == h).sum():.2f}"
              f"  (n={ph.sum()})  4-class acc {(p == y).mean():.2f}")
    print(pd.crosstab(pd.Series([CLASSES[i] for i in y], name="label"), pd.Series([CLASSES[i] for i in p], name="pred")))


if __name__ == "__main__":
    main()
