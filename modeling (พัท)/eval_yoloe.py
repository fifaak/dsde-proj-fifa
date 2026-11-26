"""Zero-shot test: YOLOE (open-vocabulary detector) with text prompts vs our hand labels (modeling (พัท)/labels/labels.csv).

For every labelled box, YOLOE runs on the full frame; the YOLOE box with the highest IoU (>= 0.4) to the labelled
box gives the predicted prompt, mapped to light / bus / heavy. No matching box -> "other".
Compared with the current CLIP linear probe (5-fold CV) on the same 450 labels.

Usage: python "modeling (พัท)/eval_yoloe.py" [--model models/yoloe-11l-seg.pt] [--conf 0.05]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLOE

PROMPTS = {
    "heavy": ["box truck", "delivery truck", "cargo truck", "dump truck", "cement mixer truck", "tanker truck",
              "semi-trailer truck", "garbage truck", "flatbed truck", "lorry", "container truck"],
    "light": ["pickup truck", "van", "minivan", "passenger van", "car", "SUV", "taxi", "tuk-tuk"],
    "bus": ["bus", "city bus"],
}


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yoloe-11l-seg.pt")
    ap.add_argument("--conf", type=float, default=0.05)
    a = ap.parse_args()
    lab = pd.read_csv(Path(__file__).resolve().parent / "labels" / "labels.csv")
    names = [p for g in PROMPTS.values() for p in g]
    group = {p: g for g, ps in PROMPTS.items() for p in ps}
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = YOLOE(a.model)
    m.set_classes(names)

    frames = lab.path.unique()
    dets, t0 = {}, time.time()
    for i in range(0, len(frames), 16):
        for p, r in zip(frames[i:i + 16], m.predict(list(frames[i:i + 16]), imgsz=640, conf=a.conf, device=dev,
                                                    verbose=False)):
            dets[p] = (r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy())
    speed = len(frames) / (time.time() - t0)

    pred, conf = [], []
    for r in lab.itertuples():
        xyxy, cls, cf = dets[r.path]
        if len(xyxy) == 0:
            pred.append("other"); conf.append(0.0); continue
        ov = iou(np.array([r.x1, r.y1, r.x2, r.y2]), xyxy)
        cand = np.where(ov >= 0.4)[0]
        if len(cand) == 0:
            pred.append("other"); conf.append(0.0); continue
        j = cand[np.argmax(cf[cand])]
        pred.append(group[names[cls[j]]]); conf.append(float(cf[j]))
    lab["yoloe"], lab["yoloe_conf"] = pred, conf
    y, p = lab.y4.values, lab.yoloe.values
    print(f"{a.model}: {speed:.1f} frames/s on {dev}")
    print(pd.crosstab(pd.Series(y, name="label"), pd.Series(p, name="yoloe")))
    for th in (0.05, 0.1, 0.2, 0.3):
        ph = (p == "heavy") & (lab.yoloe_conf.values >= th)
        tp = (ph & (y == "heavy")).sum()
        print(f"conf>={th}: heavy precision {tp / max(ph.sum(), 1):.2f}  recall {tp / (y == 'heavy').sum():.2f}  "
              f"(predicted heavy {ph.sum()})")
    print(f"4-class accuracy {(p == y).mean():.2f}")
    lab.to_csv("data_out/yoloe_eval.csv", index=False)


if __name__ == "__main__":
    main()
