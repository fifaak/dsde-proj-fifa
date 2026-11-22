"""Stage 2: vehicle detection (YOLO11) on all clean CCTV images.

Backends
  coreml  (default, Apple Silicon) - YOLO11 exported to CoreML without NMS, run on the Neural Engine
                                     (~300 img/s raw on M4 Pro). Pre/post-processing done here.
  torch                            - Ultralytics PyTorch model on MPS/CUDA/CPU, fp16 (portable fallback).

Usage
  python "pipeline (เก้อ)/detect.py" <images_clean.csv> <out.parquet> [--backend coreml --model models/yolo11s_nonms.mlpackage]
  python "pipeline (เก้อ)/detect.py" <images_clean.csv> <out.parquet> --backend torch --model models/yolo11s.pt

  CoreML export (needs Python <=3.12, torch 2.7, coremltools 8):
    YOLO("yolo11s.pt").export(format="coreml", imgsz=640, nms=False, half=True)

Output: one row per box: path, camera_id, timestamp, cls, conf, x1, y1, x2, y2 (original pixels), img_w, img_h.
Resumable: results are checkpointed in <out>.partial/chunk_XXXX.parquet + chunk_XXXX.done (list of processed
images, written only after the parquet), so a stopped run never marks images done without saving their boxes.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLS = ["path", "camera_id", "timestamp", "cls", "conf", "x1", "y1", "x2", "y2", "img_w", "img_h"]


# ---------------------------------------------------------------- CoreML backend
def letterbox(path, size=640):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    left, top = round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (left, top))
    return canvas, (w, h, r, left, top)


def nms(boxes, scores, iou=0.5):
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = order[1:][inter / (area[i] + area[order[1:]] - inter + 1e-9) <= iou]
    return np.array(keep, dtype=int)


def postprocess(pred, meta, conf=0.15, iou=0.5, max_det=300):  # Ultralytics predict default IoU=0.5
    """pred: (84, 8400) = xywh (640 px) + 80 class scores. Mirrors Ultralytics class-aware NMS (predict defaults)."""
    w, h, r, left, top = meta
    scores = pred[4:]
    cls, sc = scores.argmax(0), scores.max(0)
    m = (sc > conf) & np.isin(cls, list(COCO_VEHICLES))
    if not m.any():
        return []
    xywh, cls, sc = pred[:4, m].T, cls[m], sc[m]
    b = np.c_[xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2]
    keep = nms(b + cls[:, None] * 4096.0, sc, iou)[:max_det]          # offset trick = per-class NMS
    b = (b[keep] - [left, top, left, top]) / r
    b = np.clip(b, 0, [w, h, w, h])
    return [(COCO_VEHICLES[int(c)], float(s), *map(float, bb), w, h) for c, s, bb in zip(cls[keep], sc[keep], b)]


class CoreMLDetector:
    def __init__(self, model_path, conf, imgsz=640, threads=8):
        import coremltools as ct
        self.m = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.inp = self.m.get_spec().description.input[0].name
        self.conf, self.imgsz = conf, imgsz
        self.pool = ThreadPoolExecutor(threads)

    def load(self, paths):
        return self.pool.submit(lambda: list(self.pool.map(lambda p: letterbox(p, self.imgsz), paths)))

    def run(self, loaded):
        out = self.m.predict([{self.inp: im} for im, _ in loaded])
        preds = [next(iter(o.values()))[0].astype(np.float32) for o in out]
        return list(self.pool.map(lambda pm: postprocess(pm[0], pm[1], self.conf), zip(preds, [m for _, m in loaded])))


# ---------------------------------------------------------------- PyTorch backend
class TorchDetector:
    def __init__(self, model_path, conf, imgsz=640):
        import torch
        from ultralytics import YOLO
        self.dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.m, self.conf, self.imgsz = YOLO(model_path), conf, imgsz
        self.pool = ThreadPoolExecutor(1)

    def load(self, paths):
        return self.pool.submit(lambda: paths)

    def run(self, paths):
        res = self.m.predict(paths, imgsz=self.imgsz, conf=self.conf, classes=list(COCO_VEHICLES), device=self.dev,
                             half=self.dev != "cpu", verbose=False)
        out = []
        for det in res:
            h, w = det.orig_shape
            out.append([(COCO_VEHICLES[int(c)], s, *bb, w, h) for bb, c, s in
                        zip(det.boxes.xyxy.tolist(), det.boxes.cls.tolist(), det.boxes.conf.tolist())])
        return out


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images_csv")
    ap.add_argument("out")
    ap.add_argument("--backend", choices=["coreml", "torch"], default="coreml")
    ap.add_argument("--model", default="models/yolo11s_nonms.mlpackage")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=4000, help="images per checkpoint")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    imgs = pd.read_csv(a.images_csv, parse_dates=["timestamp"])
    if a.limit:
        imgs = imgs.sample(a.limit, random_state=0)
    out = Path(a.out)
    ckpt = out.with_suffix(".partial")
    ckpt.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in ckpt.glob("chunk_*.done"):
        done |= set(f.read_text().splitlines())
    todo = imgs[~imgs.path.isin(done)].reset_index(drop=True)
    print(f"{len(imgs)} images, {len(todo)} to process, backend={a.backend} model={a.model}", flush=True)

    det = (CoreMLDetector if a.backend == "coreml" else TorchDetector)(a.model, a.conf, a.imgsz)
    chunk_id = len(list(ckpt.glob("chunk_*.done")))
    t0 = time.time()
    batches = [todo.iloc[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    rows, chunk_paths = [], []
    nxt = det.load(list(batches[0].path)) if batches else None
    for bi, part in enumerate(batches):
        loaded = nxt.result()
        if bi + 1 < len(batches):
            nxt = det.load(list(batches[bi + 1].path))                # prefetch next batch while this one runs
        for r, boxes in zip(part.itertuples(), det.run(loaded)):
            rows += [(r.path, r.camera_id, r.timestamp, *b) for b in boxes]
        chunk_paths += list(part.path)
        if len(chunk_paths) >= a.chunk or bi == len(batches) - 1:
            pd.DataFrame(rows, columns=COLS).to_parquet(ckpt / f"chunk_{chunk_id:04d}.parquet")
            (ckpt / f"chunk_{chunk_id:04d}.done").write_text("\n".join(chunk_paths) + "\n")
            n = (bi + 1) * a.batch
            el = time.time() - t0
            print(f"{min(n, len(todo))}/{len(todo)}  {n / el:.0f} img/s  eta {(len(todo) - n) / (n / el) / 60:.1f} min",
                  flush=True)
            rows, chunk_paths, chunk_id = [], [], chunk_id + 1

    parts = [pd.read_parquet(f) for f in sorted(ckpt.glob("chunk_*.parquet"))]
    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    res.to_parquet(out)
    print(f"wrote {out}: {len(res)} boxes from {res.path.nunique()} images in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
