"""Stage 3: classify candidate vehicle boxes into size classes with CLIP (zero-shot, optional linear probe).

YOLO's COCO `truck` lumps pickups/vans/SUVs together with real lorries, and the small detector sometimes calls a
big truck `car`. So every candidate box is cropped, embedded with CLIP ViT-B-16 and scored against text prompts.

Classes
    light   - pickup / van / SUV / car / tuk-tuk / songthaew  (not covered by the truck ban)
    bus     - city or tour bus                                (not covered)
    truck6  - medium truck, 6 wheels (2 axles, dual rear)     (ban: 06-09, 16-20 weekdays)
    truck10 - heavy truck, >=10 wheels / trailer / mixer...   (ban: 06-10, 15-21 weekdays)
    other   - false positive (structure, background)

Candidates: YOLO truck/bus with box height >= 20 px, plus YOLO car with height >= 40 px and conf < 0.6.

Backends: coreml (image encoder on the Apple Neural Engine, ~260 crops/s on M4 Pro) or torch (MPS/CUDA/CPU).
Text embeddings are always computed with open_clip in PyTorch (tiny).

Usage
    python "pipeline (เก้อ)/classify_trucks.py" embed <detections.parquet> <out_prefix> [--backend coreml]
        -> <out_prefix>.parquet (candidate rows + zero-shot probs) and <out_prefix>_emb.npy (L2-normed embeddings)
    python "pipeline (เก้อ)/classify_trucks.py" train <out_prefix> modeling (พัท)/labels/labels.csv models/truck_probe.joblib
    python "pipeline (เก้อ)/classify_trucks.py" predict <out_prefix> models/truck_probe.joblib data_out/vehicles.parquet

Zero-shot alone was poor on these crops (heavy precision 0.39 on a 250-crop hand-labelled sample), so the final
class comes from a linear probe trained on hand labels (see labels/README.md for CV numbers).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

MODEL, PRETRAINED = "ViT-B-16", "datacomp_xl_s13b_b90k"
COREML_PATH = "models/clip_ViT-B-16.mlpackage"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

PROMPTS = {
    "light": ["a pickup truck", "a pickup truck with a canopy", "a van", "a white passenger van",
              "an SUV", "a car", "a sedan", "a minivan", "a tuk-tuk", "a songthaew pickup taxi", "a taxi"],
    "bus": ["a city bus", "a tour bus", "a public bus", "a double-decker bus", "the roof of a bus seen from above",
            "a green and yellow public bus", "an orange public bus", "a blue public bus"],
    "truck6": ["a six-wheel box truck", "a medium delivery truck", "a small lorry", "a light-duty cargo truck",
               "a garbage truck"],
    "truck10": ["a ten-wheel truck", "a heavy lorry", "a semi-trailer truck", "a dump truck",
                "a concrete mixer truck", "a tanker truck", "a container truck", "a large cargo truck"],
    "other": ["a bridge", "a staircase", "a building", "an empty road", "a traffic sign", "a tree", "a wall",
              "a blurry dark image", "a motorcycle", "a person"],
}
CLASSES = list(PROMPTS)
TEMPLATES = ["a CCTV photo of {}.", "a low-resolution traffic camera image of {}.", "a photo of {}."]


def select_candidates(det: pd.DataFrame) -> pd.DataFrame:
    h = det.y2 - det.y1
    m = (det.cls.isin(["truck", "bus"]) & (h >= 20)) | ((det.cls == "car") & (h >= 40) & (det.conf < 0.6))
    return det[m].sort_values(["path", "y1"]).reset_index(drop=True)


def crop_square(im: Image.Image, r, pad=0.15, size=224) -> np.ndarray:
    """Crop box (+15% context), pad to square with the CLIP mean colour (keeps long trucks whole), normalise."""
    w, h = r.x2 - r.x1, r.y2 - r.y1
    c = im.crop(tuple(int(v) for v in (max(0, r.x1 - pad * w), max(0, r.y1 - pad * h),
                                         min(r.img_w, r.x2 + pad * w), min(r.img_h, r.y2 + pad * h))))
    s = max(c.size)
    sq = Image.new("RGB", (s, s), tuple(int(v * 255) for v in MEAN))
    sq.paste(c, ((s - c.width) // 2, (s - c.height) // 2))
    a = np.asarray(sq.resize((size, size), Image.BICUBIC), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


def text_embeddings():
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tok = open_clip.get_tokenizer(MODEL)
    names, feats = [], []
    with torch.no_grad():
        for cls, items in PROMPTS.items():
            f = model.encode_text(tok([t.format(p) for p in items for t in TEMPLATES]))
            f = (f / f.norm(dim=-1, keepdim=True)).view(len(items), len(TEMPLATES), -1).mean(1)
            feats.append((f / f.norm(dim=-1, keepdim=True)).numpy())
            names += [cls] * len(items)
    return np.array(names), np.concatenate(feats)


def zero_shot(emb: np.ndarray, names: np.ndarray, T: np.ndarray) -> np.ndarray:
    logits = 100 * emb.astype(np.float32) @ T.T
    cl = np.stack([logits[:, names == c].max(1) for c in CLASSES], 1)     # best prompt per class
    e = np.exp(cl - cl.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class Encoder:
    def __init__(self, backend):
        self.backend = backend
        if backend == "coreml":
            import coremltools as ct
            self.m = ct.models.MLModel(COREML_PATH, compute_units=ct.ComputeUnit.CPU_AND_NE)
        else:
            import open_clip
            import torch
            self.torch = torch
            self.dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            self.m = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=self.dev)[0].eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "coreml":
            out = self.m.predict([{"x": xi[None]} for xi in x])
            return np.concatenate([o["emb"] for o in out]).astype(np.float16)
        with self.torch.no_grad():
            f = self.m.encode_image(self.torch.from_numpy(x).to(self.dev))
            return (f / f.norm(dim=-1, keepdim=True)).half().cpu().numpy()


def embed(det_path, out_prefix, backend="coreml", batch=256, limit=None):
    cand = select_candidates(pd.read_parquet(det_path))
    if limit:
        cand = cand.sample(limit, random_state=0).sort_values("path").reset_index(drop=True)
    print(f"{len(cand)} candidate boxes: {cand.cls.value_counts().to_dict()}", flush=True)
    enc, pool = Encoder(backend), ThreadPoolExecutor(8)

    def load(rows):  # decode each image once per batch, crop all its boxes
        by_path = {p: Image.open(p).convert("RGB") for p in rows.path.unique()}
        return np.stack(list(pool.map(lambda r: crop_square(by_path[r.path], r), rows.itertuples())))

    batches = [cand.iloc[i:i + batch] for i in range(0, len(cand), batch)]
    embs, nxt = [], pool.submit(load, batches[0])
    for i in range(len(batches)):
        x = nxt.result()
        if i + 1 < len(batches):
            nxt = pool.submit(load, batches[i + 1])
        embs.append(enc(x))
        if i % 50 == 0:
            print(f"{(i + 1) * batch}/{len(cand)}", flush=True)
    emb = np.concatenate(embs)
    names, T = text_embeddings()
    p = zero_shot(emb, names, T)
    for j, c in enumerate(CLASSES):
        cand[f"zs_{c}"] = p[:, j]
    cand["zs_class"] = np.array(CLASSES)[p.argmax(1)]
    cand.to_parquet(f"{out_prefix}.parquet")
    np.save(f"{out_prefix}_emb.npy", emb)
    print(cand.zs_class.value_counts().to_dict())


PROBE_CLASSES = ["light", "bus", "heavy", "other"]   # truck6 + truck10 merged: not separable by eye at 352x288


def probe_features(cand: pd.DataFrame, emb: np.ndarray) -> np.ndarray:
    zs = np.log(cand[[f"zs_{c}" for c in CLASSES]].values + 1e-6)
    return np.c_[emb.astype(np.float32), zs]


def train_probe(prefix, labels_csv, model_out, C=20.0):
    """Linear probe (logistic regression) on CLIP embedding + zero-shot log-probs; reports 5-fold CV first."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cand, emb = pd.read_parquet(f"{prefix}.parquet"), np.load(f"{prefix}_emb.npy")
    lab = pd.read_csv(labels_csv)                       # columns: rid (row in candidates), y4 (light/bus/heavy/other)
    X, y = probe_features(cand, emb)[lab.rid.to_numpy()], np.asarray(lab.y4, dtype=object)
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    report = pd.crosstab(pd.Series(y, name="label"), pd.Series(p, name="pred"))
    tp = ((y == "heavy") & (p == "heavy")).sum()
    print(report, f"\nCV accuracy {(p == y).mean():.2f}  heavy precision {tp / (p == 'heavy').sum():.2f}"
                  f"  recall {tp / (y == 'heavy').sum():.2f}  (n={len(y)})")
    joblib.dump(clf.fit(X, y), model_out)


def predict_probe(prefix, model_path, out_path):
    import joblib
    cand, emb = pd.read_parquet(f"{prefix}.parquet"), np.load(f"{prefix}_emb.npy")
    clf = joblib.load(model_path)
    P = clf.predict_proba(probe_features(cand, emb))
    for j, c in enumerate(clf.classes_):
        cand[f"p_{c}"] = P[:, j]
    cand["vclass"] = clf.classes_[P.argmax(1)]
    # heavy sub-type from zero-shot only (NOT validated): used for the 10-wheel extra ban windows
    cand["heavy_type"] = np.where(cand.vclass != "heavy", "",
                                  np.where(cand.zs_truck10 > cand.zs_truck6, "truck10", "truck6"))
    cand.drop(columns=[c for c in cand.columns if c.startswith("zs_") and c != "zs_class"]).to_parquet(out_path)
    print(cand.vclass.value_counts().to_dict(), cand[cand.vclass == "heavy"].heavy_type.value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("detections")
    e.add_argument("out_prefix")
    e.add_argument("--backend", choices=["coreml", "torch"], default="coreml")
    e.add_argument("--limit", type=int)
    t = sub.add_parser("train")
    t.add_argument("prefix")
    t.add_argument("labels_csv")
    t.add_argument("model_out")
    p = sub.add_parser("predict")
    p.add_argument("prefix")
    p.add_argument("model")
    p.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "embed":
        embed(a.detections, a.out_prefix, a.backend, limit=a.limit)
    elif a.cmd == "train":
        train_probe(a.prefix, a.labels_csv, a.model_out)
    else:
        predict_probe(a.prefix, a.model, a.out)


if __name__ == "__main__":
    main()
