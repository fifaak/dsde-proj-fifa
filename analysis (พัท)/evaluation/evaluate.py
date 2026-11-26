"""Reproduce the crop and weather evaluations from the repository root."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT / "modeling (พัท)"
CLASSES = ["light", "bus", "heavy", "other"]


def render_crop(image, row):
    """Match the production 15% context, square padding and bicubic resize."""
    width = row.x2 - row.x1
    height = row.y2 - row.y1
    crop = image.convert("RGB").crop((
        int(max(0, row.x1 - 0.15 * width)),
        int(max(0, row.y1 - 0.15 * height)),
        int(min(image.width, row.x2 + 0.15 * width)),
        int(min(image.height, row.y2 + 0.15 * height)),
    ))
    side = max(crop.size)
    square = Image.new("RGB", (side, side), (124, 116, 104))
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return np.asarray(square.resize((128, 128), Image.Resampling.BICUBIC))


def crop_features(crops):
    """Extract spatial gradient histograms and RGB cell means."""
    x = np.asarray(crops, dtype=np.float32) / 255
    gray = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    dy, dx = np.gradient(gray, axis=(1, 2))
    angle = ((np.arctan2(dy, dx) % np.pi) * (9 / np.pi)).astype(int) % 9
    magnitude = np.hypot(dx, dy)

    def cells(a):
        return a.reshape(len(x), 4, 32, 4, 32).sum(axis=(2, 4))

    hist = np.stack([cells(magnitude * (angle == i)) for i in range(9)], axis=-1)
    hist /= np.sqrt((hist**2).sum(axis=-1, keepdims=True) + 1e-6)
    color = x.reshape(len(x), 4, 32, 4, 32, 3).mean(axis=(2, 4))
    return np.concatenate((hist.reshape(len(x), -1), color.reshape(len(x), -1)), axis=1)


def classify_checkpoint(crops, checkpoint):
    import timm
    import torch
    import torch.nn.functional as F

    arch = "vit_base_patch16_clip_224.datacompxl"
    model = timm.create_model(arch, pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    mean = torch.tensor(model.pretrained_cfg["mean"], device=device)[None, :, None, None]
    std = torch.tensor(model.pretrained_cfg["std"], device=device)[None, :, None, None]
    probabilities = []
    with torch.inference_mode():
        for batch in np.array_split(crops, (len(crops) + 15) // 16):
            x = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            probabilities.append(model((x - mean) / std).softmax(1).cpu().numpy())
    scores = np.concatenate(probabilities)
    heavy = CLASSES.index("heavy")
    other = scores.copy()
    other[:, heavy] = -1
    return np.where(scores[:, heavy] >= 0.6, "heavy", np.asarray(CLASSES)[other.argmax(1)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels = pd.read_csv(MODELING / "labels/labels.csv")
    assert labels.rid.is_unique
    crops = []
    for r in labels.itertuples():
        path = args.images / str(r.camera_id) / Path(r.path).name
        with Image.open(path) as image:
            crops.append(render_crop(image, r))

    features = crop_features(np.stack(crops))
    output = labels[["rid", "camera_id", "timestamp", "y4"]].copy()
    output["fold"] = 0
    output["baseline_oof"] = ""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(
        splitter.split(features, labels.y4, groups=labels.camera_id), start=1
    ):
        assert set(labels.camera_id.iloc[train]).isdisjoint(labels.camera_id.iloc[test])
        assert set(labels.y4.iloc[train]) == set(CLASSES)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(features[train], labels.y4.iloc[train])
        assert output.loc[test, "fold"].eq(0).all()
        output.loc[test, "fold"] = fold
        output.loc[test, "baseline_oof"] = model.predict(features[test])
    assert output.fold.gt(0).all()
    if args.checkpoint:
        output["clip_checkpoint_in_sample"] = classify_checkpoint(np.stack(crops), args.checkpoint)
    output.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
