"""Camera-held-out cross-validation of a linear classifier on CLIP features."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from evaluate import CLASSES, MODELING, render_crop

ARCH = "vit_base_patch16_clip_224.datacompxl"


def embeddings(labels, images, weights):
    import timm
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    model = timm.create_model(ARCH, pretrained=False)
    model.load_state_dict(load_file(str(weights)), strict=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    mean = torch.tensor(model.pretrained_cfg["mean"], device=device)[None, :, None, None]
    std = torch.tensor(model.pretrained_cfg["std"], device=device)[None, :, None, None]
    output = []
    with torch.inference_mode():
        for start in range(0, len(labels), 16):
            crops = []
            for row in labels.iloc[start:start + 16].itertuples():
                with Image.open(images / str(row.camera_id) / Path(row.path).name) as frame:
                    crops.append(render_crop(frame, row))
            x = torch.from_numpy(np.stack(crops)).to(device).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            output.append(model((x - mean) / std).cpu().numpy())
            if start % 80 == 0:
                print(f"Encoded {min(start + len(crops), len(labels))}/{len(labels)} crops on {device}", flush=True)
    features = np.concatenate(output)
    assert np.isfinite(features).all() and np.all(np.linalg.norm(features, axis=1) > 0)
    return normalize(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    original = pd.read_csv(args.predictions)
    labels = pd.read_csv(MODELING / "labels/labels.csv").merge(
        original[["rid", "fold"]], on="rid", how="left", validate="one_to_one"
    )
    assert set(labels.fold) == set(range(1, 6))
    assert labels.groupby("camera_id").fold.nunique().eq(1).all()
    features = embeddings(labels, args.images, args.weights)
    for fold in range(1, 6):
        test = labels.fold.eq(fold)
        assert set(labels.loc[~test, "y4"]) == set(CLASSES)
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(features[~test], labels.loc[~test, "y4"])
        labels.loc[test, "clip_linear_oof"] = model.predict(features[test])
    assert labels.clip_linear_oof.notna().all()
    result = original.drop(columns=["clip_linear_oof"], errors="ignore").merge(
        labels[["rid", "clip_linear_oof"]], on="rid", validate="one_to_one"
    )
    assert len(result) == len(original) == len(labels)
    result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
