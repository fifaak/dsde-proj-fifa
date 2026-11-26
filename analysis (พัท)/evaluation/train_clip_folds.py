"""Train five independent CLIP models with the repo's CUDA recipe."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate import MODELING, render_crop

sys.path.insert(0, str(MODELING))
import train_gpu as recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--thai-tensors", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("The repository training recipe requires a CUDA GPU.")

    from PIL import Image

    labels = pd.read_csv(MODELING / "labels/labels.csv").merge(
        pd.read_csv(args.splits)[["rid", "fold"]], on="rid", how="left", validate="one_to_one"
    )
    assert labels.fold.notna().all() and set(labels.fold) == set(range(1, 6))
    assert labels.groupby("camera_id").fold.nunique().max() == 1
    crops = []
    for row in labels.itertuples():
        with Image.open(args.images / str(row.camera_id) / Path(row.path).name) as frame:
            crops.append(render_crop(frame, row))
    ours_x = torch.tensor(np.stack(crops), device="cuda")
    ours_y = torch.tensor([recipe.CLASSES.index(y) for y in labels.y4], device="cuda")
    thai_x = torch.from_numpy(np.load(args.thai_tensors / "thai_x.npy")).to("cuda")
    thai_y = torch.from_numpy(np.load(args.thai_tensors / "thai_y.npy")).long().to("cuda")
    assert thai_x.ndim == 5 and tuple(thai_x.shape[2:]) == (128, 128, 3)
    assert thai_x.shape[1] == len(thai_y) and len(thai_y) > 0
    recipe.CFG.update(arch="vit_base_patch16_clip_224.datacompxl", size=224)
    recipe.LR, recipe.BS = args.lr, args.batch_size
    heavy = recipe.CLASSES.index("heavy")

    for fold in range(1, 6):
        test = labels.fold.eq(fold).to_numpy()
        train_indices = torch.as_tensor(np.flatnonzero(~test), device="cuda")
        test_indices = torch.as_tensor(np.flatnonzero(test), device="cuda")
        print(f"Training fold {fold} on {len(train_indices)} crops", flush=True)
        model = recipe.train(thai_x, thai_y, ours_x[train_indices], ours_y[train_indices], seed=fold)
        probabilities = recipe.predict(model, ours_x[test_indices])
        labels.loc[test, "p_heavy"] = probabilities[:, heavy]
        probabilities[:, heavy] = -1
        labels.loc[test, "clip_oof"] = np.where(
            labels.loc[test, "p_heavy"] >= 0.6, "heavy", np.asarray(recipe.CLASSES)[probabilities.argmax(1)]
        )
        del model
    labels[["rid", "camera_id", "timestamp", "y4", "fold", "clip_oof", "p_heavy"]].to_csv(
        args.out, index=False
    )


if __name__ == "__main__":
    main()
