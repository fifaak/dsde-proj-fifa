"""GPU (CUDA) training of the truck crop classifier from pre-rendered tensors (see prep_gpu_data.py).

Same model/recipe as train_cnn.py (EfficientNet-B0, 4 classes, Thai crops + our labels x6, class-weighted loss,
OneCycle AdamW), but all data lives on the GPU, fp16 autocast + channels_last + cudnn.benchmark, batch 256.

Usage (on the GPU host):
  python train_gpu.py holdout data/ [--arch timm_name --size 224 --lr 3e-5 --bs 128]
  python train_gpu.py fit data/ --out out.pt [same options]
--arch effnet_b0 (torchvision, default) or any timm model name, e.g. convnextv2_tiny.fcmae_ft_in22k_in1k,
vit_base_patch14_reg4_dinov2.lvd142m, vit_base_patch16_clip_224.datacompxl. Inputs are stored at 128 px and resized
on the GPU to --size.
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from torchvision import models

CLASSES = ["light", "bus", "heavy", "other"]
BS, OURS_REPEAT, LR = 256, 6, 1e-3
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
DEV = "cuda"
torch.backends.cudnn.benchmark = True


CFG = {"arch": "effnet_b0", "size": 128}


def to_input(u8):                                          # (B,H,W,3) uint8 on GPU -> normalised NCHW channels_last
    x = u8.permute(0, 3, 1, 2).float()
    if CFG["size"] != x.shape[-1]:
        x = F.interpolate(x, size=(CFG["size"], CFG["size"]), mode="bilinear", align_corners=False)
    return ((x - MEAN.to(x.device)) / STD.to(x.device)).contiguous(memory_format=torch.channels_last)


def make_model():
    global MEAN, STD
    if CFG["arch"] == "effnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    else:
        import timm
        kw = {"img_size": CFG["size"]} if "vit" in CFG["arch"] else {}
        m = timm.create_model(CFG["arch"], pretrained=True, num_classes=len(CLASSES), **kw)
        MEAN = torch.tensor(m.pretrained_cfg["mean"]).view(1, 3, 1, 1) * 255
        STD = torch.tensor(m.pretrained_cfg["std"]).view(1, 3, 1, 1) * 255
    return m.to(DEV, memory_format=torch.channels_last)


def train(thai_x, thai_y, ours_x, ours_y, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    torch.manual_seed(seed)
    E, N = thai_x.shape[:2]
    m = make_model()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    ys = torch.cat([thai_y, ours_y.repeat(OURS_REPEAT)])
    w = 1.0 / torch.bincount(ys, minlength=len(CLASSES)).clamp(min=1).float()
    loss_fn = nn.CrossEntropyLoss(weight=w / w.sum() * len(CLASSES))
    total = N + len(ours_y) * OURS_REPEAT
    nb = total // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=E * nb)
    scaler = torch.amp.GradScaler()
    m.train()
    for e in range(E):
        X = torch.cat([thai_x[e], ours_x.repeat(OURS_REPEAT, 1, 1, 1)])          # epoch e uses Thai variant e
        perm = torch.randperm(total, device=DEV, generator=g)
        for b in range(nb):
            idx = perm[b * BS:(b + 1) * BS]
            x = to_input(X[idx])
            flip = torch.rand(len(idx), device=DEV, generator=g) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(m(x).float(), ys[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return m


@torch.no_grad()
def predict(m, x_u8):
    m.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        return torch.cat([torch.softmax(m(to_input(x_u8[i:i + 512])).float(), 1)
                          for i in range(0, len(x_u8), 512)]).cpu().numpy()


def main():
    global BS, LR
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["holdout", "fit"])
    ap.add_argument("data")
    ap.add_argument("--arch", default="effnet_b0")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--out")
    a = ap.parse_args()
    CFG.update(arch=a.arch, size=a.size)
    BS, LR = a.bs, a.lr
    cmd, d = a.cmd, a.data.rstrip("/")
    t0 = time.time()
    thai_x = torch.from_numpy(np.load(f"{d}/thai_x.npy")).to(DEV)
    thai_y = torch.from_numpy(np.load(f"{d}/thai_y.npy")).long().to(DEV)
    ours_x = torch.from_numpy(np.load(f"{d}/ours_x.npy")).to(DEV)
    ours_y_np = np.load(f"{d}/ours_y.npy")
    ours_y = torch.from_numpy(ours_y_np).long().to(DEV)
    print(f"loaded to GPU in {time.time() - t0:.1f}s: thai {tuple(thai_x.shape)}, ours {tuple(ours_x.shape)}", flush=True)
    if cmd == "fit":
        t0 = time.time()
        m = train(thai_x, thai_y, ours_x, ours_y)
        torch.save(m.state_dict(), a.out)
        print(f"trained in {time.time() - t0:.1f}s, saved {a.out}")
        return
    tr, te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(ours_y_np, ours_y_np))   # same split as Mac
    t0 = time.time()
    m = train(thai_x, thai_y, ours_x[tr], ours_y[tr])
    torch.cuda.synchronize()
    print(f"[{a.arch} @{a.size}px lr={a.lr}] trained in {time.time() - t0:.1f}s ({thai_x.shape[0]} epochs)", flush=True)
    P = predict(m, ours_x[te])
    y, p, h = ours_y_np[te], P.argmax(1), CLASSES.index("heavy")
    for th in (None, 0.5, 0.6, 0.7):
        ph = (p == h) if th is None else (P[:, h] >= th)
        tp = (ph & (y == h)).sum()
        print(f"{'argmax' if th is None else f'p>={th}':8s} heavy P {tp / max(ph.sum(), 1):.2f}  "
              f"R {tp / (y == h).sum():.2f}  (n={ph.sum()})  4-class acc {(p == y).mean():.2f}")


if __name__ == "__main__":
    main()
