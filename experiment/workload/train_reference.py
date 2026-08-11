"""CALIBRATION ONLY -- a strong reference recipe. NEVER shown to the agent.

Purpose: bound the headroom available at a given training budget. The baseline
alone cannot tell you whether a budget is a good choice, because a saturated
*baseline* says nothing about whether an *improved* recipe can exploit more
compute.

DESIGN CONSTRAINT LEARNED FROM THE PILOT. The first version of this file was a
much larger network (64-512 channels, ~4.7M parameters, batch 256). Under a fixed
WALL-CLOCK budget it was catastrophic: 1 352 steps against the baseline's 33 735
at 240 s -- 25x slower per step, 7 epochs versus 98 -- and it scored *worse* than
the baseline at every budget tested (-23.6 pp at 20 s, still -3.8 pp at 240 s).

That is the fixed-time budget doing exactly what it is designed to do: it rewards
recipes that are cheap per step, and punishes capacity that cannot be trained to
convergence inside the budget. A competent agent optimising under this rule would
never scale the model up that far, so a bloated network is not a valid stand-in
for what the agent would find.

This version therefore keeps the baseline's ARCHITECTURAL SCALE and improves only
how it is trained -- batch normalisation, cheap augmentation, a one-cycle
schedule, mixed precision. Roughly 1.5x the cost per step, not 25x. That is the
realistic upper bound on what the agent can reach, and the gap between it and the
baseline, in units of the noise floor, is the SNR the agent operates in.

Keep it OUT of the recipe repository and out of every prompt.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_cifar import load_splits

TRAIN_SECONDS = 240.0
BATCH_SIZE = 128            # same as the baseline, so steps stay comparable
MAX_LR = 0.2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEED = 0


def build_model() -> nn.Module:
    """Baseline topology (32-64-128 channels) plus batch norm.

    Deliberately NOT wider than the baseline: under a fixed wall-clock budget,
    extra capacity costs steps, and steps are what the budget actually buys.
    """
    def blk(cin, cout, pool=False):
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                  nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return layers

    return nn.Sequential(
        *blk(3, 32), *blk(32, 32, pool=True),        # 16x16
        *blk(32, 64), *blk(64, 64, pool=True),       # 8x8
        *blk(64, 128, pool=True),                    # 4x4
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256, bias=False),
        nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        nn.Linear(256, 10),
    )


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Cheap augmentation: one random crop offset and one flip decision per BATCH.

    Per-sample boolean indexing (`xb[flip] = torch.flip(xb[flip], ...)`) is
    several times more expensive and buys little at these batch sizes. Under a
    wall-clock budget, augmentation that costs steps can lose more than it gains.
    """
    xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2,))
    xb = xb[:, :, i:i + 32, j:j + 32]
    if torch.rand(()) < 0.5:
        xb = torch.flip(xb, dims=[3])
    return xb


@torch.no_grad()
def evaluate(model, x, y, device) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), 1000):
        xb = x[i:i + 1000].to(device, non_blocking=True)
        correct += int((model(xb).argmax(1).cpu() == y[i:i + 1000]).sum())
    model.train()
    return correct / len(y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--checkpoint", default="model.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train-seconds", type=float, default=None)
    args = ap.parse_args()

    budget = float(args.train_seconds) if args.train_seconds else TRAIN_SECONDS
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    d = load_splits(args.data_dir)
    x_tr = torch.from_numpy(d["x_train"])
    y_tr = torch.from_numpy(d["y_train"])
    x_va = torch.from_numpy(d["x_val"])
    y_va = torch.from_numpy(d["y_val"])

    model = build_model().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=MAX_LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY, nesterov=True)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Warm-up outside the budget: CUDA context, cudnn autotune, first allocation.
    xb0 = x_tr[:BATCH_SIZE].to(device)
    with torch.autocast("cuda", enabled=use_amp):
        loss0 = F.cross_entropy(model(xb0), y_tr[:BATCH_SIZE].to(device))
    scaler.scale(loss0).backward()
    opt.zero_grad(set_to_none=True)
    if use_amp:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    n = len(x_tr)
    steps_per_epoch = math.ceil(n / BATCH_SIZE)
    steps = epochs = 0
    model.train()
    t0 = time.monotonic()

    # The one-cycle schedule needs a horizon. Probe the achievable step rate for
    # one second, then fix the horizon; re-estimating mid-run would make the
    # schedule depend on thermal state.
    probe_end = t0 + 1.0
    while time.monotonic() < probe_end:
        idx = torch.randint(0, n, (BATCH_SIZE,))
        xb = augment(x_tr[idx].to(device, non_blocking=True))
        yb = y_tr[idx].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        steps += 1
    rate = max(1.0, steps / max(1e-6, time.monotonic() - t0))
    est_total = max(steps_per_epoch, int(rate * budget))

    while time.monotonic() - t0 < budget:
        perm = torch.randperm(n)
        for i in range(steps_per_epoch):
            if time.monotonic() - t0 >= budget:
                break
            frac = min(1.0, steps / est_total)
            lr = MAX_LR * (frac / 0.15 if frac < 0.15 else
                           0.5 * (1 + math.cos(math.pi * (frac - 0.15) / 0.85)))
            for g in opt.param_groups:
                g["lr"] = max(lr, 1e-4)

            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            xb = augment(x_tr[idx].to(device, non_blocking=True))
            yb = y_tr[idx].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            steps += 1
        else:
            epochs += 1

    if use_amp:
        torch.cuda.synchronize()
    elapsed = time.monotonic() - t0

    val_acc = evaluate(model, x_va, y_va, device)
    torch.save(model.state_dict(), args.checkpoint)
    peak = int(torch.cuda.max_memory_allocated() / 1e6) if use_amp else 0
    result = {"val_acc": round(float(val_acc), 6), "epochs_completed": epochs,
              "steps": steps, "train_seconds": round(elapsed, 2),
              "peak_vram_mb": peak}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
