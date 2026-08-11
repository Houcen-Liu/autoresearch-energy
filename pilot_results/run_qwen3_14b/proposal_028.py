"""CIFAR-10 training recipe -- THE SINGLE FILE THE AGENT EDITS.

Contract with the harness (do not break any of these; they are checked statically
by harness/guards.py before this file is ever executed):

  * TRAIN_SECONDS is a fixed wall-clock training budget and MUST keep its value.
    Training stops at the budget boundary, mid-epoch if necessary. Buying accuracy
    with extra time destroys comparability between experiments.
  * Data comes only from prepare_cifar.load_splits(); the test split is off limits.
  * On success this file writes result.json with keys:
        val_acc, epochs_completed, steps, train_seconds, peak_vram_mb
    and saves the trained weights to the checkpoint path.
  * build_model() must exist and return the model, so final_eval.py can rebuild it.

BASELINE NOTE: this recipe is deliberately unoptimised (no augmentation, constant
LR, no normalisation layers) so that the agent has real headroom. Calibrated in
week 1 to land in the 0.60-0.85 validation band (gate G2). Once calibrated it is a
FIXED VARIABLE: pin the commit and never touch it again.
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
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from prepare_cifar import load_splits

# ---------------------------------------------------------------- fixed budget
TRAIN_SECONDS = 240.0          # DO NOT MODIFY -- enforced by harness/guards.py

# ------------------------------------------------------------- tunable recipe
BATCH_SIZE = 128
LEARNING_RATE = 0.0025
WEIGHT_DECAY = 0.0005
SEED = 0


def build_model() -> nn.Module:
    """Return the model. Must stay importable for final_eval.py."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),                                    # 16x16
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),                                    # 8x8
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),                                    # 4x4
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Dropout(0.5),  # Added dropout for regularization
        nn.Linear(256, 10),
    )


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Data augmentation hook. Baseline: none."""
    # Simple augmentation that works across all torchvision versions
    if torch.rand(1) < 0.5:
        xb = TF.hflip(xb)
    # Add color jitter
    transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    xb = transform(xb)
    # Add RandomErasing augmentation
    transform = T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5)
    xb = transform(xb)
    return xb


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device) -> float:
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
    ap.add_argument("--train-seconds", type=float, default=None,
                    help="harness override for pilot mode; ignored on the server")
    args = ap.parse_args()

    budget = float(args.train_seconds) if args.train_seconds else TRAIN_SECONDS

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    d = load_splits(args.data_dir)
    x_tr = torch.from_numpy(d["x_train"])
    y_tr = torch.from_numpy(d["y_train"])
    x_va = torch.from_numpy(d["x_val"])
    y_va = torch.from_numpy(d["y_val"])

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    # Mixed precision training setup
    scaler = torch.cuda.amp.GradScaler()

    # Warm-up outside the budget: CUDA context, kernel autotune, first allocation.
    xb0 = x_tr[:BATCH_SIZE].to(device)
    F.cross_entropy(model(xb0), y_tr[:BATCH_SIZE].to(device)).backward()
    opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    n = len(x_tr)
    steps_per_epoch = math.ceil(n / BATCH_SIZE)
    steps = epochs = 0
    model.train()
    t0 = time.monotonic()                        # <-- budget clock starts here

    while time.monotonic() - t0 < budget:
        perm = torch.randperm(n)
        for i in range(steps_per_epoch):
            if time.monotonic() - t0 >= budget:
                break
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            xb = augment(x_tr[idx].to(device, non_blocking=True))
            yb = y_tr[idx].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():  # Enable mixed precision
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            steps += 1
        else:
            epochs += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - t0

    val_acc = evaluate(model, x_va, y_va, device)
    torch.save(model.state_dict(), args.checkpoint)

    peak = int(torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0
    result = {
        "val_acc": round(float(val_acc), 6),
        "epochs_completed": epochs,
        "steps": steps,
        "train_seconds": round(elapsed, 2),
        "peak_vram_mb": peak,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
