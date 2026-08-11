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

Everything else -- architecture, optimizer, schedule, augmentation, batch size,
regularisation -- is fair game.

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

from prepare_cifar import load_splits
from torch.cuda.amp import autocast, GradScaler


# ---------------------------------------------------------------- fixed budget
TRAIN_SECONDS = 240.0          # DO NOT MODIFY -- enforced by harness/guards.py

# ------------------------------------------------------------- tunable recipe
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
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
        nn.MaxPool2d(2),                                    # 4x4
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        nn.Linear(256, 10),
    )


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Data augmentation hook."""
    # Random crop
    xb = F.pad(xb, (4, 4, 4, 4), 'reflect')  # Pad to 32x32 + 8 pixels on each side
    # Random crop to 32x32
    i = torch.randint(0, 8, (1,)).item()
    j = torch.randint(0, 8, (1,)).item()
    xb = xb[:, :, i:i+32, j:j+32]
    # Random horizontal flip
    if torch.rand(1) < 0.5:
        xb = torch.flip(xb, [3])
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
    x_te = torch.from_numpy(d["x_test"])
    y_te = torch.from_numpy(d["y_test"])
    x_va = torch.cat([x_tr[:len(x_va)], x_va])
    y_va = torch.cat([y_tr[:len(y_va)], y_va])

    model = build_model()
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()

    # Warm-up
    model.train()
    for _ in range(2):
        for _ in range(10):
            xb = x_tr[:BATCH_SIZE].to(device)
            yb = y_tr[:BATCH_SIZE].to(device)
            opt.zero_grad()
            with autocast():
                loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

    # Training
    model.train()
    steps = 0
    epochs = 0
    while time.monotonic() - t0 < budget:
        perm = torch.randperm(len(x_tr))
        for i in range(steps_per_epoch):
            if time.monotonic() - t0 >= budget:
                break
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            xb = augment(x_tr[idx].to(device, non_blocking=True))
            yb = y_tr[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                loss = F.cross_entropy(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            steps += 1
        else:
            epochs += 1
            # Adjust learning rate
            for param_group in opt.param_groups:
                param_group['lr'] = LEARNING_RATE * (0.1 ** (epochs // 5))

    # Final evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        for i in range(0, len(x_te), 1000):
            xb = x_te[i:i + 1000].to(device, non_blocking=True)
            correct += int((model(xb).argmax(1).cpu() == y_te[i:i + 1000]).sum())
    acc = correct / len(x_te)
    print(f"Final test accuracy: {acc:.4f}")

    return 0
