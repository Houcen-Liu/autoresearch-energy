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
import random  # Added for random horizontal flip
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_cifar import load_splits

# ---------------------------------------------------------------- fixed budget
TRAIN_SECONDS = 240.0          # DO NOT MODIFY -- enforced by harness/guards.py

# ------------------------------------------------------------- tunable recipe
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001  # Changed from 0.0 to 0.0001 for regularization
SEED = 0


def build_model() -> nn.Module:
    """Return the model. Must stay importable for final_eval.py."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),                                    # 16x16
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),                                    # 8x8
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),                                    # 4x4
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),  # Added dropout layer to prevent overfitting
        nn.Linear(256, 10),
    )


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Apply random horizontal flip with 50% probability."""
    if random.random() < 0.5:
        return torch.flip(xb, dims=(3,))
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
    ap.add_argument("--train-seconds", type=float, default=240.0)
    return 0


if __name__ == "__main__":
    main()
