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
        nn.Linear(128 * 4 * 4, 512),  # Increased from 256 to 512
        nn.ReLU(),
        nn.Dropout(p=0.5),  # Added dropout layer to prevent overfitting
        nn.Linear(512, 10),
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
        yb = y[i:i + 1000].to(device, non_blocking=True)
        pred = model(xb)
        correct += int((pred.argmax(1) == yb).sum().item())
    model.train()
    return correct / len(y)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splits()
    x_train, y_train = data['train']
    x_val, y_val = data['val']
    
    model = build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] * 10))
    
    # Warmup phase
    for _ in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()
    
    # Training loop
    steps = 0
    epochs = 0
    start_time = time.time()
    while time.time() - start_time < TRAIN_SECONDS:
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            steps += 1
            if steps % 100 == 0:
                print(f"Step {steps}, Loss: {loss.item():.4f}")
        
        epochs += 1
    
    # Evaluate
    val_acc = evaluate(model, x_val, y_val, device)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Save results
    result = {
        "val_acc": float(val_acc),
        "epochs_completed": epochs,
        "steps": steps,
        "train_seconds": round(time.time() - start_time, 2),
        "peak_vram_mb": int(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0
    }
    with open("result.json", "w") as f:
        json.dump(result, f)
    
    # Save model
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
