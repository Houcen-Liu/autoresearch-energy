"""CIFAR-10 training recipe -- THE SINGLE FILE THE AGENT EDITS.

Contract with the harness (do not break any of these; they are checked statically
by harness/guards.py before this file is ever executed:

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
import torchvision.transforms as T

from prepare_cifar import load_splits

# ENVIRONMENT -- this file runs against these exact versions.
#   torch 2.13, torchvision 0.28, Python 3.12, CUDA 13.0, one NVIDIA GPU.
# Several APIs common in older code have been REMOVED or changed. Do not use:
#   * `verbose=` on any torch.optim.lr_scheduler.* constructor  -> removed
#   * torchvision.transforms.functional.pad(fill=...) with a non-int/tuple fill
#   * torch.cuda.amp.autocast(...)      -> use torch.amp.autocast("cuda", ...)
#   * Variable(), .data, torch.set_grad_enabled as a decorator
# If you are unsure whether an argument still exists in torch 2.13, leave it out.
# A proposal that raises TypeError scores nothing: the iteration is spent, the
# energy is spent, and the recipe is reverted.

# ---------------------------------------------------------------- fixed budget
TRAIN_SECONDS = 240.0          # DO NOT MODIFY -- enforced by harness/guards.py

# ------------------------------------------------------------- tunable recipe
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
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
    """Data augmentation with random crop and horizontal flip."""
    # Random crop with padding
    padding = 4
    xb = F.pad(xb, (padding, padding, padding, padding), "replicate")
    # Random crop
    i = torch.randint(0, 2*padding, (1,))
    j = torch.randint(0, 2*padding, (1,))
    h, w = 32, 32
    xb = xb[:, :, i:i+h, j:j+w]
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

    # DATA CONTRACT -- read this before changing anything about the inputs.
    #   x_train / x_val : float32, shape (N, 3, 32, 32), NCHW, ALREADY
    #                     normalised per channel (mean/std applied in
    #                     prepare_cifar.py). Do NOT divide by 255 and do NOT
    #                     subtract a mean again -- the data is not raw uint8.
    #   y_train / y_val : int64 class indices in [0, 10).
    #   Channel stats, if you need them: mean [0.4914, 0.4822, 0.4465],
    #   std [0.2470, 0.2435, 0.2616]. They are already applied.
    # A per-channel tensor must be shaped (1, 3, 1, 1) to broadcast against
    # NCHW; a bare (3,) tensor broadcasts against the width axis and raises
    # "size of tensor a (32) must match tensor b (3) at dimension 3".
    d = load_splits(args.data_dir)
    x_tr = torch.from_numpy(d["x_train"])
    y_tr = torch.from_numpy(d["y_train"])
    x_va = torch.from_numpy(d["x_val"])
    y_va = torch.from_numpy(d["y_val"])

    model = build_model().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

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
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
        else:
            epochs += 1
            scheduler.step()

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
        "peak_vRAM_mb": peak,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
