"""Torch-free stand-in recipe, used only by the integration test.

Satisfies the same contract as workload/train.py (fixed TRAIN_SECONDS, load_splits,
build_model, main, result.json keys) so that the harness -- git history, guards,
subprocess execution, result parsing, keep/revert -- can be exercised end to end
on a machine with no torch and no GPU. Not part of the experiment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from prepare_cifar import load_splits

TRAIN_SECONDS = 240.0
LEARNING_RATE = 0.01
BATCH_SIZE = 128
SEED = 0


def build_model():
    rng = np.random.default_rng(SEED)
    return {"W": rng.normal(0, 0.01, size=(3 * 32 * 32, 10)), "b": np.zeros(10)}


def _forward(m, x):
    return x.reshape(len(x), -1) @ m["W"] + m["b"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--checkpoint", default="model.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train-seconds", type=float, default=None)
    a = ap.parse_args()

    budget = float(a.train_seconds) if a.train_seconds else TRAIN_SECONDS
    d = load_splits(a.data_dir)
    xt, yt = d["x_train"], d["y_train"]
    xv, yv = d["x_val"], d["y_val"]

    m = build_model()
    n = len(xt)
    steps = epochs = 0
    t0 = time.monotonic()
    rng = np.random.default_rng(SEED)

    while time.monotonic() - t0 < budget:
        perm = rng.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            if time.monotonic() - t0 >= budget:
                break
            idx = perm[i:i + BATCH_SIZE]
            xb = xt[idx].reshape(len(idx), -1)
            logits = xb @ m["W"] + m["b"]
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            p[np.arange(len(idx)), yt[idx]] -= 1
            p /= len(idx)
            m["W"] -= LEARNING_RATE * (xb.T @ p)
            m["b"] -= LEARNING_RATE * p.sum(0)
            steps += 1
        else:
            epochs += 1

    val_acc = float((_forward(m, xv).argmax(1) == yv).mean())
    with open(a.checkpoint, "wb") as fh:      # exact path, no .npz suffix appended
        np.savez(fh, **m)
    result = {"val_acc": round(val_acc, 6), "epochs_completed": epochs, "steps": steps,
              "train_seconds": round(time.monotonic() - t0, 2), "peak_vram_mb": 0}
    Path(a.out).write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
