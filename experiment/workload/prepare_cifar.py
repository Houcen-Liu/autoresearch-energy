"""One-time data preparation for the inner CIFAR-10 workload.

FIXED VARIABLE OF THE EXPERIMENT. The agent may not modify this file, and train.py
must obtain its data exclusively through `load_splits()`.

Split policy (seeded, identical in every session):
  50 000 CIFAR-10 train images -> 45 000 train / 5 000 validation
  10 000 CIFAR-10 test images  -> touched once per session, by final_eval.py only
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

SPLIT_SEED = 20260810
VAL_SIZE = 5_000
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _cache_path(data_dir: Path) -> Path:
    return Path(data_dir) / "cifar10_splits.npz"


def prepare_synthetic(data_dir: str | Path = "./data") -> Path:
    """DEV ONLY: random-noise splits with the real shapes and dtypes.

    Lets the harness be smoke-tested with no download and no network. Accuracy
    from a synthetic run is meaningless by construction (the labels are random);
    the point is to exercise every code path. Never use for real runs -- the cache
    records synthetic=True and load_splits() prints a warning.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(data_dir)
    rng = np.random.default_rng(SPLIT_SEED)

    def blob(n):
        return (rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8),
                rng.integers(0, 10, size=(n,), dtype=np.int64))

    xt, yt = blob(4_500)
    xv, yv = blob(500)
    xs, ys = blob(1_000)
    np.savez_compressed(cache, x_train=xt, y_train=yt, x_val=xv, y_val=yv,
                        x_test=xs, y_test=ys, split_seed=SPLIT_SEED, synthetic=True)
    print(f"[prepare] SYNTHETIC cache written to {cache} -- results are meaningless")
    return cache


def prepare(data_dir: str | Path = "./data") -> Path:
    """Download CIFAR-10 and materialise the fixed splits as a single .npz cache."""
    from torchvision.datasets import CIFAR10

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(data_dir)
    if cache.exists():
        print(f"[prepare] cache present: {cache}")
        return cache

    tr = CIFAR10(root=str(data_dir), train=True, download=True)
    te = CIFAR10(root=str(data_dir), train=False, download=True)

    x_all = np.asarray(tr.data, dtype=np.uint8)            # (50000, 32, 32, 3)
    y_all = np.asarray(tr.targets, dtype=np.int64)
    x_test = np.asarray(te.data, dtype=np.uint8)
    y_test = np.asarray(te.targets, dtype=np.int64)

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(x_all))
    val_idx, train_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]

    np.savez_compressed(
        cache,
        x_train=x_all[train_idx], y_train=y_all[train_idx],
        x_val=x_all[val_idx], y_val=y_all[val_idx],
        x_test=x_test, y_test=y_test,
        split_seed=SPLIT_SEED,
    )
    print(f"[prepare] wrote {cache} "
          f"(train={len(train_idx)}, val={len(val_idx)}, test={len(x_test)})")
    return cache


def load_splits(data_dir: str | Path | None = None, include_test: bool = False):
    """Return normalised float32 NCHW arrays.

    train.py MUST call this with include_test=False (the default). Requesting the
    test split from train.py is a guard violation and will be rejected before the
    proposal is ever executed.
    """
    data_dir = Path(data_dir or os.environ.get("AR_DATA_DIR", "./data"))
    cache = _cache_path(data_dir)
    if not cache.exists():
        prepare(data_dir)
    d = np.load(cache)
    if "synthetic" in d.files and bool(d["synthetic"]):
        print("[prepare] WARNING: synthetic data cache -- accuracy is meaningless")

    def norm(x):
        x = x.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))

    out = {
        "x_train": norm(d["x_train"]), "y_train": d["y_train"],
        "x_val": norm(d["x_val"]), "y_val": d["y_val"],
    }
    if include_test:
        out["x_test"] = norm(d["x_test"])
        out["y_test"] = d["y_test"]
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--synthetic", action="store_true",
                   help="DEV ONLY: random-noise splits, no download")
    a = p.parse_args()
    prepare_synthetic(a.data_dir) if a.synthetic else prepare(a.data_dir)
