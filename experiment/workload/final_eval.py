"""End-of-session evaluation on the held-out CIFAR-10 test set.

Run EXACTLY ONCE per session, by the harness, on the best recipe found. The test
set is never visible to train.py or to the proposer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from prepare_cifar import load_splits


def _load_train_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_train", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_train"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True, help="the winning train.py")
    ap.add_argument("--checkpoint", required=True, help="model weights saved by train.py")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    mod = _load_train_module(Path(args.train_file))
    device = torch.device(args.device)

    model = mod.build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    d = load_splits(args.data_dir, include_test=True)
    x = torch.from_numpy(d["x_test"])
    y = torch.from_numpy(d["y_test"])

    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), 1000):
            xb = x[i:i + 1000].to(device, non_blocking=True)
            pred = model(xb).argmax(dim=1).cpu()
            correct += int((pred == y[i:i + 1000]).sum())

    acc = correct / len(y)
    Path(args.out).write_text(json.dumps({"test_acc": acc, "n": int(len(y))}, indent=2))
    print(f"FINAL_TEST_ACC {acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
