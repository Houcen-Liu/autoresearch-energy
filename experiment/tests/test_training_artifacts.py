"""Training outputs must belong to the candidate that just ran."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.agent_loop import run_training  # noqa: E402


def test_run_training_does_not_reuse_stale_checkpoint(tmp_path):
    work = tmp_path / "recipe"
    run = tmp_path / "run"
    work.mkdir()
    run.mkdir()
    (work / "model.pt").write_bytes(b"stale checkpoint")
    (work / "train.py").write_text(
        """\
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
args, _ = ap.parse_known_args()
with open(args.out, "w", encoding="utf-8") as fh:
    json.dump({"val_acc": 0.75, "train_seconds": 0.01}, fh)
""",
        encoding="utf-8",
    )
    cfg = {
        "gpus": {"train": "cpu"},
        "workload": {
            "data_dir": str(tmp_path / "data"),
            "train_seconds": 0.01,
            "train_timeout_s": 10,
        },
    }

    result = run_training(work, run, iteration=7, cfg=cfg, device="cpu")

    assert result["errored"] is False
    assert result["checkpoint"] is False
    assert not (work / "model.pt").exists()
    assert not (run / "model_007.pt").exists()
