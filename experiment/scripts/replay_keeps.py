"""Does the loop overfit the split it selects on?

The protocol touches the test set exactly once per session, on the final best
recipe. That is the right rule during a session -- but it means the val-test gap
can only be observed at one point, and the interesting question is whether that
gap GROWS as the agent makes more selections against a 5 000-image validation
split.

This answers it without ever breaking the rule, by replaying afterwards: every
kept revision is checked out of the session's own git bundle, retrained from
scratch at the same budget, and evaluated on validation and test. Nothing here
influenced any decision the agent made; it is post-hoc measurement on a finished
session.

Cost: one training run per keep, i.e. `keeps x train_seconds` plus overhead.
A 100-iteration session with 12 keeps costs about 15 minutes.

    python scripts/replay_keeps.py --run-dir ../experiments/long_horizon/moe_100it_...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.session_log import read_log            # noqa: E402
from harness.agent_loop import run_training, _final_eval, _device   # noqa: E402


def kept_shas(records: list[dict]) -> list[tuple[int, str]]:
    """Return ``(iteration, sha)`` for each mutation the harness kept."""
    sha_of, out = {}, []
    for r in records:
        if r.get("ev") == "train_start" and r.get("sha"):
            sha_of[r.get("iter")] = r["sha"]
        if r.get("ev") == "decision" and r.get("decision") == "keep":
            i = r.get("iter")
            if i in sha_of:
                out.append((i, sha_of[i]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--profile", default=str(ROOT / "profiles" / "server.yaml"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rd = Path(a.run_dir)
    cfg = yaml.safe_load(Path(a.profile).read_text())
    records = read_log(rd / "session.jsonl")
    keeps = kept_shas(records)
    bundle = rd / "recipe_history.bundle"
    if not bundle.exists():
        print(f"[FAIL] no recipe_history.bundle in {rd}")
        return 1
    if not keeps:
        out = Path(a.out) if a.out else rd / "replay_keeps.json"
        out.write_text("[]\n")
        print(f"no kept revisions in this session; wrote an empty replay to {out}")
        return 0

    print(f"replaying {len(keeps)} kept revision(s) from {bundle.name}")
    print(f"estimated {len(keeps) * (cfg['workload']['train_seconds'] + 25) / 60:.0f} min\n")

    rows = []
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "replay"
        subprocess.run(["git", "clone", "-q", str(bundle), str(work)], check=True)
        for n, (it, sha) in enumerate(keeps, 1):
            subprocess.run(["git", "-C", str(work), "checkout", "-q", sha, "--", "train.py"],
                           check=True)
            out_dir = rd / "replay"; out_dir.mkdir(exist_ok=True)
            res = run_training(work, out_dir, 900 + n, cfg, _device(cfg))
            if res["errored"]:
                print(f"  keep {n} (iter {it}): FAILED to retrain -- skipped")
                continue
            ckpt = work / "model.pt"
            test = _final_eval(work, out_dir, ckpt, cfg) if ckpt.exists() else None
            rows.append({"keep_index": n, "iteration": it, "sha": sha[:8],
                         "val_acc": res["val_acc"], "test_acc": test,
                         "gap_pp": None if test is None
                                   else round((res["val_acc"] - test) * 100, 3)})
            print(f"  keep {n:2d} (iter {it:3d})  val {res['val_acc']:.4f}  "
                  f"test {test if test is None else f'{test:.4f}'}  "
                  f"gap {rows[-1]['gap_pp']} pp")

    gaps = [r["gap_pp"] for r in rows if r["gap_pp"] is not None]
    if len(gaps) >= 3:
        n = len(gaps); xs = list(range(1, n + 1))
        mx, my = sum(xs) / n, sum(gaps) / n
        num = sum((x - mx) * (g - my) for x, g in zip(xs, gaps))
        den = sum((x - mx) ** 2 for x in xs)
        slope = num / den if den else 0.0
        print(f"\nval-test gap across the keep sequence: "
              f"first {gaps[0]:+.2f} pp -> last {gaps[-1]:+.2f} pp, "
              f"slope {slope:+.3f} pp per keep")
        print("  A clearly positive slope means the loop is buying validation "
              "accuracy that\n  does not transfer -- i.e. it is overfitting the "
              "metric it selects on.")

    out = Path(a.out) if a.out else rd / "replay_keeps.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
