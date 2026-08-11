"""Generate a fake-but-real-shaped experiment tree.

Purpose: validate the whole analysis half of the pipeline -- alignment,
aggregation, statistics, Pareto, figures -- before a single real session exists.
Run this in week 0. The numbers are invented; the schema, the file layout and the
code paths are exactly the real ones.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

ARMS = ["dense", "moe"]
PATIENCE = ["greedy", "patience3"]
BUDGETS = [10, 20]

# Invented effect structure, so the analysis has something to find:
PROP_POWER_W = {"dense": 165, "moe": 95}      # MoE cheaper per second of inference
PROP_SECS = {"dense": 38, "moe": 55}          # ...but slower to a usable answer
KEEP_RATE = {"dense": 0.42, "moe": 0.33}
TRAIN_POWER_W = 120
IDLE_W = 18


def make_run(out: Path, arm: str, pat: str, budget: int, rep: int, seed: int,
             train_seconds: float = 240.0) -> None:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    patience = 1 if pat == "greedy" else 3

    t = 1_760_000_000.0 + seed * 20_000
    log, nvml = [], []

    def emit(ev, **kw):
        log.append({"ev": ev, "t": t, "m": t, **kw})

    def sample(dur, p0, p1):
        """Append 10 Hz NVML rows for [t, t+dur) at the given per-device powers."""
        nonlocal t
        n = max(2, int(dur * 10))
        for k in range(n):
            tt = t + k / 10
            for dev, w in ((0, p0), (1, p1)):
                jitter = rng.gauss(0, w * 0.02)
                nvml.append((tt, dev, max(0.0, w + jitter)))
        t += dur

    emit("session_start", cell={"proposer": arm, "patience": pat, "loop_budget": budget},
         seed=seed, attribution="per_device", train_seconds=train_seconds,
         stub=False, synthetic_data=False)
    # The generator must produce REAL-shaped logs, including the proposer
    # manifest -- the analysis quarantines runs that lack it.
    emit("proposer_config", endpoint="http://127.0.0.1:8000/v1", model=arm,
         params={"temperature": 0.0, "max_tokens": 8192}, params_reduced=False)

    base_acc = rng.uniform(0.68, 0.73)
    emit("baseline_start")
    sample(train_seconds, TRAIN_POWER_W, IDLE_W)
    emit("baseline_eval", val_acc=base_acc, train_seconds=train_seconds)

    best, regressions, provisional = base_acc, 0, []
    counts = {"kept": 0, "reverted": 0, "rejected": 0, "errored": 0}

    for i in range(1, budget + 1):
        emit("propose_start", iter=i)
        dur = PROP_SECS[arm] * rng.uniform(0.8, 1.25)
        sample(dur, IDLE_W, PROP_POWER_W[arm])
        emit("propose_end", iter=i, prompt_tokens=rng.randint(3000, 6000),
             completion_tokens=rng.randint(600, 2200), latency_s=dur, attempts=1)

        if rng.random() < 0.08:                       # guard rejection
            emit("guard", iter=i, ok=False, violations=["TRAIN_SECONDS was changed"])
            emit("decision", iter=i, decision="rejected", best_acc=best,
                 regressions=regressions)
            counts["rejected"] += 1
            sample(3, IDLE_W, IDLE_W)
            continue
        emit("guard", iter=i, ok=True, violations=[])

        emit("train_start", iter=i)
        sample(train_seconds, TRAIN_POWER_W, IDLE_W)
        if rng.random() < 0.04:                       # training crash
            emit("train_end", iter=i, val_acc=None, exit=1, error="CUDA OOM")
            emit("decision", iter=i, decision="errored", best_acc=best,
                 regressions=regressions)
            counts["errored"] += 1
            continue

        improved = rng.random() < KEEP_RATE[arm]
        acc = best + rng.uniform(0.004, 0.02) if improved else best - rng.uniform(0.001, 0.03)
        emit("train_end", iter=i, val_acc=acc, exit=0,
             epochs=rng.randint(8, 16), steps=rng.randint(2500, 5000))

        if acc > best + 0.001:
            best, regressions, provisional = acc, 0, []
            counts["kept"] += 1
            emit("decision", iter=i, decision="keep", best_acc=best, regressions=0)
        else:
            regressions += 1
            provisional.append(i)
            if regressions >= patience:
                emit("rollback", iter=i, to_sha="deadbeef", discarded_iters=list(provisional))
                counts["reverted"] += len(provisional)
                regressions, provisional = 0, []
                emit("decision", iter=i, decision="revert", best_acc=best, regressions=0)
            else:
                emit("decision", iter=i, decision="provisional", best_acc=best,
                     regressions=regressions)
        sample(4, IDLE_W, IDLE_W)

    if provisional:
        emit("rollback", iter=budget, to_sha="deadbeef", discarded_iters=list(provisional),
             reason="budget exhausted")
        counts["reverted"] += len(provisional)

    sample(30, TRAIN_POWER_W * 0.6, IDLE_W)
    test_acc = best - rng.uniform(0.005, 0.03)
    emit("final_eval", test_acc=test_acc, best_sha="deadbeef", best_iter=1)
    summary = {"iterations": budget, **counts, "baseline_val_acc": base_acc,
               "best_val_acc": best, "test_acc": test_acc,
               "no_progress": counts["kept"] == 0}
    emit("session_end", **summary)

    (out / "session.jsonl").write_text(
        "\n".join(json.dumps(r) for r in log) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # NVML rows -> cumulative energy counter in mJ, per device, exactly as the
    # real sampler emits it.
    lines = ["t_wall,t_mono,dev,power_mw,energy_mj,util_gpu,util_mem,temp_c,mem_used_mb"]
    cum = {0: 0.0, 1: 0.0}
    last = {}
    for tt, dev, w in sorted(nvml, key=lambda r: (r[0], r[1])):
        dt = tt - last.get(dev, tt)
        cum[dev] += w * dt * 1000
        last[dev] = tt
        lines.append(f"{tt:.4f},{tt:.4f},{dev},{int(w*1000)},{int(cum[dev])},"
                     f"{90 if w > 60 else 2},40,62,{8000 if dev == 0 else 17000}")
    (out / "nvml.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../experiments/synthetic_phase1")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--train-seconds", type=float, default=240.0)
    a = ap.parse_args()

    root = Path(a.out_dir)
    n = 0
    for rep in range(a.reps):
        for arm, pat, bud in itertools.product(ARMS, PATIENCE, BUDGETS):
            make_run(root / f"run_{n}_repetition_{rep}", arm, pat, bud, rep,
                     seed=n * 13 + rep, train_seconds=a.train_seconds)
            n += 1
    print(f"wrote {n} synthetic runs to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
