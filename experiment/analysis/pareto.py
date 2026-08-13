"""RQ2: the energy/accuracy Pareto frontier.

Non-dominated set over (E_total minimised, test_acc maximised), computed on cell
means with the individual sessions kept visible underneath. The baseline cell
(dense x greedy x budget 10) is marked so that every other configuration can be
read as a delta from it -- which is the practitioner-facing output of the study.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# The proposal designates dense x greedy x budget-10 as the baseline cell, the
# reference against which configuration effects are reported. The run table
# names patience levels "greedy"/"patience3"; the tidy table stores the enforced
# integer. Match on both, or is_baseline is False for every row and the figure's
# "outlined: baseline" legend marks nothing.
BASELINE_CELL = {"proposer": "dense", "patience": ("greedy", "1", 1),
                 "loop_budget": (10, "10")}
# Phase 1's factors by default; Stage 2 experiments vary different ones, so the
# cell definition follows the data rather than being fixed here (see
# aggregate.ALL_FACTORS).
ALL_FACTORS = ["proposer", "patience", "loop_budget", "thinking", "temperature"]
CELL_KEYS = ["proposer", "patience", "loop_budget"]


def cell_keys(tidy) -> list:
    """Columns that actually vary in this table, falling back to whatever exists."""
    present = [k for k in ALL_FACTORS if k in tidy.columns]
    varying = [k for k in present if tidy[k].nunique(dropna=True) > 1]
    return varying or present


def non_dominated(points: pd.DataFrame, x: str = "E_gpu_total_J",
                  y: str = "test_acc") -> pd.Series:
    """True where no other point has lower-or-equal x AND higher-or-equal y."""
    flags = []
    for _, p in points.iterrows():
        dominated = ((points[x] <= p[x]) & (points[y] >= p[y]) &
                     ((points[x] < p[x]) | (points[y] > p[y]))).any()
        flags.append(not dominated)
    return pd.Series(flags, index=points.index)


def cell_summary(tidy: pd.DataFrame, x: str = "E_gpu_total_J",
                 y: str = "test_acc") -> pd.DataFrame:
    keys = cell_keys(tidy)
    g = (tidy.groupby(keys, dropna=False)
         .agg(n=("run", "count"),
              E_mean=(x, "mean"), E_sd=(x, "std"),
              acc_mean=(y, "mean"), acc_sd=(y, "std"),
              E_wasted_mean=("E_wasted_J", "mean"),
              kept_mean=("kept", "mean"))
         .reset_index())
    g["on_frontier"] = non_dominated(g.rename(columns={"E_mean": x, "acc_mean": y}), x, y)
    base = g
    for k, v in BASELINE_CELL.items():
        accepted = {str(x) for x in (v if isinstance(v, tuple) else (v,))}
        base = base[base[k].astype(str).isin(accepted)]
    g["is_baseline"] = g.index.isin(base.index)
    if len(base):
        b = base.iloc[0]
        g["dE_vs_baseline_pct"] = 100 * (g.E_mean - b.E_mean) / b.E_mean
        g["dAcc_vs_baseline_pp"] = 100 * (g.acc_mean - b.acc_mean)
    return g.sort_values("E_mean")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy", required=True)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    tidy = pd.read_csv(a.tidy)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = cell_summary(tidy)
    cells.to_csv(out / "pareto_cells.csv", index=False)
    print(cells.to_string(index=False))
    print("\nFrontier:")
    print(cells[cells.on_frontier][cell_keys(cells) + ["E_mean", "acc_mean"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
