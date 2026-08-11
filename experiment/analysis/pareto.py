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

BASELINE_CELL = {"proposer": "dense", "patience": "greedy", "loop_budget": 10}
CELL_KEYS = ["proposer", "patience", "loop_budget"]


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
    g = (tidy.groupby(CELL_KEYS, dropna=False)
         .agg(n=("run", "count"),
              E_mean=(x, "mean"), E_sd=(x, "std"),
              acc_mean=(y, "mean"), acc_sd=(y, "std"),
              E_wasted_mean=("E_wasted_J", "mean"),
              kept_mean=("kept", "mean"))
         .reset_index())
    g["on_frontier"] = non_dominated(g.rename(columns={"E_mean": x, "acc_mean": y}), x, y)
    base = g
    for k, v in BASELINE_CELL.items():
        base = base[base[k].astype(str) == str(v)]
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
    print(cells[cells.on_frontier][CELL_KEYS + ["E_mean", "acc_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
