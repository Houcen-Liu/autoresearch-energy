"""All report figures. One function per figure, all writing to --out-dir.

Figure 1 (headline) : energy/accuracy Pareto frontier
Figure 2            : proposer vs training energy decomposition per cell
Figure 3            : per-iteration power time series for one representative session
Figure 4            : where the energy went -- kept / provisional / reverted / rejected
Figure 5            : validation-accuracy trajectories per cell
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import pandas as pd                                               # noqa: E402

import sys                                                        # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pareto import CELL_KEYS, cell_keys, cell_summary                        # noqa: E402

MARKERS = {"dense": "o", "moe": "s"}
# The run table names the levels "greedy"/"patience3"; the tidy table stores
# the enforced integer (1/3). Key on both, or every lookup silently falls
# through to black and the figure's own subtitle promises an encoding that is
# not there -- which is how the first Phase-1 figure shipped.
COLORS = {"greedy": "#1f77b4", "patience3": "#d62728",
          "1": "#1f77b4", "3": "#d62728", 1: "#1f77b4", 3: "#d62728"}
SIZES = {10: 70, 20: 170}


def _label(r) -> str:
    return f"{r.proposer}/{r.patience}/b{int(r.loop_budget)}"


def fig_pareto(tidy: pd.DataFrame, out: Path) -> Path:
    cells = cell_summary(tidy)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.scatter(tidy["E_gpu_total_J"] / 1000, tidy["test_acc"] * 100,
               c="0.8", s=22, zorder=1, label="individual sessions")

    for _i, (_, r) in enumerate(cells.iterrows()):
        ax.errorbar(r.E_mean / 1000, r.acc_mean * 100,
                    xerr=(r.E_sd or 0) / 1000, yerr=(r.acc_sd or 0) * 100,
                    fmt=MARKERS.get(r.proposer, "o"),
                    color=COLORS.get(r.patience, COLORS.get(str(r.patience), "k")),
                    markersize=(10 if r.loop_budget == 20 else 7),
                    markeredgecolor="black" if r.is_baseline else "none",
                    markeredgewidth=2 if r.is_baseline else 0,
                    capsize=3, zorder=3)
        # alternate the offset so the low-energy cluster does not overprint
        dy = 9 if (_i % 2 == 0) else -13
        ax.annotate(_label(r), (r.E_mean / 1000, r.acc_mean * 100),
                    textcoords="offset points", xytext=(7, dy), fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    front = cells[cells.on_frontier].sort_values("E_mean")
    ax.plot(front.E_mean / 1000, front.acc_mean * 100, "k--", lw=1.2,
            zorder=2, label="Pareto frontier")

    ax.set_xlabel("Session energy $E_{total}$ (kJ)")
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("Energy/accuracy Pareto frontier\n"
                 "marker: proposer  ·  colour: patience  ·  size: loop budget  ·  "
                 "outlined: baseline", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = out / "fig1_pareto.png"
    fig.savefig(p, dpi=200)
    fig.savefig(out / "fig1_pareto.pdf")
    plt.close(fig)
    return p


def fig_decomposition(tidy: pd.DataFrame, out: Path) -> Path:
    g = tidy.groupby(cell_keys(tidy), dropna=False)[["E_prop_J", "E_train_J"]].mean().reset_index()
    labels = [f"{r.proposer}\n{r.patience}\nb{int(r.loop_budget)}" for _, r in g.iterrows()]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(labels, g.E_train_J / 1000, label="$E_{train}$ (GPU0)", color="#4c72b0")
    ax.bar(labels, g.E_prop_J / 1000, bottom=g.E_train_J / 1000,
           label="$E_{prop}$ (GPU1)", color="#dd8452")
    ax.set_ylabel("Energy (kJ)")
    ax.set_title("Session energy decomposition by physical device")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out / "fig2_decomposition.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_power_trace(run_dir: Path, out: Path, gpu_train: int = 0,
                    gpu_prop: int = 1) -> Path | None:
    f = run_dir / "nvml.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    t0 = df.t_wall.min()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for dev, name, c in ((gpu_train, "GPU0 — training", "#4c72b0"),
                         (gpu_prop, "GPU1 — proposer", "#dd8452")):
        d = df[df.dev == dev]
        ax.plot(d.t_wall - t0, d.power_mw / 1000, lw=0.8, label=name, color=c)
    ax.set_xlabel("time since session start (s)")
    ax.set_ylabel("board power (W)")
    ax.set_title(f"Per-device power trace — {run_dir.name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig3_power_trace.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_waste(iters: pd.DataFrame, out: Path) -> Path:
    d = iters.copy()
    d["bucket"] = d.decision.fillna("rejected")
    g = (d.groupby(cell_keys(d) + ["bucket"])["E_iter_J"].sum().unstack(fill_value=0) / 1000)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    g.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("Energy (kJ)")
    ax.set_title("Where the energy went, by iteration outcome")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out / "fig4_waste.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_trajectories(iters: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for (run, prop, pat), d in iters.groupby(["run", "proposer", "patience"]):
        d = d.sort_values("iter")
        ax.plot(d["iter"], d.val_acc.cummax() * 100, lw=1,
                color=COLORS.get(pat, COLORS.get(str(pat), "k")), alpha=0.6,
                ls="-" if prop == "dense" else "--")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best validation accuracy so far (%)")
    ax.set_title("Search trajectories (solid: dense, dashed: MoE)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig5_trajectories.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy", required=True)
    ap.add_argument("--iterations", default=None)
    ap.add_argument("--trace-run", default=None, help="run dir for the power-trace figure")
    ap.add_argument("--out-dir", default="figures")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy = pd.read_csv(a.tidy)
    made = [fig_pareto(tidy, out)]
    if {"E_prop_J", "E_train_J"} <= set(tidy.columns) and tidy.E_prop_J.notna().any():
        made.append(fig_decomposition(tidy, out))
    if a.iterations and Path(a.iterations).exists():
        it = pd.read_csv(a.iterations)
        if "E_iter_J" in it.columns:
            made.append(fig_waste(it, out))
        if "val_acc" in it.columns:
            made.append(fig_trajectories(it, out))
    if a.trace_run:
        made.append(fig_power_trace(Path(a.trace_run), out))
    print("\n".join(str(p) for p in made if p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
