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
import numpy as np
import pandas as pd                                               # noqa: E402

import sys                                                        # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pareto import CELL_KEYS, cell_keys, cell_summary                        # noqa: E402

def _subtitle(cells) -> str:
    """Describe only the encodings this experiment actually uses.

    A fixed subtitle promised "colour: patience" on a Stage-2 figure that varies
    no patience, which is worse than saying nothing: it tells the reader to look
    for a distinction that is not in the plot.
    """
    parts = []
    if "proposer" in cells.columns and cells.proposer.nunique() > 1:
        parts.append("marker: proposer")
    if "patience" in cells.columns and cells.patience.nunique() > 1:
        parts.append("colour: patience")
    if "thinking_requested" in cells.columns and cells.thinking_requested.nunique() > 1:
        parts.append("colour: reasoning")
    if "temperature" in cells.columns and cells.temperature.nunique() > 1:
        parts.append("colour: temperature")
    if "loop_budget" in cells.columns and cells.loop_budget.nunique() > 1:
        parts.append("size: loop budget")
    if "is_baseline" in cells.columns and cells.is_baseline.any():
        parts.append("outlined: baseline")
    return "  ·  ".join(parts)


def _enc(row, field, table, default):
    """Look a value up in an encoding table, tolerating an absent factor."""
    v = getattr(row, field, None)
    if v is None:
        return default
    return table.get(v, table.get(str(v), default))


MARKERS = {"dense": "o", "moe": "s"}
# The run table names the levels "greedy"/"patience3"; the tidy table stores
# the enforced integer (1/3). Key on both, or every lookup silently falls
# through to black and the figure's own subtitle promises an encoding that is
# not there -- which is how the first Phase-1 figure shipped.
ALT_COLORS = {"off": "#1f77b4", "on": "#d62728",
              0.0: "#1f77b4", 0.4: "#55a868", 0.7: "#d62728", 1.0: "#8172b3",
              "0.0": "#1f77b4", "0.4": "#55a868", "0.7": "#d62728", "1.0": "#8172b3"}
COLORS = {"greedy": "#1f77b4", "patience3": "#d62728",
          "1": "#1f77b4", "3": "#d62728", 1: "#1f77b4", 3: "#d62728"}
SIZES = {10: 70, 20: 170}


def _label(r) -> str:
    """Cell label built from whichever factors the experiment varied."""
    bits = []
    for field, fmt in (("proposer", str), ("patience", str),
                       ("thinking_requested", lambda v: f"think-{v}"),
                       ("temperature", lambda v: f"T{v}"),
                       ("loop_budget", lambda v: f"b{int(v)}")):
        v = getattr(r, field, None)
        if v is not None:
            bits.append(fmt(v))
    return "/".join(bits)


def fig_pareto(tidy: pd.DataFrame, out: Path) -> Path:
    """Energy/accuracy frontier, legible at column width.

    The earlier version encoded proposer, patience and loop budget in three
    simultaneous channels, labelled all eight cells, and drew full error bars on
    each -- at 8 cm wide the labels collided and the bars merged into a grid.
    Here the headline factor (proposer) gets the strong channel (colour and
    marker), loop budget gets size, and only the frontier and the baseline are
    labelled; the remaining cells are identified in Table 4. Error bars are drawn
    thin and behind the markers so they inform without dominating.
    """
    cells = cell_summary(tidy)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.scatter(tidy["E_gpu_total_J"] / 1000, tidy["test_acc"] * 100,
               c="0.85", s=18, zorder=1, label="individual sessions")

    arm_style = {"dense": ("#1f77b4", "o"), "moe": ("#d62728", "s")}
    for _, r in cells.iterrows():
        arm = getattr(r, "proposer", None)
        col, mk = arm_style.get(arm, ("0.3", "^"))
        ax.errorbar(r.E_mean / 1000, r.acc_mean * 100,
                    xerr=(r.E_sd or 0) / 1000, yerr=(r.acc_sd or 0) * 100,
                    fmt="none", ecolor=col, elinewidth=0.8, alpha=0.45, zorder=2)
        ax.plot(r.E_mean / 1000, r.acc_mean * 100, mk, color=col,
                markersize=(11 if getattr(r, "loop_budget", None) == 20 else 7),
                markeredgecolor="black" if getattr(r, "is_baseline", False) else "white",
                markeredgewidth=1.6 if getattr(r, "is_baseline", False) else 0.6,
                zorder=4)

    front = cells[cells.on_frontier].sort_values("E_mean")
    ax.plot(front.E_mean / 1000, front.acc_mean * 100, "k--", lw=1.1, zorder=3,
            label="Pareto frontier")

    # Label only what the reader must identify: frontier cells and the baseline.
    to_label = cells[cells.on_frontier | cells.get("is_baseline", False)]
    for i, (_, r) in enumerate(to_label.sort_values("E_mean").iterrows()):
        ax.annotate(_label(r), (r.E_mean / 1000, r.acc_mean * 100),
                    textcoords="offset points",
                    xytext=(0, 14 if i % 2 == 0 else -20),
                    ha="center", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7",
                              alpha=0.9),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.5"))

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker=m, ls="", markersize=8,
                      markeredgecolor="white", label=a)
               for a, (c, m) in arm_style.items()
               if "proposer" in cells.columns and a in set(cells.proposer)]
    handles += [Line2D([], [], color="0.85", marker="o", ls="", markersize=5,
                       label="individual sessions"),
                Line2D([], [], color="k", ls="--", lw=1.1, label="Pareto frontier")]
    if "loop_budget" in cells.columns and cells.loop_budget.nunique() > 1:
        handles += [Line2D([], [], color="0.4", marker="o", ls="", markersize=5,
                           label="budget 10"),
                    Line2D([], [], color="0.4", marker="o", ls="", markersize=9,
                           label="budget 20")]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.95)

    ax.set_xlabel("Session energy $E_{total}$ (kJ)")
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("Energy/accuracy Pareto frontier")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p_ = out / "fig1_pareto.png"
    fig.savefig(p_, dpi=200)
    fig.savefig(p_.with_suffix(".pdf"))
    plt.close(fig)
    return p_


def fig_decomposition(tidy: pd.DataFrame, out: Path) -> Path:
    g = tidy.groupby(cell_keys(tidy), dropna=False)[["E_prop_J", "E_train_J"]].mean().reset_index()
    labels = [_label(r).replace("/", "\n") for _, r in g.iterrows()]
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
    keys = cell_keys(d)
    # Divide by the number of sessions in the cell. Summing over repetitions
    # produced totals three times any session's energy, which does not compare
    # with any other figure or table in the study.
    n_sessions = d.groupby(keys)["run"].nunique()
    g = (d.groupby(keys + ["bucket"])["E_iter_J"].sum().unstack(fill_value=0) / 1000)
    g = g.div(n_sessions, axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    g.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("Energy per session (kJ)")
    ax.set_title("Where the energy went, by iteration outcome")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out / "fig4_waste.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_trajectories(iters: pd.DataFrame, out: Path,
                     baselines: dict | None = None) -> Path:
    """Median trajectory per arm with an interquartile band.

    Twenty-four individual step functions on one axis is not readable: the lines
    overlap, share a colour scale nobody can decode, and the eye cannot recover
    a central tendency. What the figure needs to show is *when* improvement
    happens and *how much*, per arm. A median with an IQR band shows exactly
    that, with the session count stated so the reader knows what is being
    summarised.

    Improvement is expressed relative to each session's own baseline, because
    baselines vary by ~1 pp between sessions and absolute curves inherit that
    spread for no reason.
    """
    grp = [c for c in ("run", "proposer") if c in iters.columns]
    if "proposer" not in grp:
        return _fig_trajectories_single(iters, out, baselines)

    curves = {}
    for key, d in iters.groupby(grp):
        info = dict(zip(grp, key if isinstance(key, tuple) else (key,)))
        run, arm = info.get("run"), info.get("proposer")
        base = (baselines or {}).get(run)
        if base is None:
            continue
        d = d[d["iter"] > 0].sort_values("iter")
        running, series = base, []
        for v in d.val_acc:
            if pd.notna(v) and v > running:
                running = float(v)
            series.append((running - base) * 100)
        curves.setdefault(arm, []).append(pd.Series(series, index=d["iter"].values))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style = {"dense": ("#1f77b4", "-"), "moe": ("#d62728", "--")}
    for arm, ss in sorted(curves.items()):
        m = pd.concat(ss, axis=1).sort_index()
        m = m.ffill()                      # a finished session holds its value
        med, lo, hi = m.median(axis=1), m.quantile(0.25, axis=1), m.quantile(0.75, axis=1)
        c, ls = style.get(arm, ("k", "-"))
        ax.plot(med.index, med.values, color=c, ls=ls, lw=2,
                label=f"{arm} (n={m.shape[1]})")
        ax.fill_between(med.index, lo.values, hi.values, color=c, alpha=0.15, lw=0)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("iteration")
    ax.set_ylabel("improvement over session baseline (pp)")
    ax.set_title("Search trajectories: median and interquartile range")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p_ = out / "fig5_trajectories.png"
    fig.savefig(p_, dpi=200)
    fig.savefig(p_.with_suffix(".pdf"))
    plt.close(fig)
    return p_


def _fig_trajectories_single(iters, out, baselines):
    """Fallback for an experiment with a single arm (Stage 2b)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ss = []
    for run, d in iters.groupby("run"):
        base = (baselines or {}).get(run)
        if base is None:
            continue
        d = d[d["iter"] > 0].sort_values("iter")
        running, series = base, []
        for v in d.val_acc:
            if pd.notna(v) and v > running:
                running = float(v)
            series.append((running - base) * 100)
        ss.append(pd.Series(series, index=d["iter"].values))
    if ss:
        m = pd.concat(ss, axis=1).sort_index().ffill()
        ax.plot(m.index, m.median(axis=1), color="#1f77b4", lw=2,
                label=f"median (n={m.shape[1]})")
        ax.fill_between(m.index, m.quantile(0.25, axis=1), m.quantile(0.75, axis=1),
                        color="#1f77b4", alpha=0.15, lw=0)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("iteration")
    ax.set_ylabel("improvement over session baseline (pp)")
    ax.set_title("Search trajectories: median and interquartile range")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p_ = out / "fig5_trajectories.png"
    fig.savefig(p_, dpi=200); plt.close(fig)
    return p_


def fig_diagnostics(tidy: pd.DataFrame, out: Path) -> Path:
    """Three checks adapted from the Green Lab course analysis template.

    (a) ENERGY vs WALL-CLOCK, fitted per arm. This separates two claims that are
        easy to conflate: "the sparse model is faster" and "the sparse model is
        more efficient". If both arms lay on one line, energy would be nothing
        but time in different units and the architectural result would reduce to
        a speed result. Different slopes mean different mean power, i.e. a real
        efficiency difference on top of the speed difference.

    (b) DISTRIBUTIONS per arm, which the analysis plan promises and the summary
        tables cannot show: whether a cell mean represents its sessions or is
        pulled by one outlier.

    (c) ENERGY vs RUN ORDER. The run table is randomised precisely so that
        thermal or environmental drift cannot align with a factor; this shows
        whether any drift is present at all across the campaign.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    arm_style = {"dense": "#1f77b4", "moe": "#d62728"}
    has_arm = "proposer" in tidy.columns and tidy.proposer.nunique() > 1

    ax = axes[0]
    for arm, d in (tidy.groupby("proposer") if has_arm else [("all", tidy)]):
        c = arm_style.get(arm, "0.3")
        x, y = d.wallclock_s / 60, d.E_gpu_total_J / 1000
        ax.scatter(x, y, s=26, color=c, alpha=0.8, label=arm)
        if len(d) >= 3:
            b, a = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(xs, a + b * xs, color=c, lw=1.4, ls="--")
            ax.annotate(f"{b*1000/60:.0f} W", (xs[-1], a + b * xs[-1]),
                        color=c, fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("session wall-clock (min)"); ax.set_ylabel("$E_{total}$ (kJ)")
    ax.set_title("(a) energy vs time; slope = mean power")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    if has_arm:
        groups = [(a, d.E_gpu_total_J / 1000) for a, d in tidy.groupby("proposer")]
        bp = ax.boxplot([g for _, g in groups], labels=[a for a, _ in groups],
                        patch_artist=True, widths=0.55)
        for patch, (a, _) in zip(bp["boxes"], groups):
            patch.set_facecolor(arm_style.get(a, "0.6")); patch.set_alpha(0.35)
        for i, (a, g) in enumerate(groups, start=1):
            ax.scatter(np.random.default_rng(0).normal(i, 0.04, len(g)), g,
                       s=18, color=arm_style.get(a, "0.3"), zorder=3, alpha=0.9)
    ax.set_ylabel("$E_{total}$ (kJ)")
    ax.set_title("(b) session distribution per arm")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    order = "seed" if "seed" in tidy.columns else None
    if order:
        for arm, d in (tidy.groupby("proposer") if has_arm else [("all", tidy)]):
            ax.scatter(d[order], d.E_gpu_total_J / 1000, s=26,
                       color=arm_style.get(arm, "0.3"), alpha=0.85, label=arm)
        if len(tidy) >= 3:
            b, a = np.polyfit(tidy[order], tidy.E_gpu_total_J / 1000, 1)
            xs = np.linspace(tidy[order].min(), tidy[order].max(), 20)
            ax.plot(xs, a + b * xs, color="0.35", lw=1.2, ls=":")
            r = np.corrcoef(tidy[order], tidy.E_gpu_total_J)[0, 1]
            ax.annotate(f"r = {r:+.2f}", (0.04, 0.92), xycoords="axes fraction",
                        fontsize=8)
    ax.set_xlabel("run order"); ax.set_ylabel("$E_{total}$ (kJ)")
    ax.set_title("(c) drift across the campaign")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    p_ = out / "fig6_diagnostics.png"
    fig.savefig(p_, dpi=200)
    plt.close(fig)
    return p_


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
    if {"wallclock_s", "E_gpu_total_J"} <= set(tidy.columns):
        made.append(fig_diagnostics(tidy, out))
    if {"E_prop_J", "E_train_J"} <= set(tidy.columns) and tidy.E_prop_J.notna().any():
        made.append(fig_decomposition(tidy, out))
    if a.iterations and Path(a.iterations).exists():
        it = pd.read_csv(a.iterations)
        if "E_iter_J" in it.columns:
            made.append(fig_waste(it, out))
        if "val_acc" in it.columns:
            baselines = (tidy.set_index("run")["baseline_val_acc"].to_dict()
                         if {"run", "baseline_val_acc"} <= set(tidy.columns) else {})
            made.append(fig_trajectories(it, out, baselines))
    if a.trace_run:
        made.append(fig_power_trace(Path(a.trace_run), out))
    print("\n".join(str(p) for p in made if p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
