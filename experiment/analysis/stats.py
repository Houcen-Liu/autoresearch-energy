"""Hypothesis testing, pre-registered in EXPERIMENT_PLAN.md section 7.

Effect sizes with confidence intervals are the PRIMARY output; p-values are
secondary and Holm-corrected across the three registered hypotheses. With n=3 per
cell, three-way interactions are not interpretable and are not tested.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FACTORS = ["proposer", "patience", "loop_budget"]
OUTCOMES = ["E_gpu_total_J", "E_per_kept_J", "test_acc"]


# ------------------------------------------------------------------ effect size
def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def cliffs_delta_ci(a, b, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    boots = [cliffs_delta(rng.choice(a, len(a), replace=True),
                          rng.choice(b, len(b), replace=True)) for _ in range(n_boot)]
    return tuple(np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def interpret_delta(d: float) -> str:
    ad = abs(d)
    return ("negligible" if ad < 0.147 else "small" if ad < 0.33
            else "medium" if ad < 0.474 else "large")


# ------------------------------------------------------------------ assumptions
def check_assumptions(df: pd.DataFrame, outcome: str) -> dict:
    groups = [g[outcome].dropna().values for _, g in df.groupby(FACTORS, dropna=False)]
    groups = [g for g in groups if len(g) >= 3]
    resid = np.concatenate([g - g.mean() for g in groups]) if groups else np.array([])
    out = {"n_cells": len(groups)}
    if len(resid) >= 3:
        w, p = stats.shapiro(resid)
        out["shapiro_W"], out["shapiro_p"], out["normal"] = float(w), float(p), bool(p > 0.05)
    if len(groups) >= 2:
        s, p = stats.levene(*groups)
        out["levene_p"], out["homoscedastic"] = float(p), bool(p > 0.05)
    out["use_art"] = not (out.get("normal", False) and out.get("homoscedastic", True))
    return out


def aligned_rank_transform(df: pd.DataFrame, outcome: str, effect: tuple[str, ...]) -> pd.Series:
    """ART for one effect: subtract all other effects' cell means, then rank."""
    y = df[outcome].astype(float)
    grand = y.mean()
    cell_means = df.groupby(FACTORS)[outcome].transform("mean")
    effect_mean = df.groupby(list(effect))[outcome].transform("mean")
    residual = y - cell_means
    aligned = residual + effect_mean - grand
    return stats.rankdata(aligned)


def anova(df: pd.DataFrame, outcome: str, use_art: bool = False) -> pd.DataFrame:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    d = df.dropna(subset=[outcome]).copy()
    for f in FACTORS:
        d[f] = d[f].astype(str)
    formula_terms = FACTORS + [f"{a}:{b}" for a, b in itertools.combinations(FACTORS, 2)]

    if not use_art:
        d["_y"] = d[outcome].astype(float)
        model = ols("_y ~ " + " + ".join(f"C({t})" if ":" not in t else
                                         ":".join(f"C({x})" for x in t.split(":"))
                                         for t in formula_terms), data=d).fit()
        tbl = sm.stats.anova_lm(model, typ=2)
        ss_resid = tbl.loc["Residual", "sum_sq"]
        tbl["partial_eta_sq"] = tbl["sum_sq"] / (tbl["sum_sq"] + ss_resid)
        return tbl

    rows = []
    for term in formula_terms:
        eff = tuple(term.split(":"))
        d["_y"] = aligned_rank_transform(d, outcome, eff)
        model = ols("_y ~ " + " * ".join(f"C({f})" for f in FACTORS), data=d).fit()
        tbl = sm.stats.anova_lm(model, typ=2)
        key = ":".join(f"C({x})" for x in eff)
        match = [ix for ix in tbl.index if set(ix.split(":")) == set(key.split(":"))]
        if match:
            r = tbl.loc[match[0]]
            ss_resid = tbl.loc["Residual", "sum_sq"]
            rows.append({"term": term, "F": r["F"], "p": r["PR(>F)"],
                         "partial_eta_sq": r["sum_sq"] / (r["sum_sq"] + ss_resid)})
    return pd.DataFrame(rows).set_index("term")


# ------------------------------------------------------------------ hypotheses
HYPOTHESES = [
    ("H_prop", "proposer", ["E_gpu_total_J", "test_acc"]),
    ("H_pat", "patience", ["E_wasted_J", "E_per_kept_J"]),
    ("H_bud", "loop_budget", ["E_per_kept_J"]),
]


def contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, factor, outcomes in HYPOTHESES:
        levels = sorted(df[factor].dropna().unique())
        if len(levels) != 2:
            continue
        a_df, b_df = df[df[factor] == levels[0]], df[df[factor] == levels[1]]
        for out in outcomes:
            if out not in df.columns:
                continue
            a, b = a_df[out].dropna(), b_df[out].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            d = cliffs_delta(a, b)
            lo, hi = cliffs_delta_ci(a, b)
            rows.append({
                "hypothesis": name, "factor": factor, "outcome": out,
                "level_a": levels[0], "level_b": levels[1],
                "mean_a": a.mean(), "mean_b": b.mean(),
                "median_a": a.median(), "median_b": b.median(),
                "pct_change": 100 * (b.mean() - a.mean()) / a.mean() if a.mean() else np.nan,
                "cliffs_delta": d, "delta_ci_lo": lo, "delta_ci_hi": hi,
                "magnitude": interpret_delta(d), "U": u, "p_raw": p,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p_raw.values)
    return out


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment, returned in the input order."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy", required=True)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    df = pd.read_csv(a.tidy)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = {}
    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue
        assumptions = check_assumptions(df, outcome)
        report[outcome] = {"assumptions": assumptions}
        try:
            tbl = anova(df, outcome, use_art=assumptions["use_art"])
            tbl.to_csv(out / f"anova_{outcome}.csv")
            report[outcome]["anova"] = tbl.to_dict()
            report[outcome]["method"] = "ART-ANOVA" if assumptions["use_art"] else "ANOVA"
        except Exception as e:                                    # noqa: BLE001
            report[outcome]["anova_error"] = str(e)

    c = contrasts(df)
    c.to_csv(out / "contrasts.csv", index=False)
    (out / "stats_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(c.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
