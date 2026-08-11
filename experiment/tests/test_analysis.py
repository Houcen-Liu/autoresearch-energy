"""Analysis pipeline on synthetic-but-real-shaped data."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from measurement.energy_align import align                        # noqa: E402
from scripts.make_synthetic_runs import make_run                  # noqa: E402


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    import itertools
    n = 0
    for rep in range(3):
        for arm, pat, bud in itertools.product(["dense", "moe"],
                                               ["greedy", "patience3"], [10, 20]):
            make_run(root / f"run_{n}_repetition_{rep}", arm, pat, bud, rep,
                     seed=n * 13 + rep, train_seconds=120.0)
            n += 1
    for d in root.glob("run_*"):
        align(d, 0, 1)
    return root


def test_alignment_reconstructs_the_session(tree):
    import json
    for d in list(tree.glob("run_*"))[:4]:
        e = json.loads((d / "energy_summary.json").read_text())
        assert e["E_gpu_total_J"] > 0
        assert e["E_train_J"] > e["E_prop_J"]        # training dominates
        assert e["alignment_ok"]
        assert e["E_wasted_J"] >= 0


def test_aggregate_and_pareto(tree):
    from analysis.aggregate import collect
    from analysis.pareto import cell_summary
    tidy, iters, quar = collect(tree)
    assert len(tidy) == 24
    assert len(quar) == 0
    assert iters.E_iter_J.sum() > 0
    cells = cell_summary(tidy)
    assert len(cells) == 8
    assert cells.on_frontier.sum() >= 1
    assert cells.is_baseline.sum() == 1


def test_stats_runs(tree):
    from analysis.aggregate import collect
    from analysis.stats import check_assumptions, contrasts
    tidy, _, _ = collect(tree)
    a = check_assumptions(tidy, "E_gpu_total_J")
    assert "use_art" in a
    c = contrasts(tidy)
    assert set(c.hypothesis) >= {"H_prop", "H_pat"}
    assert c.cliffs_delta.notna().all()
    assert "p_holm" in c.columns
