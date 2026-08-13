# Overleaf bundle

Self-contained copy of the manuscript for compilation. **This is a copy** — the
working source is `../../greenlab_proposal/`. Edit there, then re-copy, or
Overleaf and the repository will drift apart.

## Compiling

Main file: `Liu_ProjectProposal.tex`. Class `acmart` (`sigconf`) is provided by
Overleaf, so no `.cls` file ships here. Bibliography is `IEEEtran` over
`references.bib`; the project needs two LaTeX passes plus BibTeX, which is
Overleaf's default recipe.

## Figures

| file | produced by | shown in |
|---|---|---|
| `gqm.drawio.pdf`, `infra.drawio.pdf`, `autoresearch_loop.drawio.pdf` | drawn by hand | definition, execution, background |
| `fig1_pareto.pdf` | `analysis/figures.py` on Phase 1 | results, RQ2 |
| `fig2_decomposition.png` | `analysis/figures.py` on Phase 1 | results, mechanism |
| `fig3_reasoning_decomposition.png` | `analysis/figures.py` on Stage 2a | results, reasoning |
| `fig4_waste.png`, `fig5_trajectories.png` | `analysis/figures.py` | not currently referenced |

Regenerate the measured figures with:

```bash
python analysis/figures.py --tidy <results>/tidy.csv \
       --iterations <results>/iterations.csv --out-dir <results>/figures
```

## Known things to check on the first compile

* Eight tables and three figures have never been through LaTeX. Watch float
  placement and whether any table overflows the two-column measure —
  `tab:anova`, `tab:contrasts` and `tab:reasoning` are the widest.
* `fig4_waste.png` and `fig5_trajectories.png` are unreferenced; either cite them
  or drop them from the bundle.
* Only `tab:headline` is referenced in the prose; the rest rely on placement.
