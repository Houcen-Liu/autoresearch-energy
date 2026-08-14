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
| `gqm.drawio.pdf`, `infra.drawio.pdf` | drawn by hand | definition, execution |
| `autoresearch_loop.drawio.pdf` | editable source `sources/autoresearch_loop.drawio`; workspace export via `figures/export_figures.ps1` | background |
| `fig1_pareto.pdf` | `analysis/figures.py` on Phase 1 | results, RQ2 |
| `fig2_decomposition.png` | `analysis/figures.py` on Phase 1 | results, mechanism |
| `fig3_reasoning_decomposition.png` | `analysis/figures.py` on Stage 2a | results, reasoning |
| `fig4_waste.png`, `fig5_trajectories.png`, `fig6_diagnostics.png` | `analysis/figures.py` on Phase 1 | results, outcome accounting, retained progress, diagnostics |

Regenerate the measured figures with:

```bash
python analysis/figures.py --tidy <results>/tidy.csv \
       --iterations <results>/iterations.csv --out-dir <results>/figures
```

Regenerate the editable conceptual diagrams from the workspace root:

```powershell
powershell -ExecutionPolicy Bypass -File figures\export_figures.ps1
bash sync_overleaf.sh
```

The tracked `sources/autoresearch_loop.drawio` copy lets the editable source
travel with the GitHub repository. In the full Windows workspace,
`figures/autoresearch_loop.drawio` is canonical; keep the two files identical
when the diagram changes.

## Verification

After regenerating an asset, rebuild the manuscript and inspect every affected
page. The current 16-page build includes Figures 1--9 and was checked for float
placement, clipping, undefined references, and caption-to-asset consistency.
