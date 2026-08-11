# Evaluating self-hosted LLM architectures under an agentic AutoML workload

Energy and accuracy of a sparse Mixture-of-Experts proposer against a dense sibling,
driving an `autoresearch`-style agentic AutoML loop on hardware you own.

S2 Research Project, VU Amsterdam — Houcen Liu.

---

## What this is

A measurement harness, not a model. It re-implements the `autoresearch` protocol — one
editable training file, a fixed wall-clock training budget, one scalar metric, keep or
revert — with loop control in Python so that patience and loop budget are *enforced*
rather than requested, and with per-GPU energy attribution so that proposer joules and
training joules are separate measurements rather than modelled estimates.

| Document | Read it for |
|---|---|
| `EXPERIMENT_PLAN.md` | the design, the 15 recorded deviations from the proposal, gates, risk register |
| `SERVER_RUNBOOK.md` | what to type on the server, in order, day by day |
| `pilot_results/` | evidence from the laptop pilot that set the fixed variables |

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd experiment
python -m pytest tests/ -q                    # 46 tests, no GPU or model needed
python workload/prepare_cifar.py --data-dir ./data
python scripts/preflight.py --profile profiles/server.yaml
```

Then follow `SERVER_RUNBOOK.md` from Day 1. **Do not skip the gates** — each one exists
because the pilot hit the failure it prevents.

To exercise the whole analysis pipeline with no hardware at all:

```bash
python scripts/make_synthetic_runs.py --out-dir ../experiments/synthetic
for d in ../experiments/synthetic/run_*; do python measurement/energy_align.py --run-dir "$d"; done
python analysis/aggregate.py --experiments-dir ../experiments/synthetic
python analysis/figures.py   --tidy ../experiments/synthetic/tidy.csv \
                             --iterations ../experiments/synthetic/iterations.csv \
                             --out-dir ../experiments/synthetic/figures
```

The numbers are invented; the schema, file layout and code paths are the real ones.

---

## Layout

```
experiment/
  RunnerConfig.py       experiment-runner config: one run = one agent session
  profiles/             server.yaml (2x RTX 4000 Ada) and pilot.yaml (single GPU)
  harness/              the agent loop: propose -> guard -> train -> keep/revert
    agent_loop.py         loop control, patience and budget enforced in code
    proposer.py           OpenAI-compatible client (vLLM, llama.cpp, Ollama)
    guards.py             static checks; rejects a proposal before it costs a GPU run
    errors.py             infra failure vs agent failure — they are not the same thing
    recipe_repo.py        git-backed mutation history; revert is a checkout
  workload/
    train.py              THE FILE THE AGENT EDITS — fixed variable once calibrated
    train_reference.py    calibration only, never shown to the agent
    prepare_cifar.py      fixed splits; the test set is touched once per session
  measurement/          per-device NVML sampling, energy/iteration alignment
  analysis/             tidy tables, ANOVA/ART, Pareto frontier, figures
  scripts/              the gates: preflight, vram_check, headroom_check, noise_floor,
                        align_check, probe_proposer, smoke_test
  serving/              vLLM and llama.cpp launchers, plus the model-swap manager
  tests/                46 tests: loop semantics, guards, error taxonomy, analysis
experiment-runner/      the S2 framework, vendored and pinned
pilot_results/          calibration evidence from the laptop pilot
```

---

## The five things most likely to waste a night

All learned the hard way on the pilot; all cheap to check.

1. **Context length.** Left at a model's native default, a 4B model reserved 43 GB of KV
   cache and ran 66 % on CPU — every request hit the timeout. Pin `--max-model-len`.
2. **Thinking mode.** Vendor request fields for disabling reasoning are silently ignored
   by OpenAI-compatible endpoints. Verify with `probe_proposer.py`; check
   `params_reduced` and `thinking_tokens` in the log rather than trusting the config.
3. **Cooldown.** 30 s between runs cut the apparent noise floor by 2.7×. Without it you
   measure thermal drift and call it variance.
4. **EPS below the noise floor.** Measure it (`noise_floor.py`), never assume it. At
   0.001 against a 0.4 pp SD, "kept mutation" means nothing.
5. **Scaffolding in the analysis.** Stub and synthetic-data sessions score ~0.10 by
   construction. They are flagged and quarantined automatically — keep it that way.

---

## Provenance

- `autoresearch` protocol after Karpathy, <https://github.com/karpathy/autoresearch> (MIT)
- `experiment-runner` — Karsten et al., S2 group, VU Amsterdam
- `EnergiBridge` — Sallou et al.
