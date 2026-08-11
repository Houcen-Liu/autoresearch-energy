# experiment/ — build and run guide

Implementation of the design in `../EXPERIMENT_PLAN.md`. Read that first; this file
is the operating manual.

## Install

```bash
cd experiment
pip install -r requirements.txt
```

## Verify without any hardware (week 0)

Nothing here needs a GPU, a model, or the server.

```bash
pytest tests/ -v                                   # 17 tests: guards, patience, analysis

python scripts/make_synthetic_runs.py --out-dir ../experiments/synthetic
for d in ../experiments/synthetic/run_*; do
  python measurement/energy_align.py --run-dir "$d"
done
python analysis/aggregate.py --experiments-dir ../experiments/synthetic
python analysis/stats.py    --tidy ../experiments/synthetic/tidy.csv --out-dir ../experiments/synthetic
python analysis/pareto.py   --tidy ../experiments/synthetic/tidy.csv --out-dir ../experiments/synthetic
python analysis/figures.py  --tidy ../experiments/synthetic/tidy.csv \
                            --iterations ../experiments/synthetic/iterations.csv \
                            --trace-run ../experiments/synthetic/run_0_repetition_0 \
                            --out-dir ../experiments/synthetic/figures
```

The numbers are invented; the schema, file layout and code paths are the real ones.
If the headline Pareto figure comes out of this, the analysis half of the study is
already working before a single real session exists.

## Pilot on the laptop (RTX 3080 Laptop, 16 GB)

One GPU means training and the proposer share a device, so **energy attribution is
invalid** and the harness refuses to emit `E_prop`/`E_train` under `profiles/pilot.yaml`.
The pilot proves the pipeline runs; it does not produce data.

```bash
python workload/prepare_cifar.py --data-dir ./data       # ~170 MB, one time

# 1. no model at all — scripted proposer, exercises every code path
python scripts/smoke_test.py --profile profiles/pilot.yaml --stub --iterations 4

# 2. with a small local model (Ollama shown; llama.cpp-server works identically)
ollama serve &
ollama pull qwen3.6:4b
python scripts/smoke_test.py --profile profiles/pilot.yaml --iterations 4
```

Point `profiles/pilot.yaml` at whatever small instruct model fits alongside training.
Adjust `model_names` to match what your server reports at `/v1/models`.

## Server (2× RTX 4000 Ada)

Order matters. Gate G1 first — it is the one that can change the model choice.

```bash
# day 1
python scripts/preflight.py --profile profiles/server.yaml
python workload/prepare_cifar.py --data-dir ./data

# day 2 — GATE G1, the highest-risk item in the build
PROPOSER_GPU=1 DENSE_MODEL=... REVISION=... ./serving/serve_vllm.sh dense 8000 &
python scripts/vram_check.py --endpoint http://127.0.0.1:8000/v1 --model dense --gpu 1
PROPOSER_GPU=1 MOE_MODEL=... REVISION=... ./serving/serve_vllm.sh moe 8001 &
python scripts/vram_check.py --endpoint http://127.0.0.1:8001/v1 --model moe --gpu 1

# day 3 — GATE G2, baseline headroom
python scripts/smoke_test.py --profile profiles/server.yaml --calibrate

# day 3 — GATE G3, the agent can actually improve it
python scripts/smoke_test.py --profile profiles/server.yaml --proposer dense --iterations 5
python scripts/smoke_test.py --profile profiles/server.yaml --proposer moe   --iterations 5

# day 4 — GATES G4/G5 (alignment) and G6 (idle stability)
python scripts/align_check.py --run-dir <smoke run dir> --gpu-train 0 --gpu-prop 1
python measurement/idle_baseline.py --minutes 30

# day 5 onward — Phase 1
sudo AR_PROFILE=$PWD/profiles/server.yaml python ../experiment-runner/ RunnerConfig.py
```

`experiment-runner` writes to `../experiments/autoresearch_energy_phase1/`, resumes
where it left off if interrupted, and calls `analysis/aggregate.py` when the run
table is complete.

## What each piece is responsible for

| Path | Responsibility |
|---|---|
| `RunnerConfig.py` | one run = one session; cooldowns, instrumentation lifecycle, run-table row |
| `harness/agent_loop.py` | propose → guard → train → keep/revert, patience and budget enforced in code |
| `harness/guards.py` | rejects a proposal before it costs a 240 s training run |
| `harness/recipe_repo.py` | git-backed mutation history; revert is a checkout, never a textual undo |
| `workload/train.py` | the file the agent edits; fixed wall-clock budget |
| `measurement/nvml_sampler.py` | per-device energy counters — the attribution channel |
| `measurement/energy_align.py` | joins energy to iteration boundaries, asserts nothing is lost |
| `analysis/` | tidy tables → assumptions → ANOVA/ART → effect sizes → Pareto → figures |

## Things that will bite you

- **`sudo` changes the Python.** EnergiBridge needs root; run `sudo -E` or install
  into the system interpreter, or the harness will not find its imports.
- **`CUDA_VISIBLE_DEVICES` remaps indices.** The harness passes `cuda:0` to a child
  that already has `CUDA_VISIBLE_DEVICES` set to the training GPU. NVML indices are
  *physical* and unaffected — that mismatch is exactly what gate G5 checks.
- **Never edit `workload/train.py` after gate G2 passes.** It is a fixed variable of
  the experiment. Pin the commit and record the sha in the report.
- **The test split is sacred.** `final_eval.py` touches it once per session. Guard
  rule 4 exists to keep it that way; do not relax it.
