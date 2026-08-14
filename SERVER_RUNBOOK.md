# Server runbook — Phase 1 on the 2× RTX 4000 Ada machine

Companion to `EXPERIMENT_PLAN.md`. That document says *why*; this one says *what to type*,
in order, with the decision points marked. Every gate here exists because the pilot hit
the failure it prevents.

**Rule for the whole runbook:** if a gate fails, stop and fix it. Do not proceed "to see
if it works anyway". Four of the pilot's five wasted sessions came from skipping ahead.

---

## Before you touch the server

Clear these with Vincenzo first — they change what you run, not just how.

| # | Item | Why it blocks |
|---|---|---|
| 1 | D1: we re-implement the loop rather than orchestrate a coding agent | Biggest deviation from the reviewed proposal |
| 2 | D6: EPS becomes a measured quantity (~0.007), not 0.001 | Changes what "kept mutation" means |
| 3 | D15: training budget 45 s, not 240 s | Changes the schedule and the energy axis |
| 4 | D3 fallback: if the MoE will not fit, llama.cpp for both arms, or the Granite pair | Decide *before* G1, not during |
| 5 | Baseline cell: dense × greedy × 10 | Everything is reported as a delta from it |

Also settle: sustained overnight access, sudo (EnergiBridge needs root), and disk
(~50 GB for weights and traces).

---

## Day 1 — bring-up

### 1.1 Get the code and environment on the box

```bash
git clone <your-repo> ~/autoresearch-energy
cd ~/autoresearch-energy
bash setup_server.sh
```

**Install vLLM in a SEPARATE virtualenv.** vLLM pins its own torch build; installing
it beside the harness will replace the torch that training uses, and a silently
downgraded CUDA build is a miserable thing to debug at 2 a.m.

```bash
python3 -m venv .venv-serve
.venv-serve/bin/pip install --upgrade pip
.venv-serve/bin/pip install vllm            # or llama-cpp-python, per D3
.venv-serve/bin/python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
```

The serving scripts only need `vllm` on PATH, so activate `.venv-serve` in the shell
that serves and `.venv` in the shell that runs the experiment. Confirm the harness
venv is untouched afterwards:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Install EnergiBridge per its README and confirm it runs as root:

```bash
sudo energibridge --help
```

### 1.2 Tests first — they need no hardware

```bash
cd experiment && python -m pytest tests/ -q
```

**46 tests must pass.** If they don't, the code did not survive the transfer; fix that
before touching a GPU.

### 1.3 Data

```bash
python workload/prepare_cifar.py --data-dir ./data
```

### 1.4 Preflight

```bash
python scripts/preflight.py --profile profiles/server.yaml
```

Expect green on: torch+CUDA, **two distinct GPU indices**, NVML energy counter readable,
energibridge on PATH, disk space. The proposer endpoints will show red — nothing is
serving yet, which is correct at this point.

### 1.5 Confirm the device map

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

`profiles/server.yaml` assumes **GPU 0 = training, GPU 1 = proposer**. If the physical
layout differs, change the profile now — gate G5 will catch a mismatch later, but only
after you have burned a session.

---

## Day 2 — G1, the gate that can change the study

This is the highest-risk item in the whole build (D3). Do it before anything else is
built on top of the model choice.

### 2.1 Pin the weights

Fill in `experiment/serving/models.yaml`: exact repo, exact revision, exact quantizer for
both arms. **AWQ and GPTQ of the same model are not the same subject** — record which.

### 2.2 Serve the dense arm

```bash
export DENSE_MODEL=<pinned int4 repo>
export REVISION=<sha>
PROPOSER_GPU=1 bash serving/serve_vllm.sh dense 8000
```

### 2.3 Probe before you trust it

```bash
python scripts/probe_proposer.py --profile profiles/server.yaml \
    --model dense --endpoint http://127.0.0.1:8000/v1 --timeout 600
```

Read four lines specifically:

- **context length** — the pilot's single most expensive mistake. A 4B model reserved
  43 GB of KV cache at a 262 k context and ran 66 % on CPU. `serve_vllm.sh` pins
  `--max-model-len 16384`; confirm the probe agrees.
- **resident … GPU** — any CPU spill means minutes-long replies.
- **real prompt answered in Ns** — this is your per-iteration proposer cost. At 45 s
  training, anything over ~120 s means the proposer dominates the session.
- **extra_params accepted / rejected** — if `chat_template_kwargs` is rejected, thinking
  is **not** pinned and D9's confound is live. Fix it now.

### 2.4 VRAM stress, then repeat for the MoE

```bash
python scripts/vram_check.py --endpoint http://127.0.0.1:8000/v1 --model dense --gpu 1
python serving/manager.py --stop

export MOE_MODEL=<pinned int4 repo>
PROPOSER_GPU=1 bash serving/serve_vllm.sh moe 8001
python scripts/probe_proposer.py --profile profiles/server.yaml \
    --model moe --endpoint http://127.0.0.1:8001/v1 --timeout 600
python scripts/vram_check.py --endpoint http://127.0.0.1:8001/v1 --model moe --gpu 1
```

**GATE G1: 40 consecutive real-shape requests, no OOM, stable memory, on both arms.**

If the MoE fails: drop `--max-model-len` to 8192 first (our prompt is ~2.5 k). Still
failing → llama.cpp GGUF Q4_K_M **for both arms** (fairness rule, D3) → still failing →
the Granite 4.1 8B/30B backup pair, which turns the study into a size contrast rather
than a sparsity contrast. That is Vincenzo's call, made now.

### 2.5 Record what the models actually cost

Both probes print tokens/s and latency. Put them in the report — they are the first real
evidence about self-hosted proposer economics on this hardware.

---

## Day 3 — calibration, and freezing the fixed variables

### 3.1 Baseline sanity

```bash
python scripts/smoke_test.py --profile profiles/server.yaml --calibrate
```

The RTX 4000 Ada is slower than the pilot GPU, so expect fewer epochs in 45 s and a
baseline below the pilot's 0.76. Anywhere in 0.60–0.85 is fine.

### 3.2 GATE G11 — does the agent have signal to work with?

```bash
python scripts/headroom_check.py --profile profiles/server.yaml \
    --budgets 45,90 --repeats 2 --cooldown 120
```

Pilot values were +14.5 pp headroom at SNR 30 (45 s) and +16.3 pp at SNR 63 (240 s).
**Require SNR ≥ 3 and `instrument_ok: yes`.** If the reference recipe completes under
25 % of the baseline's steps the script refuses to recommend anything — that means the
reference is throughput-bound on this card (D14), and it needs shrinking before the
number means anything.

### 3.3 GATES G8′, G6b, G10 — noise floor, drift, utilisation

```bash
python scripts/noise_floor.py --profile profiles/server.yaml \
    --repeats 8 --train-seconds 45 --cooldown 120 \
    --out ../experiments/noise_floor_server_45s.json
```

Three numbers decide three things:

| Output | Gate | Action |
|---|---|---|
| `suggested_eps` | G8′ | write it into `profiles/server.yaml` → `loop.eps` |
| `step_drift_pct_total` + `spearman` | G6b | drift flagged → raise `cooldown_s` above 120 and repeat |
| `mean_util_pct` | G10 | below 50 % → `E_train` is idling, widen the baseline (D10) |

The pilot went from SD 1.01 pp to 0.37 pp purely by adding a 30 s cooldown. Do not skip
the cooldown here to save ten minutes.

### 3.4 Freeze

```bash
git add -A && git commit -m "server calibration: baseline, eps=<measured>, budget=45s"
git rev-parse HEAD          # this sha goes in the report
```

**After this commit, `workload/train.py` and `loop.eps` do not change.** They are fixed
variables. Changing them mid-experiment invalidates every session run before the change.

---

## Day 4 — measurement integrity

### 4.1 One real session per arm

```bash
python scripts/smoke_test.py --profile profiles/server.yaml --proposer dense --iterations 5
python scripts/smoke_test.py --profile profiles/server.yaml --proposer moe   --iterations 5
```

**GATE G3: at least one kept mutation, on both arms.**

Also read the error taxonomy in the summary. Contract violations and guard rejections are
*data* (the pilot's 4B model was guard-rejected on 4 of 6 iterations for emitting
fragments); infrastructure errors are *not*, and a session above 25 % infra rate is
quarantined automatically.

### 4.2 GATES G4 and G5 — energy attribution

```bash
python scripts/align_check.py --run-dir <the smoke run dir> --gpu-train 0 --gpu-prop 1
```

- **G4**: per-iteration energy + gaps reconstruct the session total.
- **G5**: during training GPU 0 draws ≫ GPU 1, and the reverse while proposing.

G5 failing means the device indices are crossed somewhere and **every attribution in the
study is wrong**. This is the check that protects the paper's central claim.

### 4.3 GATE G6 — idle stability

```bash
IDLE_DIR=../experiments/idle_before
mkdir -p "$IDLE_DIR"
energibridge -o "$IDLE_DIR/energibridge.csv" -i 100 --summary -- \
  python measurement/idle_baseline.py --minutes 30 --out-dir "$IDLE_DIR"
```

Record both GPU boards and the host counter with the proposer model resident in the
same state as the campaign. Gross energy remains the primary outcome. A matched
before/after pair permits an optional idle-subtracted sensitivity analysis; do not
subtract a single unmatched rate or silently replace the gross result.

### 4.4 Model swapping works

```bash
python serving/manager.py --arm dense --profile profiles/server.yaml
python serving/manager.py --arm moe   --profile profiles/server.yaml   # should swap
```

Both arms cannot be resident on one 20 GB card, and the run table is shuffled, so
`RunnerConfig` swaps the served model whenever the cell's arm changes. Time one swap —
it is dead time in every session that changes arm, roughly 1–3 minutes.

---

## Day 5 onward — Phase 1

### 5.1 Launch

```bash
cd ~/green-lab-experiment
sudo -E AR_PROFILE=$PWD/experiment/profiles/server.yaml \
    .venv/bin/python experiment-runner/ experiment/RunnerConfig.py
```

`sudo -E` preserves the environment; plain `sudo` uses a different Python and the imports
fail. 24 sessions, shuffled, ≥120 s cooldown, ~10 GPU-hours at the 45 s budget.

It resumes: `experiment-runner` skips rows already marked `DONE`, so an interrupted batch
picks up where it stopped.

### 5.2 Watch the first two sessions

The harness prints per-iteration progress (proposal latency, tokens, training, decision).
If the first session looks wrong, kill it — a wrong session is cheaper than 24 of them.

Watch for: proposal latency ≫ 45 s (proposer dominates), `thinking_tokens` non-zero
(thinking not pinned, D9), infra errors (serving unstable).

### 5.3 Roll the analysis forward daily

```bash
cd experiment
python analysis/aggregate.py --experiments-dir ../experiments/autoresearch_energy_phase1
```

Check `quarantine.csv` every morning. Quarantined sessions need re-running; both the
failure and the re-run go in the report.

**Do not change analysis decisions after seeing data.** The plan is the pre-registration.

### 5.4 When the run table is complete

```bash
python analysis/stats.py   --tidy ../experiments/.../tidy.csv --out-dir ../experiments/.../stats
python analysis/pareto.py  --tidy ../experiments/.../tidy.csv --out-dir ../experiments/.../stats
python analysis/figures.py --tidy ../experiments/.../tidy.csv \
    --iterations ../experiments/.../iterations.csv \
    --trace-run ../experiments/.../run_0_repetition_0 \
    --out-dir ../experiments/.../figures
IDLE_DIR=../experiments/idle_after
mkdir -p "$IDLE_DIR"
energibridge -o "$IDLE_DIR/energibridge.csv" -i 100 --summary -- \
  python measurement/idle_baseline.py --minutes 30 --out-dir "$IDLE_DIR"
```

The second matched idle measurement bounds drift across the campaign. Report the gross
measurement first and any idle-subtracted estimate as a labelled sensitivity analysis.

---

## Week 3 — Phase 2, CPU-only proposer

```bash
python serving/manager.py --stop
GGUF_PATH=<pinned Q4_K_M gguf> THREADS=$(nproc) bash serving/serve_cpu.sh dense 8000
```

Then run the 4 cells nearest the Phase-1 frontier at 2 repetitions.

**State this in the report:** on CPU, `E_prop` is a RAPL package measurement that also
contains OS and training-host work, so it is an *upper bound*. Subtract the measured
idle-plus-host baseline and report both raw and corrected numbers. This is a construct
change, not just a hardware change.

---

## The pilot's traps, as a checklist

Every one of these cost a wasted session. They are all cheap to check.

- [ ] **Context length** pinned (`--max-model-len 16384`). 262 k default → 43 GB KV cache → CPU spill → 600 s timeouts.
- [ ] **Thinking mode** verified *accepted*, not just configured. Check `params_reduced: false` in the session manifest and `thinking_tokens: 0` per iteration.
- [ ] **Endpoint is a URL**, not a model name. The client now refuses non-URLs, but check `proposer_config` in the log.
- [ ] **`max_tokens` ≥ 6144.** A full `train.py` rewrite is ~1500 tokens; truncation presents as "no fenced block" and is now diagnosed explicitly.
- [ ] **Timeouts exceed the training budget.** Derived automatically now (`budget × 1.25 + 120`), but check the profile reads sensibly.
- [ ] **Cooldown ≥ 120 s.** 30 s cut the pilot's apparent noise by 2.7×.
- [ ] **Scaffolding excluded.** Stub and synthetic-data sessions are flagged and quarantined; never let one into a cell mean.
- [ ] **`git status` clean** before each batch, so the run-table metadata records a real sha.

---

## If something goes wrong

| Symptom | First thing to check |
|---|---|
| Every proposal times out | `nvidia-smi` / probe → KV cache size, CPU spill |
| Every proposal is a contract violation | `rejected_*.txt` in the run dir → truncation vs fragments |
| Same crash repeatedly | Tracebacks are fed back now; if it persists the model cannot fix it — that is data |
| Session marked invalid | `invalid_reason` in `summary.json`; infra → re-run, agent-side → keep it |
| Energy numbers implausible | `align_check.py` → G4/G5 before believing anything |
| Model won't load after a swap | `experiments/vllm_<arm>.log` → almost always OOM |

---

## Stage 2a — reasoning as a factor (12 runs, ~5.5 h)

*Naming: the proposal's "Phase 2" is the CPU-only study. Reasoning mode is an
intra-model setting, so varying it is Stage 2. Stage 2a = reasoning,
Stage 2b = temperature.*

Feasibility already probed on both arms; `gate_evidence/probe_thinking_*.json`
holds the numbers. `max_tokens` was raised 8192 -> 12288 as a result.

```bash
cd ~/autoresearch-energy
AR_PROFILE=$PWD/experiment/profiles/server.yaml \
    .venv/bin/python experiment-runner/ experiment/RunnerConfigStage2a.py
```

Factors: `proposer` x `thinking`, 3 repetitions. `patience` and `loop_budget` are
fixed at greedy/10. Phase 1 did not show a clear benefit from additional
patience, and the pooled energy per retained mutation was lower at budget 10;
the exploratory long-horizon session below asks the separate trajectory question.

**Watch the manipulation check.** After each run the console prints
`thinking=<on|off>, reasoning tokens=<n>`. Expect ~0 for off and several
thousand for on. If a `thinking=on` run reports 0, the endpoint accepted the
parameter and ignored it: the run is marked invalid automatically, and you should
stop rather than collect a null result that is really a plumbing failure.

Expected per-session wall-clock, from the probe: dense 20 min (off) / 32 min
(on); MoE 15 min / 24 min.

---

## Probing CPU-only serving before scheduling it (~30 min)

The CPU phase estimate spans 7–20 h only because CPU throughput here is unknown.
Measure it first.

```bash
# build llama.cpp (once)
cd ~ && git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build -DGGML_NATIVE=ON && cmake --build build -j"$(nproc)"

# fetch GGUF weights for BOTH arms at the same quant (D3 fairness rule)
#   Qwen3-14B  Q4_K_M  ~9 GB
#   Qwen3-30B-A3B Q4_K_M ~18 GB
# then serve on CPU only:
./build/bin/llama-server -m <model>.gguf --host 127.0.0.1 --port 8080 \
    -c 16384 -ngl 0 -t "$(nproc)" --alias dense

# in another shell
cd ~/autoresearch-energy/experiment
../.venv/bin/python scripts/probe_cpu.py --model dense \
    --endpoint http://127.0.0.1:8080/v1
```

`-ngl 0` forces all layers onto CPU — verify with `nvidia-smi` that GPU memory
stays at zero, or you are measuring GPU serving with extra steps.

The probe reports tokens/s, projected session length, and whether the proposal
exceeds `request_timeout_s`. **The dense arm is the one at risk**: 14.8B dense
parameters are memory-bandwidth bound on CPU, and if a proposal takes longer than
600 s then every session dies on infrastructure errors rather than producing
data. Raise `request_timeout_s` and `time_budget_s` before running the phase if
the probe says so.

A plausible outcome is that the MoE is workable on CPU and the dense arm is not.
That asymmetry is itself the practitioner-facing result — ~3B active parameters
is exactly what makes CPU serving viable — and is worth reporting rather than
engineering around.

---

## Exploratory: how far can the loop climb? (~2.5 h, MoE)

Descriptive, n = 1. Phase 1 established that the loop can improve the baseline,
but its sessions ended after at most 20 iterations. This run asks whether a much
longer search keeps climbing, plateaus, or increasingly selects validation gains
that do not transfer to the test set.

```bash
# Shell A: serve the pinned MoE on GPU 1. Leave this running through the
# long-horizon session and both idle measurements.
cd ~/autoresearch-energy && source .venv-serve/bin/activate && export HF_HOME=~/hf-cache
pkill -u "$USER" -f "vllm serve" || true
sleep 8
PROPOSER_GPU=1 MOE_MODEL=QuixiAI/Qwen3-30B-A3B-AWQ \
REVISION=$(grep -A3 '^moe:' experiment/serving/models.yaml | grep revision | awk '{print $2}') \
  bash experiment/serving/serve_vllm.sh moe 8001

# Shell B: preflight and warm the exact request before opening a measured run.
cd ~/autoresearch-energy/experiment && source ../.venv/bin/activate
python scripts/preflight.py --profile profiles/server.yaml
python scripts/probe_proposer.py --profile profiles/server.yaml \
  --model moe --endpoint http://127.0.0.1:8001/v1 --timeout 600 \
  --max-tokens 8192

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$HOME/autoresearch-energy/experiments/long_horizon/moe_100it_seed0_$STAMP"
mkdir -p "$RUN_DIR/idle_before"
printf '%s\n' "$RUN_DIR" > "$HOME/autoresearch-energy/experiments/long_horizon/LATEST"
git -C .. rev-parse HEAD > "$RUN_DIR/code_commit.txt"
cp profiles/server.yaml "$RUN_DIR/server.yaml"
cp serving/models.yaml "$RUN_DIR/models.yaml"
python -m pip freeze > "$RUN_DIR/train_requirements.txt"
../.venv-serve/bin/python -m pip freeze > "$RUN_DIR/serve_requirements.txt"
nvidia-smi -q > "$RUN_DIR/nvidia-smi-q.txt"

# Matched idle baseline, with the MoE already resident.
energibridge -o "$RUN_DIR/idle_before/energibridge.csv" -i 100 --summary -- \
  python measurement/idle_baseline.py --minutes 5 \
    --out-dir "$RUN_DIR/idle_before" \
  2>&1 | tee "$RUN_DIR/idle_before.log"

# Measured 100-iteration session. The harness records both GPU boards; the
# wrapper records the AMD CPU-package counter.
set -o pipefail
energibridge -o "$RUN_DIR/energibridge.csv" -i 100 --summary -- \
  python -u scripts/long_horizon.py \
    --profile "$PWD/profiles/server.yaml" --proposer moe \
    --iterations 100 --patience 1 --seed 0 --thinking off \
    --max-tokens 8192 --history-max-rows 20 \
    --run-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/console.log"
RUN_RC=${PIPESTATUS[0]}

# Matched idle measurement before changing model residency.
mkdir -p "$RUN_DIR/idle_after"
energibridge -o "$RUN_DIR/idle_after/energibridge.csv" -i 100 --summary -- \
  python measurement/idle_baseline.py --minutes 5 \
    --out-dir "$RUN_DIR/idle_after" \
  2>&1 | tee "$RUN_DIR/idle_after.log"
if [ "$RUN_RC" -ne 0 ]; then
  echo "long-horizon session failed with exit code $RUN_RC; preserve this run directory and diagnose it"
  exit "$RUN_RC"
fi
```

Validate and analyse the finished session before stopping the server:

```bash
python measurement/energy_align.py --run-dir "$RUN_DIR" \
  --gpu-train 0 --gpu-prop 1 | tee "$RUN_DIR/energy_align.log"
python measurement/idle_subtract.py --run-dir "$RUN_DIR" \
  --gpu-train 0 --gpu-prop 1 | tee "$RUN_DIR/idle_subtract.log"
python analysis/trajectory.py --run-dir "$RUN_DIR" | tee "$RUN_DIR/trajectory.log"
python -c 'import json,sys; s=json.load(open(sys.argv[1])); print(json.dumps(s,indent=2)); assert s["iterations"]==100 and s["iterations_completed"]==100 and s["valid"] and not s["aborted_early"] and s["infra_errors"]==0 and s["thinking_tokens_total"]==0' \
  "$RUN_DIR/summary.json"
```

`idle_subtracted_summary.json` is a matched-idle **sensitivity analysis**, not
the primary result. Report gross measured energy first; do not clamp a negative
idle-subtracted component if the sensitivity calculation produces one.

`replay_keeps.py` is the scientifically interesting half. It checks every kept
revision out of the session's own git bundle, retrains it from scratch at the
same budget, and evaluates validation **and** test. That measures whether the
val–test gap grows as the agent makes more selections against a 5 000-image
validation split — i.e. whether the loop overfits the metric it selects on.
Stop vLLM only after the second idle trace, then replay the kept checkpoints:

```bash
pkill -u "$USER" -f "vllm serve" || true
python scripts/replay_keeps.py --profile profiles/server.yaml \
  --run-dir "$RUN_DIR" | tee "$RUN_DIR/replay.log"
test -f "$RUN_DIR/replay_keeps.json"
```

The run directory is experimental data, not source code. Archive it and copy it
back rather than adding it to Git:

```bash
RUN_NAME=$(basename "$RUN_DIR")
RUN_PARENT=$(dirname "$RUN_DIR")
tar --exclude="$RUN_NAME/recipe/.git" -C "$RUN_PARENT" \
  -czf "$RUN_PARENT/$RUN_NAME.tar.gz" "$RUN_NAME"
sha256sum "$RUN_PARENT/$RUN_NAME.tar.gz" \
  > "$RUN_PARENT/$RUN_NAME.tar.gz.sha256"
```
Nothing in the replay influenced any decision the agent made, so the
"test set touched once per session" rule is not broken.

Phase 1 shows no sign of such overfitting (gap flat at ~0.87 pp, r = −0.29
against kept count), but no session kept more than 4 mutations, so there has been
no real selection pressure yet. A positive slope here would be a genuine
methodological finding about agentic AutoML.
