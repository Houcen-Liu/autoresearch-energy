# Experiment & Code Plan — Self-Hosted LLM Architectures under an Agentic AutoML Workload

**Status:** build plan, v7 (2026-08-11) — subjects changed after G1 failed on the server (D16)
**Owner:** Houcen Liu
**Source of truth for design:** `greenlab_proposal/` (Liu_ProjectProposal.tex and sections)
**This document:** how the proposal becomes runnable code, plus the deviations that must be reported.

---

## 0. How to use this document

Sections 1–2 are decisions you should read before writing any code, and section 1 contains items
you must clear with Vincenzo. Sections 3–7 are the build spec: every file, its responsibility, its
inputs and outputs. Section 8 is the schedule. Section 9 lists the gates that decide whether the
main run may start. Section 10 is the risk register.

The companion scaffold lives in `experiment/`. Every file listed in section 4 exists there already,
either complete or as a clearly-marked stub with a `TODO(hw)` where real hardware is required.

---

## 1. Decisions and deviations from the proposal

Sixteen things changed between the proposal and reality. D1-D5 came from reading the `autoresearch`
repository and the available hardware; **D6-D15 came from running the pilot, and D16 from the server**, and D6 in
particular would have invalidated the headline analysis had it gone unnoticed. All of them need a
line in the final report's *Experiment Execution* and *Threats to Validity* sections.

### D1 — `autoresearch` has no agent loop to orchestrate. We write one.

The proposal says `experiment-runner` "launches the pinned, unmodified `autoresearch` as the target
program". The repository (`github.com/karpathy/autoresearch`, MIT, ~36 commits) contains only:

```
prepare.py      constants, data prep, eval utilities   (not modified)
train.py        the single file the agent edits
program.md      the instructions the agent reads
pyproject.toml
```

There is no runner, no loop, no keep/revert logic. The "agent" is whatever coding-agent CLI you
point at the directory (Claude Code, Codex). That is unusable as an experimental subject: patience
and loop budget would be prompt instructions a 27B local model can silently violate, and the CLI's
own scaffolding tokens would land inside the proposer energy measurement.

**Decision:** we implement the loop ourselves (`experiment/harness/`, ~500 lines), faithful to the
`autoresearch` protocol — propose a mutation to a single training file, run a fixed-wall-clock
training experiment, read one scalar metric, keep or revert — but with patience and budget enforced
in Python, deterministic logging, and no hidden agent scaffolding.

**Reporting consequence:** the framing "unmodified `autoresearch` as a black box" is no longer
accurate and should be replaced by: *"we re-implement the `autoresearch` protocol as an instrumented
harness, preserving its design choices (single editable file, fixed wall-clock training budget,
single scalar metric, keep/revert on that metric), because the reference implementation delegates
loop control to a third-party coding agent whose scaffolding would confound proposer energy."*
This is a strengthening of construct validity, not a weakening — say so explicitly. The
`program.md` we hand the proposer is derived from the upstream one at a pinned commit and shipped
in the replication package for comparison.

### D2 — Hardware is 2× RTX 4000 Ada. No RTX 4090.

Confirmed available: one server, two RTX 4000 Ada (20 GB each). The proposal's 4090 is gone, and so
is the idle third card that was going to serve as the swap card.

**Decision:** GPU 0 = inner training, GPU 1 = proposer serving. Per-GPU NVML attribution — the
methodological core of the design — is unaffected; it needs two physical devices, not two
particular devices.

**Why the schedule survives:** the training budget is *fixed wall-clock*, inherited from
`autoresearch`'s central design choice. A slower training GPU does not lengthen a session; it lowers
how many epochs the CNN completes inside the same budget. (The budget itself was later resolved to
45 s by measurement — see D15 — bringing Phase 1 to ≈ 10 GPU-hours: ≈ 16 min at budget 10 and ≈ 32
min at budget 20 per session.)

**Reporting consequence:** absolute accuracy numbers shift down slightly (fewer epochs in 240 s
than a 4090 would deliver). Comparability across cells is untouched, because every cell trains under
the same budget on the same device. State the device in the results, and note that the fixed-time
budget makes the design hardware-portable but the absolute metrics platform-specific — this is
exactly the trade-off Karpathy documents for `autoresearch`.

### D3 — No spare card: the MoE proposer must fit 20 GB, and it is tight.

`Qwen/Qwen3.6-35B-A3B` (MoE, ~3B active) at 4-bit is ≈ 18 GB of weights. On a 20 GB card that
leaves ~2 GB for KV cache, activations and the CUDA context. It will not serve at a long context
with default vLLM settings, and there is no longer a second idle Ada to fall back on.

**Decision:** this is the single highest-risk item in the build. Week 1 runs `scripts/vram_check.py`
before anything else, in this order:

1. `Qwen/Qwen3.6-35B-A3B`, AWQ/GPTQ int4, `--max-model-len 16384`, `--gpu-memory-utilization 0.94`,
   `--kv-cache-dtype fp8`, `--enforce-eager`. Target: serves and sustains 40+ requests of the real
   prompt shape without OOM.
2. If that fails: GGUF Q4_K_M via `llama.cpp` server for both proposers (better memory control,
   costs throughput; must then be used for *both* arms for fairness).
3. If that fails: fall back to the pre-registered backup pair, IBM Granite 4.1 8B vs 30B.

`nvidia/Qwen3.6-35B-A3B-NVFP4` is **not** an option: NVFP4 needs Blackwell (sm_100+); RTX 4000 Ada
is Ada Lovelace (sm_89). Official `Qwen/Qwen3.6-27B-FP8` is also out — FP8 weights of a 27B model
are ~28 GB. Int4 community quantizations are the only route; pin exact revisions and record the
quantizer, because AWQ and GPTQ of the same model are not the same subject.

**Fairness rule:** both arms must use the same quantization method, the same serving stack, the same
sampling parameters, and the same `--max-model-len`. If the MoE only fits under llama.cpp, the dense
model runs under llama.cpp too. Note this in the run-table metadata per run.

### D4 — Inner workload is the CIFAR-10 CNN, per the proposal.

Kept as reviewed with Vincenzo. Ground truth = CIFAR-10 test accuracy at session end; the in-loop
signal the agent optimizes = validation accuracy on a held-out 5 000-image split carved from the
50 000 training images with a fixed seed. The 10 000-image test set is touched **once per session**,
after the loop ends, and is never visible to the proposer.

For the record (worth one sentence in *Threats*): the schedule argument originally used to justify
swapping out `nanochat` does not actually hold, because `autoresearch`'s training budget is fixed
wall-clock and therefore hardware-independent. The swap remains defensible on other grounds — a
compact CNN gives a lower-variance, faster-saturating ground-truth signal at a 240 s budget than
`val_bpb` on a depth-8 GPT — but justify it that way rather than on cost. `nanochat` stays as the
first specified extension.

### D5 — Patience and loop budget get exact operational definitions.

The proposal names the levels but not their semantics. Locked here (section 5.2), enforced in code,
and asserted in unit tests, so that a reader can reproduce the loop from the paper alone.

### D6 — The keep/revert threshold must be measured, not assumed. (from the pilot)

Three runs of the **identical** baseline on the pilot machine, at a fixed seed, gave val_acc 0.7632,
0.7670, 0.7590: a spread of 0.8 accuracy points, SD ≈ 0.4 pp. Step counts differed too (8461 vs
8620).

This is not a bug and cannot be seeded away. `autoresearch` budgets training by **wall clock**, so
the number of optimisation steps depends on machine state. A fixed-time budget makes the recipe's
score a random variable by construction.

The original `EPS = 0.001` (0.1 pp) sat **four times below** that noise. Under it, a mutation that
changes nothing is kept roughly half the time — meaning kept mutations, `E/kept`, and the entire
patience factor would have been measuring noise rather than progress. This is the most valuable
thing the pilot found.

**Decision:** `EPS` becomes a measured quantity. `scripts/noise_floor.py` repeats the unmodified
baseline N times at a given budget and reports mean, SD, range, step variability, GPU utilisation,
and a suggested `EPS = 2 × SD`. It runs on the server before Phase 1, per candidate budget, and the
chosen value is a fixed variable reported alongside the noise floor that justifies it.

**Reporting consequence:** gate G8 (reproducibility) is *unsatisfiable by construction* and is
replaced by G8′ (noise floor). *Conclusion Validity* gains a paragraph: under a wall-clock budget,
run-to-run variance of the inner workload sets a floor on the smallest detectable effect, and every
reported effect is compared against it.

### D7 — Infrastructure failure and agent failure are different things. (from the pilot)

The pilot's real-model session recorded four `errored` iterations. All four were Ollama read
timeouts — our serving stack, not the proposer. Under the original coding that session would have
entered the run table as "this proposer produced four useless mutations," and one flaky night would
have become a finding about model architecture.

**Decision:** failures are classified (`harness/errors.py`):

| Class | Meaning | Treatment |
|---|---|---|
| `infra_timeout`, `infra_transport` | our stack failed | session quarantined above a 25 % rate, re-run, both facts recorded |
| `contract_violation` | model could not produce a parseable reply in budget | **data** — a dependent variable |
| `guard_rejection` | well-formed but broke a task rule | **data** |
| `train_crash`, `train_timeout` | the proposed recipe does not run | **data** |

Contract-compliance rate joins the dependent variables. For a study about the viability of
self-hosting, "can this model reliably emit a valid artifact at all" is a first-class result, not an
inconvenience.

### D8 — Proposer time is bounded per iteration, and every attempt is accounted. (from the pilot)

Three retries at a 300 s request timeout burned **15 minutes per failed iteration**, each retry
re-sending a growing prompt, and the failed iteration recorded **zero** tokens and zero latency — so
the most expensive iterations were invisible in the token and energy analysis. The pilot session
spent 62 minutes producing nothing.

**Decision:** one iteration gets a total proposer time budget across all attempts (`time_budget_s`,
900 s on the server); retries drop to 1 and use a *minimal repair prompt* rather than the full
history; every attempt records its tokens, latency and outcome whether it succeeded or not.

### D9 — Thinking mode is a pinned fixed variable. (from the pilot)

Qwen3.6 is a reasoning model. Thinking tokens are pure proposer energy that produces no artifact,
and reasoning models emit extra fenced code blocks that break a naive output contract. If the dense
and MoE arms differ in default thinking behaviour, that difference will swamp the sparsity effect
the study exists to measure.

**Decision:** thinking is set explicitly and identically for both arms
(`chat_template_kwargs.enable_thinking: false` on vLLM, `think: false` on Ollama), the full request
manifest is written into every session log, and the setting is reported as a fixed variable. The
response parser disambiguates multiple code blocks on the `TRAIN_SECONDS` marker rather than
failing.

### D10 — Open question: does the training GPU actually do work? (from the pilot)

The baseline peaked at **438 MB of VRAM**. If GPU utilisation during training is also low, then
`E_train` is dominated by idle power × time rather than by work done, and "energy wasted on reverted
mutations" is largely "energy spent idling" — which would make the energy story about elapsed time
rather than computation. `scripts/noise_floor.py` now reports mean utilisation and power during
training and warns below 50 %. If the server confirms low utilisation, widen the baseline (more
channels, larger batch) until the card is loaded, *before* freezing it as a fixed variable.

### D11 — Thermal drift is a first-order confound, not a footnote. (from the pilot)

Five back-to-back baseline runs with no cooldown gave step counts of 8500, 7652, 7159, 6486, 5780
(**-37 %**) and mean power of 115, 91, 81, 76, 72 W (**-37 %**), at a constant 88 % GPU utilisation.
That is sustained-load power/thermal throttling, and it means **identical work costs different
joules depending on where in a batch it runs**.

The proposal treated thermal drift as an internal-validity nuisance handled by a 60 s cool-down.
The pilot says it is larger than that. Randomised run order (already in the design) prevents drift
from *biasing* any factor, but it inflates within-cell variance, and with n=3 that directly costs
statistical power.

**Decision:** `noise_floor.py` now inserts a cooldown between repeats and reports any residual trend
separately from the scatter around it (`val_acc_sd_detrended`, `step_drift_pct_total`,
`power_slope_w_per_run`). Gate G6 is extended from idle stability to **loaded** stability: repeat
the baseline 8x with the intended cooldown and require throughput drift below 10 % across the
series. Temperature is recorded per run and reported. A workstation with blower cards should drift
far less than a laptop — but that must be *measured on the server*, not assumed.

### D12 — The training budget is chosen by signal-to-noise, not by tradition. (from the pilot)

Measured on the pilot machine, 5 repeats each with cooldown:

| budget | val_acc | SD | steps | GPU util |
|---|---|---|---|---|
| 45 s | 0.7580 | 1.01 pp | 7 115 | 88 % |
| 240 s | 0.7671 | 0.91 pp | 31 375 | 96 % |

**4.4x the training compute bought +0.91 pp against a pooled noise SD of 0.96 pp — an SNR of 0.95.**
The baseline is saturated: within this range, training compute does not buy accuracy.

*A correction to an earlier reading of this result.* Saturation does **not** degenerate the Pareto
analysis, because `loop_budget` is the number of *agent iterations*, not the training time per
iteration. More iterations mean more mutations tried and therefore a better recipe; that channel is
untouched by the baseline saturating. The training budget is a **fixed variable**, and what its
saturation actually costs is different and more subtle:

1. **It sets the noise floor** the agent must beat. SD ≈ 0.9 pp at both budgets, so EPS ≈ 0.017–0.020
   either way; only mutations worth ~2 pp are distinguishable from a re-run.
2. **A saturated baseline says nothing about an improved recipe.** The baseline is deliberately weak;
   what matters is whether the recipes the agent *reaches* can exploit the budget. Augmentation — the
   single most valuable move available — only repays over many epochs, so a budget that looks
   generous for the baseline can be far too short for the agent's best work.
3. **The real degeneracy risk is agent plateau, not compute saturation.** If the agent's wins all
   land in the first few iterations and everything after sits inside the noise band, budget 20 buys
   energy and no accuracy over budget 10, and the frontier flattens for that reason.

**Decision:** the budget is chosen by measuring the signal-to-noise ratio it gives the agent.
`scripts/headroom_check.py` runs the baseline and a strong reference recipe
(`workload/train_reference.py`: batch norm, random crop and flip, one-cycle schedule, wider
channels — calibration scaffolding, never shown to the agent) at each candidate budget, and reports

    headroom = mean(reference) - mean(baseline)      SNR = headroom / pooled SD

Pick the budget with the largest **SNR**, not the largest headroom. Below SNR ≈ 3 the agent cannot
tell its own improvements from noise and keep/revert decays toward coin flipping; the fix is then to
reduce noise (average `val_acc` over repeats inside the recipe, or evaluate on a larger split)
rather than to spend more joules. Gate G11 becomes this measurement.

### D13 — Drift detection needs a monotonicity test, not a slope. (from the pilot)

The 45 s series without cooldown was perfectly monotonic — steps 8500, 7652, 7159, 6486, 5780
(Spearman ρ = −1.00), power 115 → 72 W — genuine thermal throttling. With a 30 s cooldown at 240 s,
the series was 32837, 26400, 30355, 32510, 34771: one dip and a recovery, ρ = +0.40, no drift. A
least-squares slope alone called that "+12.7 % drift" and raised a false alarm.

**Decision:** drift requires **both** a material magnitude (>10 % across the series) **and** a
monotonic trend (|ρ| ≥ 0.8). Both pilot series are regression tests. The practical finding stands:
**a 30 s cooldown was enough to eliminate the throttling drift** on the pilot laptop, which is
strong evidence that the plan's 120 s inter-session cooldown is adequate on a better-cooled
workstation — to be confirmed by G6b on the server.

### D14 — The fixed wall-clock budget systematically penalises added capacity. (from the pilot)

A finding worth reporting in its own right, discovered by accident while calibrating.

The first reference recipe used for `headroom_check.py` was a larger network (64-512 channels,
~4.7M parameters, batch 256, mixed precision). Under the fixed wall-clock budget it lost to the
crippled baseline at **every** budget tested:

| budget | baseline | reference | reference steps | baseline steps | reference epochs |
|---|---|---|---|---|---|
| 20 s | 0.7544 | 0.5184 | 133 | 3 669 | 0 |
| 45 s | 0.7632 | 0.5172 | 296 | 8 211 | 1 |
| 90 s | 0.7623 | 0.6721 | 530 | 15 121 | 2–3 |
| 240 s | 0.7617 | 0.7238 | 1 343 | 34 142 | 7 |

25x slower per step, so it never trains to convergence and never gets to use its capacity. The gap
closes steadily with budget (−23.6 → −3.8 pp), implying a crossover somewhere past ~400 s.

**Two consequences.**

*For the report:* `autoresearch`'s fixed-time budget makes throughput a first-class part of recipe
quality. An agent optimising under it is rewarded for cheap-per-step changes and punished for
scaling capacity it cannot train — which is a genuine and non-obvious property of the design, and
directly relevant to the energy question, since it means the loop implicitly optimises for compute
efficiency rather than model size.

*For the calibration:* a bloated reference is not a valid stand-in for what a competent agent would
find, so the headroom measurement it produces is meaningless. `train_reference.py` now keeps the
baseline's architectural scale and improves only *how* it is trained (batch norm, cheap
per-batch augmentation, one-cycle schedule, AMP) at roughly 1.5x the per-step cost.
`headroom_check.py` gained an **instrument-validity check**: if the reference completes under 25 %
of the baseline's steps or fewer than 3 epochs, the run is rejected as measuring throughput rather
than recipe quality, and no budget recommendation is issued.

### D15 — RESOLVED: the training budget is 45 s, and a short budget makes the experiment *better*.

Gate G11 with a correctly-scaled reference recipe (pilot hardware, 2 repeats each, 30 s cooldown):

| budget | baseline | reference | headroom | noise | SNR | step ratio |
|---|---|---|---|---|---|---|
| 45 s | 0.7556 | 0.9008 | **+14.52 pp** | 0.48 pp | 30.2 | 0.87 |
| 240 s | 0.7608 | 0.9241 | **+16.33 pp** | 0.26 pp | 62.8 | 0.97 |

Both valid. The agent has ~15 accuracy points of headroom — far more than the ~1 pp noise floor — so
keep/revert has real signal at either budget. 240 s buys **1.8 pp more headroom for 5.3x the
joules**.

**Decision: 45 s.** Two reasons, and the second is the important one.

*Cost.* Phase 1 is 384 training runs (360 iterations + 24 baselines). At 45 s that is 4.8 GPU-hours
of training plus ~5 h of proposer inference, ≈ **10 h total**. At 240 s it is 25.6 h + 5 h ≈ **31 h**.
The short budget saves roughly 20 GPU-hours and two to three overnight batches on a single shared
machine.

*Sensitivity — the decisive argument.* The study's headline factor is the **proposer**, and a
shorter training budget raises the proposer's share of session energy:

| budget | proposer share of loop time |
|---|---|
| 45 s | ~53 % |
| 240 s | ~17 % |

At 240 s, a dense-vs-MoE difference in `E_prop` would be diluted roughly threefold by training
energy that has nothing to do with the factor under test, and would have to be detected against
training-energy variance. At 45 s the proposer is half the signal. **A short training budget makes
this experiment about three times more sensitive to the very thing it exists to measure**, while
costing a third as much. State this explicitly in *Experiment Design*; it is a methodological
argument, not a convenience.

**EPS, measured (gate G8').** 8 repeats at 45 s with a 30 s cooldown:

| | SD | steps CV | drift | EPS |
|---|---|---|---|---|
| without cooldown (first attempt) | 1.01 pp | 14.7 % | −37 %, ρ = −1.00 | 0.0202 |
| **with 30 s cooldown** | **0.37 pp** (detrended) | **1.5 %** | −3.9 %, ρ = −0.98 | **0.0073** |

A 30 s cooldown cut the apparent noise by **2.7x**. The first measurement was mostly thermal drift
wearing a lab coat, and had it been trusted, EPS would have been set nearly three times too strict —
the agent would have discarded most genuine improvements as noise. `eps: 0.0073` is now set in both
profiles.

Note the drift detector behaving correctly here: the residual −3.9 % trend is perfectly monotonic
(ρ = −0.98) but too small in magnitude to matter, so the AND rule from D13 correctly stays quiet.
Mean power still declined 114.5 → 101.8 W across the series, so a longer cooldown may remove even
that; check on the server whether the planned 120 s flattens it.

All of this re-runs on the server regardless — the RTX 4000 Ada is slower than the pilot GPU, so
G11 and G8' are both repeated there before Phase 1.

### D16 — Gate G1 failed on Qwen3.6. The subjects change to the Qwen3 pair. (from the server)

**What happened.** Both Qwen3.6 arms exceed the 20475 MiB (20.0 GiB) card *on weights alone*,
before any KV cache exists:

| candidate | on disk | GiB | why |
|---|---|---|---|
| cyankiwi/Qwen3.6-27B-AWQ-INT4 | 20.46 GB | **19.05** | 3.85B params left in 16-bit |
| cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | ~24 GB | **~22.4** | same, plus more experts |

**Root cause.** Qwen3.6 is **multimodal**. The quantizer's `ignore` list excludes the entire vision
tower, `lm_head`, and every `linear_attn.in_proj` projection, leaving 3.85B parameters at 16 bits
(~7.7 GB) on top of 12.7 GB of int4 weights. The Hub metadata shows it directly:
`I32: 25.47B, F16: 3.33B, BF16: 0.52B`. **No `--max-model-len` setting rescues this** — the failure
is in weights, not cache.

**Why the estimate was wrong.** `approx_weights_gb` was inferred from parameter count and
bit-width, which assumes everything gets quantized. It doesn't. `pin_models.py` now reads the Hub's
authoritative `usedStorage`, reports the dtype breakdown, warns when >0.5B parameters remain
unquantized or when the checkpoint is multimodal, and **fails G1 before downloading**. The old
behaviour cost 44 GB of transfer to discover.

**Qwen3-32B was also evaluated and rejected on arithmetic.** At 18.0 GiB of weights it leaves ~2.0
GiB. Qwen3-32B has 64 layers x 8 KV heads x 128 dim, so fp8 KV costs ~131 KB/token; the workload
needs >=6k context (~2.5k prompt + ~2k completion), i.e. ~0.8 GiB, plus ~0.4 GiB CUDA context and
~0.7 GiB activations. Total ~19.9 GiB against a 19.99 GiB card. Unservable, not merely tight.

**Decision: Qwen3-30B-A3B (MoE) vs Qwen3-14B (dense), both AWQ int4.**

| arm | repo | GiB | active params |
|---|---|---|---|
| MoE | QuixiAI/Qwen3-30B-A3B-AWQ | 15.6 | ~3B of 30.5B |
| dense | Qwen/Qwen3-14B-AWQ | 9.3 | 14.8B of 14.8B |

**The research question survives, in a arguably stronger form.** Both models fit the same card, so
the contrast becomes the one a self-hoster actually faces: *given one 20 GB GPU, does a 30B sparse
model activating ~3B parameters per token beat a 14B dense model activating all of them?* That is
the proposal's claim — "large-model quality at small-model inference cost" — tested at 2x the total
capacity against 4.7x fewer active parameters. It is more practitioner-relevant than the original
27B-vs-35B pairing, which contrasted two models of near-identical active cost.

**Two limitations to report.**

1. *Publisher asymmetry.* No single publisher ships AWQ builds of both models, so the dense arm is
   official Qwen and the MoE is a third party. Same method, same bit-width; the difference is
   calibration corpus. Second-order, but it belongs in *Threats to Validity* and the run-table
   metadata records both repos and revisions.
2. *Unequal total capacity.* 30.5B vs 14.8B. If the MoE wins, sparsity and capacity cannot be fully
   separated. State this plainly: the study answers "which is better per joule on one card", not
   "is sparsity better at matched capacity" — the latter needed Qwen3-32B, which does not fit.

**Reporting consequence.** The proposal's subject section changes family (Qwen3.6 -> Qwen3) and the
sparsity contrast is re-specified as above. This needs Vincenzo's sign-off; the measurements are
reproducible with `scripts/pin_models.py` and take under a minute.

---

## 2. System architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ experiment-runner  (session-level orchestrator)                          │
│   run table: 8 cells × 3 reps = 24 runs, shuffled, ≥120 s cooldown       │
│   per run: 1 complete agent session                                      │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ RunnerConfig.py lifecycle hooks
   ┌────────────┴─────────────┬───────────────────────────┐
   │                          │                           │
┌──▼───────────────┐  ┌───────▼──────────────┐  ┌─────────▼───────────────┐
│ EnergiBridge     │  │ NVML sampler         │  │ arloop (agent harness)  │
│ CPU pkg + DRAM   │  │ per-device, 10 Hz    │  │                         │
│ RAPL, 10 Hz      │  │ GPU0 = train         │  │  ┌───────────────────┐  │
│ whole session    │  │ GPU1 = proposer      │  │  │ propose (GPU1)    │  │
└──────────────────┘  └──────────────────────┘  │  │  vLLM OpenAI API  │  │
                                                │  └─────────┬─────────┘  │
                                                │  ┌─────────▼─────────┐  │
                                                │  │ guard + git commit│  │
                                                │  └─────────┬─────────┘  │
                                                │  ┌─────────▼─────────┐  │
                                                │  │ train 240 s (GPU0)│  │
                                                │  └─────────┬─────────┘  │
                                                │  ┌─────────▼─────────┐  │
                                                │  │ keep / revert     │  │
                                                │  └─────────┬─────────┘  │
                                                │       loop until budget │
                                                │  ┌─────────▼─────────┐  │
                                                │  │ final test eval   │  │
                                                │  └───────────────────┘  │
                                                └─────────────────────────┘
                     ▼ per run directory
        session.jsonl · energibridge.csv · nvml.csv · train.py history (git)
                     ▼ analysis/
        tidy.csv → stats → Pareto figure
```

Two instrumentation channels, deliberately redundant:

- **EnergiBridge** — required by the course, gives CPU package + DRAM energy, and a session-level
  cross-check. Its GPU support is a single `--gpu` flag whose multi-device behaviour we do not want
  to depend on.
- **NVML sampler** (`measurement/nvml_sampler.py`, built on `pynvml`, mirroring
  `experiment-runner/Plugins/Profilers/NvidiaML.py`) — samples `nvmlDeviceGetTotalEnergyConsumption`
  (cumulative mJ, per device, hardware-integrated, Ampere+) and instantaneous power per device at
  10 Hz. `E_train` and `E_prop` come from the energy-counter deltas, which are exact integrals and
  immune to sampling aliasing; the power trace is kept for the per-iteration time series.

Both start before the first proposer call and stop after the final evaluation, so their windows are
comparable.

---

## 3. Repository layout

```
green-lab-experiment/
├─ EXPERIMENT_PLAN.md              ← this file
├─ experiment/
│  ├─ RunnerConfig.py              experiment-runner config (session = 1 run)
│  ├─ profiles/
│  │  ├─ server.yaml               2× RTX 4000 Ada, real experiment
│  │  └─ pilot.yaml                RTX 3080 Laptop 16 GB, functional test only
│  ├─ harness/
│  │  ├─ agent_loop.py             the loop: propose→guard→train→eval→keep/revert
│  │  ├─ proposer.py               OpenAI-compatible client, token + latency accounting
│  │  ├─ guards.py                 static validation of proposed train.py
│  │  ├─ recipe_repo.py            git-backed keep/revert of the editable file
│  │  ├─ session_log.py            JSONL event schema + writer
│  │  └─ templates/
│  │     ├─ program.md.j2          task spec, patience + budget rendered per cell
│  │     └─ system.txt             system prompt
│  ├─ workload/
│  │  ├─ prepare_cifar.py          download, fixed 45k/5k split, cache tensors
│  │  ├─ train.py                  THE EDITABLE FILE — CNN + fixed 240 s budget
│  │  └─ final_eval.py             test-set evaluation, run once per session
│  ├─ measurement/
│  │  ├─ nvml_sampler.py           per-device 10 Hz power + energy counters
│  │  ├─ energy_align.py           align energy to iteration boundaries
│  │  └─ idle_baseline.py          idle draw of both GPUs + CPU
│  ├─ analysis/
│  │  ├─ aggregate.py              run dirs → tidy.csv
│  │  ├─ stats.py                  assumptions, ANOVA/ART, effect sizes
│  │  ├─ pareto.py                 non-dominated set
│  │  └─ figures.py                all report figures
│  ├─ serving/
│  │  ├─ serve_vllm.sh             GPU proposer (server)
│  │  ├─ serve_cpu.sh              llama.cpp CPU proposer (Phase 2)
│  │  └─ models.yaml               pinned repos + revisions + quantizer
│  ├─ scripts/
│  │  ├─ preflight.py              environment / GPU / permission checks
│  │  ├─ vram_check.py             D3 gate: does the MoE actually serve?
│  │  ├─ smoke_test.py             one short session end to end
│  │  └─ align_check.py            week-1 energy/iteration alignment proof
│  ├─ tests/                       unit tests for patience, guards, parsers
│  └─ requirements.txt
└─ experiments/                    experiment-runner output (git-ignored)
```

---

## 4. Component specifications

### 4.1 `workload/train.py` — the editable file

The only file the proposer may rewrite. Design constraints, inherited from `autoresearch`:

- Self-contained: model, optimizer, schedule, augmentation, training loop, in one file.
- Trains for exactly `TRAIN_SECONDS` (default 240) of wall clock, measured from after data loading
  and CUDA warm-up, excluding compilation. Training stops at the budget boundary mid-epoch if
  needed.
- Prints exactly one machine-readable result line to stdout and writes `result.json`:
  `{"val_acc": float, "epochs_completed": int, "steps": int, "train_seconds": float, "peak_vram_mb": int}`
- Loads data through `prepare_cifar.load_splits()`; never re-splits, never touches the test set.
- Fails loudly (non-zero exit) rather than silently, so the harness can classify errors.

Baseline recipe: a ~600k-parameter 6-layer CNN with BatchNorm, SGD + momentum, one-cycle LR,
random crop + horizontal flip. It should reach roughly 70–80 % validation accuracy in 240 s on an
RTX 4000 Ada, leaving clear headroom for the agent to improve — a baseline that is already saturated
produces no keep/revert signal, which would gut the experiment.

**Calibrate this in week 1** (`scripts/smoke_test.py --calibrate`): if the baseline exceeds ~85 %,
weaken it (fewer channels, no augmentation); if below ~60 %, strengthen it. Record the final
baseline as a pinned commit — it is a fixed variable of the experiment.

### 4.2 `harness/proposer.py`

Thin OpenAI-compatible client (`/v1/chat/completions`) so vLLM, llama.cpp-server and Ollama are all
usable without code changes. Responsibilities:

- Sampling parameters fixed across all cells: `temperature=0`, `top_p=1`, `seed=<run seed>`,
  `max_tokens=8192`. Recorded per call.
- Returns `ProposerResponse(text, prompt_tokens, completion_tokens, latency_s, t_start, t_end)`.
  Timestamps are `time.time()` and `time.monotonic()` — the wall-clock pair is what the energy
  aligner joins on.
- Retries: up to 3 attempts on transport error or on a response failing the output contract, with
  the failure fed back into the next attempt. A 4th failure marks the iteration `errored`.
- A `--proposer stub` mode returns scripted mutations from a fixture file. This makes the whole
  pipeline testable without a GPU and is what the CI tests use.

**Output contract.** Full-file replacement, not a diff — local models produce unappliable diffs far
too often, and `train.py` is small enough that a rewrite is cheap. The model must answer with:

````
RATIONALE: <one or two sentences>
```python
<complete new train.py>
```
````

Anything else is a contract violation and triggers a retry.

### 4.3 `harness/guards.py`

Static checks on a proposed `train.py` before it is ever executed. A proposal failing any check is
rejected without spending training energy; the rejection is logged and fed back to the proposer.

1. Parses as Python 3.11 (`ast.parse`).
2. Declares `TRAIN_SECONDS` and its value is unchanged. (Otherwise the agent can buy accuracy with
   time and destroy comparability — the single most important guard.)
3. Imports `load_splits` from `prepare_cifar` and calls it; no other data source.
4. No reference to `test` splits, `final_eval`, or the test tensors.
5. No `subprocess`, `os.system`, `socket`, `requests`, `urllib`, `open(...,'w')` outside the run dir.
6. No import outside the pinned allowlist (torch, torchvision, numpy, math, time, json, random).
7. Writes `result.json` with the required keys (checked structurally, verified at runtime).

### 4.4 `harness/recipe_repo.py`

The mutation history is a git repository inside the run directory, seeded with the baseline
`train.py`. One commit per accepted iteration, message
`iter=<n> decision=<keep|revert> val_acc=<x>`. Revert is `git checkout <sha> -- train.py`, never a
textual undo. At session end the repo is archived into the run dir, giving a complete, inspectable
mutation history per session for free.

### 4.5 `harness/agent_loop.py`

The loop. Pseudocode with the exact semantics locked in section 5.2:

```
best_acc      = baseline_val_acc          (measured once, before iteration 1)
current_sha   = baseline_sha
best_sha      = baseline_sha
regressions   = 0                          consecutive non-improving kept iterations
for i in 1..loop_budget:
    prompt   = render(program.md, history, current train.py)
    proposal = proposer.complete(prompt)          # energy: GPU1
    if not guards.ok(proposal): log(rejected); continue      # counts against budget
    write train.py; commit
    result   = run_training(timeout = TRAIN_SECONDS + 120)   # energy: GPU0
    if result.errored: log(errored); revert to current_sha; continue
    if result.val_acc > best_acc + EPS:
        keep; best_acc = result.val_acc; best_sha = head; regressions = 0
    else:
        regressions += 1
        if regressions >= patience:
            revert train.py to best_sha; current_sha = best_sha; regressions = 0
        else:
            keep provisionally; current_sha = head
final: checkout best_sha; run final_eval.py on the test set once
```

`EPS = 0.001` (0.1 accuracy points) so that floating-point noise is not counted as improvement.

### 4.6 `RunnerConfig.py`

One `experiment-runner` run = one session.

| Hook | Action |
|---|---|
| `before_experiment` | preflight; assert GPUs visible, serving endpoints up, disk space; write env manifest (git shas, pip freeze, driver, CUDA, model revisions) |
| `before_run` | ≥120 s cooldown; assert GPU temps back to idle band; assert idle power within tolerance of the week-1 baseline |
| `start_run` | create run dir; seed git recipe repo; render `program.md` for the cell; point the harness at the right serving endpoint |
| `start_measurement` | start NVML sampler thread; start EnergiBridge wrapping the harness process |
| `interact` | block until the session finishes or the hard timeout (3× expected duration) fires |
| `stop_measurement` | stop EnergiBridge; stop the sampler; flush traces |
| `stop_run` | archive git repo; free GPU memory; verify the serving endpoint is still healthy |
| `populate_run_data` | parse `session.jsonl` + `nvml.csv` + `energibridge.csv` into the run-table row |

Factors: `proposer ∈ {dense, moe}`, `patience ∈ {greedy, patience3}`, `loop_budget ∈ {10, 20}`,
`repetitions=3`, `shuffle=True`.

Data columns: `E_total_J, E_prop_J, E_train_J, E_cpu_J, E_wasted_J, E_per_kept_J, test_acc,
best_val_acc, baseline_val_acc, iterations, kept, reverted, rejected, errored, wallclock_s,
prompt_tokens, completion_tokens, proposer_latency_s_mean, no_progress`.

### 4.7 `measurement/energy_align.py`

Joins the 10 Hz per-device traces to the iteration boundaries in `session.jsonl` on wall-clock
timestamps, producing per-iteration energy:

- `E_prop[i]` — GPU1 energy-counter delta over `[proposer.t_start, proposer.t_end]`
- `E_train[i]` — GPU0 delta over `[train.t_start, train.t_end]`
- `E_idle_gap` — energy in the gaps between phases, reported separately so nothing is silently lost
- `E_wasted` — `Σ (E_prop[i] + E_train[i])` over iterations whose mutation was ultimately reverted,
  including iterations discarded by a patience-chain rollback

**Invariant, asserted in code:** `Σ_i (E_prop[i] + E_train[i]) + E_idle_gap` must equal the
session-level counter delta within 1 %. A violation means the alignment is wrong and the run is
quarantined rather than analysed.

### 4.8 `analysis/`

- `aggregate.py` — walks `experiments/<name>/run_*`, validates each against the schema, emits
  `tidy.csv` (one row per session) and `iterations.csv` (one row per iteration).
- `stats.py` — Shapiro–Wilk + Levene per cell; if satisfied, three-way full-factorial ANOVA on
  `E_total`, `E_per_kept`, `test_acc` with all two-way interactions; otherwise Aligned Rank
  Transform then the same ANOVA. Effect sizes are primary: partial η² with CIs, Cliff's δ with
  bootstrap CIs for pairwise contrasts. Multiple comparisons: Holm across the three registered
  hypotheses.
- `pareto.py` — non-dominated set over (`E_total` ↓, `test_acc` ↑), computed on cell means with the
  24 individual sessions plotted underneath.
- `figures.py` — Pareto plot (headline), energy decomposition stacked bars per cell, per-iteration
  power time series, accuracy trajectories, waste breakdown.

---

## 5. Protocol — exact operational definitions

### 5.1 Session

One session = one full agent run at one cell: baseline evaluation, `loop_budget` iterations, one
final test evaluation. A session is `valid` if it completed `loop_budget` iterations without
infrastructure failure. Infrastructure failures (serving crash, OOM, machine reboot) invalidate the
session; it is re-run and both the failure and the re-run are recorded. Agent-level failures (bad
proposals, training errors, zero kept mutations) are **data**, never grounds for a re-run.

### 5.2 Factor semantics

| Factor | Level | Meaning |
|---|---|---|
| `proposer` | `dense` | `Qwen/Qwen3.6-27B`, int4, served on GPU1 |
| | `moe` | `Qwen/Qwen3.6-35B-A3B`, int4, same stack and settings |
| `patience` | `greedy` | patience = 1: any iteration that does not beat `best_acc + EPS` is reverted immediately to `best_sha` |
| | `patience3` | patience = 3: non-improving mutations are kept provisionally; on the 3rd *consecutive* non-improving iteration the whole provisional chain is rolled back to `best_sha` and the counter resets |
| `loop_budget` | `10` / `20` | maximum iterations per session. Guard-rejected and errored iterations **count** against the budget (they consume real energy, which is the point) |

Fixed: 240 s training budget, int4 quantization, temperature 0, seeded, same serving stack, same
`max-model-len`, same CIFAR split seed, same baseline `train.py`, pinned software.

### 5.3 Phase 2 — CPU-only proposer

The 4 cells adjacent to the Phase-1 frontier, 2 repetitions, proposer served on CPU via
`llama.cpp` (`serving/serve_cpu.sh`), training unchanged on GPU 0. `E_prop` becomes a CPU RAPL
measurement rather than an NVML one — this is a construct change, not just a hardware change, and
must be stated: CPU package energy includes the OS and the training process's host-side work, so
`E_prop` in Phase 2 is an *upper bound*. Subtract the measured idle-plus-training-host baseline and
report both the raw and corrected numbers.

Expect 10–30× slower proposer inference. Cap per-call latency at 900 s; a session that exceeds 4×
its Phase-1 wall clock is truncated and reported as a timeout, which is itself a finding.

### 5.4 Pilot on the RTX 3080 Laptop (16 GB)

The laptop has one GPU, so training and proposer share it. **Energy attribution is invalid in pilot
mode** and the harness refuses to emit `E_prop`/`E_train` under `profiles/pilot.yaml`; it emits
`E_total` only, flagged `attribution=none`. Pilot exists to prove the pipeline works, not to produce
data.

Pilot settings: proposer = a small instruct model on the same OpenAI-compatible API (Ollama or
llama.cpp; `Qwen3.6-4B`-class int4 at ~3 GB, or 1.7B if VRAM is tight next to training),
`TRAIN_SECONDS=45`, `loop_budget=4`, 1 repetition, 2 cells. A full pilot pass is ~15 minutes and
exercises every code path: rendering, proposing, guarding, committing, training, reverting,
logging, aligning, aggregating, plotting.

Windows notes: EnergiBridge needs administrator rights and RAPL access is unreliable on consumer
laptops — run the pilot with `--energy nvml-only`, or in WSL2. Do not spend time fixing Windows
energy measurement; the server is where energy numbers come from.

---

## 6. Data schema

`session.jsonl`, one JSON object per event, append-only:

```jsonc
{"ev":"session_start","t":1754812800.12,"cell":{"proposer":"moe","patience":"patience3","loop_budget":20},
 "run_id":"run_7_repetition_1","seed":7,"versions":{...}}
{"ev":"baseline_eval","t":...,"val_acc":0.7412,"train_seconds":240.0}
{"ev":"propose_start","t":...,"iter":1}
{"ev":"propose_end","t":...,"iter":1,"prompt_tokens":3412,"completion_tokens":1180,"latency_s":41.2}
{"ev":"guard","t":...,"iter":1,"ok":true,"violations":[]}
{"ev":"train_start","t":...,"iter":1}
{"ev":"train_end","t":...,"iter":1,"val_acc":0.7551,"epochs":9,"exit":0}
{"ev":"decision","t":...,"iter":1,"decision":"keep","best_acc":0.7551,"regressions":0}
{"ev":"rollback","t":...,"iter":7,"to_sha":"a1b2c3","discarded_iters":[5,6,7]}
{"ev":"final_eval","t":...,"test_acc":0.7893,"best_sha":"a1b2c3"}
{"ev":"session_end","t":...,"iterations":20,"kept":6,"reverted":11,"rejected":2,"errored":1}
```

`nvml.csv`: `t_wall,t_mono,dev,power_mw,energy_mj,util_gpu,util_mem,temp_c,mem_used_mb`
`energibridge.csv`: EnergiBridge native format.

Everything downstream reads these three files and nothing else. That is what makes the replication
package self-contained.

---

## 7. Analysis plan

Registered before data collection:

- **Primary output:** effect sizes with confidence intervals. p-values secondary, Holm-corrected
  across H_prop, H_pat, H_bud.
- **H_prop** (dense vs MoE): contrast on `E_total` and `test_acc`. Report the mechanism split too —
  `E_prop` per call, iterations-to-first-keep, kept fraction — since the interesting result is
  whether per-token savings survive worse steering.
- **H_pat** (greedy vs patience-3): contrast on `E_wasted` and `E_per_kept`.
- **H_bud** (10 vs 20): contrast on `E_per_kept`; also report marginal accuracy per extra
  iteration.
- **RQ2:** Pareto frontier over cell means of (`E_total`, `test_acc`), with all 24 sessions shown;
  baseline cell (dense × greedy × 10) marked; Phase-2 CPU points overlaid.
- Sessions with zero kept mutations are reported as no-progress sessions, included in energy
  analyses, and excluded only from `E_per_kept` (where the denominator is zero) with the count
  stated.
- With n=3 per cell, three-way interactions are not interpretable. Say so once, in *Conclusion
  Validity*, and do not test them.

---

## 8. Schedule

**Week 0 — pilot on the laptop (before server access).** Build and validate the whole pipeline in
stub and pilot mode. Deliverable: a `tidy.csv` and a Pareto figure produced from fake-but-real-shaped
data. Nothing about this week requires the server.

**Week 1 — server bring-up and gates.**
- Day 1: access, environment, `preflight.py` green, EnergiBridge with sudo, NVML counters readable.
- Day 2: **D3 gate** — `vram_check.py` on both proposers. This is the gate that can force a model
  change; run it first, not last.
- Day 3: baseline calibration (`smoke_test.py --calibrate`); pin the baseline `train.py`.
- Day 4: `align_check.py` — the energy/iteration alignment proof, plus the idle baseline with both
  GPUs idle, and a sleep-stub session to measure harness overhead.
- Day 5: two full pilot sessions at the real scale (one per proposer, budget 10). Launch the first
  Phase-1 batch overnight.

**Week 2 — Phase 1.** 24 sessions ≈ 10 GPU-hours at the 45 s budget (D15) over 2–3 overnight batches. Daytime: run
`aggregate.py` on completed sessions, inspect, fold re-runs into the next batch. Rule: no analysis
decisions are changed after seeing the data — the plan above is the pre-registration.

**Week 3 — Phase 2 (8 CPU sessions) + Stage 2 if the schedule holds.** Stage 2 is the intra-model
sweep (int4 vs int8, `max-model-len`, batching) on the Stage-1-preferred architecture, one factor at
a time.

**Week 4 — analysis and report.** Figures frozen by day 2, writing after.

Priority under pressure, as agreed with Vincenzo: Phase 1 > Phase 2 > Stage 2.

---

## 9. Gates

Do not start the main run until every one of these passes. Each has a script.

| # | Gate | Script | Pass criterion |
|---|---|---|---|
| G1 | Both proposers serve on one 20 GB card | `vram_check.py` | 40 consecutive real-shape requests, no OOM, stable memory |
| G2 | Baseline leaves headroom | `smoke_test.py --calibrate` | baseline val_acc in 0.60–0.85 at 240 s |
| G3 | Agent can actually improve it | `smoke_test.py` | ≥1 kept mutation in a 5-iteration session, both proposers |
| G4 | Energy alignment is sound | `align_check.py` | per-iteration sum + gaps = session total within 1 % |
| G5 | Devices are correctly attributed | `align_check.py` | during training, GPU0 power ≫ GPU1; during proposing, the reverse |
| G6 | Idle is stable | `idle_baseline.py` | idle draw drifts < 2 % across 30 min |
| G6b | **Loaded** throughput is stable | `noise_floor.py --repeats 8` | step-count drift < 10 % across the series (pilot laptop: -37 %) |
| G11 | **The agent has signal to work with** | `headroom_check.py` | SNR = headroom / noise >= 3 at the chosen budget; budget chosen by max SNR |
| G7 | Loop semantics are correct | `pytest tests/` | patience and rollback tests pass |
| G8' | The noise floor is known and EPS exceeds it | `noise_floor.py --repeats 8` | EPS >= 2x SD of the repeated baseline; SD recorded |
| G9 | Infra errors are rare | any smoke session | infra error rate < 25 %, else the serving stack is not ready |
| G10 | The training GPU is loaded | `noise_floor.py` | mean utilisation >= 50 %, else widen the baseline (D10) |

G8 as originally written (run the same cell twice, expect the same decisions) is **unsatisfiable
by construction** and has been replaced by G8'. Two independent sources of nondeterminism make it
so: vLLM's continuous batching means even temperature-0 output is not bitwise reproducible, and
the fixed *wall-clock* training budget means step counts differ run to run regardless of seeding
(D6). Rather than chase reproducibility, measure the resulting variance and require every reported
effect to exceed it. The divergence rate is itself a legitimate result about self-hosted
determinism and belongs in *Threats*.

---

## 10. Risk register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| MoE int4 does not fit 20 GB with usable context | Kills the headline factor | **High** | G1 first; llama.cpp fallback; Granite backup pair pre-registered |
| Baseline CNN saturates in 240 s → no keep/revert signal | Experiment measures noise | Medium | G2 calibration; weaken baseline; it is a fixed variable, tune before, never during |
| Local 27B models produce unusable proposals | Sessions all errored | Medium | Full-file contract + retries + guard feedback; measure the rejection rate in G3 and report it as a finding |
| Only one server, shared access | Schedule slip | Medium | Batches are ~2 h; priority order; sessions are independent and resumable |
| Non-determinism swamps n=3 | Weak conclusions | Medium | Effect sizes with CIs as primary; report within-cell dispersion; G8 quantifies it |
| EnergiBridge multi-GPU behaviour differs from expectation | Attribution broken | Low | NVML sampler is the primary attribution channel; EnergiBridge is corroboration |
| Proposer 30-minute sessions blow the schedule at budget 20 | Phase 1 overruns | Low | Hard per-session timeout at 3× expected; timeout is recorded as data |
| **Effects smaller than the noise floor** (D6) | Conclusions unsupported | **High** | EPS from `noise_floor.py`; effects compared against measured SD; gate G8' |
| **Serving instability read as a model result** (D7) | Wrong conclusions | Medium | Error taxonomy; sessions quarantined above a 25 % infra rate |
| **Training GPU mostly idle** (D10) | Energy story becomes elapsed time, not work | Medium | Utilisation measured in `noise_floor.py`; pilot showed 88 %, so this one is resolved |
| **Thermal drift between sessions** (D11) | Inflated variance, weaker power | **High** | Cooldown, randomised order, drift reported; gate G6b |
| **Headroom too small relative to noise** (D12) | keep/revert becomes coin flipping | **High** | `headroom_check.py` as blocking gate G11; budget chosen by max SNR; reduce noise before spending joules |
| **Agent plateaus early** (D12) | budget 20 buys energy but no accuracy | Medium | Report iterations-to-last-keep per cell; it is a finding, not a flaw, but must be stated |
| **Calibration instrument mis-specified** (D14) | Budget chosen on a meaningless number | Medium | Instrument-validity check in `headroom_check.py`: reference must reach >=25 % of baseline steps and >=3 epochs |

---

## 11. Open items for Vincenzo

1. **D1** — re-implementing the loop instead of orchestrating a coding agent. This is the biggest
   deviation from the reviewed proposal and needs his sign-off on the framing.
2. **D2** — no 4090; training on an RTX 4000 Ada. Confirm this is the machine and that there is not
   a second one with the 4090 in it.
3. **D3** — if the MoE will not serve on 20 GB, does he prefer the llama.cpp route (both arms, lower
   throughput) or the Granite 4.1 8B/30B backup pair (loses the sparse-vs-dense contrast, becomes a
   size contrast)?
4. **D4** — the CIFAR swap justification changes from "schedule" to "signal quality at a 240 s
   budget". Confirm he is content with that.
5. Baseline cell confirmation: dense × greedy × 10.
6. **D6 (needs sign-off).** EPS rises from 0.001 to a measured value, likely 0.008-0.02. This makes
   the agent markedly more conservative: fewer kept mutations, and possibly sessions with none at
   all. The alternative is a threshold below the noise floor, under which "kept mutation" means
   nothing. Recommend measuring, setting EPS = 2x SD, and reporting both. No-progress sessions
   become more common and are reported, not dropped.
7. **D7.** Confirm that sessions above a 25 % infrastructure-error rate are re-run rather than
   analysed, and that contract-compliance rate is acceptable as a reported dependent variable.

---

## 12. Immediate next actions

0. Decide the training budget from measurement, not assumption:
   `python scripts/headroom_check.py --budgets 45,240 --repeats 2` (gate G11), then
   `python scripts/noise_floor.py --repeats 8 --train-seconds <chosen>` (gates G6b, G8')
1. `cd experiment && pip install -r requirements.txt`
2. `pytest tests/ -v` — loop semantics and parsers, no hardware needed
3. `python scripts/smoke_test.py --profile pilot --proposer stub` — full pipeline on fake data
4. Point `profiles/pilot.yaml` at your local small model, re-run without `--proposer stub`
5. Send Vincenzo the section 11 items
