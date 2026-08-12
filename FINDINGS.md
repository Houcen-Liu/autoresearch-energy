# Findings notebook

Everything measured so far that belongs in the report, organised by **where it goes
in the paper** rather than by when it happened. `EXPERIMENT_PLAN.md` keeps the
chronological record (D1-D20); this file is the writing source.

Status: pilot complete (laptop, RTX 3080 16 GB), server gates G1/G3/G4/G5/G8'/G10/G11
passed (2x RTX 4000 Ada 20 GB). Phase 1 not yet run. **Nothing here answers the
research question** -- these are one-session-per-arm observations and calibration
results. They fix the method and flag what to watch.

---

## 1. Numbers to cite

| quantity | value | where measured |
|---|---|---|
| training budget | 45 s | G11, server |
| EPS (keep threshold) | 0.0123 | G8', server, 8 repeats |
| baseline val_acc | 0.7614 +/- 0.0062 | G8', server |
| noise floor (detrended SD) | 0.62 pp | G8', server |
| headroom at 45 s | +15.27 pp | G11, server |
| SNR at 45 s | 24.6 | headroom / 8-repeat noise |
| cooldown | 120 s | G8', no residual drift |
| training GPU utilisation | 94 % at 116 W | G10 |
| dense throughput | 35 tok/s | G1 probe |
| MoE throughput | 69 tok/s | G1 probe |
| dense proposal latency | 47-59 s | G1, G3 |
| MoE proposal latency | 25-29 s | G1, G3 |
| session energy (dense, 5 iter) | 60 204 J | G4 |
| wasted energy (dense) | 38 242 J (64 %) | G4 |
| propose / train power | 112.8 W / 115.7 W | G5 |

---

## 2. Methods -- decisions made from measurement, not convention

Each of these was originally a guess in the proposal and is now an experimental
result. That framing is worth keeping in the writeup: the calibration phase is
itself a contribution.

### 2.1 The training budget was chosen by signal-to-noise, not tradition

A fixed wall-clock training budget is the `autoresearch` protocol's core
constraint, but the protocol does not say how long. Too short and the agent cannot
distinguish an improvement from noise; too long and the proposer becomes a
rounding error in session energy, which defeats the study.

Measured at two budgets on server hardware:

| budget | baseline | reference (strong recipe) | headroom | steps |
|---|---|---|---|---|
| 45 s | 0.7634 | 0.9161 | +15.27 pp | ~15.1k |
| 240 s | 0.7801 | 0.9233 | +14.32 pp | ~79.4k |

5.3x the compute buys 1.7 pp of baseline accuracy and *reduces* headroom -- the
workload saturates. Using the 8-repeat noise floor for both, SNR is 24.6 at 45 s
against 26.5 at 240 s: within 8 % of each other.

45 s therefore wins on cost (~10 vs ~53 GPU-hours for Phase 1) and, more
importantly, on sensitivity to the factor under study: the proposer occupies
40-57 % of session wall-clock at 45 s versus ~17 % at 240 s. **A longer budget
would have made the experiment less able to see its own independent variable.**

### 2.2 EPS was measured; the proposal's value was four times too small

The proposal specified EPS = 0.001. The measured run-to-run standard deviation of
the *unmodified* baseline is 0.0062 on server hardware (0.0037 on the pilot). At
0.001, "kept mutation" would have meant "won a coin flip".

EPS = 2 x detrended SD = **0.0123**.

This is not a formality. In the dense smoke session, iteration 5 scored 0.8296
against a kept best of 0.8178 -- an improvement of 1.18 pp, just under the
threshold -- and was correctly reverted. Under the pilot's EPS it would have been
kept on evidence indistinguishable from re-running the same code.

### 2.3 Cooldown: thermal drift masquerades as variance

On the pilot, 8 baseline repeats with no cooldown gave SD 1.01 pp; the same
measurement with a 30 s cooldown gave 0.37 pp. **The apparent noise floor was
2.7x too high, and it was thermal, not stochastic.** Without the cooldown, EPS
would have been set to 0.0202 -- so strict that almost nothing could be kept.

On the server with 120 s cooldown, raw SD equals detrended SD to four decimals:
no residual trend at all.

Generalisable lesson for energy experiments: **run-to-run variance measured
back-to-back is not variance, it is drift.** Detrend, and test for monotonicity
(Spearman rho) rather than slope alone -- a slope test on a non-monotonic series
produced a false drift alarm during the pilot.

### 2.4 Infrastructure failure and agent failure are different things

The error taxonomy separates:

- **infra**: timeouts, transport errors -> session invalid above 25 %, re-run
- **agent**: contract violations, guard rejections, crashed recipes -> **data**

This distinction turned out to carry real weight. Both smoke sessions had a 0 %
infrastructure error rate and a 20-40 % agent error rate. Collapsing the two would
have thrown away valid sessions and hidden a dependent variable (see 4.3).

---

## 3. Subject selection -- a negative result worth reporting

The proposal named a Qwen3.6 pair. **Neither arm fits a 20 GB card**, on weights
alone, before any KV cache:

| candidate | on disk | GiB |
|---|---|---|
| Qwen3.6-27B AWQ int4 | 20.46 GB | 19.05 |
| Qwen3.6-35B-A3B AWQ int4 | ~24 GB | ~22.4 |

The cause is instructive: Qwen3.6 is multimodal, and the quantizer leaves the
vision tower, `lm_head` and the linear-attention projections at 16 bits. Hub
metadata shows `I32: 25.47B, F16: 3.33B, BF16: 0.52B` -- 3.85B unquantized
parameters, ~7.7 GB, on top of 12.7 GB of int4 weights. **"27B at int4" does not
imply 13.5 GB.**

Qwen3-32B-AWQ was also rejected, on arithmetic rather than measurement: 18.0 GiB
of weights + ~0.8 GiB fp8 KV at 6k context + ~1.1 GiB CUDA context and activations
= ~19.9 GiB against a 19.99 GiB card.

**Final pair, both AWQ int4:**

| arm | repo | GiB | active params |
|---|---|---|---|
| MoE | QuixiAI/Qwen3-30B-A3B-AWQ | 15.6 | ~3B of 30.5B |
| dense | Qwen/Qwen3-14B-AWQ | 9.3 | 14.8B of 14.8B |

The research question arguably improves: given one 20 GB GPU, does a 30B sparse
model activating ~3B parameters per token beat a 14B dense model activating all of
them? That is the self-hosting decision a practitioner actually faces.

**Recommendation for the writeup:** check `usedStorage` from the Hub API and the
dtype breakdown *before* downloading. `scripts/pin_models.py` does this and fails
the gate in under a minute, versus 44 GB of downloads discovering it the slow way.

---

## 4. Behavioural findings about the agents

### 4.1 Format compliance and content quality are independent axes

The clearest finding of the pilot. Two models from the same family:

| model | guard rejections | accuracy gain |
|---|---|---|
| qwen3:4b | 4 of 6 iterations (emitted file fragments) | **+8.8 pp** |
| qwen3:14b | 0 (perfect compliance) | **0.0 pp** |

The smaller model could not follow the output contract but proposed changes that
worked. The larger model followed the contract exactly and proposed a change (LR
0.01 -> 0.001 with batch 128 -> 256) that left an already-undertrained recipe
worse off.

**Implication:** an agentic-coding benchmark that scores only parse rate or tool
compliance would rank these two models in exactly the wrong order. Report
compliance and quality separately.

### 4.2 Undocumented context is a hidden validity threat

First G3 attempt: **4 of 5 iterations crashed**, all on the same line. The
proposer added per-channel normalisation to data that `load_splits()` already
returns normalised, using a bare `(3,)` tensor that broadcasts against the width
axis of an `(N, 3, 32, 32)` tensor.

`train.py` never stated the shape, dtype, or that normalisation was applied, and
`prepare_cifar.py` is not in the prompt. **The information did not exist anywhere
the proposer could reach.** A human given the same context would have guessed the
same way.

Left unfixed, the experiment would have measured "can the model guess an
undocumented data contract" -- a floor effect unrelated to sparsity. After
documenting the contract (comments only, so calibration stayed valid), the crash
rate fell from 4/5 to 1/5.

**Generalisable:** when an agent edits a file, everything it needs to reason about
must be *in that file*. Anything in an unshown import is invisible.

### 4.3 Crash rate may be a dependent variable, not noise

Even with the contract documented, both arms crashed: dense 1/5, MoE 2/5. If that
holds over 24 sessions, "proportion of proposals that run at all" is a legitimate
outcome alongside accuracy and joules -- and it is directly relevant to the energy
question, because a crashed proposal costs a full proposer inference and returns
nothing.

### 4.4 The crash-feedback loop works, but only with information to reason from

Feeding tracebacks into the next prompt was a pilot fix (the 14B had repeated one
hallucinated API call four times without it). Under test: iteration 2 crashed, the
traceback reached iteration 3, and iteration 3 *changed approach* rather than
re-emitting identical code. It still failed, because the traceback says the shapes
disagree but not which is correct. Mechanism sound; starved of context (4.2).

### 4.5 Proposals are large and mostly harmful

Per-iteration val_acc across the dense session ranged 0.4266 to 0.8296 against a
0.7494 baseline. The proposer is not producing no-ops or single-constant tweaks;
it rewrites substantial parts of the recipe, most attempts are worse, and the
protocol's value is keeping the one that is not.

---

## 5. Measurement integrity

### 5.1 Per-device attribution is sound (G5)

| phase | hot GPU | cold GPU | ratio |
|---|---|---|---|
| propose | 112.8 W (GPU 1) | 16.8 W | 6.73 |
| train | 115.7 W (GPU 0) | 24.1 W | 4.81 |

Per-window ratios 4.69-7.65, no exceptions among genuine windows.

**One methodological catch worth reporting.** The gate initially failed on a
single window: a crashed recipe produced a 2.6 s "training" phase where GPU 0 read
15.9 W (never loaded) and GPU 1 read 69.6 W (cooling from the previous request),
inverting the ratio to 0.23. Nine real windows passed. A correct measurement was
defeated by an aggregate that treated a non-event as an event. Phases shorter than
5 s are now excluded, visibly rather than silently.

### 5.2 Energy reconstructs (G4)

E_train 28 439 J + E_prop 31 765 J = 49 006 J accounted, against 60 204 J total:
an 18.6 % gap covering cooldown, checkpoint I/O and evaluation.

### 5.3 The training GPU is genuinely busy (G10)

94 % mean utilisation at 116 W sustained. The inner workload is GPU-bound, so
attributing training joules to the training device measures what it claims to.

### 5.4 Determinism

At temperature 0 with a fixed prompt, dense proposal latency had p50 58.87 s and
p95 58.88 s over 40 requests. Session-level energy differences will come from what
the agent does, not from sampling variance in how long it takes to say it.

---

## 6. Preliminary arm comparison -- **n = 1 per arm, not a result**

| | dense (14B) | MoE (30B-A3B) |
|---|---|---|
| baseline val_acc | 0.7494 | 0.7590 |
| best val_acc | 0.8178 | 0.8312 |
| gain | +6.84 pp | +7.22 pp |
| test_acc | 0.8129 | 0.8181 |
| kept | 1 | 1 |
| crashed | 1 | 2 |
| evaluated | 4 | 3 |
| best found at iteration | 3 | 1 |
| proposal latency | ~56 s | ~27 s |
| iteration wall-clock | ~104 s | ~74 s |

The MoE proposes **2.0x faster** at 2.2x the parameter count, which is the
sparsity mechanism doing exactly what the proposal predicted. Whether that
converts into fewer joules per kept mutation, and whether proposal *quality*
holds up over 24 sessions, is what Phase 1 exists to answer.

Baselines differ by ~1 pp between the two sessions from training stochasticity
alone -- a useful reminder of why the design uses repeated sessions and effect
sizes rather than single comparisons.

---

## 7. Threats to validity

1. **Publisher asymmetry.** No single publisher ships AWQ builds of both models:
   dense is Qwen's official quantization, MoE is third-party. Same method and
   bit-width; the calibration corpus differs. Both repos and revisions are pinned
   in the run table.
2. **Unequal total capacity** (30.5B vs 14.8B). If the MoE wins, sparsity and
   capacity cannot be fully separated. The study answers "which is more efficient
   per joule on one 20 GB card", not "is sparsity better at matched capacity" --
   the latter needed Qwen3-32B, which does not fit.
3. **JIT-compiled kernels.** Attention kernels are compiled for sm89 on this
   machine. A replication on different silicon gets different kernels and
   therefore different absolute joules. Both arms share one build, so the
   *contrast* is sound.
4. **Prefix caching is on** (vLLM default), so from iteration 2 the shared prompt
   prefix skips prefill. Realistic for self-hosting and identical across arms, but
   per-iteration proposer energy is not uniform within a session.
5. **Single workload.** CIFAR-10 CNN under a 45 s budget. Findings about proposal
   quality may not transfer to other AutoML tasks.
6. **The reference recipe defines headroom.** A stronger reference would raise
   measured headroom; the instrument-validity check (reference must reach a
   comparable step count) guards against comparing a differently-scaled model.

---

## 8. Reproducibility hazards

Not scientific findings, but they cost an evening and belong in an appendix or the
replication README.

1. `--disable-log-requests` was removed in vLLM 0.27.
2. vLLM 0.27 ignores `VLLM_ATTENTION_BACKEND` and has no `--attention-backend`, so
   FlashInfer cannot be swapped out.
3. FlashInfer cannot be uninstalled either -- vLLM's sampler imports it unguarded.
4. FlashInfer's JIT needs a self-consistent CUDA toolkit, and pip will not give
   you one: `flashinfer-python` depends on `nvidia-cuda-nvcc` unpinned, which
   depends on `nvidia-nvvm` and `nvidia-cuda-crt` unpinned. The result was 13.0
   headers, 13.3 frontend, 13.0 ptxas. Pin all five to the same minor version.
5. The pip CUDA wheels use `cu13/lib` while the generated build file expects
   `cu13/lib64`, ship only `libcudart.so.13`, and ship no `stubs/libcuda.so`.
   Three symlinks; `serve_vllm.sh` now creates them.
6. Installing an older `flashinfer-python` silently **downgrades torch**. Serving
   lives in a separate venv (`.venv-serve`) for exactly this reason.
7. Ollama defaults to the model's native context length -- a 4B model reserved
   43 GB of KV cache and ran 66 % on CPU. Always pin context length.
8. Vendor "disable thinking" request fields are silently ignored by some
   OpenAI-compatible endpoints. Verify acceptance *and* effect.

---

## 9. Open questions for Phase 1

- Does the MoE's 2x latency advantage survive as a **joules** advantage? Power
  draw during MoE inference is not yet measured per-arm.
- Is proposal **quality** equal? One session each is not evidence.
- Is crash rate arm-dependent (4.3)?
- Does `iters_to_first_keep` differ? MoE found its best at iteration 1, dense at
  iteration 3 -- with n=1 this is noise, but it is a cheap metric to track.
- What is the **energy cost of a model swap**, and does it bias arms unevenly
  given the shuffled run table?
