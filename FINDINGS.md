# Findings notebook

Everything measured so far that belongs in the report, organised by **where it goes
in the paper** rather than by when it happened. `EXPERIMENT_PLAN.md` keeps the
chronological record (D1-D20); this file is the writing source.

Status: pilot complete (laptop, RTX 3080 16 GB); server gates G1/G3/G4/G5/G6/G8'/
G10/G11 passed (2x RTX 4000 Ada 20 GB); Phase 1 **restarted** after a first
attempt was discarded at 3/24 runs (section 2.4). **Nothing here answers the
research question** -- these are calibration results and one-session-per-arm
observations. They fix the method and flag what to watch.

**If you read only one thing:** section 2.4. A reproducibility choice
(`temperature: 0`) turned a 10-iteration search into one proposal repeated ten
times, voided an experimental factor, and was invisible to every health metric
the harness reports.

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
| true GPU idle | 8.6 W / 7.9 W | G6 |
| resident-model standby | ~13 W above idle | G5/G6 |
| model swap time | 25 s | Day 4 |
| proposal similarity at temperature 0 | **0.982 mean pairwise** | section 2.4 |
| dense throughput | 35 tok/s | G1 probe |
| MoE throughput | 69 tok/s | G1 probe |
| dense proposal latency | 47-59 s | G1, G3 |
| MoE proposal latency | 25-29 s | G1, G3 |
| session energy (dense, 5 iter) | 60 204 J | G4 |
| session energy (MoE, 5 iter) | 33 837 J | G4 |
| E_prop per proposal, dense | 6 353 J | G4/G5 |
| E_prop per proposal, MoE | 2 366 J (**0.37x**) | G4/G5 |
| proposer power, dense / MoE | 112.8 W / 72.5 W | G5 |
| wasted energy (dense) | 38 242 J (64 %) | G4 |
| training power | ~115.5 W both arms | G5 |

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

### 2.4 Sampling temperature is a search parameter, not a reproducibility knob

**The strongest methodological finding so far, and the least expected.**

`temperature: 0.0` was chosen so sessions would be reproducible. Measured
consequence, over one 10-iteration session:

| proposal pair | similarity |
|---|---|
| 1-2 | 0.969 |
| 2-3 | 0.999 |
| 6-7 | **1.000 (byte-identical)** |
| mean over all 45 pairs | **0.982** |

The session bought **one idea, ten times**. Under greedy patience the recipe
resets to baseline after every failure, so the prompt is nearly identical each
iteration and a zero-temperature model returns a nearly identical answer.

Two consequences, neither visible in any conventional health metric:

1. All three completed sessions ended `no_progress: true`, 0 kept, with **no
   near-misses** -- best observed accuracies -2.72, -0.56, -7.84 pp against
   baseline. The floor was the harness, not the proposer.
2. **`loop_budget` (10 vs 20) silently measured nothing.** Iterations 11-20 were
   the same proposal again. An entire experimental factor was void.

The sessions looked perfectly healthy throughout: 0 errors, 0 infrastructure
failures, `valid: true`, `alignment_ok: true`, sensible energy totals. Only
diffing the proposals against each other revealed it.

After changing to `temperature: 0.7, top_p: 0.95`, the first session kept on
**iteration 1**: 0.7656 -> 0.8366, **+7.1 pp**.

**Generalisable claim for the paper.** In a propose-evaluate-keep loop, greedy
selection restores the same input after every rejection, so a deterministic
proposer is a *degenerate search*: N experiments cost N times the energy of one
and return one result. Any study of agentic loops that fixes temperature at 0 for
reproducibility should check proposal diversity before trusting its iteration
count. Reproducibility belongs at the level of the distribution -- fixed seed,
recorded sampling parameters -- not the single sample.

### 2.5 Infrastructure failure and agent failure are different things

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

### 4.3 What the agent may be told, and what it must work out

Three times the experiment was floored by information the agent could not
observe, and each time the fix was comments only -- no executable line changed,
so the D18 calibration remained valid throughout.

| | missing information | crash / failure rate before | after |
|---|---|---|---|
| D19 | data is already normalised, NCHW, `(1,3,1,1)` broadcast | 4 of 5 crashed | 1 of 5 |
| D21 | torch 2.13 / torchvision 0.28; removed APIs (`verbose=`, ...) | 7 of 9 crashed | 0 of 10 |
| D22 | what a 45 s budget buys (~43 epochs, ~15k steps) | -- | -- |

The line held consistently:

> **Give the agent facts it cannot observe. Never give it the judgement being
> measured.**

Tensor shapes, library versions and the step count of the budget are properties
of the environment that a human collaborator would simply read from the
surrounding code -- withholding them measures guessing, not capability. By
contrast, "change one thing at a time", "retune the learning rate when you swap
optimiser", or "don't augment normalised data" are *reasoning*, and reasoning
quality is the dependent variable. Supplying those would have been tuning the
experiment toward a preferred answer.

This distinction is itself a contribution: **an agentic harness's prompt is part
of the instrument, and an under-specified prompt produces a floor effect that
looks exactly like poor model capability.**

### 4.4 A concrete reasoning failure worth quoting

The dense arm replaced

```python
opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,      # 0.01
                      momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
```

with

```python
opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,    # still 0.01
                        weight_decay=WEIGHT_DECAY)
```

An LR tuned for SGD is roughly 10x too large for Adam. Result: **0.0924
validation accuracy -- chance level for 10 classes.** In the same proposal it
applied `adjust_brightness` and `adjust_saturation` to data the contract states
is already standardised, where photometric transforms are meaningless.

Both are the kind of error that a reasoning pass plausibly catches, which is one
motivation for the thinking-mode study in section 9.

### 4.5 Crash rate may be a dependent variable, not noise

Even with the contract documented, both arms crashed: dense 1/5, MoE 2/5. If that
holds over 24 sessions, "proportion of proposals that run at all" is a legitimate
outcome alongside accuracy and joules -- and it is directly relevant to the energy
question, because a crashed proposal costs a full proposer inference and returns
nothing.

### 4.6 The crash-feedback loop works, but only with information to reason from

Feeding tracebacks into the next prompt was a pilot fix (the 14B had repeated one
hallucinated API call four times without it). Under test: iteration 2 crashed, the
traceback reached iteration 3, and iteration 3 *changed approach* rather than
re-emitting identical code. It still failed, because the traceback says the shapes
disagree but not which is correct. Mechanism sound; starved of context (4.2).

### 4.7 Proposals are large and mostly harmful

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

### 5.4 Idle and the standing cost of residency (G6)

30 minutes with both servers stopped and no workload:

| device | mean power | energy over 30 min |
|---|---|---|
| GPU 0 | 8.63 W | 15 529 J |
| GPU 1 | 7.90 W | 14 230 J |

**True idle is ~8 W per card. The "cold" GPU during a live session drew 17-24 W**
(G5 table). The difference is residency: a loaded model costs roughly **13 W
continuously** whether or not a request is in flight, and the training card holds
a CUDA context and the dataset between phases.

This matters for the self-hosting framing. Over the 523 s dense session, the idle
GPU alone accounted for ~9-12 kJ -- comparable to a third of the proposer's total
draw -- purely for being ready. A practitioner who serves a proposer continuously
pays this even at zero utilisation, and it scales with how long the model stays
resident rather than with how much work it does. Baseline subtraction uses the
measured idle figures above rather than assuming zero.

Notably the two arms' resident standby draw was similar (dense 24.1 W, MoE
21.9 W) despite the MoE holding 15.6 GiB of weights against dense's 9.3 GiB, so
residency cost appears driven more by the card being powered and contexted than by
how much VRAM is occupied. Worth confirming over Phase 1's larger sample.

### 5.5 Determinism

At temperature 0 with a fixed prompt, dense proposal latency had p50 58.87 s and
p95 58.88 s over 40 requests. Session-level energy differences will come from what
the agent does, not from sampling variance in how long it takes to say it.

---

## 6. Preliminary arm comparison -- SUPERSEDED by section 10 (kept for the mechanism)

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

### 6.1 Energy: the mechanism compounds

| quantity | dense | MoE | ratio |
|---|---|---|---|
| proposer power (mean) | 112.8 W | **72.5 W** | 0.64x |
| proposal duration | ~56 s | ~27 s | 0.48x |
| **E_prop, 5 iterations** | **31 765 J** | **11 831 J** | **0.37x** |
| E_prop per proposal | 6 353 J | 2 366 J | 0.37x |
| E_train | 28 439 J | 22 006 J | -- (see caveat) |
| session total | 60 204 J | 33 837 J | 0.56x |
| session wall-clock | 523 s | 335 s | 0.64x |

**The MoE spends 2.7x less energy per proposal.** Two independent effects
multiply: ~3B active parameters draw ~36 % less instantaneous power than 14.8B,
*and* finish in half the time. Neither alone would give this; together they do.

This is the study's central mechanism, visible in a single session pair. Phase 1
determines whether it survives 24 sessions and, crucially, whether proposal
*quality* is preserved -- 2.7x less energy is only interesting if the MoE finds
comparable improvements.

**Caveat on the session total.** The MoE session crashed twice against dense's
once, so it ran three full training phases against dense's four. Training energy
is therefore not comparable between these two sessions, and the 0.56x session
ratio is partly an artifact of crash count. **E_prop per proposal (0.37x) is the
clean comparison**, because both arms issued exactly 5 proposals.

### 6.2 Read this with appropriate caution

The MoE proposes **2.0x faster** at 2.2x the parameter count, which is the
sparsity mechanism doing exactly what the proposal predicted. Whether that
converts into fewer joules *per kept mutation*, and whether proposal quality
holds up over 24 sessions, is what Phase 1 exists to answer. With one session per
arm, the accuracy figures above are indistinguishable from noise.

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
9. EnergiBridge warns `Interval must be at least 200ms to accurately measure CPU
   usage` when run at 100 ms. **The warning does not apply to energy.** It
   concerns the `CPU_USAGE` column, derived from /proc/stat scheduler-tick
   accounting, which is noisy when sampled faster. `PACKAGE_ENERGY` and
   `DRAM_ENERGY` -- the only columns this study consumes -- are RAPL
   accumulators read as deltas, so the interval does not affect the totals, and
   sampling faster slightly reduces the risk of missing a counter wraparound.
   Sampling stayed at 100 ms. Check which columns you actually use before
   changing an instrument setting in response to a warning.

---

### 8.1 Orchestration hazards that produce *plausible but empty* data

These are worse than crashes, because every health indicator stayed green.

1. **A profiler that wraps your program runs it as its own child.** EnergiBridge
   launches the target itself, so the `cwd` and `PYTHONPATH` the runner had
   prepared were never applied: `python -m harness.agent_loop` could not find its
   own package, died instantly, and its traceback went into a pipe nobody read.
   EnergiBridge then sampled an **idle machine** for the full session and wrote a
   perfectly valid CSV. Twenty-four sessions of that would have looked like data.
   Fixed by passing `cwd`/`env`/output redirection through to the child, plus a
   liveness check 8 s after start that fails loudly with the log tail.
2. **`stdout=PIPE` with nobody reading it deadlocks** any target that emits more
   than ~64 KB. A 20-iteration session does. Redirect to a file.
   Consequence: `communicate()` then returns `None` for the redirected stream and
   vendored code calling `.decode()` on it crashes at teardown -- read the file
   instead.
3. **Git ignore rules match case-insensitively on Windows and at any depth.** An
   unanchored `models/` (intended for model weights) silently excluded the
   vendored `ConfigValidator/Config/Models/`, `EventManager/Models/` and
   `ProgressManager/RunTable/Models/` packages. The clone succeeded, `git status`
   was clean on both machines, and 46 unit tests passed -- because the tests
   never import the orchestrator. It failed only at first launch. Anchor the rule
   (`/models/`).
4. **A vendored plugin hardcoded `sudo`.** `EnergiBridge.requires_admin = True`
   blocks forever on a password prompt on a machine where you have no sudo --
   even though `energibridge --summary -- sleep 2` returned joules unprivileged
   on that same host. Made it a profile switch. Verify, do not assume, whether
   elevation is needed.
5. **The orchestrator's md5 check is a feature.** It refuses to merge runs made
   under different configurations. Answering "yes" would have produced a run
   table whose early rows came from a config that never ran.

## 9. Open questions for Phase 1

- Does the MoE's 2x latency advantage survive as a **joules** advantage? Power
  draw during MoE inference is not yet measured per-arm.
- Is proposal **quality** equal? One session each is not evidence.
- Is crash rate arm-dependent (4.3)?
- Does `iters_to_first_keep` differ? MoE found its best at iteration 1, dense at
  iteration 3 -- with n=1 this is noise, but it is a cheap metric to track.
- What is the **energy cost of a model swap**? Measured at **25 s** per swap,
  with ~12 swaps expected across a shuffled 24-run table -- ~5 minutes total, so
  randomisation costs essentially nothing versus blocking by arm. The swap
  happens in `before_run`, outside the measured window.

### 9.1 Analysis-side decisions already required

- **Report `max_val_acc_observed`, not only `best_val_acc`.** When nothing clears
  EPS, `best_val_acc` collapses to the baseline and the outcome has no variance.
  The best accuracy *observed* (kept or not) retains signal and distinguishes a
  proposer that repeatedly lands within 1 pp from one that collapses to 0.09. It
  is recoverable retroactively from `session.jsonl`, so no re-runs are needed.
- **Keep rate is a legitimate binary outcome**, and `E_per_kept_J` is undefined
  when nothing is kept -- state the convention explicitly rather than dropping
  those sessions.
- **Verify the thinking pin empirically** by summing `thinking_tokens` over every
  `propose_end` event per arm. Config acceptance is not proof of effect.
- The energy comparison stands regardless of any of the above, because `E_prop`
  per proposal does not depend on anything being kept.

### 9.2 Stage 2a (reasoning) feasibility PROBED (2026-08-13) -- and a registered prediction

Both arms probed with the real harness prompt, thinking off then on, 2 repeats
each. **Both pass**; Stage 2a is viable as designed.

| | dense 14B | MoE 30B-A3B |
|---|---|---|
| completion tokens, off -> on | 2067 -> 4550 | 2118 -> 5853 |
| **token inflation** | **2.20x** | **2.76x** |
| latency, off -> on | 59.7 s -> 131.6 s | 30.7 s -> 85.1 s |
| throughput | 34.6 tok/s (unchanged) | 68.8 tok/s (unchanged) |
| max_tokens headroom | 2.70x | 1.40x -> raised ceiling to 12288 |
| latency vs 600 s timeout | 22 % | 14 % |
| fenced block present | every reply | every reply |

Throughput is **identical** with thinking on and off in both arms, so reasoning
costs *more tokens*, not slower tokens. Thinking energy is therefore proportional
to thinking length -- a clean interpretation for the analysis.

**Registered prediction, from the probe alone.** Combining inflation with the
measured proposal-phase power draws (dense 112.8 W, MoE 72.5 W):

| configuration | energy per proposal |
|---|---|
| MoE, thinking off | 2.23 kJ |
| **MoE, thinking on** | **6.17 kJ** |
| **dense, thinking off** | **6.73 kJ** |
| dense, thinking on | 14.84 kJ |

**The sparse model can afford to reason for slightly less energy than the dense
model spends not reasoning.** If reasoning also improves proposal quality, the
practitioner claim becomes concrete: on a fixed 20 GB card, sparsity buys you
reasoning for free relative to a dense model of half the size. Stage 2a tests
whether the quality half of that holds; the energy half is already measured.

Note the direction of the inflation difference: the MoE emits *more* reasoning
(2.76x vs 2.20x) and is still cheaper, because its tokens are ~2x cheaper in both
time and power. Reasoning length and reasoning cost are separable.

### 9.3 Stage 2a design (unchanged by the probe)

Thinking tokens are pure proposer energy that yields no artifact, which puts the
question squarely on the thesis: **does reasoning improve proposal quality enough
to justify its joules, and does the answer differ by architecture?** If the MoE's
cheaper tokens make reasoning affordable where dense's do not, that sharpens the
sparsity argument considerably. The AdamW/SGD blunder in 4.4 is exactly the class
of error a reasoning pass tends to catch.

Design: `proposer` x `thinking`, 2x2, fixing `patience=greedy` and
`loop_budget=10`, 3 repetitions = **12 runs, ~5-7 hours**. Adding thinking as a
fourth factor to Phase 1 would instead mean 48 runs and much longer sessions.

Measured schedule from the probe: dense sessions 20 min (off) and 32 min (on),
MoE 15 min and 24 min. **Total ~5.5 h for 12 runs**, ~6.5 h with the usual
re-run buffer.

One configuration change was required and applied: `max_tokens` 8192 -> 12288.
The MoE's reasoning traces averaged 5853 tokens with run-to-run variation of
270 tokens, leaving only 1.40x headroom -- a longer trace would have truncated
mid-study, and a truncated reply is a contract violation confounded with the
factor under test. Raising the ceiling does not change thinking-off behaviour
(those replies stop at ~2100 tokens regardless), so it costs nothing.


---

# 10. PHASE 1 RESULTS (24 sessions, 2026-08-12)

Full factorial: 2 proposers x 2 patience x 2 loop budgets x 3 repetitions.
All 24 sessions valid, 0 quarantined, 0 infrastructure errors.

## 10.1 Headline: the MoE dominates -- cheaper *and* better

| outcome | dense (14B) | MoE (30B-A3B) | change |
|---|---|---|---|
| session GPU energy | 156.4 kJ | **90.7 kJ** | **-42.0 %** |
| proposer energy | 94.3 kJ | **35.7 kJ** | **-62.2 %** |
| training energy | 62.1 kJ | 55.0 kJ | -11.4 % |
| wasted energy | 126.1 kJ | 67.0 kJ | -46.9 % |
| test accuracy | 0.7939 | **0.8224** | **+2.85 pp** |
| kept mutations (total) | 12 | **17** | +42 % |
| sessions with 0 kept | **5 / 12** | **1 / 12** | -- |
| session wall-clock | 1515 s | 1023 s | -32.5 % |
| proposal latency | 61.8 s | 31.9 s | -48.4 % |

**This is not a Pareto trade-off.** The sparse model is cheaper on every energy
measure *and* more accurate. The proposal's hypothesis -- large-model quality at
small-model inference cost -- holds in the strong form on this workload.

Mechanism (section 6.1): ~3B active parameters draw ~36 % less instantaneous
power *and* finish in half the time; the two effects multiply.

## 10.2 Statistics

Primary factor, session GPU energy (ART-ANOVA; Shapiro p = 0.012 so ranks were
aligned, Levene p = 0.46):

| effect | F | p | partial eta^2 |
|---|---|---|---|
| **proposer** | 42.61 | 6.96e-06 | **0.727 (large)** |
| **loop_budget** | 50.58 | 2.47e-06 | **0.760 (large)** |
| patience | 2.77 | 0.116 | 0.147 |
| proposer x loop_budget | ~0 | 1.000 | ~0 |
| proposer x patience | 0.70 | 0.416 | 0.042 |

Pairwise contrasts (Mann-Whitney, Cliff's delta, Holm-corrected over 5 tests):

| contrast | outcome | change | Cliff's delta | p_holm |
|---|---|---|---|---|
| dense vs moe | E_gpu_total | -42.0 % | 0.556 **large** | 0.090 |
| dense vs moe | test_acc | +3.6 % | -0.444 medium | 0.207 |
| budget 10 vs 20 | E_per_kept | +107.6 % | -0.802 **large** | **0.024** |
| patience 1 vs 3 | E_wasted | -9.8 % | 0.042 negligible | 1.000 |
| patience 1 vs 3 | E_per_kept | -9.4 % | 0.117 negligible | 1.000 |

**Report the effect sizes, and be honest about the p-values.** The ART-ANOVA on
`proposer` is unambiguous (p = 7e-06, partial eta^2 = 0.73), but the *pairwise*
Holm-corrected test is p = 0.090. These are not contradictory: the ANOVA models
`loop_budget`, which is itself a huge energy effect, while the pairwise test pools
across it and inherits that variance. With n = 12 per arm, the honest statement is
**a large, consistent effect that the pairwise nonparametric test is underpowered
to certify at alpha = 0.05 after correction.**

## 10.3 A reporting trap: two conventions for energy-per-kept-mutation

`E_per_kept_J` is undefined when a session keeps nothing, and 5 of 12 dense
sessions kept nothing. Averaging the defined values silently **drops the dense
arm's five worst sessions**:

| convention | dense | MoE | ratio |
|---|---|---|---|
| mean of per-session ratios (drops 0-kept) | 112.1 kJ | 74.4 kJ | 1.51x |
| **pooled: total energy / total kept** | **156.4 kJ** | **64.0 kJ** | **2.44x** |

The first convention *understates the effect by 38 %* by discarding exactly the
sessions where the dense arm did worst. **Report the pooled figure**, state the
convention explicitly, and report the zero-kept counts alongside it.

Zero-kept sessions: dense 5/12 vs MoE 1/12 (Fisher exact p = 0.155 -- suggestive,
underpowered at this n).

## 10.4 Reliability, not just central tendency

Best accuracy *observed* in a session, relative to that session's baseline
(kept or not -- this retains signal when nothing clears EPS):

| arm | mean | median |
|---|---|---|
| dense | **-2.21 pp** | +4.69 pp |
| MoE | **+7.23 pp** | +7.42 pp |

The dense arm's mean is *negative* while its median is positive: a minority of
sessions collapse catastrophically (proposals reaching ~0.09, chance level for 10
classes -- section 4.4) and drag the mean below baseline. The MoE's mean and
median nearly coincide.

**So the finding is about reliability as much as quality.** The dense proposer
sometimes destroys the recipe; the MoE consistently improves it. For a
practitioner this matters more than the 2.85 pp mean difference: a session that
ends below baseline has spent its entire energy budget for a negative result.

## 10.5 Doubling the iteration budget does not pay

`loop_budget` 10 -> 20 raised energy-per-kept-mutation by **+107.6 %**
(Cliff's delta -0.80, large, p_holm = 0.024 -- the only Holm-significant pairwise
contrast). Twice the experiments cost twice the energy per useful result.

This factor was void in the discarded first attempt (section 2.4) because
temperature 0 made iterations 11-20 duplicates. With diverse proposals it now
measures something real: within a session, later iterations are *less* productive
per joule -- the cheap wins are found early.

## 10.6 Patience did not matter

Neither `patience` contrast reached even a small effect (Cliff's delta 0.04 and
0.12, both p_holm = 1.000). Allowing a chain of provisionally-kept regressions
neither wasted meaningfully more energy nor found meaningfully better recipes on
this workload.

## 10.7 Thinking mode was pinned, verified empirically

**0 thinking tokens across all 360 proposals**, both arms. D9's confound is closed
by measurement, not by configuration. This also establishes the clean baseline for
the Phase 2 thinking study (section 9.2).

## 10.8 What this does not establish

- **Unequal total capacity** (30.5B vs 14.8B) means sparsity and capacity are not
  separated. The claim is "more efficient per joule on one 20 GB card", not
  "sparsity beats density at matched capacity".
- **Publisher asymmetry**: dense is Qwen's official AWQ build, MoE is third-party.
- **One workload, one budget.** CIFAR-10 CNN at 45 s.
- n = 12 per arm. Effect sizes are large and consistent; pairwise significance
  after correction is not achieved for the proposer contrast.

---

# 11. STAGE 2a RESULTS — does reasoning pay for itself? (12 sessions, 2026-08-13)

`proposer` x `thinking`, 3 repetitions, MoE and dense, greedy, budget 10.
All 12 sessions valid, 0 quarantined.

**Manipulation verified by measurement:** 0 reasoning tokens in every
`thinking=off` session; 23 909-47 904 in every `thinking=on` session. The factor
took, and this is stated from the log rather than from the config.

## 11.1 The cells

| arm | thinking | E_total kJ | E_prop kJ | test acc | max obs | kept | errors | latency s |
|---|---|---|---|---|---|---|---|---|
| dense | off | 108.1 | 59.2 | 0.7572 | 0.7577 | 0.33 | 1.0 | 60.0 |
| dense | **on** | 172.6 | 133.3 | **0.8490** | 0.8622 | 1.33 | 4.3 | 139.3 |
| **MoE** | **off** | **52.1** | **22.0** | 0.8469 | 0.8593 | **2.00** | 3.3 | 31.2 |
| MoE | on | 92.0 | 59.0 | 0.8378 | 0.8470 | 1.00 | 4.3 | 88.4 |

## 11.2 The registered prediction held, but only just

Predicted (FINDINGS 9.2, from probe + power, before running): MoE-with-reasoning
6.17 kJ per proposal against dense-without at 6.73 kJ.

Measured: **MoE-on 5.90 kJ vs dense-off 5.92 kJ.**

The direction is correct and the absolute values are within 5 % of prediction,
but the margin collapsed from 0.56 kJ to **0.02 kJ**. The honest statement is
that *a sparse model can reason for approximately what a dense model of half the
size spends not reasoning* -- not "for less". Reporting this as a confirmed
inequality would over-read a 0.3 % difference.

## 11.3 The finding: reasoning and sparsity are substitutes, not complements

| | effect of turning reasoning on |
|---|---|
| dense | +9.18 pp test accuracy, 0.33 -> 1.33 kept |
| MoE | -0.91 pp test accuracy, 2.00 -> 1.00 kept |

The MoE difference (0.91 pp) is **below the measured keep threshold** (1.23 pp),
so the correct claim is that reasoning does **nothing** for the MoE, not that it
harms it. The dense difference (9.18 pp) is far above it.

Reasoning appears to substitute for whatever the sparse model already brings.
Buying both is buying the same thing twice.

**Correction for an unlucky cell.** Stage 2a's `dense/off` cell is nominally
identical to Phase 1's `dense/greedy/budget-10` cell, and the two disagree: 0.7572
vs 0.8214 test accuracy, 0.33 vs 1.67 kept. Pooling both (n = 6) gives dense-off
at **0.7893**, so the effect of reasoning on the dense arm is **+6.0 pp**, not
+9.2 pp. The larger figure is partly an unlucky draw. Report the pooled version.

## 11.4 The practitioner conclusion inverts the intuitive one

| configuration | E_total | test acc |
|---|---|---|
| **MoE, no reasoning** | **52.1 kJ** | **0.8469** |
| dense, with reasoning | 172.6 kJ | 0.8490 |

**3.3x less energy for an accuracy difference of 0.21 pp** -- six times smaller
than the noise floor, i.e. indistinguishable. For a self-hoster the guidance is
therefore: **spend the budget on sparsity, not on reasoning tokens.** Reasoning is
the expensive way to reach a place the sparse model reaches for free.

## 11.5 Reasoning did not reduce coding errors

Errors per session rose in both arms: dense 1.0 -> 4.3, MoE 3.3 -> 4.3.

This contradicts the expectation stated in 9.2, that a reasoning pass would catch
errors of the AdamW-with-an-SGD-learning-rate class. It did not. Longer
deliberation produced more ambitious proposals, not more correct ones. (Note the
dense-off error count of 1.0 is low partly because that cell barely proposed
anything that ran: 0.33 kept.)

## 11.6 A control worth reporting: the instrument is stable, the agent is not

Phase 1's `dense/greedy/b10` and Stage 2a's `dense/off` are the same
configuration, executed in independent experiments days apart, on the same
hardware:

| | Phase 1 | Stage 2a | agreement |
|---|---|---|---|
| session energy | 107.9 kJ | 108.1 kJ | **0.20 %** |
| test accuracy | 0.8214 | 0.7572 | **6.42 pp apart** |
| kept | 1.67 | 0.33 | -- |

**Energy reproduces to a fifth of a percent; accuracy does not reproduce at all
at n = 3.** The measurement apparatus is therefore not the source of variance --
the agent is. This is the strongest available justification for the design's
pre-registered decision to report effect sizes over significance tests, and a
concrete warning against reading any single three-session cell as a point
estimate of accuracy.

It also means energy comparisons in this study are far better powered than
accuracy comparisons, and should carry more of the argument.
