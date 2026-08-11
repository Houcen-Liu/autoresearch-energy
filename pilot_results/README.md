# Pilot results — calibration evidence

Measured on an RTX 3080 Laptop (16 GB), single GPU. **Energy attribution is invalid on
one GPU**, so these are behavioural and calibration measurements only. Joules come from
the two-GPU server.

They are committed because they are what set the experiment's fixed variables, and
because the report needs to show the noise floor that justifies EPS.

| File | What it establishes |
|---|---|
| `noise_floor_45s.json` | **EPS = 0.0073** from 8 repeats with a 30 s cooldown (detrended SD 0.37 pp). Without cooldown the same measurement gave 1.01 pp and EPS 0.0202 — 2.7x too strict (D6, D11, D13). |
| `noise_floor_240s.json` | The 240 s comparison: SD 0.91 pp, and the non-monotonic series that exposed a false positive in the drift detector (D13). |
| `headroom.json` | **Gate G11.** +14.5 pp headroom at SNR 30 (45 s) vs +16.3 pp at SNR 63 (240 s). 240 s buys 1.8 pp more for 5.3x the joules → budget set to 45 s (D15). |
| `budget_sensitivity.json` | The baseline's accuracy-vs-compute curve: knee at ~20 s, flat to 240 s. Shows why the baseline's own curve cannot pick the budget (D12). |
| `run_qwen3_4b/` | A working session: 0.7604 → 0.8486 val, 0.8344 test, 3 kept mutations. Guard-rejected on 4 of 6 iterations for emitting file fragments. |
| `run_qwen3_14b/` | Zero kept. Every proposal was textbook-correct and budget-hostile (LR 0.01 → 0.001 with batch 128 → 256), so every recipe was undertrained at 45 s. |

The last two together are the substantive behavioural finding: **format compliance and
content quality are independent axes.** The 14B complied perfectly and improved nothing;
the 4B broke the output contract two thirds of the time and gained 8.8 points. A single
"did it work" number ranks them backwards.

`session.jsonl` and `proposal_*.py` are included; model checkpoints and NVML traces are
not (size, and meaningless without per-device attribution).
