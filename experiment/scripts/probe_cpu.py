"""Feasibility probe for CPU-only proposer serving (proposal Phase 2 / Stage 1).

The schedule estimate for CPU serving spans 7-20 h purely because CPU inference
throughput on this host has never been measured. This measures it, and turns the
range into a number before a night is spent on it.

It also checks the failure mode that would silently destroy the phase: CPU
proposals are slow enough that `request_timeout_s` may be exceeded on every
iteration, in which case every session dies on INFRASTRUCTURE errors -- which
are not data, and would be quarantined rather than analysed.

Point it at a llama.cpp (or any OpenAI-compatible) endpoint serving on CPU:

    python scripts/probe_cpu.py --model dense --endpoint http://127.0.0.1:8080/v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(ROOT / "profiles" / "server.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--loop-budget", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.profile).read_text())
    p = cfg["proposer"]
    system = (ROOT / "harness" / "templates" / "system.txt").read_text()
    program = (ROOT / "harness" / "templates" / "program.md.j2").read_text()
    train = (ROOT / "workload" / "train.py").read_text()
    user = program.replace("{{ current_source }}", train)

    print(f"probing CPU serving: {a.model} at {a.endpoint}")
    print("one real-shape proposal; this may take several minutes\n")

    t0 = time.time()
    try:
        r = requests.post(f"{a.endpoint.rstrip('/')}/chat/completions",
                          json={"model": a.model, "stream": False,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}],
                                "temperature": p.get("temperature", 0.7),
                                "top_p": p.get("top_p", 0.95),
                                "max_tokens": p.get("max_tokens", 8192)},
                          timeout=a.timeout)
        r.raise_for_status()
    except Exception as e:                                             # noqa: BLE001
        print(f"{BAD} request failed after {time.time()-t0:.0f}s: {str(e)[:200]}")
        return 1
    dt = time.time() - t0
    b = r.json()
    u = b.get("usage", {})
    ctok = u.get("completion_tokens", 0)
    text = b["choices"][0]["message"].get("content") or ""
    rate = ctok / dt if dt else 0

    print(f"{OK} answered in {dt:.0f}s ({u.get('prompt_tokens',0)} prompt + "
          f"{ctok} completion tokens, {rate:.1f} tok/s)")
    has_block = "```" in text
    print(f"{OK if has_block else BAD} fenced code block "
          f"{'present' if has_block else 'ABSENT'}")

    rt = p.get("request_timeout_s", 600)
    tb = p.get("time_budget_s", 900)
    print(f"\n--- against the current harness limits ---")
    for name, lim in (("request_timeout_s", rt), ("time_budget_s", tb)):
        frac = dt / lim
        tag = OK if frac < 0.5 else (WARN if frac < 1.0 else BAD)
        print(f"  {tag} {dt:.0f}s is {frac:.0%} of {name}={lim}s")
    need = max(rt, dt * 2.5)
    if dt > rt * 0.5:
        print(f"  -> raise request_timeout_s to >= {need:.0f}s and time_budget_s "
              f"to >= {need*1.5:.0f}s BEFORE running this phase, or every "
              f"session dies on infrastructure errors.")

    per_iter = dt + 45 + 5
    sess_min = (45 + a.loop_budget * per_iter + 30) / 60
    print(f"\n--- schedule ---")
    print(f"  proposal {dt:.0f}s + 45s training -> session (budget "
          f"{a.loop_budget}) {sess_min:.0f} min")
    for reps, label in ((2, "4 runs (2 cells x 2 reps)"), (4, "8 runs, both arms")):
        print(f"  {label:26s} {(sess_min*reps + reps*2)/60:5.1f} h")

    verdict = has_block and dt < rt * 2
    print(f"\n{'FEASIBLE' if verdict else 'PROBLEMATIC'} — "
          + ("CPU serving is workable for this arm."
             if verdict else
             "this arm is impractical on CPU at the current settings; consider "
             "restricting the CPU phase to the MoE arm and reporting the dense "
             "arm's infeasibility as the result."))

    out = Path(a.out) if a.out else ROOT.parent / "gate_evidence" / f"probe_cpu_{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": a.model, "endpoint": a.endpoint,
                               "latency_s": dt, "completion_tokens": ctok,
                               "tok_per_s": rate, "has_block": has_block,
                               "session_min_budget10": sess_min,
                               "feasible": verdict}, indent=2))
    print(f"written to {out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
