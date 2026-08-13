"""Feasibility probe for the thinking-mode study: measure, do not assume.

Phase 2's cost hinges on one unknown -- how much reasoning inflates output --
and on two failure modes that would invalidate the study before it starts:

  * if reasoning overruns max_tokens the reply truncates, and the experiment
    measures truncation rather than reasoning;
  * if latency approaches request_timeout_s, sessions die on infrastructure
    errors, which are not data.

Sends the REAL prompt the harness sends, with thinking off and then on, N times
each, and reports the inflation factor with the schedule implications worked out.

    python scripts/probe_thinking.py --model moe --endpoint http://127.0.0.1:8001/v1
    python scripts/probe_thinking.py --model dense --endpoint http://127.0.0.1:8000/v1 -n 3
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"


def one(endpoint, model, system, user, params, timeout):
    t0 = time.time()
    r = requests.post(f"{endpoint.rstrip('/')}/chat/completions",
                      json={"model": model, "stream": False,
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}],
                            **params}, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    b = r.json()
    ch = b["choices"][0]
    text = ch["message"].get("content") or ""
    u = b.get("usage", {})
    think = sum(len(m.split()) for m in re.findall(r"<think>(.*?)</think>", text, re.S))
    return {"latency_s": dt,
            "completion_tokens": u.get("completion_tokens", 0),
            "prompt_tokens": u.get("prompt_tokens", 0),
            "finish_reason": ch.get("finish_reason"),
            "has_block": "```" in text,
            "think_words": think}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(ROOT / "profiles" / "server.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("-n", "--repeats", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.profile).read_text())
    p = cfg["proposer"]
    system = (ROOT / "harness" / "templates" / "system.txt").read_text()
    program = (ROOT / "harness" / "templates" / "program.md.j2").read_text()
    train = (ROOT / "workload" / "train.py").read_text()
    user = program.replace("{{ current_source }}", train)

    base = {"temperature": p.get("temperature", 0.7), "top_p": p.get("top_p", 0.95),
            "max_tokens": p.get("max_tokens", 8192)}
    modes = {
        "thinking_off": {**base, "chat_template_kwargs": {"enable_thinking": False}},
        "thinking_on":  {**base, "chat_template_kwargs": {"enable_thinking": True}},
    }

    print(f"probing {a.model} at {a.endpoint}, {a.repeats} repeat(s) per mode")
    print(f"max_tokens={base['max_tokens']}  request_timeout_s="
          f"{p.get('request_timeout_s')}  time_budget_s={p.get('time_budget_s')}\n")

    res = {}
    for name, params in modes.items():
        runs = []
        for i in range(a.repeats):
            try:
                r = one(a.endpoint, a.model, system, user, params, a.timeout)
            except Exception as e:                                     # noqa: BLE001
                print(f"{BAD} {name} run {i+1} failed: {str(e)[:200]}")
                return 1
            runs.append(r)
            print(f"  {name:13s} run {i+1}: {r['latency_s']:6.1f}s  "
                  f"{r['completion_tokens']:5d} tok  "
                  f"{r['completion_tokens']/max(r['latency_s'],1e-9):5.1f} tok/s  "
                  f"finish={r['finish_reason']}  block={'y' if r['has_block'] else 'N'}")
        res[name] = {
            "latency_s": st.mean(x["latency_s"] for x in runs),
            "completion_tokens": st.mean(x["completion_tokens"] for x in runs),
            "truncated": any(x["finish_reason"] == "length" for x in runs),
            "all_have_block": all(x["has_block"] for x in runs),
            "runs": runs,
        }

    off, on = res["thinking_off"], res["thinking_on"]
    tok_x = on["completion_tokens"] / max(off["completion_tokens"], 1)
    lat_x = on["latency_s"] / max(off["latency_s"], 1e-9)
    print(f"\n--- inflation ---")
    print(f"  tokens   {off['completion_tokens']:.0f} -> {on['completion_tokens']:.0f}"
          f"   ({tok_x:.2f}x)")
    print(f"  latency  {off['latency_s']:.1f}s -> {on['latency_s']:.1f}s   ({lat_x:.2f}x)")

    ok = True
    if on["truncated"]:
        ok = False
        print(f"\n{BAD} TRUNCATED at max_tokens={base['max_tokens']}. The study would "
              f"measure truncation,\n       not reasoning. Raise max_tokens above "
              f"{on['completion_tokens']*1.5:.0f} and re-probe.")
    else:
        head = base["max_tokens"] / max(on["completion_tokens"], 1)
        tag = OK if head > 1.3 else WARN
        print(f"\n{tag} max_tokens headroom {head:.2f}x "
              f"({on['completion_tokens']:.0f} of {base['max_tokens']})")
        if head <= 1.3:
            ok = False
            print("       too tight -- a longer reasoning trace will truncate mid-study.")

    rt = p.get("request_timeout_s", 600)
    frac = on["latency_s"] / rt
    tag = OK if frac < 0.5 else (WARN if frac < 0.8 else BAD)
    print(f"{tag} latency is {frac:.0%} of request_timeout_s={rt}s")
    if frac >= 0.8:
        ok = False
        print(f"       raise request_timeout_s to >= {on['latency_s']*2.5:.0f}s, "
              f"and time_budget_s with it.")

    if not on["all_have_block"]:
        ok = False
        print(f"{BAD} a thinking reply had no fenced code block -- reasoning models "
              f"sometimes bury\n       the answer. Contract violations would be "
              f"confounded with the factor.")
    else:
        print(f"{OK} fenced code block present in every reply")

    # schedule implication for the proposed 2x2, budget 10, 3 reps
    per_iter_off = off["latency_s"] + 45 + 5
    per_iter_on = on["latency_s"] + 45 + 5
    sess = lambda pi: (45 + 10 * pi + 30) / 60
    total = 3 * (sess(per_iter_off) + sess(per_iter_on)) + 6 * 2  # +cooldown/swap
    print(f"\n--- schedule (this arm only: 3 reps x {{off,on}}, budget 10) ---")
    print(f"  session off {sess(per_iter_off):.0f} min | on {sess(per_iter_on):.0f} min"
          f"  ->  {total/60:.1f} h for this arm; both arms ~{2*total/60:.1f} h")

    print(f"\n{'PROBE PASSED' if ok else 'PROBE FAILED -- fix the above before running Phase 2'}")
    out = Path(a.out) if a.out else ROOT.parent / "gate_evidence" / f"probe_thinking_{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": a.model, "endpoint": a.endpoint,
                               "max_tokens": base["max_tokens"],
                               "token_inflation": tok_x, "latency_inflation": lat_x,
                               "passed": ok, "modes": res}, indent=2, default=str))
    print(f"written to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
