"""Diagnose a proposer endpoint in about a minute, instead of burning sessions.

The pilot lost three Stage-4 attempts to three different causes -- a model name
in the endpoint slot, non-standard parameters rejected by a cloud proxy, and a
local model that simply could not answer inside its timeout. Each cost a full
session to discover. This probes the endpoint directly and reports what happens.

    python scripts/probe_proposer.py --model qwen3:4b
    python scripts/probe_proposer.py --model glm-5.2:cloud
    python scripts/probe_proposer.py --model qwen3:4b --profile profiles/pilot.yaml

Escalates deliberately, cheapest first:

  1. is the server reachable, and does it list the model?
  2. is the model resident, and on GPU or CPU?  (Ollama only)
  3. a tiny prompt with standard parameters      -> is it alive at all?
  4. the real prompt shape with standard params  -> realistic latency and tok/s
  5. the same, plus the profile's extra_params   -> are they accepted?

Anything that fails tells you which layer is broken before a session pays for it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"


def _post(endpoint: str, body: dict, timeout: float) -> tuple[bool, object, float]:
    t0 = time.time()
    try:
        r = requests.post(f"{endpoint.rstrip('/')}/chat/completions",
                          json=body, timeout=timeout)
        dt = time.time() - t0
        if r.status_code >= 400:
            return False, f"HTTP {r.status_code}: {r.text[:300]}", dt
        return True, r.json(), dt
    except requests.Timeout:
        return False, f"timed out after {time.time() - t0:.0f}s", time.time() - t0
    except requests.RequestException as e:
        return False, str(e)[:300], time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(ROOT / "profiles" / "pilot.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", default=None,
                    help="defaults to the profile's dense endpoint")
    ap.add_argument("--timeout", type=float, default=180)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.profile).read_text())
    pcfg = cfg["proposer"]
    endpoint = a.endpoint or pcfg["endpoints"]["dense"]
    base = endpoint.rstrip("/").removesuffix("/v1")

    print(f"probing {a.model} at {endpoint}\n")

    # 1 ------------------------------------------------------------ reachable
    try:
        tags = requests.get(f"{base}/api/tags", timeout=10).json()
        names = [m["name"] for m in tags.get("models", [])]
        print(f"{OK} server reachable, {len(names)} model(s) installed")
        if a.model in names:
            print(f"{OK} '{a.model}' is installed")
        elif a.model.endswith(":cloud"):
            print(f"{WARN} '{a.model}' is a cloud tag; not listed locally, that is "
                  f"normal. It needs `ollama signin`.")
        else:
            print(f"{BAD} '{a.model}' NOT installed. Try: ollama pull {a.model}")
            print(f"       installed: {', '.join(names[:10])}")
            return 1
    except Exception as e:                                             # noqa: BLE001
        print(f"{WARN} could not list models ({e}); continuing")

    # 2 --------------------------------------------------------------- loaded
    try:
        ps = requests.get(f"{base}/api/ps", timeout=10).json()
        for m in ps.get("models", []):
            if m.get("name") == a.model or m.get("model") == a.model:
                size = m.get("size", 0) / 1e9
                vram = m.get("size_vram", 0) / 1e9
                ctx = m.get("context_length") or m.get("context") or 0
                if ctx:
                    print(f"       context length: {ctx:,} tokens")
                    if ctx > 32768:
                        print(f"{WARN} that context allocates a huge KV cache. The pilot "
                              f"saw a 4B model\n"
                              f"       reserve 43 GB at 262,144 tokens and spill 66 %% to "
                              f"CPU, turning\n"
                              f"       seconds into minutes. Set OLLAMA_CONTEXT_LENGTH "
                              f"(8192 is ample:\n"
                              f"       ~2.5k prompt + ~1.5k completion) and restart the "
                              f"server.")
                where = ("GPU" if vram >= size * 0.95 else
                         f"PARTLY ON CPU ({vram:.1f} of {size:.1f} GB on GPU)")
                tag = OK if "GPU" == where else WARN
                print(f"{tag} resident: {size:.1f} GB, {where}")
                if where != "GPU":
                    print("       CPU offload is the usual cause of multi-minute "
                          "latencies. Lower the context length or use a smaller model.")
                break
        else:
            print(f"{WARN} not resident; the first request pays the load cost")
    except Exception:                                                  # noqa: BLE001
        pass

    # 3 ----------------------------------------------------------- tiny probe
    std = {"temperature": 0, "max_tokens": 16}
    ok, body, dt = _post(endpoint, {"model": a.model, "stream": False,
                                    "messages": [{"role": "user",
                                                  "content": "Reply with the word OK."}],
                                    **std}, min(a.timeout, 120))
    if not ok:
        print(f"{BAD} tiny prompt failed after {dt:.0f}s: {body}")
        if "401" in str(body) or "403" in str(body):
            print("       authentication. For a :cloud tag run `ollama signin`.")
        return 1
    print(f"{OK} tiny prompt answered in {dt:.1f}s")

    # 4 ------------------------------------------------------- realistic size
    program = (ROOT / "harness" / "templates" / "program.md.j2").read_text()
    train = (ROOT / "workload" / "train.py").read_text()
    system = (ROOT / "harness" / "templates" / "system.txt").read_text()
    user = program.replace("{{ current_source }}", train)

    # Send exactly what the harness would, including system_suffix and
    # extra_params. Probing with bare parameters measures a different request
    # than the one the experiment will actually make.
    suffix = pcfg.get("system_suffix", "")
    if suffix:
        system = f"{system}\n\n{suffix}"
        print(f"       (system_suffix in effect: {suffix!r})")
    real = {"temperature": 0, "max_tokens": pcfg.get("max_tokens", 4096),
            **(pcfg.get("extra_params") or {})}
    ok, body, dt = _post(endpoint, {"model": a.model, "stream": False,
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}],
                                    **real}, a.timeout)
    if not ok:
        print(f"{BAD} real-size prompt failed after {dt:.0f}s: {body}")
        print("       If this is a reasoning model, thinking tokens are the usual")
        print("       cause: set proposer.system_suffix to '/no_think' (Qwen) or")
        print("       pick a non-reasoning model.")
        print(f"       the harness allows {pcfg.get('request_timeout_s')}s per request "
              f"and {pcfg.get('time_budget_s')}s per iteration.")
        print("       Options: raise those, use a smaller model, or lower max_tokens.")
        return 1

    usage = body.get("usage", {})
    ptok, ctok = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    text = body["choices"][0]["message"]["content"]
    rate = ctok / dt if dt else 0
    print(f"{OK} real prompt answered in {dt:.0f}s "
          f"({ptok} prompt + {ctok} completion tokens, {rate:.0f} tok/s)")

    budget = pcfg.get("request_timeout_s", 240)
    if dt > budget * 0.5:
        print(f"{WARN} that is over half the {budget}s request timeout; sessions will "
              f"be slow and fragile")

    has_block = "```" in text
    print(f"{OK if has_block else WARN} reply "
          f"{'contains' if has_block else 'does NOT contain'} a fenced code block")
    if not has_block:
        print(f"       first 200 chars: {text[:200]!r}")
        print("       this model may not follow the output contract; expect "
              "contract_violation errors (that is DATA, not a bug).")

    # 5 --------------------------------------------------------- extra params
    extra = pcfg.get("extra_params") or {}
    if extra:
        ok, body2, dt2 = _post(endpoint, {"model": a.model, "stream": False,
                                          "messages": [{"role": "user",
                                                        "content": "Reply with OK."}],
                                          **std, **extra}, 60)
        if ok:
            print(f"{OK} extra_params accepted: {list(extra)}")
        else:
            print(f"{WARN} extra_params REJECTED: {list(extra)} -> {body2}")
            print("       the harness will drop them and continue, but the settings "
                  "they encode (e.g. thinking mode) will NOT be in effect.")
            print("       Fix profiles/*.yaml before Phase 1 -- see D9.")

    print("\nprobe complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
