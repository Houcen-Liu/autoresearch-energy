"""Resolve exact commit shas for the proposer weights and write them into models.yaml.

Pinning by hand invites two mistakes that are expensive later: recording `main`
(which moves) instead of a sha, and accidentally pairing arms from different
quantizers. An AWQ build and a GPTQ build of the same weights are not the same
subject, so the fairness rule in D3 requires one publisher and one method for
both arms.

    python scripts/pin_models.py \
        --dense cyankiwi/Qwen3.6-27B-AWQ-INT4 \
        --moe   cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
        --quantizer awq --write

Without --write it prints the block for you to paste. Uses the public HF API over
plain HTTP, so it needs no huggingface_hub and no token for public repos.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
API = "https://huggingface.co/api/models/"


def resolve(repo: str) -> dict:
    r = requests.get(API + repo, timeout=30)
    if r.status_code == 404:
        raise SystemExit(f"[FAIL] '{repo}' not found on the Hub. Check the spelling.")
    r.raise_for_status()
    d = r.json()

    files = [f["rfilename"] for f in d.get("siblings", [])]

    # `usedStorage` is the authoritative on-disk size. Estimating from parameter
    # count is what let a "27B int4" model turn out to be 20.5 GB: Qwen3.6 is
    # multimodal and its quantizer leaves the vision tower, lm_head and the
    # linear-attention projections in 16-bit.
    total_gb = round(d.get("usedStorage", 0) / 1e9, 1) or None

    st = d.get("safetensors") or {}
    dtypes = {k: v for k, v in (st.get("parameters") or {}).items()}
    quantized = sum(v for k, v in dtypes.items() if k in ("I32", "I8", "U8", "I4"))
    unquantized = sum(v for k, v in dtypes.items() if k in ("F16", "BF16", "F32"))

    # The upstream model this was quantized from, straight off the Hub tags.
    # Hardcoding a family here would silently record the wrong base_repo the
    # moment the subjects change -- which they did, when Qwen3.6 failed G1.
    base = ""
    for t in d.get("tags", []):
        if t.startswith("base_model:") and not t.startswith("base_model:quantized:"):
            base = t.split(":", 1)[1]
            break

    return {
        "repo": repo,
        "base_repo": base or "unknown",
        "sha": d.get("sha"),
        "last_modified": d.get("lastModified"),
        "downloads": d.get("downloads"),
        "pipeline": d.get("pipeline_tag"),
        "weight_files": len([f for f in files if f.endswith((".safetensors", ".bin"))]),
        "approx_gb": total_gb,
        "dtypes": {k: round(v / 1e9, 2) for k, v in dtypes.items()},
        "unquantized_b": round(unquantized / 1e9, 2),
        "quantized_b": round(quantized / 1e9, 2),
        "config": [f for f in files if f in ("config.json", "quant_config.json")],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--moe", required=True)
    ap.add_argument("--quantizer", required=True,
                    choices=["awq", "gptq", "gguf-q4_k_m"])
    ap.add_argument("--models-yaml", default=str(ROOT / "serving" / "models.yaml"))
    ap.add_argument("--card-gib", type=float, default=20.0,
                    help="usable VRAM per card; RTX 4000 Ada = 20475 MiB = 20.0 GiB")
    ap.add_argument("--kv-headroom-gib", type=float, default=2.5,
                    help="reserve for KV cache, activations and CUDA context")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    # Fairness check before anything else (D3).
    dense_pub, moe_pub = a.dense.split("/")[0], a.moe.split("/")[0]
    if dense_pub != moe_pub:
        print(f"[warn] different publishers: '{dense_pub}' vs '{moe_pub}'.")
        print("       D3 requires the same quantizer and method for both arms;")
        print("       mixing them makes the comparison a quantizer contrast too.")

    budget_gib = a.card_gib - a.kv_headroom_gib
    info = {"dense": resolve(a.dense), "moe": resolve(a.moe)}
    fails = []

    for arm, d in info.items():
        gib = (d["approx_gb"] / 1.073741824) if d["approx_gb"] else None
        print(f"\n{arm}: {d['repo']}")
        print(f"  sha           {d['sha']}")
        print(f"  last modified {d['last_modified']}")
        if d["downloads"]:
            print(f"  downloads     {d['downloads']:,}")
        print(f"  base model    {d['base_repo']}")
        print(f"  pipeline      {d['pipeline']}")
        if gib:
            print(f"  ON-DISK SIZE  {d['approx_gb']} GB = {gib:.1f} GiB")
        if d["dtypes"]:
            print(f"  dtypes        {d['dtypes']}")
            if d["unquantized_b"] > 0.5:
                print(f"  [warn] {d['unquantized_b']}B parameters are NOT quantized "
                      f"(16-bit) ~= {d['unquantized_b'] * 2:.1f} GB.")
                print( "         Multimodal towers, lm_head and linear-attention")
                print( "         projections are commonly excluded by quantizers.")
        if d["pipeline"] and "image" in str(d["pipeline"]):
            print( "  [warn] multimodal checkpoint: the vision tower ships with the")
            print( "         weights and is usually left in 16-bit. A text-only")
            print( "         sibling is far smaller if one exists.")

        if gib and gib > budget_gib:
            fails.append((arm, gib))
            print(f"  *** WILL NOT FIT *** {gib:.1f} GiB of weights against a "
                  f"{a.card_gib:.1f} GiB card")
            print(f"      minus {a.kv_headroom_gib:.1f} GiB for KV cache and context "
                  f"= {budget_gib:.1f} GiB available.")
        elif gib:
            print(f"  [ ok ] fits: {gib:.1f} GiB <= {budget_gib:.1f} GiB available")

    if fails:
        print("\n*** GATE G1 FAILS ON WEIGHTS, BEFORE SERVING ***")
        for arm, gib in fails:
            print(f"  {arm}: {gib:.1f} GiB")
        print("""
  No --max-model-len setting fixes this: the weights alone exceed the card.
  Options, in order of preference:
    1. A smaller matched pair from the same family (text-only if possible).
    2. The pre-registered Granite backup (becomes a size contrast, not sparsity).
    3. llama.cpp GGUF at a lower quant for BOTH arms (fairness rule, D3).
  Re-run this script on candidates BEFORE downloading them.""")
        if not a.write:
            return 1

    block = f"""dense:
  base_repo: {info['dense']['base_repo']}
  quant_repo: {info['dense']['repo']}
  revision: {info['dense']['sha']}
  quantizer: {a.quantizer}
  approx_weights_gb: {info['dense']['approx_gb'] or 'unknown'}

moe:
  base_repo: {info['moe']['base_repo']}
  quant_repo: {info['moe']['repo']}
  revision: {info['moe']['sha']}
  quantizer: {a.quantizer}
  approx_weights_gb: {info['moe']['approx_gb'] or 'unknown'}
"""

    if not a.write:
        print("\n--- paste into serving/models.yaml (or re-run with --write) ---\n")
        print(block)
        return 0

    p = Path(a.models_yaml)
    lines = p.read_text().splitlines(keepends=True)

    # Line-based, not regex: the file carries comment blocks between sections and
    # a regex that walks over them is easy to get subtly wrong.
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("dense:"))
        stop = next(i for i, l in enumerate(lines) if l.startswith("backup_dense:"))
    except StopIteration:
        raise SystemExit(f"[FAIL] could not find the dense/backup_dense markers in {p}")

    # keep any comment block that introduces the backups
    while stop > start and (lines[stop - 1].startswith("#") or not lines[stop - 1].strip()):
        stop -= 1

    p.write_text("".join(lines[:start]) + block + "\n" + "".join(lines[stop:]))
    print(f"\n[ ok ] written to {p}")
    print("       commit this before Phase 1 — the revisions are part of the result.")

    check = yaml.safe_load(p.read_text())
    for arm in ("dense", "moe"):
        rev = check[arm]["revision"]
        if not rev or rev == "TODO" or len(str(rev)) < 20:
            print(f"[FAIL] {arm} revision looks wrong: {rev!r}")
            return 1
    print("[ ok ] both arms pinned to exact shas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
