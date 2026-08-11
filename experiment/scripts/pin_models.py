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
    total_gb = None
    try:                                   # size, when the Hub exposes it
        total = sum(f.get("size") or 0 for f in d.get("siblings", []))
        total_gb = round(total / 1e9, 1) if total else None
    except Exception:                                              # noqa: BLE001
        pass

    return {
        "repo": repo,
        "sha": d.get("sha"),
        "last_modified": d.get("lastModified"),
        "downloads": d.get("downloads"),
        "weight_files": len([f for f in files if f.endswith((".safetensors", ".bin"))]),
        "approx_gb": total_gb,
        "config": [f for f in files if f in ("config.json", "quant_config.json")],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--moe", required=True)
    ap.add_argument("--quantizer", required=True,
                    choices=["awq", "gptq", "gguf-q4_k_m"])
    ap.add_argument("--models-yaml", default=str(ROOT / "serving" / "models.yaml"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    # Fairness check before anything else (D3).
    dense_pub, moe_pub = a.dense.split("/")[0], a.moe.split("/")[0]
    if dense_pub != moe_pub:
        print(f"[warn] different publishers: '{dense_pub}' vs '{moe_pub}'.")
        print("       D3 requires the same quantizer and method for both arms;")
        print("       mixing them makes the comparison a quantizer contrast too.")

    info = {"dense": resolve(a.dense), "moe": resolve(a.moe)}
    for arm, d in info.items():
        print(f"\n{arm}: {d['repo']}")
        print(f"  sha           {d['sha']}")
        print(f"  last modified {d['last_modified']}")
        print(f"  downloads     {d['downloads']:,}" if d["downloads"] else "")
        print(f"  weight files  {d['weight_files']}"
              + (f", ~{d['approx_gb']} GB" if d["approx_gb"] else ""))
        if d["approx_gb"] and d["approx_gb"] > 19:
            print(f"  [warn] ~{d['approx_gb']} GB of weights on a 20.4 GB card leaves")
            print( "         almost nothing for KV cache. Gate G1 will decide it.")

    block = f"""dense:
  base_repo: Qwen/Qwen3.6-27B
  quant_repo: {info['dense']['repo']}
  revision: {info['dense']['sha']}
  quantizer: {a.quantizer}
  approx_weights_gb: {info['dense']['approx_gb'] or 15}

moe:
  base_repo: Qwen/Qwen3.6-35B-A3B
  quant_repo: {info['moe']['repo']}
  revision: {info['moe']['sha']}
  quantizer: {a.quantizer}
  approx_weights_gb: {info['moe']['approx_gb'] or 18}
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
