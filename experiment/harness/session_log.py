"""Append-only JSONL session log. The single source of truth for everything the
analysis pipeline reads about agent behaviour and iteration timing.

Timestamps: `t` is time.time() (wall clock, joinable with the energy traces) and
`m` is time.monotonic() (immune to clock adjustments, used for durations).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class SessionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, ev: str, **fields: Any) -> dict:
        rec = {"ev": ev, "t": time.time(), "m": time.monotonic(), **fields}
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        return rec

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_log(path: str | Path) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iterations_from_log(records: list[dict]) -> list[dict]:
    """Fold the event stream into one row per iteration.

    Produces the phase boundaries the energy aligner joins on.
    """
    iters: dict[int, dict] = {}

    def slot(i: int) -> dict:
        return iters.setdefault(i, {
            "iter": i,
            "propose_t0": None, "propose_t1": None,
            "train_t0": None, "train_t1": None,
            "prompt_tokens": 0, "completion_tokens": 0, "proposer_latency_s": None,
            "guard_ok": None, "guard_violations": [],
            "val_acc": None, "decision": None, "exit_code": None,
            "discarded_by_rollback": False,
        })

    for r in records:
        ev, i = r.get("ev"), r.get("iter")
        if i is None:
            continue
        s = slot(i)
        if ev == "propose_start":
            s["propose_t0"] = r["t"]
        elif ev == "propose_end":
            s["propose_t1"] = r["t"]
            s["prompt_tokens"] = r.get("prompt_tokens", 0)
            s["completion_tokens"] = r.get("completion_tokens", 0)
            s["proposer_latency_s"] = r.get("latency_s")
        elif ev == "guard":
            s["guard_ok"] = r.get("ok")
            s["guard_violations"] = r.get("violations", [])
        elif ev == "train_start":
            s["train_t0"] = r["t"]
        elif ev == "train_end":
            s["train_t1"] = r["t"]
            s["val_acc"] = r.get("val_acc")
            s["exit_code"] = r.get("exit")
        elif ev == "decision":
            s["decision"] = r.get("decision")

    # A patience rollback retroactively discards a chain of provisionally-kept
    # iterations; their energy belongs to E_wasted.
    for r in records:
        if r.get("ev") == "rollback":
            for i in r.get("discarded_iters", []):
                if i in iters:
                    iters[i]["discarded_by_rollback"] = True
                    iters[i]["decision"] = "reverted"

    return [iters[k] for k in sorted(iters)]
