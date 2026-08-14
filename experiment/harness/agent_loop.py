"""arloop -- the agentic AutoML loop.

Re-implements the `autoresearch` protocol (single editable training file, fixed
wall-clock training budget, one scalar metric, keep/revert on that metric) with
loop control in Python rather than delegated to a coding-agent CLI. See
EXPERIMENT_PLAN.md section D1 for why.

Usage:
    python -m harness.agent_loop --profile profiles/server.yaml \
        --proposer moe --patience 3 --loop-budget 20 --run-dir <dir> --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import guards
from harness.errors import (DEFAULT_MAX_INFRA_ERROR_RATE, ErrorClass,
                            ProposerError)
from harness.proposer import Proposer, ProposerResponse, StubProposer
from harness.recipe_repo import RecipeRepo
from harness.session_log import SessionLog

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "harness" / "templates"


def _thinking_override(extra: dict | None, thinking: bool | None) -> dict | None:
    """Return extra_params with reasoning forced on or off.

    Phase 1 pinned reasoning off for both arms as a fixed variable (D9). Phase 2
    promotes it to a factor, so the run table -- not the profile -- decides. The
    override is applied here rather than by editing the profile per run, because
    a profile mutated mid-experiment is exactly the kind of state that makes runs
    non-comparable after the fact.
    """
    if thinking is None:
        return extra
    out = dict(extra or {})
    kw = dict(out.get("chat_template_kwargs") or {})
    kw["enable_thinking"] = bool(thinking)
    out["chat_template_kwargs"] = kw
    return out


# --------------------------------------------------------------------- render
def render_program_md(**ctx) -> str:
    """Minimal Jinja-subset renderer, so the harness has no template dependency."""
    try:
        from jinja2 import Template
        return Template((TEMPLATES / "program.md.j2").read_text(encoding="utf-8")).render(**ctx)
    except ImportError:
        text = (TEMPLATES / "program.md.j2").read_text(encoding="utf-8")
        # Strip the conditional block we do not need, then substitute.
        import re
        keep_greedy = ctx["patience"] == 1
        text = re.sub(
            r"\{%\s*if patience == 1\s*-%\}(.*?)\{%-\s*else\s*-%\}(.*?)\{%-\s*endif\s*%\}",
            lambda m: m.group(1) if keep_greedy else m.group(2), text, flags=re.DOTALL)
        for k, v in ctx.items():
            text = text.replace("{{ " + k + " }}", str(v))
        return text


def history_table(rows: list[dict], max_rows: int | None = None) -> str:
    if not rows:
        return "_(no experiments yet)_"
    prefix = ""
    if max_rows is not None and len(rows) > max_rows:
        omitted = rows[:-max_rows]
        counts: dict[str, int] = {}
        for row in omitted:
            outcome = str(row.get("outcome", "unknown")).split(" (")[0]
            counts[outcome] = counts.get(outcome, 0) + 1
        compact = ", ".join(
            f"{n} {outcome}" for outcome, n in sorted(counts.items())
        )
        prefix = (f"_({len(omitted)} earlier attempts omitted from this prompt: "
                  f"{compact}; full history remains in session.jsonl)_\n\n")
        rows = rows[-max_rows:]
    head = "| # | change | val_acc | outcome |\n|---|---|---|---|"
    body = "\n".join(
        f"| {r['iter']} | {r['rationale'][:70]} | "
        f"{('%.4f' % r['val_acc']) if r['val_acc'] is not None else '—'} | {r['outcome']} |"
        for r in rows)
    return prefix + head + "\n" + body


# --------------------------------------------------------------------- runner
def run_training(workdir: Path, run_dir: Path, iteration: int, cfg: dict,
                 device: str) -> dict:
    """Execute a candidate train.py in a subprocess and return its result."""
    result_path = workdir / "result.json"
    ckpt_path = workdir / "model.pt"
    result_path.unlink(missing_ok=True)
    # Every candidate trains from scratch. A prior iteration's checkpoint must
    # not make a recipe that writes only result.json look checkpoint-capable.
    ckpt_path.unlink(missing_ok=True)

    env = dict(os.environ)
    env["AR_DATA_DIR"] = str(Path(cfg["workload"]["data_dir"]).resolve())
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"]["train"])
    env["PYTHONPATH"] = str(ROOT / "workload") + os.pathsep + env.get("PYTHONPATH", "")

    # The subprocess timeout must always exceed the training budget itself,
    # plus startup, data load, CUDA warm-up and evaluation. Deriving it here
    # makes a profile whose train_timeout_s is smaller than train_seconds
    # impossible to misconfigure -- the pilot lost five runs to exactly that.
    budget = float(cfg["workload"]["train_seconds"])
    timeout_s = max(float(cfg["workload"].get("train_timeout_s", 0)),
                    budget * 1.25 + 120)

    log_path = run_dir / f"train_{iteration:03d}.log"
    cmd = [sys.executable, str(workdir / "train.py"),
           "--data-dir", env["AR_DATA_DIR"],
           "--out", str(result_path),
           "--checkpoint", str(ckpt_path),
           "--device", "cuda:0" if device != "cpu" else "cpu",
           "--train-seconds", str(cfg["workload"]["train_seconds"])]

    with log_path.open("w", encoding="utf-8") as lf:
        try:
            proc = subprocess.run(cmd, cwd=str(workdir), env=env, stdout=lf,
                                  stderr=subprocess.STDOUT,
                                  timeout=timeout_s)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            return {"errored": True, "exit": -9, "val_acc": None,
                    "error": f"training exceeded {timeout_s:.0f}s "
                             f"(budget {budget:.0f}s)",
                    "error_class": str(ErrorClass.TRAIN_TIMEOUT)}

    if code != 0 or not result_path.exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        return {"errored": True, "exit": code, "error": tail, "val_acc": None,
                "error_class": str(ErrorClass.TRAIN_CRASH)}

    res = json.loads(result_path.read_text())
    res["errored"] = False
    res["exit"] = 0
    # A recipe that reports a result but saves no checkpoint is not a crash: the
    # iteration still counts, it simply cannot win the final evaluation.
    if ckpt_path.exists():
        shutil.copy(ckpt_path, run_dir / f"model_{iteration:03d}.pt")
        res["checkpoint"] = True
    else:
        res["checkpoint"] = False
    return res


# ----------------------------------------------------------------------- loop
def run_session(cfg: dict, *, proposer_arm: str, patience: int, loop_budget: int,
                run_dir: Path, seed: int, stub: bool = False,
                proposer_model: str | None = None,
                thinking: bool | None = None,
                temperature: float | None = None,
                baseline_path: Path | None = None,
                proposer_max_tokens: int | None = None,
                history_max_rows: int | None = None) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = run_dir / "recipe"

    if cfg.get("attribution") == "none":
        print("[arloop] WARNING: profile declares attribution=none. "
              "E_prop/E_train will NOT be emitted. Functional test only.")

    baseline_path = Path(baseline_path) if baseline_path else ROOT / "workload" / "train.py"
    baseline_source = baseline_path.read_text(encoding="utf-8")
    repo = RecipeRepo(workdir, baseline_source)
    for aux in ("prepare_cifar.py",):
        shutil.copy(ROOT / "workload" / aux, workdir / aux)

    # Energy sampling lives here, not only in RunnerConfig, so that any session --
    # a smoke test, a gate check, a pilot run -- carries its own energy trace.
    sampler = None
    if cfg.get("energy", {}).get("nvml", False):
        try:
            from measurement.nvml_sampler import NvmlSampler
            gpus = cfg["gpus"]
            devices = sorted({int(gpus["train"]), int(gpus["proposer"])}) \
                if str(gpus["train"]).isdigit() else None
            sampler = NvmlSampler(run_dir / "nvml.csv",
                                  hz=cfg["energy"].get("nvml_hz", 10),
                                  devices=devices)
            sampler.start()
        except Exception as e:                                     # noqa: BLE001
            print(f"[arloop] NVML sampling unavailable: {e}")
            sampler = None

    log = SessionLog(run_dir / "session.jsonl")
    train_seconds = float(cfg["workload"]["train_seconds"])      # effective runtime
    # The value the agent must not change is the one the baseline declares; in
    # pilot mode the effective runtime is shorter, and that is the harness's
    # business, not the agent's.
    contract_seconds = guards.extract_train_seconds(baseline_source)
    if contract_seconds is None:
        raise RuntimeError(f"baseline recipe {baseline_path} declares no TRAIN_SECONDS")
    eps = float(cfg["loop"]["eps"])
    if history_max_rows is not None and history_max_rows < 1:
        raise ValueError("history_max_rows must be at least 1")

    log.emit("session_start", cell={"proposer": proposer_arm, "patience": patience,
                                    "loop_budget": loop_budget},
             seed=seed, attribution=cfg.get("attribution"),
             train_seconds=train_seconds, contract_seconds=contract_seconds,
             stub=bool(stub), synthetic_data=_is_synthetic(cfg),
             history_max_rows=history_max_rows,
             versions=_versions())

    # --- proposer ----------------------------------------------------------
    pcfg = cfg["proposer"]
    if stub:
        prop = StubProposer(seed=seed)
    else:
        prop = Proposer(
            endpoint=pcfg["endpoints"][proposer_arm],
            model=proposer_model or pcfg.get("model_names", {}).get(proposer_arm, proposer_arm),
            temperature=(pcfg["temperature"] if temperature is None else temperature),
            top_p=pcfg["top_p"],
            max_tokens=(pcfg["max_tokens"] if proposer_max_tokens is None
                        else proposer_max_tokens), seed=seed,
            timeout_s=pcfg["request_timeout_s"],
            max_retries=pcfg.get("max_retries", 1),
            time_budget_s=pcfg.get("time_budget_s"),
            extra_params=_thinking_override(pcfg.get("extra_params"), thinking),
            system_suffix=pcfg.get("system_suffix", ""),
            prompt_suffix=pcfg.get("prompt_suffix", ""),
        )
    system = (TEMPLATES / "system.txt").read_text(encoding="utf-8")
    # The exact request is a fixed variable of the experiment (sampling params,
    # thinking mode, context length). Record it, do not assume it.
    log.emit("proposer_config", **prop.request_manifest())

    # --- baseline ----------------------------------------------------------
    print(f"[arloop] baseline training {train_seconds:.0f}s ...", flush=True)
    log.emit("baseline_start")
    base = run_training(workdir, run_dir, 0, cfg, _device(cfg))
    if base["errored"]:
        log.emit("session_abort", reason="baseline training failed", detail=base.get("error"))
        log.close()
        raise RuntimeError("baseline training failed; the experiment cannot start")
    print(f"[arloop] baseline val_acc {base['val_acc']:.4f}; "
          f"{loop_budget} iterations to go", flush=True)
    log.emit("baseline_eval", val_acc=base["val_acc"], train_seconds=base["train_seconds"])

    best_acc = base["val_acc"]
    best_sha = repo.head()
    # Keep the winning checkpoint identity alongside the winning recipe SHA.
    # Recovering it later by matching validation accuracy is unsafe: accuracy is
    # discrete, so a reverted proposal can tie the best score and would then be
    # mistaken for the winner by a reverse search through history.
    best_iter = 0
    regressions = 0
    think_total = 0
    provisional: list[int] = []
    history: list[dict] = []
    counts = {"kept": 0, "reverted": 0, "rejected": 0, "errored": 0}
    # Errors split by who failed: see harness/errors.py.
    err_counts = {str(e): 0 for e in ErrorClass}
    max_infra_rate = float(cfg.get("loop", {}).get("max_infra_error_rate",
                                                   DEFAULT_MAX_INFRA_ERROR_RATE))
    max_consecutive_infra = int(cfg.get("loop", {}).get(
        "max_consecutive_infra_errors", 3))
    aborted_early = False
    guard_feedback = ""

    # --- iterations --------------------------------------------------------
    for i in range(1, loop_budget + 1):
        if i > 1 and _should_abort(err_counts, i - 1, max_infra_rate, history,
                                   max_consecutive_infra):
            aborted_early = True
            log.emit("session_abort_early", iter=i - 1,
                     reason="persistent infrastructure errors",
                     infra_errors=_infra_total(err_counts))
            print(f"[arloop] aborting after {i - 1} iterations: persistent "
                  f"infrastructure errors. Check the last HTTP response and server log.")
            break

        prompt = render_program_md(
            train_seconds=train_seconds, patience=patience, loop_budget=loop_budget,
            eps=eps, history=history_table(history, history_max_rows),
            current_source=repo.read(),
            # Measured facts about the budget, not advice. Without these the
            # agent has no idea that the budget buys ~43 epochs, and proposes
            # schedules built for hundreds (CosineAnnealingLR(T_max=100) was
            # proposed repeatedly). The harness knows these numbers; the agent
            # cannot see them.
            baseline_val_acc=f"{base['val_acc']:.4f}",
            baseline_epochs=base.get("epochs_completed", "?"),
            baseline_steps=base.get("steps", "?"),
        )
        if guard_feedback:
            prompt += "\n\n## Feedback on your last proposal\n\n" + guard_feedback
            guard_feedback = ""

        _t_iter = time.time()
        print(f"[arloop] iter {i}/{loop_budget}  proposing "
              f"(best {best_acc:.4f}, kept {counts['kept']}, "
              f"errored {counts['errored']}) ...", flush=True)
        prompt_chars = len(system) + len(prompt)
        log.emit("propose_start", iter=i, prompt_chars=prompt_chars,
                 prompt_tokens_estimate=(prompt_chars + 3) // 4)
        try:
            resp: ProposerResponse = prop.complete(system, prompt)
        except ProposerError as e:
            ec = e.error_class
            for _a in e.attempts:
                if getattr(_a, "raw", ""):
                    (run_dir / f"rejected_{i:03d}_attempt{_a.n}.txt").write_text(
                        f"# outcome: {_a.outcome}\n# finish_reason: "
                        f"{_a.finish_reason}\n# detail: {_a.detail}\n\n{_a.raw}",
                        encoding="utf-8")
            print(f"[arloop] iter {i}: {ec} -- "
                  f"{e.attempts[-1].detail if e.attempts else e}", flush=True)
            # Tokens and latency of the FAILED attempts are still recorded: that
            # energy was spent, and the previous version lost it entirely.
            log.emit("propose_end", iter=i,
                     prompt_tokens=sum(a.prompt_tokens for a in e.attempts),
                     completion_tokens=sum(a.completion_tokens for a in e.attempts),
                     latency_s=sum(a.latency_s for a in e.attempts),
                     attempts=len(e.attempts), rationale="",
                     attempt_log=[{k: v for k, v in vars(a).items() if k != "raw"}
                                  for a in e.attempts])
            log.emit("propose_error", iter=i, error=str(e), error_class=str(ec),
                     is_infra=ec.is_infra)
            log.emit("decision", iter=i, decision="errored", error_class=str(ec),
                     best_acc=best_acc, regressions=regressions)
            counts["errored"] += 1
            err_counts[str(ec)] += 1
            history.append({"iter": i,
                            "rationale": f"({ec} — no proposal)",
                            "val_acc": None, "outcome": "errored",
                            "error_class": str(ec)})
            continue
        except Exception as e:                                    # noqa: BLE001
            log.emit("propose_error", iter=i, error=str(e),
                     error_class=str(ErrorClass.INFRA_TRANSPORT), is_infra=True)
            log.emit("decision", iter=i, decision="errored", best_acc=best_acc,
                     regressions=regressions)
            counts["errored"] += 1
            err_counts[str(ErrorClass.INFRA_TRANSPORT)] += 1
            history.append({"iter": i, "rationale": "(proposer failed)",
                            "val_acc": None, "outcome": "errored",
                            "error_class": str(ErrorClass.INFRA_TRANSPORT)})
            continue
        _think = sum(a.thinking_tokens for a in resp.attempt_log)
        think_total += _think
        print(f"[arloop] iter {i}: proposal in {resp.latency_s:.0f}s, "
              f"{resp.total_completion_tokens} completion tokens"
              + (f" ({_think} of them reasoning)" if _think else "")
              + f" ({resp.attempts} attempt(s))", flush=True)
        log.emit("propose_end", iter=i, prompt_tokens=resp.total_prompt_tokens,
                 completion_tokens=resp.total_completion_tokens,
                 latency_s=resp.latency_s, attempts=resp.attempts,
                 rationale=resp.rationale,
                 thinking_tokens=sum(a.thinking_tokens for a in resp.attempt_log),
                 attempt_log=[{k: v for k, v in vars(a).items() if k != "raw"}
                              for a in resp.attempt_log])
        (run_dir / f"proposal_{i:03d}.py").write_text(resp.source, encoding="utf-8")

        g = guards.check(resp.source, contract_seconds)
        log.emit("guard", iter=i, ok=g.ok, violations=g.violations)
        if not g.ok:
            counts["rejected"] += 1
            err_counts[str(ErrorClass.GUARD_REJECTION)] += 1
            guard_feedback = g.feedback()
            log.emit("decision", iter=i, decision="rejected", best_acc=best_acc,
                     regressions=regressions)
            history.append({"iter": i, "rationale": resp.rationale, "val_acc": None,
                            "outcome": "rejected (rule violation)"})
            continue

        # Preserve the state from which this candidate was proposed. If its
        # training crashes, restore this exact tip: with patience > 1 it can be
        # a valid provisional chain rather than the global best recipe.
        previous_sha = repo.head()
        sha = repo.write_and_commit(resp.source, f"iter={i} proposal")
        print(f"[arloop] iter {i}: training {train_seconds:.0f}s ...", flush=True)
        log.emit("train_start", iter=i, sha=sha)
        res = run_training(workdir, run_dir, i, cfg, _device(cfg))
        log.emit("train_end", iter=i, val_acc=res.get("val_acc"), exit=res["exit"],
                 epochs=res.get("epochs_completed"), steps=res.get("steps"),
                 train_seconds_actual=res.get("train_seconds"),
                 error=res.get("error", "")[:500] if res["errored"] else "")

        if res["errored"]:
            counts["errored"] += 1
            ec = res.get("error_class", str(ErrorClass.TRAIN_CRASH))
            # Hand the traceback back to the proposer. Without it the same
            # hallucinated API gets proposed over and over.
            tail = str(res.get("error", ""))[-1200:]
            guard_feedback = ("Your previous proposal ran but CRASHED. Fix the "
                              "cause; do not repeat it.\n\n```\n" + tail + "\n```")
            err_counts[ec] = err_counts.get(ec, 0) + 1
            repo.checkout(previous_sha)
            log.emit("decision", iter=i, decision="errored", error_class=ec,
                     best_acc=best_acc, regressions=regressions)
            history.append({"iter": i, "rationale": resp.rationale, "val_acc": None,
                            "outcome": "crashed"})
            continue

        acc = res["val_acc"]
        if acc > best_acc + eps:
            best_acc, best_sha = acc, repo.head()
            best_iter = i
            regressions, provisional = 0, []
            counts["kept"] += 1
            outcome = "kept (new best)"
            log.emit("decision", iter=i, decision="keep", best_acc=best_acc, regressions=0)
        else:
            regressions += 1
            provisional.append(i)
            if regressions >= patience:
                repo.checkout(best_sha)
                log.emit("rollback", iter=i, to_sha=best_sha,
                         discarded_iters=list(provisional))
                counts["reverted"] += len(provisional)
                regressions, provisional = 0, []
                outcome = "reverted (patience exhausted)"
                log.emit("decision", iter=i, decision="revert", best_acc=best_acc,
                         regressions=0)
            else:
                outcome = f"kept provisionally ({regressions}/{patience})"
                log.emit("decision", iter=i, decision="provisional",
                         best_acc=best_acc, regressions=regressions)

        print(f"[arloop] iter {i}: val_acc {acc:.4f} -> {outcome}  "
              f"[{time.time() - _t_iter:.0f}s]", flush=True)
        history.append({"iter": i, "rationale": resp.rationale, "val_acc": acc,
                        "outcome": outcome})

        # Fail fast on a dead endpoint. A session that has produced nothing but
        # infrastructure errors will not recover by trying harder, and every
        # further iteration burns a full training budget for no data. The pilot
        # spent six iterations discovering that a model name was wrong.
        if _should_abort(err_counts, i, max_infra_rate, history,
                         max_consecutive_infra):
            aborted_early = True
            log.emit("session_abort_early", iter=i,
                     reason="persistent infrastructure errors",
                     infra_errors=_infra_total(err_counts))
            print(f"[arloop] aborting after {i} iterations: persistent "
                  f"infrastructure errors. Check the last HTTP response and server log.")
            break

    # Any still-provisional chain at budget exhaustion never became the best.
    if provisional:
        repo.checkout(best_sha)
        log.emit("rollback", iter=loop_budget, to_sha=best_sha,
                 discarded_iters=list(provisional), reason="budget exhausted")
        counts["reverted"] += len(provisional)

    # --- final evaluation --------------------------------------------------
    test_acc = None
    ckpt = run_dir / f"model_{best_iter:03d}.pt"
    if ckpt.exists():
        log.emit("final_eval_start", best_sha=best_sha, best_iter=best_iter)
        test_acc = _final_eval(workdir, run_dir, ckpt, cfg)
        log.emit("final_eval", test_acc=test_acc, best_sha=best_sha, best_iter=best_iter)
    else:
        log.emit("final_eval_skipped", reason="no checkpoint for best iteration")

    repo.archive(run_dir / "recipe_history.bundle")

    infra_errors = sum(err_counts[str(e)] for e in ErrorClass if e.is_infra)
    iterations_completed = len(history)
    infra_rate = infra_errors / iterations_completed if iterations_completed else 0.0
    max_rate = float(cfg.get("loop", {}).get("max_infra_error_rate",
                                             DEFAULT_MAX_INFRA_ERROR_RATE))
    evaluated = sum(1 for h in history if h["val_acc"] is not None)

    # Best accuracy SEEN, kept or not. When nothing clears eps, best_val_acc
    # collapses to the baseline and carries no signal; this retains it, and
    # distinguishes a proposer that repeatedly lands just short from one that
    # collapses to chance.
    div = _proposal_diversity(run_dir)
    _seen = [h["val_acc"] for h in history if h.get("val_acc") is not None]
    max_obs = max(_seen) if _seen else base["val_acc"]

    summary = {
        "iterations": loop_budget, "iterations_completed": iterations_completed,
        "aborted_early": aborted_early, **counts,
        **{f"err_{k}": v for k, v in err_counts.items()},
        "infra_errors": infra_errors, "infra_error_rate": round(infra_rate, 4),
        "evaluated_iterations": evaluated,
        "baseline_val_acc": base["val_acc"], "best_val_acc": best_acc,
        "test_acc": test_acc, "best_iter": best_iter,
        "no_progress": counts["kept"] == 0,
        "eps": eps,
        # Reasoning is a FACTOR in Phase 2 and a fixed variable in Phase 1.
        # Either way, record what actually happened: a serving stack that
        # accepts `enable_thinking` and ignores it would otherwise be
        # indistinguishable from one that honours it.
        "thinking_requested": ("unset" if thinking is None else
                               ("on" if thinking else "off")),
        "thinking_tokens_total": think_total,
        "max_val_acc_observed": max_obs,
        "temperature": (pcfg["temperature"] if temperature is None else temperature),
        # Search diversity. Near 1.0 means the session proposed the same thing
        # repeatedly -- a degenerate search that costs N inferences and returns
        # one idea (see EXPERIMENT_PLAN D22).
        "proposal_similarity_mean": div["mean"],
        "proposal_similarity_max": div["max"],
        "proposals_compared": div["n"],
        # A session is valid evidence about the proposer only if the machine
        # behaved and at least one mutation was actually evaluated.
        "valid": bool(not aborted_early and infra_rate <= max_rate and evaluated > 0),
        "invalid_reason": ("" if (not aborted_early and infra_rate <= max_rate
                                      and evaluated > 0) else
                           ("session aborted after persistent infrastructure errors"
                            if aborted_early else
                            (f"infra error rate {infra_rate:.0%} > {max_rate:.0%}"
                             if infra_rate > max_rate else
                             "no iteration was ever evaluated"))),
    }
    log.emit("session_end", **summary)
    log.close()
    if sampler:
        totals = sampler.stop()
        (run_dir / "nvml_totals.json").write_text(
            json.dumps(totals, indent=2, default=str))
    if not summary["valid"]:
        print(f"[arloop] SESSION INVALID: {summary['invalid_reason']} "
              f"-- re-run it; do not analyse it as a result")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _proposal_diversity(run_dir: Path) -> dict:
    """Mean pairwise similarity of the proposals a session produced.

    A propose-evaluate-keep loop under greedy selection restores the same recipe
    after every rejection, so the prompt barely changes between iterations. A
    proposer that is too deterministic then returns the same answer repeatedly:
    N experiments cost N inferences and deliver one idea. Phase 1 measured 0.982
    mean similarity at temperature 0, with one consecutive pair byte-identical,
    and no other metric the harness records revealed it.

    Reported per session so that search degeneracy is observable directly rather
    than inferred from a null result.
    """
    import difflib
    from itertools import combinations
    texts = [p.read_text(encoding="utf-8", errors="replace")
             for p in sorted(run_dir.glob("proposal_*.py"))]
    if len(texts) < 2:
        return {"mean": None, "max": None, "n": len(texts)}
    # Cap the pair count: similarity is O(n^2) in file size and a 100-iteration
    # session would otherwise spend minutes here.
    pairs = list(combinations(range(len(texts)), 2))
    if len(pairs) > 300:
        step = len(pairs) // 300 + 1
        pairs = pairs[::step]
    rs = [difflib.SequenceMatcher(None, texts[i], texts[j]).ratio() for i, j in pairs]
    return {"mean": round(sum(rs) / len(rs), 4), "max": round(max(rs), 4),
            "n": len(texts)}


# ---------------------------------------------------------------------- utils
def _is_synthetic(cfg: dict) -> bool:
    """True when the data cache is random noise (prepare_cifar --synthetic).

    A synthetic session scores ~0.10 by construction. Without this flag such a
    run looks like a catastrophically bad proposer in the comparison table.
    """
    try:
        import numpy as np
        cache = Path(cfg["workload"]["data_dir"]) / "cifar10_splits.npz"
        if not cache.exists():
            return False
        d = np.load(cache)
        return "synthetic" in d.files and bool(d["synthetic"])
    except Exception:                                                  # noqa: BLE001
        return False


def _infra_total(err_counts: dict) -> int:
    return sum(err_counts[str(e)] for e in ErrorClass if e.is_infra)


def _should_abort(err_counts: dict, done: int, max_rate: float,
                  history: list[dict], max_consecutive: int = 3,
                  min_iters: int = 3) -> bool:
    """Stop a session that is persistently producing infrastructure failures.

    Consecutive infrastructure failures abort even after earlier useful results;
    otherwise require a few iterations so one transient blip does not kill an
    otherwise healthy session.
    """
    if done < min_iters:
        return False
    infra_classes = {str(e) for e in ErrorClass if e.is_infra}
    recent = history[-max_consecutive:] if max_consecutive > 0 else []
    if (len(recent) == max_consecutive
            and all(row.get("error_class") in infra_classes for row in recent)):
        return True
    if any(h["val_acc"] is not None for h in history):
        return False                      # something worked; keep going
    return _infra_total(err_counts) / done > max_rate


def _device(cfg: dict) -> str:
    return "cpu" if str(cfg["gpus"]["train"]).lower() == "cpu" else "cuda"


def _final_eval(workdir: Path, run_dir: Path, ckpt: Path, cfg: dict) -> float | None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"]["train"])
    env["PYTHONPATH"] = str(ROOT / "workload") + os.pathsep + env.get("PYTHONPATH", "")
    out = run_dir / "test_result.json"
    cmd = [sys.executable, str(ROOT / "workload" / "final_eval.py"),
           "--train-file", str(workdir / "train.py"),
           "--checkpoint", str(ckpt),
           "--data-dir", str(Path(cfg["workload"]["data_dir"]).resolve()),
           "--out", str(out),
           "--device", "cuda:0" if _device(cfg) != "cpu" else "cpu"]
    proc = subprocess.run(cmd, cwd=str(workdir), env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        (run_dir / "final_eval.log").write_text(proc.stdout + proc.stderr)
        return None
    return json.loads(out.read_text())["test_acc"]


def _versions() -> dict:
    def sh(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True).stdout.strip()
        except Exception:                                          # noqa: BLE001
            return "n/a"
    return {
        "python": sys.version.split()[0],
        "git_sha": sh("git", "rev-parse", "HEAD"),
        "nvidia_smi": sh("nvidia-smi", "--query-gpu=name,driver_version",
                         "--format=csv,noheader"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--proposer", required=True, choices=["dense", "moe"])
    ap.add_argument("--patience", type=int, required=True)
    ap.add_argument("--loop-budget", type=int, required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stub", action="store_true", help="offline scripted proposer")
    ap.add_argument("--proposer-model", default=None)
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the profile's sampling temperature for this "
                         "session; Stage 2b varies this as a factor")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override proposer output allowance; long prompts need "
                         "room inside the server context window")
    ap.add_argument("--history-max-rows", type=int, default=None,
                    help="retain only the most recent N history rows in prompts; "
                         "the full history remains in session.jsonl")
    ap.add_argument("--thinking", choices=["on", "off"], default=None,
                    help="override the profile's reasoning setting for this "
                         "session; Phase 2 varies this as a factor")
    ap.add_argument("--baseline", default=None,
                    help="override the baseline recipe (integration testing)")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.profile).read_text())
    s = run_session(cfg, proposer_arm=a.proposer, patience=a.patience,
                    loop_budget=a.loop_budget, run_dir=Path(a.run_dir),
                    seed=a.seed, stub=a.stub, proposer_model=a.proposer_model,
                    thinking=(None if a.thinking is None else a.thinking == "on"),
                    temperature=a.temperature,
                    baseline_path=a.baseline,
                    proposer_max_tokens=a.max_tokens,
                    history_max_rows=a.history_max_rows)
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
