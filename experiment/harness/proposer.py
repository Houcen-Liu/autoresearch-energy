"""Proposer client.

Speaks plain OpenAI-compatible /v1/chat/completions so that vLLM, llama.cpp-server
and Ollama (including Ollama's `:cloud` tags) are interchangeable without code
changes -- which is what makes the GPU (Phase 1) and CPU (Phase 2) arms
comparable, and what lets a cloud reference model be run through the identical
harness.

Output contract: full-file replacement, not a diff. Local 20-40B models produce
unappliable diffs often enough to dominate the error budget, and train.py is
small enough that a rewrite is cheap.

    RATIONALE: <one or two sentences>
    ```python
    <complete new train.py>
    ```

Two things the pilot forced into this design:

  * A TIME BUDGET per iteration, not just per request. Three retries at a 300 s
    request timeout burned 15 minutes per failed iteration and produced nothing.
    Total proposer time per iteration is now bounded, and exceeding it is an
    infrastructure error, not a verdict on the model.
  * PER-ATTEMPT ACCOUNTING. Previously a failed iteration recorded zero tokens
    and zero latency, so the most expensive iterations were invisible in the
    energy and token analysis. Every attempt is now recorded, successful or not.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from harness.errors import (ProposerContract, ProposerTimeout, ProposerTransport)

STANDARD_PARAMS = {"temperature", "top_p", "max_tokens", "seed", "stop",
                   "presence_penalty", "frequency_penalty"}

HTTP_BODY_LIMIT = 2_000
EXTRA_PARAM_REJECTION_MARKERS = (
    "unknown field", "unknown parameter", "unrecognized field",
    "unsupported parameter", "unexpected keyword", "extra fields not permitted",
    "extra inputs are not permitted",
)

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
RATIONALE = re.compile(r"RATIONALE:\s*(.+?)(?:\n|$)", re.IGNORECASE)

REPAIR_PROMPT = """Your previous reply could not be used: {reason}

Reply again with nothing but:

RATIONALE: <one sentence>
```python
<the complete new train.py>
```

Exactly one fenced block, containing the whole file. Your previous reply began:
{excerpt}
"""


@dataclass
class Attempt:
    """One HTTP round trip, recorded whether it succeeded or not."""
    n: int
    t_start: float
    t_end: float
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    outcome: str = "ok"                    # ok | contract | transport | timeout
    detail: str = ""
    finish_reason: str = ""
    raw: str = ""                          # kept so a rejected reply is inspectable
    thinking_tokens: int = 0               # reasoning emitted but not delivered


@dataclass
class ProposerResponse:
    source: str
    rationale: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    t_start: float
    t_end: float
    attempts: int
    raw: str
    attempt_log: list[Attempt] = field(default_factory=list)

    @property
    def total_prompt_tokens(self) -> int:
        """Includes tokens spent on failed attempts -- the energy was real."""
        return sum(a.prompt_tokens for a in self.attempt_log) or self.prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return sum(a.completion_tokens for a in self.attempt_log) or self.completion_tokens


class Proposer:
    """One instance per session; bound to a single endpoint and model."""

    def __init__(self, endpoint: str, model: str, *, temperature: float = 0.0,
                 top_p: float = 1.0, max_tokens: int = 8192, seed: int = 0,
                 timeout_s: int = 600, max_retries: int = 1,
                 time_budget_s: float | None = None,
                 extra_params: dict | None = None,
                 system_suffix: str = "", prompt_suffix: str = ""):
        # Fail loudly on a malformed endpoint. A config bug once put the model
        # name here, and requests reported only "No connection adapters were
        # found for 'qwen3:4b/chat/completions'" -- classified as a transport
        # error and blamed on the model.
        if not str(endpoint).startswith(("http://", "https://")):
            raise ValueError(
                f"proposer endpoint must be an http(s) URL, got {endpoint!r}. "
                f"Check profiles/*.yaml -> proposer.endpoints")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.params = {"temperature": temperature, "top_p": top_p,
                       "max_tokens": max_tokens, "seed": seed,
                       **(extra_params or {})}
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        # Total wall clock allowed for one iteration's proposal, across all
        # attempts. Defaults to 1.5x a single request timeout.
        self.time_budget_s = time_budget_s or (timeout_s * 1.5)
        # Appended to the system prompt. Prompt-level control works on every
        # serving stack, unlike vendor-specific request fields.
        self.system_suffix = system_suffix
        # Appended to the USER turn. Qwen3's /no_think soft switch only takes
        # effect there -- in the system prompt it is silently ignored, which cost
        # the pilot ~4000 reasoning tokens per reply.
        self.prompt_suffix = prompt_suffix
        self._reduced = False

    def request_manifest(self) -> dict:
        """Exactly what is sent, recorded in the session log as a fixed variable."""
        return {"endpoint": self.endpoint, "model": self.model,
                "params": dict(self.params), "timeout_s": self.timeout_s,
                "max_retries": self.max_retries, "time_budget_s": self.time_budget_s,
                "params_reduced": self._reduced,
                "system_suffix": self.system_suffix,
                "prompt_suffix": self.prompt_suffix}

    # ------------------------------------------------------------------ call
    def complete(self, system: str, user: str) -> ProposerResponse:
        if self.system_suffix:
            system = f"{system}\n\n{self.system_suffix}"
        if self.prompt_suffix:
            user = f"{user}\n\n{self.prompt_suffix}"
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        t_start = time.time()
        attempts: list[Attempt] = []

        for n in range(1, self.max_retries + 2):        # 1 try + max_retries
            remaining = self.time_budget_s - (time.time() - t_start)
            # The first attempt always runs: a budget smaller than the threshold
            # must not silently skip the call and report a timeout with no
            # attempts, which would lose the failure's tokens and latency.
            if attempts and remaining <= 0:
                raise ProposerTimeout(
                    f"proposer time budget of {self.time_budget_s:.0f}s exhausted "
                    f"after {len(attempts)} attempt(s)", attempts)
            remaining = max(remaining, 1.0)

            a_start = time.time()
            response = None
            try:
                response = requests.post(
                    f"{self.endpoint}/chat/completions",
                    json={"model": self.model, "messages": messages,
                          "stream": False, **self.params},
                    timeout=min(self.timeout_s, remaining),
                )
                # Serving stacks differ on which non-standard fields they accept
                # (`think` on Ollama, `chat_template_kwargs` on vLLM, `options`
                # on neither /v1 endpoint). A 4xx here usually means one of our
                # extras was rejected, so retry once with standard fields only
                # and record that the pinned settings did not take effect.
                if 400 <= response.status_code < 500 and not self._reduced:
                    dropped = sorted(set(self.params) - STANDARD_PARAMS)
                    if dropped and _rejects_extra_params(response, dropped):
                        body = _response_body(response)
                        attempts.append(Attempt(
                            n=n, t_start=a_start, t_end=time.time(),
                            latency_s=time.time() - a_start,
                            outcome="transport",
                            detail=_http_detail(response), raw=body,
                        ))
                        self._reduced = True
                        print(f"[proposer] endpoint rejected {dropped}; retrying "
                              f"with standard parameters only. Those settings are "
                              f"NOT in effect -- fix them before treating the "
                              f"session as evidence.")
                        self.params = {k: v for k, v in self.params.items()
                                       if k in STANDARD_PARAMS}
                        continue
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                text = choice["message"]["content"]
                usage = body.get("usage", {})
                finish = choice.get("finish_reason")
                a = Attempt(n=n, t_start=a_start, t_end=time.time(),
                            latency_s=time.time() - a_start,
                            prompt_tokens=int(usage.get("prompt_tokens", 0)),
                            completion_tokens=int(usage.get("completion_tokens", 0)),
                            finish_reason=finish or "", raw=text,
                            thinking_tokens=_thinking_tokens(text))
                attempts.append(a)

                try:
                    if finish == "length":
                        raise ValueError(
                            f"reply truncated at the {self.params.get('max_tokens')} "
                            f"token limit, so it has no closing fence. Raise "
                            f"proposer.max_tokens: a full train.py rewrite needs "
                            f"~2000 tokens, plus any reasoning the model emits")
                    source, rationale = self._parse(text)
                except ValueError as e:
                    a.outcome, a.detail = "contract", str(e)
                    messages = self._repair(messages, str(e), text)
                    continue

                t_end = time.time()
                return ProposerResponse(
                    source=source, rationale=rationale,
                    prompt_tokens=a.prompt_tokens, completion_tokens=a.completion_tokens,
                    latency_s=t_end - t_start, t_start=t_start, t_end=t_end,
                    attempts=n, raw=text, attempt_log=attempts,
                )

            except requests.Timeout as e:
                attempts.append(Attempt(n=n, t_start=a_start, t_end=time.time(),
                                        latency_s=time.time() - a_start,
                                        outcome="timeout", detail=str(e)[:200]))
                raise ProposerTimeout(f"request timed out after "
                                      f"{time.time() - a_start:.0f}s", attempts) from e
            except requests.HTTPError as e:
                detail = _http_detail(response, e)
                attempts.append(Attempt(
                    n=n, t_start=a_start, t_end=time.time(),
                    latency_s=time.time() - a_start,
                    outcome="transport", detail=detail,
                    raw=_response_body(response),
                ))
                if n > self.max_retries:
                    raise ProposerTransport(detail, attempts) from e
                time.sleep(min(2 ** n, 8))
            except (requests.RequestException, KeyError, ValueError) as e:
                body = _response_body(response)
                detail = str(e)[:HTTP_BODY_LIMIT]
                if body:
                    detail = f"{detail}; response: {body}"[:HTTP_BODY_LIMIT]
                attempts.append(Attempt(n=n, t_start=a_start, t_end=time.time(),
                                        latency_s=time.time() - a_start,
                                        outcome="transport", detail=detail,
                                        raw=body))
                if n > self.max_retries:
                    raise ProposerTransport(detail, attempts) from e
                time.sleep(min(2 ** n, 8))

        raise ProposerContract(
            f"no usable recipe after {len(attempts)} attempt(s)", attempts)

    @staticmethod
    def _repair(messages: list[dict], reason: str, text: str) -> list[dict]:
        """Minimal repair turn.

        Deliberately does NOT re-send the growing conversation: a failing model
        was already given the full task, and repeating it makes every retry more
        expensive than the attempt that just failed.
        """
        return [messages[0], messages[1],
                {"role": "user",
                 "content": REPAIR_PROMPT.format(reason=reason,
                                                 excerpt=text[:400].strip())}]

    @staticmethod
    def _parse(text: str) -> tuple[str, str]:
        """Extract the proposed recipe.

        Reasoning models often emit extra fenced blocks inside their thinking.
        Rather than failing the whole call, disambiguate on the one marker every
        valid recipe must carry: a TRAIN_SECONDS declaration.
        """
        blocks = CODE_BLOCK.findall(text)
        if not blocks:
            raise ValueError("no ```python fenced block found")
        if len(blocks) > 1:
            candidates = [b for b in blocks if "TRAIN_SECONDS" in b and "def main" in b]
            if len(candidates) != 1:
                raise ValueError(
                    f"expected one recipe block, found {len(blocks)} code blocks "
                    f"({len(candidates)} of them look like a recipe)")
            blocks = candidates
        m = RATIONALE.search(text)
        return blocks[0], (m.group(1).strip() if m else "")


class StubProposer(Proposer):
    """Offline proposer for tests, CI, and the pilot dry run.

    Replays scripted mutations, so the whole pipeline -- guards, git history,
    training, keep/revert, logging, alignment, analysis -- is exercisable with no
    model, no GPU and no network.
    """

    def __init__(self, fixture: str | Path | None = None, latency_s: float = 0.4,
                 seed: int = 0, **_):
        self.fixture = Path(fixture) if fixture else None
        self.latency_s = latency_s
        self.model = "stub"
        self.endpoint = "stub://"
        self.params = {"temperature": 0.0}
        self.timeout_s = 0
        self.max_retries = 0
        self.time_budget_s = 0
        self._reduced = False
        self._rng = random.Random(seed)
        self._i = 0
        self._mutations = json.loads(self.fixture.read_text()) if self.fixture else None

    def request_manifest(self) -> dict:
        return {"endpoint": "stub://", "model": "stub", "params": {}}

    def complete(self, system: str, user: str) -> ProposerResponse:  # noqa: D102
        t0 = time.time()
        time.sleep(self.latency_s)
        current = _extract_current_source(user)
        source, rationale = self._mutate(current)
        t1 = time.time()
        self._i += 1
        a = Attempt(n=1, t_start=t0, t_end=t1, latency_s=t1 - t0,
                    prompt_tokens=len(user) // 4, completion_tokens=len(source) // 4)
        return ProposerResponse(
            source=source, rationale=rationale,
            prompt_tokens=a.prompt_tokens, completion_tokens=a.completion_tokens,
            latency_s=t1 - t0, t_start=t0, t_end=t1, attempts=1, raw=source,
            attempt_log=[a],
        )

    def _mutate(self, current: str) -> tuple[str, str]:
        if self._mutations:
            m = self._mutations[self._i % len(self._mutations)]
            return m["source"], m.get("rationale", "scripted")
        # Default: nudge the learning rate, occasionally emit an invalid proposal
        # so the guard path gets exercised too.
        if self._rng.random() < 0.15:
            return current.replace("TRAIN_SECONDS = 240.0",
                                   "TRAIN_SECONDS = 600.0"), "cheat on the budget"
        lr = round(self._rng.choice([0.005, 0.02, 0.03, 0.05, 0.1]), 4)
        new = re.sub(r"LEARNING_RATE = [\d.]+", f"LEARNING_RATE = {lr}", current)
        return new, f"try learning rate {lr}"


THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _thinking_tokens(text: str) -> int:
    """Rough token count of reasoning the model emitted.

    Thinking is pure proposer energy that produces no artifact, so it belongs in
    the measurements rather than being invisible. ~4 chars per token.
    """
    return sum(len(b) for b in THINK_BLOCK.findall(text)) // 4


def _response_body(response) -> str:
    """Return a bounded HTTP response body, including JSON error messages."""
    if response is None:
        return ""
    try:
        text = response.text
    except (AttributeError, RuntimeError):
        try:
            text = json.dumps(response.json(), ensure_ascii=True)
        except (AttributeError, TypeError, ValueError):
            return ""
    return str(text).strip()[:HTTP_BODY_LIMIT]


def _http_detail(response, error: Exception | None = None) -> str:
    """Keep the server's explanation; ``raise_for_status`` omits it."""
    status = getattr(response, "status_code", None)
    body = _response_body(response)
    prefix = f"HTTP {status}" if status is not None else str(error or "HTTP error")
    return f"{prefix}: {body}"[:HTTP_BODY_LIMIT] if body else prefix[:HTTP_BODY_LIMIT]


def _rejects_extra_params(response, params: list[str]) -> bool:
    """True only when a 4xx identifies request fields as the problem.

    A context-window overflow is also HTTP 400. Treating every 4xx as an
    unsupported-extra signal hid the actual failure in the long-horizon run and
    incorrectly disabled ``chat_template_kwargs`` for the rest of the session.
    """
    body = _response_body(response).lower()
    return (any(str(param).lower() in body for param in params)
            or any(marker in body for marker in EXTRA_PARAM_REJECTION_MARKERS))


def _extract_current_source(prompt: str) -> str:
    blocks = CODE_BLOCK.findall(prompt)
    if not blocks:
        raise ValueError("prompt contains no current train.py")
    return blocks[-1]
