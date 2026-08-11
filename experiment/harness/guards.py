"""Static validation of a proposed train.py.

Rejected proposals never reach the GPU: they cost one proposer call, not a full
240 s training experiment. Rejections are logged, counted against the loop budget
(they consumed real energy) and fed back to the proposer as an error message.
"""
from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field

ALLOWED_IMPORTS = {
    "torch", "torchvision", "numpy", "np", "math", "time", "json", "random",
    "argparse", "pathlib", "prepare_cifar", "dataclasses", "typing", "collections",
    "itertools", "functools", "__future__",
}

BANNED_NAMES = {
    "subprocess", "os", "sys", "socket", "requests", "urllib", "shutil",
    "multiprocessing", "importlib", "ctypes", "pickle", "eval", "exec", "compile",
}

TEST_TOKENS = ("x_test", "y_test", "include_test", "final_eval", "test_acc")

REQUIRED_RESULT_KEYS = {
    "val_acc", "epochs_completed", "steps", "train_seconds", "peak_vram_mb",
}


def strip_comments_and_docstrings(source: str) -> str:
    """Return only executable code text.

    Without this, a docstring mentioning `final_eval.py` (as the baseline recipe's
    own header comment does) trips the test-set guard. Rules must fire on what the
    code does, never on what it says about itself.

    A string is a docstring only when it stands alone as a statement at bracket
    depth 0. Strings inside a dict, call or list are data and must be preserved --
    that is where the result.json keys live.
    """
    out: list[str] = []
    depth = 0
    stmt_start = True          # next token begins a statement
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        return source

    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.OP:
            if tok.string in "([{":
                depth += 1
            elif tok.string in ")]}":
                depth = max(0, depth - 1)
        if tok.type == tokenize.STRING and depth == 0 and stmt_start:
            stmt_start = False
            continue                                   # docstring
        if tok.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            stmt_start = True
        elif tok.type not in (tokenize.NL, tokenize.ENCODING):
            stmt_start = False
        out.append(tok.string)

    return " ".join(out)


@dataclass
class GuardResult:
    ok: bool
    violations: list[str] = field(default_factory=list)

    def feedback(self) -> str:
        return ("Your previous proposal was rejected before execution:\n- "
                + "\n- ".join(self.violations))


def extract_train_seconds(source: str) -> float | None:
    """The TRAIN_SECONDS literal declared in a recipe, or None if absent."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "TRAIN_SECONDS" \
                        and isinstance(node.value, ast.Constant) \
                        and isinstance(node.value.value, (int, float)):
                    return float(node.value.value)
    return None


def check(source: str, expected_train_seconds: float) -> GuardResult:
    """Validate a proposed recipe.

    `expected_train_seconds` is the literal declared by the BASELINE recipe, not
    the profile's effective budget. The two differ in pilot mode, where the
    harness scales actual runtime down via --train-seconds while the file's
    contract value stays put. Comparing against the profile instead would reject
    every proposal the moment the pilot shortens the budget.
    """
    v: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return GuardResult(False, [f"file does not parse: {e}"])

    # --- imports -----------------------------------------------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    v.append(f"import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                v.append(f"import from '{node.module}' is not allowed")

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            v.append(f"use of banned name '{node.id}'")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in BANNED_NAMES:
            v.append(f"use of banned module '{node.value.id}'")

    # --- the time budget ---------------------------------------------------
    found_budget = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "TRAIN_SECONDS":
                    if isinstance(node.value, ast.Constant) and \
                            isinstance(node.value.value, (int, float)):
                        found_budget = float(node.value.value)
                    else:
                        v.append("TRAIN_SECONDS must be a literal number")
    if found_budget is None:
        v.append("TRAIN_SECONDS is missing; the fixed time budget is mandatory")
    elif abs(found_budget - expected_train_seconds) > 1e-6:
        v.append(f"TRAIN_SECONDS was changed to {found_budget}; "
                 f"it must remain {expected_train_seconds}")

    # --- data access -------------------------------------------------------
    # Scan executable code only; a docstring naming final_eval.py is not an access.
    code = strip_comments_and_docstrings(source)
    if "load_splits" not in code:
        v.append("must load data via prepare_cifar.load_splits()")
    for tok in TEST_TOKENS:
        if tok in code:
            v.append(f"reference to the held-out test set ('{tok}') is forbidden")

    # --- required interface ------------------------------------------------
    fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for required in ("build_model", "main"):
        if required not in fns:
            v.append(f"function '{required}()' must exist")

    for key in REQUIRED_RESULT_KEYS:
        if f'"{key}"' not in code and f"'{key}'" not in code:
            v.append(f"result.json key '{key}' is missing")

    return GuardResult(not v, v)
