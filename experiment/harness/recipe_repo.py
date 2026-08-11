"""Git-backed mutation history for the editable recipe.

Every session gets its own repository inside the run directory, seeded with the
calibrated baseline train.py. Keep = commit. Revert = checkout of a known-good
blob, never a textual undo. The repo is archived at session end and ships in the
replication package as the complete, inspectable mutation history.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class RecipeRepo:
    def __init__(self, workdir: str | Path, baseline_source: str, filename: str = "train.py"):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self._git("init", "-q")
        self._git("config", "user.email", "harness@greenlab.local")
        self._git("config", "user.name", "arloop")
        self.path.write_text(baseline_source, encoding="utf-8")
        self._git("add", filename)
        self._git("commit", "-q", "-m", "baseline")
        self.baseline_sha = self.head()

    # ------------------------------------------------------------------ util
    @property
    def path(self) -> Path:
        return self.dir / self.filename

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(self.dir), *args],
                              capture_output=True, text=True, check=True).stdout.strip()

    def head(self) -> str:
        return self._git("rev-parse", "HEAD")

    # ---------------------------------------------------------------- action
    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def write_and_commit(self, source: str, message: str) -> str:
        self.path.write_text(source, encoding="utf-8")
        self._git("add", self.filename)
        self._git("commit", "-q", "--allow-empty", "-m", message)
        return self.head()

    def checkout(self, sha: str) -> None:
        """Restore the recipe to a known-good revision, recording the restore."""
        self._git("checkout", sha, "--", self.filename)
        self._git("add", self.filename)
        self._git("commit", "-q", "--allow-empty", "-m", f"revert to {sha[:8]}")

    def archive(self, dest: str | Path) -> Path:
        dest = Path(dest)
        self._git("bundle", "create", str(dest.resolve()), "--all")
        return dest
