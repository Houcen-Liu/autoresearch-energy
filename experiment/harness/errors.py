"""Error taxonomy.

The pilot showed why this matters: four iterations died on an Ollama read
timeout, and under the previous coding they all landed in the run table as
`errored` -- indistinguishable from a proposer that produced four useless
mutations. One flaky serving night would have become a finding about model
architecture.

So failures are split by WHO failed:

  INFRA_*      our serving stack or machine. Not the proposer's fault, carries no
               information about the subject, and invalidates the session if it
               happens often enough.
  CONTRACT_*   the proposer could not produce a well-formed reply within its
               retry and time budget. This IS data: contract compliance is a real
               property of a self-hosted model and is reported as a dependent
               variable.
  GUARD_*      the proposal was well-formed but broke a task rule. Also data.
  TRAIN_*      the proposed recipe itself failed. Also data -- the proposer wrote
               code that does not run.
"""
from __future__ import annotations

from enum import Enum


class ErrorClass(str, Enum):
    """str-Enum (not enum.StrEnum) so the harness still imports on Python 3.10."""

    INFRA_TIMEOUT = "infra_timeout"          # request exceeded the time budget
    INFRA_TRANSPORT = "infra_transport"      # connection refused, 5xx, malformed body
    CONTRACT_VIOLATION = "contract_violation"  # no usable recipe after retries
    GUARD_REJECTION = "guard_rejection"      # well-formed but broke a rule
    TRAIN_CRASH = "train_crash"              # proposed recipe exited non-zero
    TRAIN_TIMEOUT = "train_timeout"          # proposed recipe overran its budget

    def __str__(self) -> str:
        return self.value

    @property
    def is_infra(self) -> bool:
        return self in (ErrorClass.INFRA_TIMEOUT, ErrorClass.INFRA_TRANSPORT)

    @property
    def is_agent(self) -> bool:
        """True when the failure tells us something about the proposer."""
        return not self.is_infra


class ProposerError(Exception):
    """Base for proposer failures, carrying its classification."""

    error_class: ErrorClass = ErrorClass.INFRA_TRANSPORT

    def __init__(self, message: str, attempts: list | None = None):
        super().__init__(message)
        self.attempts = attempts or []


class ProposerTimeout(ProposerError):
    error_class = ErrorClass.INFRA_TIMEOUT


class ProposerTransport(ProposerError):
    error_class = ErrorClass.INFRA_TRANSPORT


class ProposerContract(ProposerError):
    error_class = ErrorClass.CONTRACT_VIOLATION


# A session whose infrastructure error rate exceeds this is not evidence about
# the proposer; it is evidence that the machine misbehaved. It is quarantined by
# the analysis and re-run, and both facts are recorded.
DEFAULT_MAX_INFRA_ERROR_RATE = 0.25
