"""honest-ci — the verification layer for AI-assisted development.

Three gates (spec, review and regression) and an evidence pack: a change that
someone else submitted, an AI tool or a person, comes in, and leaves for the
merge with its evidence. Measurement is not a gate, because it decides
nothing; each run's measurements are written into its evidence pack. A model
only ever generates, and gate verdicts are decided by deterministic code.
"""

from __future__ import annotations

from .contracts import (
    GATE_SEQUENCE,
    ChangeRequest,
    EvidencePack,
    Finding,
    GateReport,
    RunRecord,
    Verdict,
)
from .runner import GateRunner, verify_change
from .settings import Settings
from .state import RunState

__version__ = "0.1.0"

__all__ = [
    "GATE_SEQUENCE",
    "ChangeRequest",
    "EvidencePack",
    "Finding",
    "GateReport",
    "GateRunner",
    "RunRecord",
    "RunState",
    "Settings",
    "Verdict",
    "__version__",
    "verify_change",
]
