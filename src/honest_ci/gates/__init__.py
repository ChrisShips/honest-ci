"""The gates.

``GATE_SEQUENCE`` (in ``contracts.py``) fixes their order: spec → review →
regression. Implementations land one at a time; the runner takes whatever
registry it is given, so a partial deployment runs the gates it has and
records the rest as skipped rather than pretending they passed.
"""

from __future__ import annotations

from .base import Gate
from .regression import RegressionGate
from .review import ReviewGate

#: Gate name → implementation. Filled in as each gate ships.
GATE_REGISTRY: dict[str, type[Gate]] = {
    ReviewGate.name: ReviewGate,
    RegressionGate.name: RegressionGate,
}

__all__ = ["GATE_REGISTRY", "Gate", "RegressionGate", "ReviewGate"]
