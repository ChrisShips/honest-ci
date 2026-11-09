"""Durable state for one change under verification.

The state is checkpointed to the ledger before and after every gate, so a
crashed or restarted process resumes at the gate that was executing instead of
re-running the whole sequence or, worse, carrying a stale approval forward.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .contracts import (
    ApprovalDecision,
    ApprovalTicket,
    ArtifactManifest,
    BlockBrief,
    ChangeRequest,
    EvidencePack,
    Finding,
    GateAttempt,
    GateReport,
    LLMCallRecord,
    PolicyDecision,
    Route,
    RunMetrics,
    Verdict,
    utc_now,
)

RunStatus = Literal["running", "awaiting_human", "mergeable", "blocked", "failed"]


class RunState(BaseModel):
    run_id: str
    change: ChangeRequest

    # Where deterministic tools run. Tests and scanners are always offline;
    # only dependency installation is granted network access (see sandbox.py).
    workspace_path: str | None = None
    sandbox_isolation: str = "process"

    # Gate output, keyed by gate name so invalidation can drop a single gate
    # and everything derived from it.
    gate_reports: dict[str, GateReport] = Field(default_factory=dict)
    policy_decision: PolicyDecision | None = None
    evidence: EvidencePack | None = None
    approval_ticket: ApprovalTicket | None = None
    approval_decision: ApprovalDecision | None = None
    auto_approve: bool = False

    # Advisory model output. It is evidence and context, never a verdict, and
    # upstream rework clears it before the evidence pack is assembled.
    briefs: list[BlockBrief] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    artifact_manifest: ArtifactManifest | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)

    current_gate: str = "spec"
    last_verdict: Verdict | None = None
    attempts: dict[str, int] = Field(default_factory=dict)
    execution_history: list[GateAttempt] = Field(default_factory=list)
    last_feedback: str | None = None
    retry_class: str | None = None
    retry_delay_seconds: float | None = None
    next_action: Route = "advance"

    trace_id: str | None = None
    status: RunStatus = "running"
    per_gate_timing: dict[str, float] = Field(default_factory=dict)
    started_at: float | None = None
    created_at: datetime = Field(default_factory=utc_now)

    def report(self, gate: str) -> GateReport | None:
        return self.gate_reports.get(gate)

    @property
    def findings(self) -> list[Finding]:
        """Every finding the gates have raised so far, in gate order."""

        return [finding for report in self.gate_reports.values() for finding in report.findings]
