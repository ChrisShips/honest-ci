"""The control plane.

One loop owns everything that is not a gate's own judgment: which gate runs
next, what a verdict routes to, how many times a transient failure is retried,
what a re-run invalidates, and when state is checkpointed. Keeping that in a
single place is what lets a gate stay a pure function of the change.

Routing:

======================  ==========================================
verdict                 route
======================  ==========================================
``PASS`` / ``SKIPPED``  advance to the next gate
``NEEDS_HUMAN``         stop and wait for an approval decision
``BLOCKED``             terminal for this head commit: fix and push
exception (transient)   retry the same gate with bounded backoff
exception (fatal)       terminal failure, recorded with its reason
======================  ==========================================

A blocked run is not retried automatically: the fix is a new commit, and a new
commit is a new fingerprint, hence a new run. That is the same property that
makes a stale approval fail loudly instead of carrying forward.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .adapters import AdapterSet, build_forge_adapter, build_llm_adapter
from .adapters.base import ForgeAdapter, LedgerStore, LLMAdapter, ScannerAdapter
from .adapters.scanners import build_scanner_adapter
from .artifacts import ArtifactStore
from .contracts import (
    GATE_SEQUENCE,
    ApprovalDecision,
    ApprovalTicket,
    ArtifactManifest,
    BlockBrief,
    ChangeRequest,
    EvidencePack,
    Finding,
    FindingAnnotation,
    GateAttempt,
    GateConfiguration,
    GateName,
    GateReport,
    ModelReviewReport,
    Route,
    RunMetrics,
    RunOutcome,
    RunRecord,
    Verdict,
    compute_evidence_fingerprint,
    utc_now,
)
from .gates import GATE_REGISTRY
from .gates.base import Gate
from .grounding import (
    deterministic_feedback,
    guarded_block_brief,
    guarded_review_report,
    is_transient_failure,
)
from .policy import PolicyConfig, PolicyEngine, unsatisfied_checks
from .report import render_evidence
from .rules import RuleSet
from .settings import Settings
from .state import RunState, RunStatus
from .storage import SQLiteLedger

#: The evidence pack's file name inside a run's artifact directory. It is the
#: file an operator is handed, so it is named in one place rather than
#: spelled out wherever someone needs to find it again.
EVIDENCE_FILENAME = "evidence.json"

#: The same pack rendered for a person. The JSON stays the record of what
#: happened; this one exists because an evidence pack nobody reads cannot
#: settle an argument about why a change was blocked.
EVIDENCE_SUMMARY_FILENAME = "evidence.md"

#: Model call telemetry, written whenever the adapter has any to report.
LLM_CALLS_FILENAME = "llm-calls.json"

#: A human decision, written beside the evidence it was given on. The ledger
#: has it too, but a run in CI loses its ledger with the job.
APPROVAL_FILENAME = "approval.json"

#: Terminal outcome → the status a checkpoint records for it.
_OUTCOME_STATUS: dict[RunOutcome, RunStatus] = {
    "mergeable": "mergeable",
    "blocked": "blocked",
    "needs_human": "awaiting_human",
    "failed": "failed",
}

#: Ordered worst-first, so a run's verdict is the worst of its gates.
_VERDICT_RANK = {
    Verdict.BLOCKED: 3,
    Verdict.NEEDS_HUMAN: 2,
    Verdict.PASS: 1,
    Verdict.SKIPPED: 0,
}


@dataclass(frozen=True)
class RuntimeServices:
    settings: Settings
    adapters: AdapterSet
    artifact_store: ArtifactStore
    policy_engine: PolicyEngine
    #: The deployment root. Gates fall back to it when nothing materialised a
    #: separate checkout, which is what makes a local run work in place.
    workspace_root: Path = Path(".")
    #: The team rule library, loaded once per runner so every gate judges one
    #: change against exactly one version of the rules.
    rule_set: RuleSet = field(default_factory=lambda: RuleSet(version="v1"))

    @property
    def ledger(self) -> LedgerStore:
        return self.adapters.ledger

    @property
    def llm(self) -> LLMAdapter | None:
        return self.adapters.llm

    @property
    def forge(self) -> ForgeAdapter | None:
        return self.adapters.forge

    @property
    def scanner(self) -> ScannerAdapter | None:
        return self.adapters.scanner


class GateRunner:
    """Drives one change through the gate sequence."""

    def __init__(
        self,
        workspace_root: Path,
        settings: Settings | None = None,
        adapters: AdapterSet | None = None,
        gates: dict[str, Gate] | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.settings = settings or Settings.from_env()
        self.artifact_store = ArtifactStore(workspace_root / self.settings.artifacts_dir)
        self.policy_config = PolicyConfig.load(
            workspace_root,
            self.settings.policy_path,
            self.settings.policy_version,
        )
        if self.policy_config.version != self.settings.policy_version:
            # The policy file on disk is the authority on its own version, so
            # evidence records the version that actually judged the change.
            self.settings = replace(self.settings, policy_version=self.policy_config.version)
        self.policy_engine = PolicyEngine(self.policy_config)
        self.rule_set = RuleSet.load(
            workspace_root,
            self.settings.rules_path,
            self.settings.rule_version,
        )
        if self.rule_set.version != self.settings.rule_version:
            # Same reasoning as the policy version: the rule file on disk is the
            # authority on its own version, and the version is in the
            # fingerprint, so editing a rule invalidates prior approvals.
            self.settings = replace(self.settings, rule_version=self.rule_set.version)
        self.adapters = adapters or AdapterSet(
            ledger=SQLiteLedger(workspace_root / self.settings.ledger_path),
            # Built here, used by whoever drives the platform entry point. The
            # runner itself never calls a forge: a gate's verdict must not
            # depend on a network round trip to the platform it came from.
            forge=build_forge_adapter(self.settings),
            llm=build_llm_adapter(self.settings),
            scanner=build_scanner_adapter(self.settings, self.artifact_store.root),
        )
        self.services = RuntimeServices(
            settings=self.settings,
            adapters=self.adapters,
            artifact_store=self.artifact_store,
            policy_engine=self.policy_engine,
            workspace_root=workspace_root,
            rule_set=self.rule_set,
        )
        #: Gate implementations available in this deployment. A gate that is
        #: absent is recorded as skipped -- never as passed.
        self.gates: dict[str, Gate] = (
            dict(gates) if gates is not None else self._registered_gates()
        )

    def _registered_gates(self) -> dict[str, Gate]:
        """Instantiate every gate this build ships.

        Deployment is the registry, not a config flag: an installation runs the
        gates that exist in it, and ``doctor`` reports the rest as undeployed.
        """

        return {
            name: gate_cls(self.services) for name, gate_cls in sorted(GATE_REGISTRY.items())
        }

    # ── entry points ─────────────────────────────────────────────────────

    def build_state(self, change: ChangeRequest, run_id: str | None = None) -> RunState:
        run_id = run_id or uuid.uuid4().hex[:12]
        return RunState(
            run_id=run_id,
            change=change,
            trace_id=run_id,
            current_gate=GATE_SEQUENCE[0],
            sandbox_isolation=self.settings.effective_sandbox_isolation(),
            policy_decision=self.policy_engine.evaluate_change(change),
            started_at=time.time(),
        )

    def run(self, state: RunState) -> RunState:
        self._checkpoint(state)
        return self._drive(state)

    def verify_change(self, change: ChangeRequest, run_id: str | None = None) -> RunState:
        return self.run(self.build_state(change, run_id))

    def resume_run(self, run_id: str, workspace_path: str | None = None) -> RunState:
        """Resume a run interrupted mid-gate.

        A terminal run is returned untouched, and a run waiting on a human is
        *not* resumable this way: mistaking "waiting for approval" for "crashed"
        is exactly how an unapproved change would slip through.

        ``workspace_path`` re-points the run at the tree its scanners should
        read. A checkout is a cache — a restart elsewhere, or after a cleanup,
        does not have the one the original run used — and resuming onto a tree
        that is simply gone would report on whatever happened to be at the
        deployment root instead. Left out, the recorded path stands.
        """

        state = self._load_state(run_id)
        if state.status in {"mergeable", "blocked", "failed"}:
            return state
        if state.status == "awaiting_human":
            raise ValueError(
                f"Run {run_id} is awaiting an approval decision; use resume_approval()."
            )
        if state.current_gate not in GATE_SEQUENCE:
            raise ValueError(f"Run {run_id} has no resumable gate ({state.current_gate!r}).")
        update: dict[str, Any] = {
            "status": "running",
            "next_action": "advance",
            "retry_class": None,
            "retry_delay_seconds": None,
        }
        if workspace_path is not None:
            update["workspace_path"] = workspace_path
        state = state.model_copy(update=update)
        # Checkpointed before the gates run again, so the tree this resume
        # judged is the tree the evidence pack names even if it crashes twice.
        self._checkpoint(state)
        return self._drive(state)

    def resume_approval(
        self,
        run_id: str,
        approved: bool,
        reason: str | None = None,
        reviewer: str | None = None,
        review_minutes: float | None = None,
        *,
        fingerprint: str | None = None,
        url: str | None = None,
        decided_at: datetime | None = None,
    ) -> RunState:
        """Record a human decision against the evidence it was made on.

        ``fingerprint`` is the evidence the decision was given on, when the
        caller knows it -- an approval read off the platform was judged against
        one -- and a decision given on other evidence is refused like any stale
        one. ``url`` and ``decided_at`` say where and when it was given, when
        that was not here and now.

        The decision is written beside the evidence pack as ``approval.json``,
        and the pack's reading is rendered again to show it, because a run in
        CI loses its ledger with the job and keeps only its artifacts.
        """

        state = self._load_state(run_id)
        if state.status != "awaiting_human":
            raise ValueError(f"Run {run_id} is not awaiting an approval decision.")
        evidence = state.evidence
        if evidence is None:
            raise ValueError(f"Run {run_id} has no evidence pack to approve.")

        current = evidence.fingerprint
        stored = self.services.ledger.load_approval(run_id)
        if (stored is not None and stored.get("fingerprint") != current) or (
            fingerprint is not None and fingerprint != current
        ):
            # The change moved under the reviewer. Approving it now would
            # approve something nobody read.
            raise ValueError("Stale approval fingerprint; re-run the gates on the new commit.")

        ticket = (state.approval_ticket or ApprovalTicket(fingerprint=current)).model_copy(
            update={"approved": approved, "reason": reason, "reviewer": reviewer}
        )
        self.services.ledger.resolve_approval(run_id, approved, reason, reviewer)
        self.services.ledger.save_approval_ticket(run_id, ticket)

        metrics = state.metrics.model_copy(
            update={"human_review_minutes": review_minutes}
            if review_minutes is not None
            else {}
        )
        decision = ApprovalDecision(
            approved=approved,
            fingerprint=current,
            reason=reason,
            reviewer=reviewer,
            decided_at=decided_at or utc_now(),
            url=url,
        )
        status: RunStatus = "mergeable" if approved else "blocked"
        manifest = self._record_decision(state, evidence, decision, status)
        evidence = evidence.model_copy(update={"artifact_manifest": manifest})
        self.services.ledger.save_evidence(run_id, evidence, current)
        state = state.model_copy(
            update={
                "approval_ticket": ticket,
                "approval_decision": decision,
                "evidence": evidence,
                "artifact_manifest": manifest,
                "metrics": metrics,
                "status": status,
                "last_feedback": reason,
            }
        )
        self._checkpoint(state)
        self._persist_run(state)
        return state

    def _record_decision(
        self,
        state: RunState,
        evidence: EvidencePack,
        decision: ApprovalDecision,
        status: RunStatus,
    ) -> ArtifactManifest:
        """Write a human decision beside the evidence it was given on.

        ``evidence.json`` stays as the gates wrote it: it records what they
        found, a signature given afterwards is not part of that, and its digest
        in the manifest keeps matching the file. Its reading is rendered again,
        because the one written with the run says the change is still waiting.
        """

        manifest = self.artifact_store.write_json(
            state.run_id, APPROVAL_FILENAME, decision, state.artifact_manifest
        )
        # Listed without itself, as the copy written with the run was: the
        # record there is of the rendering this one replaces.
        listed = manifest.model_copy(
            update={
                "artifacts": [
                    record
                    for record in manifest.artifacts
                    if record.name != EVIDENCE_SUMMARY_FILENAME
                ]
            }
        )
        ticket = state.approval_ticket
        manifest = self.artifact_store.write_text(
            state.run_id,
            EVIDENCE_SUMMARY_FILENAME,
            render_evidence(
                evidence.model_copy(update={"artifact_manifest": listed}),
                status=status,
                approval=decision,
                assistant_report=ticket.assistant_report if ticket is not None else None,
            ),
            "markdown",
            manifest,
        )
        self.services.ledger.save_artifact_manifest(state.run_id, manifest)
        return manifest

    def annotate_finding(
        self,
        run_id: str,
        finding_id: str,
        *,
        false_positive: bool,
        reason: str | None = None,
        annotator: str | None = None,
    ) -> RunState:
        """Record a human's reading of one finding without touching the verdict.

        The block stands. What changes is the finding's ``false_positive`` flag
        in the run's evidence and the false-positive count in its metrics --
        which is where revising the rule starts. Undoing a block is a new
        commit or a signed approval, never an annotation.
        """

        state = self._load_state(run_id)
        evidence = state.evidence
        if evidence is None:
            raise ValueError(f"Run {run_id} has no evidence pack yet; nothing to annotate.")
        if not any(finding.id == finding_id for finding in evidence.findings):
            known = ", ".join(finding.id for finding in evidence.findings) or "(none)"
            raise ValueError(f"Run {run_id} has no finding {finding_id!r}. Findings: {known}.")

        annotation = FindingAnnotation(
            run_id=run_id,
            finding_id=finding_id,
            fingerprint=evidence.fingerprint,
            false_positive=false_positive,
            reason=reason,
            annotator=annotator,
        )
        self.services.ledger.save_annotation(annotation)

        reports = {
            name: _mark_finding(report, finding_id, false_positive)
            for name, report in state.gate_reports.items()
        }
        evidence = evidence.model_copy(
            update={
                "gate_reports": [
                    _mark_finding(report, finding_id, false_positive)
                    for report in evidence.gate_reports
                ]
            }
        )
        # A targeted recount rather than ``_aggregate_metrics``: that one also
        # re-measures cycle time from the clock, and an annotation made a week
        # later must not make the run look like it took a week.
        metrics = state.metrics.model_copy(
            update={"false_positives": _false_positive_count(evidence.findings)}
        )
        evidence = evidence.model_copy(update={"metrics": metrics})
        state = state.model_copy(
            update={"gate_reports": reports, "evidence": evidence, "metrics": metrics}
        )
        self.services.ledger.save_evidence(run_id, evidence, evidence.fingerprint)
        self._checkpoint(state)
        self._persist_run(state)
        return state

    def record_review_minutes(self, run_id: str, minutes: float) -> RunState:
        """Record how long a person spent on a run that reached a verdict.

        ``approve --minutes`` only reaches runs that waited for a human; the
        minutes spent confirming a block, or reading a mergeable change anyway,
        are the larger part of what the gate costs people and had nowhere to go.
        """

        state = self._load_state(run_id)
        if state.evidence is None:
            raise ValueError(f"Run {run_id} has not reached a verdict; nothing to attribute to.")
        metrics = state.metrics.model_copy(update={"human_review_minutes": minutes})
        evidence = state.evidence.model_copy(update={"metrics": metrics})
        state = state.model_copy(update={"metrics": metrics, "evidence": evidence})
        self.services.ledger.save_evidence(run_id, evidence, evidence.fingerprint)
        self._checkpoint(state)
        self._persist_run(state)
        return state

    def rerun_gate(self, run_id: str, gate: str, reason: str | None = None) -> RunState:
        """Re-run one gate and everything derived from it.

        Deliberately not a bare call to ``Gate.run``: re-running a gate has to
        invalidate the evidence pack and any approval built on top of it first.
        """

        if gate not in GATE_SEQUENCE:
            raise ValueError(f"Unknown gate: {gate}. Expected one of: {', '.join(GATE_SEQUENCE)}.")
        state = self._load_state(run_id)
        state = self._invalidate_from(state, gate, reset_attempts=True)
        state = state.model_copy(
            update={
                "current_gate": gate,
                "status": "running",
                "last_feedback": reason,
                "retry_class": "manual_rerun",
                "retry_delay_seconds": None,
                "next_action": "advance",
            }
        )
        return self._drive(state)

    def status(self, run_id: str) -> RunState | None:
        return self.services.ledger.load_state(run_id)

    def evidence_path(self, state: RunState) -> Path | None:
        """Where this run's evidence pack landed, or ``None`` before it exists."""

        return self._artifact_path(state, EVIDENCE_FILENAME)

    def summary_path(self, state: RunState) -> Path | None:
        """Where the readable rendering of that pack landed."""

        return self._artifact_path(state, EVIDENCE_SUMMARY_FILENAME)

    def _artifact_path(self, state: RunState, name: str) -> Path | None:
        """Look an artifact up in the manifest, never by rebuilding its path.

        The manifest is what the SHA-256 digests were taken over, so a path that
        is not in it is a path nobody can verify.
        """

        manifest = state.artifact_manifest
        if manifest is None:
            return None
        for record in manifest.artifacts:
            if record.name == name:
                return self.artifact_store.root / record.path
        return None

    def inspect(self, run_id: str) -> dict[str, Any]:
        state = self.status(run_id)
        record = self.services.ledger.load_run_record(run_id)
        return {
            "state": state.model_dump(mode="json") if state else None,
            "approval": self.services.ledger.load_approval(run_id),
            "run_record": record.model_dump(mode="json") if record else None,
            "artifact_manifest": self.services.ledger.load_artifact_manifest(run_id),
        }

    # ── the loop ─────────────────────────────────────────────────────────

    def _drive(self, state: RunState) -> RunState:
        # A bounded budget keeps a misconfigured retry policy from spinning
        # forever; every gate can still use its own full attempt budget.
        budget = len(GATE_SEQUENCE) * (self.settings.max_attempts + 2) + 8
        for _ in range(budget):
            gate_name = state.current_gate
            if gate_name not in GATE_SEQUENCE:
                return self._finalize(state, "failed")
            state = self._execute_gate(gate_name, state)
            action: Route = state.next_action

            if action == "advance":
                index = GATE_SEQUENCE.index(gate_name)
                if index + 1 >= len(GATE_SEQUENCE):
                    return self._finalize(state, self._outcome_for(state))
                state = state.model_copy(update={"current_gate": GATE_SEQUENCE[index + 1]})
                continue
            if action == "retry_current":
                delay = state.retry_delay_seconds or 0.0
                if delay:
                    time.sleep(delay)
                state = state.model_copy(update={"retry_delay_seconds": None})
                continue
            if action == "wait_human":
                return self._finalize(state, "needs_human")
            if action == "block":
                return self._finalize(state, "blocked")
            return self._finalize(state, "failed")
        return self._finalize(state, "failed")

    def _execute_gate(self, gate_name: str, state: RunState) -> RunState:
        # Record the execution boundary before running anything: if the process
        # dies mid-gate, the checkpoint points at this gate rather than at the
        # previous one, so a resume re-runs it instead of skipping it.
        self._checkpoint(state.model_copy(update={"current_gate": gate_name, "status": "running"}))
        attempts = dict(state.attempts)
        attempts[gate_name] = attempts.get(gate_name, 0) + 1
        attempt = attempts[gate_name]
        started = time.perf_counter()

        try:
            report = self._report_for(gate_name, state)
            route, failure_class = self._route_for_verdict(gate_name, report.verdict, attempt)
            feedback, brief = self._explain(report, attempt)
            next_state = state.model_copy(
                update={
                    "attempts": attempts,
                    "current_gate": gate_name,
                    "gate_reports": {**state.gate_reports, gate_name: report},
                    "briefs": _with_brief(state.briefs, brief),
                    "last_verdict": report.verdict,
                    "last_feedback": feedback,
                    "retry_class": failure_class,
                    "retry_delay_seconds": None,
                    "per_gate_timing": {
                        **state.per_gate_timing,
                        gate_name: report.duration_s or 0.0,
                    },
                }
            )
            next_state = self._capture_artifacts(gate_name, report, next_state)
            outcome = {
                Verdict.PASS: "passed",
                Verdict.SKIPPED: "skipped",
                Verdict.NEEDS_HUMAN: "waiting",
                Verdict.BLOCKED: "blocked",
            }[report.verdict]
            next_state = self._record_attempt(
                next_state,
                gate=gate_name,
                attempt=attempt,
                outcome=outcome,
                route=route,
                feedback=feedback,
                failure_class=failure_class,
                duration_s=report.duration_s,
            )
        except Exception as exc:  # noqa: BLE001 - classify gate failures durably.
            feedback = f"{type(exc).__name__}: {exc}"
            retryable = self._is_retryable_exception(exc)
            route = (
                "retry_current"
                if retryable and attempt < self.settings.max_attempts_for(gate_name)
                else "fail"
            )
            failure_class = "transient_exception" if retryable else "fatal_exception"
            duration = round(time.perf_counter() - started, 3)
            next_state = state.model_copy(
                update={
                    "attempts": attempts,
                    "current_gate": gate_name,
                    "last_feedback": feedback,
                    "retry_class": failure_class,
                    "per_gate_timing": {**state.per_gate_timing, gate_name: duration},
                }
            )
            next_state = self._record_attempt(
                next_state,
                gate=gate_name,
                attempt=attempt,
                outcome="exception",
                route=route,
                feedback=feedback,
                failure_class=failure_class,
                duration_s=duration,
            )

        if route == "retry_current":
            next_state = next_state.model_copy(
                update={"retry_delay_seconds": self._retry_delay(attempt)}
            )
        next_state = next_state.model_copy(update={"next_action": route})
        self._checkpoint(next_state)
        return next_state

    def _report_for(self, gate_name: str, state: RunState) -> GateReport:
        gate = self.gates.get(gate_name)
        if gate is None:
            # Honest accounting for a partial deployment: the evidence pack says
            # this gate did not run, and no report can claim it passed.
            return self._skipped(
                gate_name,
                state,
                f"This deployment has no {gate_name} gate; recorded as skipped, "
                "which is not a pass.",
            )
        if gate_name in self.settings.disabled_gates:
            # Switched off is the one legitimate source of SKIPPED besides "not
            # deployed". The reason is written down so a reader can tell it
            # from a gate that ran, and the policy still gets to require it.
            return self._skipped(
                gate_name,
                state,
                f"This deployment switched the {gate_name} gate off "
                "(HONEST_CI_DISABLED_GATES); recorded as skipped, which is not a pass.",
            )
        return gate.run(state)

    def _skipped(self, gate_name: str, state: RunState, summary: str) -> GateReport:
        return GateReport(
            gate=gate_name,
            verdict=Verdict.SKIPPED,
            summary=summary,
            policy_version=self.settings.policy_version,
            rule_version=self.settings.rule_version,
            trace_id=state.trace_id,
            duration_s=0.0,
        )

    # ── model explanation ────────────────────────────────────────────────

    def _explain(self, report: GateReport, attempt: int) -> tuple[str | None, BlockBrief | None]:
        """Say why a gate blocked: deterministically first, model second.

        The block message is written by the trust boundary rather than by the
        gate, so there is exactly one place where model text can be attached to
        a verdict — and one place where it is thrown away. A brief that cites
        anything the gate did not produce never reaches this return value; the
        deterministic feedback does, either way.
        """

        if report.verdict is not Verdict.BLOCKED:
            return report.feedback, None
        llm = self._advisory_llm()
        if llm is None:
            return deterministic_feedback(report), None
        return guarded_block_brief(report, attempt, llm)

    def _reading_aid(self, evidence: EvidencePack) -> ModelReviewReport | None:
        """A grounded summary for the human this run is about to interrupt."""

        llm = self._advisory_llm()
        if llm is None:
            return None
        return guarded_review_report(evidence, llm)

    def _advisory_llm(self) -> LLMAdapter | None:
        """The model, in the only role it has here: advisory text.

        ``None`` means no credential is configured, or explanations are off.
        Both are ordinary states rather than failures — verdicts are
        deterministic, so a run without a model is complete, just quieter.
        """

        if not self.settings.model_explain:
            return None
        return self.services.llm

    # ── routing ──────────────────────────────────────────────────────────

    def _route_for_verdict(
        self,
        gate: str,
        verdict: Verdict,
        attempt: int,
    ) -> tuple[Route, str | None]:
        if verdict in {Verdict.PASS, Verdict.SKIPPED}:
            return "advance", None
        if verdict is Verdict.NEEDS_HUMAN:
            return "wait_human", f"{gate}_needs_human"
        return "block", f"{gate}_blocked"

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        explicit = getattr(exc, "retryable", None)
        if explicit is not None:
            return bool(explicit)
        if isinstance(exc, ConnectionError | TimeoutError):
            return True
        return is_transient_failure(str(exc))

    def _retry_delay(self, attempt: int) -> float:
        delay = self.settings.retry_delay_for(attempt)
        jitter = self.settings.retry_jitter_seconds
        return delay + random.uniform(0.0, jitter) if jitter > 0 else delay

    def _record_attempt(
        self,
        state: RunState,
        *,
        gate: str,
        attempt: int,
        outcome: str,
        route: str,
        feedback: str | None,
        failure_class: str | None,
        duration_s: float | None,
    ) -> RunState:
        record = GateAttempt(
            gate=gate,
            attempt=attempt,
            outcome=outcome,
            route=route,
            feedback=feedback,
            failure_class=failure_class,
            duration_s=duration_s,
        )
        return state.model_copy(
            update={
                "execution_history": [*state.execution_history, record],
                "retry_class": failure_class,
            }
        )

    # ── invalidation ─────────────────────────────────────────────────────

    def _invalidate_from(
        self,
        state: RunState,
        gate: str,
        *,
        reset_attempts: bool = False,
    ) -> RunState:
        """Drop everything derived from ``gate`` onwards.

        Re-running a gate makes the evidence pack and any approval built on it
        stale by definition, so both are cleared here rather than being
        silently reused by the next finalize.
        """

        start = GATE_SEQUENCE.index(gate)
        invalidated = set(GATE_SEQUENCE[start:])
        reports = {
            name: report
            for name, report in state.gate_reports.items()
            if name not in invalidated
        }
        update: dict[str, Any] = {
            "gate_reports": reports,
            "evidence": None,
            "approval_ticket": None,
            "approval_decision": None,
            "briefs": [brief for brief in state.briefs if brief.gate not in invalidated],
            "metrics": RunMetrics(),
            "last_verdict": None,
        }
        if reset_attempts:
            update["attempts"] = {
                name: count
                for name, count in state.attempts.items()
                if name not in invalidated
            }
        self.services.ledger.clear_approval(state.run_id)
        return state.model_copy(update=update)

    # ── evidence & persistence ───────────────────────────────────────────

    def _capture_artifacts(self, gate_name: str, report: GateReport, state: RunState) -> RunState:
        manifest = self.artifact_store.write_json(
            state.run_id,
            f"gate-{gate_name}.json",
            report,
            state.artifact_manifest,
        )
        self.services.ledger.save_artifact_manifest(state.run_id, manifest)
        return self._sync_llm_calls(state.model_copy(update={"artifact_manifest": manifest}))

    def _sync_llm_calls(self, state: RunState) -> RunState:
        """Copy the model adapter's call log into the run and re-write it.

        Called again after any late model call — the reading aid a hand-off asks
        for — so the evidence pack accounts for every token this run spent,
        including the ones spent on text that grounding then discarded.
        """

        calls = list(getattr(self.adapters.llm, "call_records", None) or [])
        if not calls:
            return state
        manifest = self.artifact_store.write_json(
            state.run_id,
            LLM_CALLS_FILENAME,
            [call.model_dump(mode="json") for call in calls],
            state.artifact_manifest,
        )
        self.services.ledger.save_artifact_manifest(state.run_id, manifest)
        return state.model_copy(update={"llm_calls": calls, "artifact_manifest": manifest})

    def _outcome_for(self, state: RunState) -> RunOutcome:
        """Decide a completed run's outcome.

        "Nothing blocked it" is not the same as "it was verified". A run whose
        shape put it outside automated judgment, whose policy asks for a gate
        this deployment never ran, or in which no gate ran at all, ends at a
        human rather than reporting itself mergeable on no evidence.
        """

        verdict = self._overall_verdict(state)
        if verdict is Verdict.BLOCKED:
            return "blocked"
        if verdict is Verdict.NEEDS_HUMAN:
            return "needs_human"
        if state.policy_decision is not None and not state.policy_decision.allowed:
            return "needs_human"
        if unsatisfied_required_checks(state):
            return "needs_human"
        if not any(
            report.verdict is not Verdict.SKIPPED for report in state.gate_reports.values()
        ):
            return "needs_human"
        return "mergeable"

    @staticmethod
    def _overall_verdict(state: RunState) -> Verdict:
        worst = Verdict.PASS
        for report in state.gate_reports.values():
            if _VERDICT_RANK[report.verdict] > _VERDICT_RANK[worst]:
                worst = report.verdict
        return worst

    def _finalize(self, state: RunState, outcome: RunOutcome) -> RunState:
        """Assemble the evidence pack, then record the run's outcome."""

        fingerprint = compute_evidence_fingerprint(
            state.change.head_sha,
            self.settings.policy_version,
            self.settings.rule_version,
        )
        state = self._reapply_annotations(state, fingerprint)
        status = _OUTCOME_STATUS[outcome]
        ticket: ApprovalTicket | None = None
        if outcome == "needs_human":
            # Built before the pack is written: the reading aid can cost a model
            # call, and a call this run made belongs in this run's evidence.
            ticket = self._approval_ticket(state, fingerprint)
            state = self._sync_llm_calls(state)

        metrics = self._aggregate_metrics(state)
        evidence = self._evidence_pack(state, fingerprint, metrics)
        manifest = self.artifact_store.write_json(
            state.run_id,
            EVIDENCE_FILENAME,
            evidence,
            state.artifact_manifest,
        )
        # The rendering is made from the pack that already knows where its own
        # JSON landed, so a reader can check the digest of the file this text
        # was derived from.
        evidence = evidence.model_copy(update={"artifact_manifest": manifest})
        manifest = self.artifact_store.write_text(
            state.run_id,
            EVIDENCE_SUMMARY_FILENAME,
            render_evidence(
                evidence,
                status=status,
                assistant_report=ticket.assistant_report if ticket else None,
            ),
            "markdown",
            manifest,
        )
        self.services.ledger.save_artifact_manifest(state.run_id, manifest)
        evidence = evidence.model_copy(update={"artifact_manifest": manifest})

        update: dict[str, Any] = {
            "evidence": evidence,
            "metrics": metrics,
            "artifact_manifest": manifest,
            "status": status,
        }
        if ticket is not None:
            update["approval_ticket"] = ticket
        state = state.model_copy(update=update)

        self.services.ledger.save_evidence(state.run_id, evidence, fingerprint)
        if state.approval_ticket is not None:
            self.services.ledger.save_approval_ticket(state.run_id, state.approval_ticket)
        self._checkpoint(state)
        self._persist_run(state)
        return state

    def _reapply_annotations(self, state: RunState, fingerprint: str) -> RunState:
        """Carry human annotations across a re-run that reached the same evidence.

        A re-run rebuilds every finding from scratch, so the flags a person set
        would vanish from the pack even though the same rules produced the same
        finding on the same commit. Only annotations made under *this*
        fingerprint come back: a rule revision changes the fingerprint, and a
        note about a finding the old rules produced is history, not a fact
        about the new one.
        """

        notes = [
            note
            for note in self.services.ledger.list_annotations(state.run_id)
            if note.fingerprint == fingerprint
        ]
        if not notes:
            return state
        reports = dict(state.gate_reports)
        for note in notes:
            reports = {
                name: _mark_finding(report, note.finding_id, note.false_positive)
                for name, report in reports.items()
            }
        return state.model_copy(update={"gate_reports": reports})

    def _gate_configuration(self, state: RunState) -> GateConfiguration:
        """The settings that judged this change, as the pack will carry them.

        Read here rather than in a gate because it is a property of the
        deployment, not of any one judgment, and read at pack time rather than
        at start-up so a resumed run records the tree it actually resumed into.

        Everything here is configuration. No credential reaches it: the
        provider and the model name say which service was asked, and
        ``active_model`` is ``"none"`` exactly when no model will be asked --
        none was enabled, or its key is missing -- which this turns back into
        ``None``, because a pack that names a model is a pack where
        judgment-class rules were really read.
        """

        # Imported here, not at module scope: ``__init__`` defines ``__version__``
        # after importing this module, so a top-level import would be circular.
        from . import __version__

        workspace = Path(state.workspace_path) if state.workspace_path else self.workspace_root
        scanner = self.services.scanner
        return GateConfiguration(
            product_version=__version__,
            rules_path=self.rule_set.source,
            rules_sha256=self.rule_set.sha256,
            rule_version=self.settings.rule_version,
            policy_path=self.policy_config.source,
            policy_sha256=self.policy_config.sha256,
            policy_version=self.settings.policy_version,
            required_checks=list(self.policy_config.required_checks),
            # What this deployment can actually run, in pipeline order -- not
            # what the registry ships. A gate that is absent is recorded as
            # skipped, and the snapshot has to show the same thing.
            deployed_gates=[name for name in GATE_SEQUENCE if name in self.gates],
            disabled_gates=list(self.settings.disabled_gates),
            fail_closed=self.settings.fail_closed,
            # From the run, not from the settings: ``docker`` degrades to
            # ``process`` when no daemon answered, and the pack records the
            # isolation the run had rather than the one it asked for.
            sandbox_isolation=state.sandbox_isolation,
            language=scanner.toolchain(workspace) if scanner is not None else "",
            docker_image=scanner.image(workspace) if scanner is not None else None,
            test_command=self.settings.test_command,
            test_report=self.settings.test_report,
            coverage_threshold=self.settings.coverage_threshold,
            changed_line_coverage_threshold=self.settings.changed_line_coverage_threshold,
            test_selection=self.settings.test_selection,
            model_provider=self.settings.model_provider or "",
            model=self.settings.active_model if self.settings.active_provider else None,
            model_explain=self.settings.model_explain,
            model_judgment=self.settings.model_judgment,
            model_testgen=self.settings.model_testgen,
            model_min_confidence=self.settings.model_min_confidence,
            prompt_version=self.settings.prompt_version,
        )

    def _evidence_pack(
        self,
        state: RunState,
        fingerprint: str,
        metrics: RunMetrics,
    ) -> EvidencePack:
        return EvidencePack(
            run_id=state.run_id,
            change=state.change,
            gate_reports=[
                state.gate_reports[name] for name in GATE_SEQUENCE if name in state.gate_reports
            ],
            verdict=self._overall_verdict(state),
            fingerprint=fingerprint,
            trace_id=state.trace_id,
            policy_version=self.settings.policy_version,
            rule_version=self.settings.rule_version,
            policy_decision=state.policy_decision,
            metrics=metrics,
            artifact_manifest=state.artifact_manifest,
            llm_calls=state.llm_calls,
            briefs=state.briefs,
            sandbox_isolation=state.sandbox_isolation,
            configuration=self._gate_configuration(state),
        )

    def _approval_ticket(self, state: RunState, fingerprint: str) -> ApprovalTicket:
        """What the reviewer signs against, plus what helps them read it.

        ``assistant_report`` is a reading aid and nothing more: it is bound to
        this evidence fingerprint, every finding in it has to land on a line the
        change added, and it is dropped whole when either check fails. The
        signature is still given to the gates' evidence, not to a summary of it.
        """

        return ApprovalTicket(
            fingerprint=fingerprint,
            evidence_summary=self._approval_summary(state),
            forge=state.change.forge,
            repo=state.change.repo,
            number=state.change.number,
            url=state.change.url,
            assistant_report=self._reading_aid(
                self._evidence_pack(state, fingerprint, self._aggregate_metrics(state))
            ),
        )

    @staticmethod
    def _approval_summary(state: RunState) -> str:
        """Tell the reviewer what ran, and why this reached them at all."""

        parts = [
            f"{name}: {report.verdict.value}" for name, report in state.gate_reports.items()
        ]
        summary = f"{state.change.ref} — " + ", ".join(parts)
        decision = state.policy_decision
        if decision is not None and not decision.allowed:
            summary += " | structural guardrails: " + "; ".join(decision.violations)
        missing = unsatisfied_required_checks(state)
        if missing:
            summary += " | required by the policy but reached no verdict: " + ", ".join(missing)
        if not any(
            report.verdict is not Verdict.SKIPPED for report in state.gate_reports.values()
        ):
            summary += " | no gate actually ran, so this change has no automated evidence"
        return summary

    def _aggregate_metrics(self, state: RunState) -> RunMetrics:
        reports = list(state.gate_reports.values())
        by_severity: dict[str, int] = {}
        blocking = 0
        findings_total = 0
        false_positives = 0
        for report in reports:
            for finding in report.findings:
                findings_total += 1
                by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
                if finding.blocking:
                    blocking += 1
                if finding.false_positive:
                    false_positives += 1
        cycle_time = round(time.time() - state.started_at, 3) if state.started_at else None
        return RunMetrics(
            gates_run=len([report for report in reports if report.verdict is not Verdict.SKIPPED]),
            gates_blocked=len([r for r in reports if r.verdict is Verdict.BLOCKED]),
            findings_total=findings_total,
            findings_by_severity=by_severity,
            blocking_findings=blocking,
            human_review_minutes=state.metrics.human_review_minutes,
            ai_generated_ratio=state.change.ai_generated_ratio,
            rework_count=max(0, sum(state.attempts.values()) - len(state.attempts)),
            cycle_time_s=cycle_time,
            false_positives=false_positives or None,
            token_usage=_token_usage(state),
        )

    def _persist_run(self, state: RunState) -> None:
        outcome: RunOutcome = {
            "mergeable": "mergeable",
            "blocked": "blocked",
            "awaiting_human": "needs_human",
            "failed": "failed",
            "running": "running",
        }[state.status]  # type: ignore[assignment]
        record = RunRecord(
            run_id=state.run_id,
            change_ref=state.change.ref,
            outcome=outcome,
            per_gate_timing=state.per_gate_timing,
            metrics=state.metrics,
            # The evidence pack itself, not the ledger that indexes it: this
            # field is what a report or a reader of the ledger follows to the
            # artifact on disk.
            evidence_path=_optional_str(self.evidence_path(state)),
            model=self.settings.active_model,
            prompt_version=self.settings.prompt_version,
            token_usage=_token_usage(state),
            attempts=state.attempts,
            failure_reasons=self._failure_reasons(state),
            sandbox_isolation=state.sandbox_isolation,
        )
        self.services.ledger.save_run_record(record)

    @staticmethod
    def _failure_reasons(state: RunState) -> list[str]:
        """Explain a non-mergeable run from durable state.

        ``last_feedback`` is the current verdict's reason; when a gate recorded
        a verdict without feedback, the blocking attempt still names its
        failure class. A mergeable run has nothing to report.
        """

        if state.status == "mergeable":
            return []
        if state.last_feedback:
            return [state.last_feedback]
        for attempt in reversed(state.execution_history):
            if attempt.outcome in {"blocked", "exception"}:
                detail = attempt.feedback or attempt.failure_class
                if detail:
                    return [f"{attempt.gate}: {detail}"]
        return []

    def _checkpoint(self, state: RunState) -> None:
        self.services.ledger.save_state(state)

    def _load_state(self, run_id: str) -> RunState:
        state = self.services.ledger.load_state(run_id)
        if state is None:
            raise ValueError(f"Unknown run_id: {run_id}")
        return state


def unsatisfied_required_checks(state: RunState) -> list[str]:
    """This run's view of ``policy.unsatisfied_checks``.

    The same reading has to hold for an evidence pack read long after the run,
    so the rule itself lives in ``policy.py`` and both callers share it.
    """

    return unsatisfied_checks(
        state.policy_decision,
        {name: report.verdict for name, report in state.gate_reports.items()},
    )


def _mark_finding(report: GateReport, finding_id: str, false_positive: bool) -> GateReport:
    """The report with one finding's annotation flag set; everything else as was."""

    if not any(finding.id == finding_id for finding in report.findings):
        return report
    return report.model_copy(
        update={
            "findings": [
                finding.model_copy(update={"false_positive": false_positive})
                if finding.id == finding_id
                else finding
                for finding in report.findings
            ]
        }
    )


def _false_positive_count(findings: list[Finding]) -> int | None:
    """``None`` until someone has annotated; a count of zero is a claim."""

    if not any(finding.false_positive is not None for finding in findings):
        return None
    return len([finding for finding in findings if finding.false_positive])


def _with_brief(briefs: list[BlockBrief], brief: BlockBrief | None) -> list[BlockBrief]:
    """One brief per gate: a re-run's explanation replaces the one it re-ran."""

    if brief is None:
        return briefs
    return [*(item for item in briefs if item.gate != brief.gate), brief]


def _optional_str(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _token_usage(state: RunState) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for call in state.llm_calls:
        if call.input_tokens:
            totals["input_tokens"] += call.input_tokens
        if call.output_tokens:
            totals["output_tokens"] += call.output_tokens
        if call.total_tokens:
            totals["total_tokens"] += call.total_tokens
    return {key: value for key, value in totals.items() if value}


def verify_change(
    change: ChangeRequest,
    workspace_root: Path,
    settings: Settings | None = None,
    adapters: AdapterSet | None = None,
    gates: dict[str, Gate] | None = None,
) -> RunState:
    """Run one change through every deployed gate and return its final state."""

    runner = GateRunner(
        workspace_root=workspace_root,
        settings=settings,
        adapters=adapters,
        gates=gates,
    )
    return runner.verify_change(change)


__all__ = [
    "EVIDENCE_FILENAME",
    "EVIDENCE_SUMMARY_FILENAME",
    "GateName",
    "GateRunner",
    "RuntimeServices",
    "unsatisfied_required_checks",
    "verify_change",
]
