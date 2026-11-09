"""One merge request, end to end: fetch it, gate it, write the verdict back.

``honest-ci run-pr`` and ``honest-ci resume`` both end here, so a run that was
interrupted halfway is finished, and written back, exactly the way it would
have been had nothing stopped it.

It sits *above* the control plane rather than inside it. ``runner.py`` never
calls a forge — a gate's verdict must not depend on a round trip to the
platform the change came from — so the two ends that do talk to a platform,
taking the change in and putting the conclusion back, live here.

The steps are separate functions because the CLI prints between them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .adapters.base import ForgeAdapter, ForgeError, InlinePublishError
from .approvals import read_approvals
from .contracts import ApprovalReading, ChangeRequest, Verdict
from .license_check import Stopped
from .redaction import redact_text
from .report import inline_findings, render_forge_comment
from .runner import GateRunner
from .settings import Settings
from .state import RunState

log = logging.getLogger(__name__)

#: Terminal state → the verdict written to the platform's commit status. A run
#: that failed maps to the platform's failure state because it produced no
#: verdict at all, and a merge rule must not be satisfied by a run that fell over.
#: It is published through ``ForgeAdapter.publish_error``, which is this same
#: blocked status unless the platform has a state of its own for "no verdict".
STATUS_VERDICT: dict[str, Verdict] = {
    "mergeable": Verdict.PASS,
    "blocked": Verdict.BLOCKED,
    "awaiting_human": Verdict.NEEDS_HUMAN,
    "failed": Verdict.BLOCKED,
    "running": Verdict.NEEDS_HUMAN,
}


@dataclass(frozen=True)
class MergeRequestSource:
    """The change to judge, and the tree the scanners will read it on."""

    change: ChangeRequest
    checkout: Path


@dataclass(frozen=True)
class Published:
    """What became of the write-back.

    ``sent`` and ``ok`` are apart on purpose. A dry run is ``ok`` — it did what
    it was asked — but nothing reached the platform, and a caller deciding an
    exit code needs to know which of the two it is looking at.
    """

    ok: bool
    sent: bool
    verdict: Verdict | None = None
    summary: str = ""
    #: The note, rendered. Present even on a dry run — that is the point of one.
    comment: str = ""
    error: str | None = None
    inline_posted: int = 0
    inline_skipped: str | None = None


@dataclass(frozen=True)
class MergeRequestRun:
    """One full pass: what the gates concluded, and whether the platform has it."""

    state: RunState
    published: Published
    checkout: Path | None = None

    @property
    def outcome(self) -> str:
        """The terminal state, downgraded to ``failed`` if nobody was told.

        A conclusion that never reached the platform is a conclusion the team
        cannot act on. The evidence pack and the ledger still have it, which is
        why this is reported as a failed *run* rather than as a lost verdict.
        """

        return self.state.status if self.published.ok else "failed"


def fetch_merge_request(
    adapter: ForgeAdapter,
    settings: Settings,
    *,
    project: str,
    number: int,
    workspace: Path,
) -> MergeRequestSource:
    """Pull the change and check its head commit out. Raises ``ForgeError``."""

    change = _with_platform_reviews(adapter, adapter.fetch_change(project, number))
    checkout = adapter.materialize(change, workspace / settings.workspace_dir)
    return MergeRequestSource(change=change, checkout=checkout)


def _with_platform_reviews(adapter: ForgeAdapter, change: ChangeRequest) -> ChangeRequest:
    """Attach what people already wrote on this change, or record that we could not.

    Fail-soft, and read-only: a platform that will not hand over its discussion
    threads must not stop the gates from running, and a ``--dry-run`` reads
    them like any other run. What it must not do is leave the change looking
    like one nobody has commented on, so a refusal lands on the change as a
    recorded limit instead of as an empty list.
    """

    try:
        reviews = adapter.fetch_reviews(change)
    except ForgeError as exc:
        reason = redact_text(str(exc))
        log.warning(
            "Review comments already on the platform were not read; the gates run anyway: %s",
            reason,
        )
        return change.model_copy(update={"platform_reviews_error": reason})
    return change.model_copy(update={"platform_reviews": list(reviews)})


def run_gates(
    runner: GateRunner,
    source: MergeRequestSource,
    *,
    adapter: ForgeAdapter | None = None,
    dry_run: bool = False,
    target_url: str | None = None,
) -> RunState:
    """Run the deployed gates over a materialised change.

    The same split as the local entry point: the deployment root owns the
    ledger, the rules and the policy, and ``workspace_path`` is the tree the
    scanners read. Recording it here is what puts the path into the evidence.

    ``target_url`` is where the "verifying" status links to: the page
    ``publish_conclusion`` will link the verdict to, so that the run can be
    opened from the change while it is still going, not only once it ends.
    """

    state = runner.build_state(source.change).model_copy(
        update={"workspace_path": str(source.checkout)}
    )
    if adapter is not None and not dry_run:
        try:
            adapter.publish_progress(
                source.change, f"Verifying · run {state.run_id}", target_url
            )
        except ForgeError as exc:
            log.warning(
                "The 'verifying' status was not sent; the gates run anyway: %s",
                redact_text(str(exc)),
            )
    return runner.run(state)


def apply_platform_approvals(
    adapter: ForgeAdapter,
    runner: GateRunner,
    state: RunState,
) -> tuple[RunState, ApprovalReading | None]:
    """Turn an approval given on the platform into the run's human decision.

    Asked only of a run the gates handed to a person. A blocked change is not
    approvable -- the fix is a new commit, not a signature -- and a mergeable
    one needs nobody's, so neither reads the platform's approvals at all.

    The reading comes back whether or not anything counted, so the conclusion
    can say what did not and why. ``None`` when nothing was asked: the run did
    not wait on a person, or the platform does not hand approvals over. A
    platform that refuses is recorded on the reading, and the change keeps
    waiting: an approval nobody could read is not an approval.
    """

    evidence = state.evidence
    if state.status != "awaiting_human" or evidence is None:
        return state, None
    try:
        found = adapter.fetch_approvals(state.change)
    except ForgeError as exc:
        reason = redact_text(str(exc))
        log.warning("Approvals on the platform were not read; the change keeps waiting: %s", reason)
        return state, ApprovalReading(error=reason)
    if found is None:
        return state, None
    reading = read_approvals(
        found, head_sha=state.change.head_sha, fingerprint=evidence.fingerprint
    )
    approval = reading.approval
    if approval is None:
        return state, reading
    try:
        approved = runner.resume_approval(
            state.run_id,
            approved=True,
            reason=approval.reason,
            reviewer=approval.reviewer,
            fingerprint=evidence.fingerprint,
            url=approval.url,
            decided_at=approval.submitted_at,
        )
    except ValueError as exc:
        # The runner refused the decision -- a stale fingerprint in the ledger,
        # say. Its refusal stands, and the conclusion says why nothing counted.
        reason = redact_text(str(exc))
        log.warning("An approval on the platform was not applied: %s", reason)
        return state, reading.model_copy(update={"approval": None, "error": reason})
    return approved, reading


def publish_conclusion(
    adapter: ForgeAdapter,
    state: RunState,
    *,
    dry_run: bool = False,
    target_url: str | None = None,
    approvals: ApprovalReading | None = None,
    evidence_at: str | None = None,
    license_note: str | None = None,
) -> Published:
    """Write the conclusion back: one note and one commit status.

    The findings worth a line comment are offered to the platform as well. A
    platform that posts none reports none posted, which is not a failure: the
    evidence pack carries every finding either way.

    A platform refusal is returned rather than raised. The gates already
    concluded; what is in doubt is only whether the platform heard about it,
    and that is the caller's decision to render or to retry. Anything that is
    *not* a platform refusal still propagates — a bug in this codebase must not
    be filed away as "the platform said no".

    ``target_url`` is where the status links to, when the caller knows a page
    about this run -- a CI job's log, say. Left out, the status has no link.
    ``approvals`` is what ``apply_platform_approvals`` read, for the note.
    ``evidence_at`` is where the note says the full evidence pack is, for a
    caller whose ledger does not outlive the run. ``license_note`` is what the
    license check says about the run, which the note ends with.
    """

    evidence = state.evidence
    if evidence is None:
        return Published(
            ok=False,
            sent=False,
            error="No evidence pack to write back: the run never reached an outcome.",
        )
    verdict = STATUS_VERDICT[state.status]
    summary = status_summary(state)
    comment = render_forge_comment(
        evidence,
        status=state.status,
        approvals=approvals,
        evidence_at=evidence_at,
        license_note=license_note,
    )
    if dry_run:
        return Published(ok=True, sent=False, verdict=verdict, summary=summary, comment=comment)

    try:
        adapter.publish_report(
            state.change,
            evidence,
            state.status,
            approvals,
            evidence_at,
            license_note=license_note,
        )
        # Only the caller can know a page about this run; a link back to the
        # merge request the status is already on would help nobody.
        if state.status == "failed":
            adapter.publish_error(state.change, summary, target_url)
        else:
            adapter.publish_status(state.change, verdict, summary, target_url)
    except ForgeError as exc:
        return Published(
            ok=False,
            sent=False,
            verdict=verdict,
            summary=summary,
            comment=comment,
            error=str(exc),
        )
    posted: list[str] = []
    skipped: str | None = None
    try:
        posted = adapter.publish_inline(state.change, evidence, inline_findings(evidence.findings))
    except InlinePublishError as exc:
        posted = exc.posted
        skipped = redact_text(str(exc))
    except ForgeError as exc:
        skipped = redact_text(str(exc))
    return Published(
        ok=True,
        sent=True,
        verdict=verdict,
        summary=summary,
        comment=comment,
        inline_posted=len(posted),
        inline_skipped=skipped,
    )


def publish_license_stop(
    adapter: ForgeAdapter,
    change: ChangeRequest,
    stopped: Stopped,
    *,
    dry_run: bool = False,
    target_url: str | None = None,
) -> Published:
    """Tell the change that no gate ran, and why: a note and an error status, never a verdict.

    The note takes the place of the last conclusion, which was about another
    commit or another run, so that nothing on the change reads as this run's
    verdict. The status is the platform's error, the one a run that fell over
    gets: a merge rule must not be satisfied by a run that checked nothing,
    and "blocked" would say the gates had found something. A platform refusal
    is returned, as by ``publish_conclusion``.
    """

    summary, comment = stopped.status, stopped.comment
    if dry_run:
        return Published(ok=True, sent=False, summary=summary, comment=comment)
    try:
        adapter.publish_note(change, comment)
        adapter.publish_error(change, summary, target_url)
    except ForgeError as exc:
        return Published(ok=False, sent=False, summary=summary, comment=comment, error=str(exc))
    return Published(ok=True, sent=True, summary=summary, comment=comment)


def resume_interrupted_run(
    *,
    runner: GateRunner,
    run_id: str,
    settings: Settings,
    workspace: Path,
    adapter: ForgeAdapter | None = None,
    dry_run: bool = False,
    license_note: str | None = None,
) -> MergeRequestRun:
    """Drive a run that was mid-gate when its process stopped, then write back.

    The tree is re-materialised when it is gone — a checkout is a cache, and a
    restart on another machine, or after a cleanup, has an empty one. Without
    it the resumed gates would scan whatever happens to be at the deployment
    root and report on the wrong tree, which is the one failure mode a resume
    must not have. ``license_note`` is for the note, as ``publish_conclusion``'s.
    """

    state = runner.status(run_id)
    if state is None:
        raise ValueError(f"Unknown run_id: {run_id}")
    checkout = _ensure_checkout(state, settings, workspace, adapter)
    state = runner.resume_run(run_id, workspace_path=_optional_str(checkout))
    if adapter is None or state.change.number is None:
        # A local run has no platform to tell. Reported as published so the
        # outcome stays the terminal state rather than turning into "failed".
        return MergeRequestRun(state=state, published=Published(ok=True, sent=False))
    state, approvals = apply_platform_approvals(adapter, runner, state)
    return MergeRequestRun(
        state=state,
        published=publish_conclusion(
            adapter, state, dry_run=dry_run, approvals=approvals, license_note=license_note
        ),
        checkout=checkout,
    )


def _ensure_checkout(
    state: RunState,
    settings: Settings,
    workspace: Path,
    adapter: ForgeAdapter | None,
) -> Path | None:
    existing = Path(state.workspace_path) if state.workspace_path else None
    if existing is not None and existing.is_dir():
        return existing
    if adapter is None or state.change.number is None:
        return existing
    return adapter.materialize(state.change, workspace / settings.workspace_dir)


def status_summary(state: RunState) -> str:
    """One line for the commit status. It has to survive being truncated."""

    blocking = len(
        [
            finding
            for report in state.gate_reports.values()
            for finding in report.findings
            if finding.blocking
        ]
    )
    findings = f"{blocking} blocking finding{'' if blocking == 1 else 's'}"
    return f"{outcome_label(state)} · {findings} · run {state.run_id}"


def outcome_label(state: RunState) -> str:
    """The terminal state as a reader is told it: ``approved`` when a person signed.

    The status stays ``mergeable`` -- that is what the merge rule reads -- but
    a status line that said so would read as "every gate passed", which a
    signed run is precisely not.
    """

    decision = state.approval_decision
    if state.status == "mergeable" and decision is not None and decision.approved:
        return "approved"
    return state.status


def _optional_str(path: Path | None) -> str | None:
    return str(path) if path is not None else None


__all__ = [
    "STATUS_VERDICT",
    "MergeRequestRun",
    "MergeRequestSource",
    "Published",
    "apply_platform_approvals",
    "fetch_merge_request",
    "outcome_label",
    "publish_conclusion",
    "publish_license_stop",
    "resume_interrupted_run",
    "run_gates",
    "status_summary",
]
