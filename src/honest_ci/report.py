"""The evidence pack, rendered for a person.

``evidence.json`` is the record; this module is the reading of it. A pack
nobody opens cannot settle the argument it exists for — "why was my change
blocked", or the harder one, "what did you actually verify before letting it
through" — and neither a developer nor an auditor answers that by scrolling
JSON.

Three rules shape what comes out:

* **Not run is not passed.** A gate this deployment never ran, a scanner that
  could not start, a file with no reviewable content: each gets a line under
  "What this run could not verify" rather than being absent from a page that
  otherwise reads like a clean bill of health.
* **Model text lives in its own section**, labelled advisory, and when there is
  none the rendering says *why* — no credential, a failed call, or output that
  ``grounding.py`` refused. "The model said nothing" and "the model was
  discarded" are different facts about the same run.
* **Every block answers four questions**: which rule, which line, why, and what
  to do about it.

Everything here is derived from the pack. No verdict is formed and nothing is
re-judged: this module only reads.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime

from .contracts import (
    GATE_SEQUENCE,
    ApprovalDecision,
    ApprovalReading,
    ApprovalWindow,
    ArtifactManifest,
    BlockBrief,
    ChangedFile,
    ChangeRequest,
    EvidencePack,
    Finding,
    GateConfiguration,
    GateReport,
    LLMCallRecord,
    ModelReviewReport,
    PlatformApproval,
    PlatformReview,
    PolicyDecision,
    RunMetrics,
    Verdict,
)
from .diffs import (
    is_safe_relative_path,
    markdown_code,
    markdown_path,
    markdown_prose,
    moved_files,
)
from .policy import unsatisfied_checks
from .redaction import redact_text
from .state import RunStatus

#: Why a run ended where it did, in one line. Shared with the CLI so a terminal
#: and a handed-over evidence pack cannot describe the same outcome differently.
#: No line names a command: the comment is read on the platform, where a run
#: in CI has left no ledger for a command to act on.
STATUS_HINT: dict[str, str] = {
    "mergeable": "Every gate that ran passed, and every gate the policy requires reached a verdict.",
    "blocked": "A blocking finding stops the merge. Fix it and run again; the new commit gets "
    "a new fingerprint.",
    "awaiting_human": "This change needs a human review: it is beyond what the gates decide "
    "on their own.",
    "failed": "The run itself failed before it reached a verdict.",
    "running": "Still running.",
}

#: The outcome of a run a person approved. Its status is ``mergeable``, and the
#: hint for that one says every gate passed -- which is exactly what an approval
#: does not mean, so a signed run is never described with it.
APPROVED_HINT = (
    "A person approved this change. No gate's verdict changed: why it needed a person is below."
)

#: What ``awaiting_human`` asks for where the platform's own reviews are read as
#: approvals (``ForgeAdapter.fetch_approvals``). Elsewhere the generic hint
#: stands, because an Approve review there would change nothing.
REVIEW_APPROVAL_HINT = (
    "This change needs a human review. To approve it, submit an Approve review on this pull "
    "request and write the reason in the review's text."
)

_VERDICT_TEXT: dict[Verdict, str] = {
    Verdict.PASS: "pass",
    Verdict.BLOCKED: "blocked",
    Verdict.NEEDS_HUMAN: "needs_human",
    Verdict.SKIPPED: "skipped, which is not a pass",
}

_DECIDED_BY_TEXT = {
    "deterministic": "deterministic code",
    "model": "a model's suggestion, not a gate verdict",
    "human": "a person",
}

#: The selection cell for a run of every test. Named because the platform note
#: leaves the default out, and has to recognise it to do so.
_FULL_SUITE = "full suite"

#: Long tables are cut here; the JSON keeps everything. A 300-file rename does
#: not make a more readable report, it makes an unread one.
_MAX_ROWS = 50

#: The one line "What this run could not verify" shows when there is nothing to
#: show.
NO_LIMITS = "Every gate that ran reached a verdict, and every required check tool ran."


def render_evidence(
    evidence: EvidencePack,
    *,
    status: RunStatus | None = None,
    approval: ApprovalDecision | None = None,
    assistant_report: ModelReviewReport | None = None,
) -> str:
    """Render one evidence pack as Markdown.

    ``status``, ``approval`` and ``assistant_report`` belong to the run rather
    than to the pack, so the caller passes whichever it holds; the pack alone
    still renders.
    """

    findings = evidence.findings
    blocking = [finding for finding in findings if finding.blocking]
    other = [finding for finding in findings if not finding.blocking]

    sections: list[tuple[str, list[str]]] = [
        ("The change", _change_section(evidence.change)),
        ("Gate verdicts", _gates_section(evidence.gate_reports)),
    ]
    tests = _tests_section(evidence.gate_reports)
    if tests:
        sections.append(("Regression tests", tests))
    integrity = _integrity_section(evidence.gate_reports)
    if integrity:
        sections.append(("Test integrity", integrity))
    reviews = _platform_reviews_section(evidence.change)
    if reviews:
        sections.append(("Review comments already on the change", reviews))
    sections += [
        (f"Blocking findings ({len(blocking)})", _blocking_section(blocking)),
        (f"Other findings ({len(other)})", _other_findings_section(other)),
        ("What this run could not verify", _limits_section(evidence)),
        (
            "What the model said",
            _model_section(
                evidence.briefs,
                assistant_report,
                evidence.llm_calls,
                evidence.gate_reports,
            ),
        ),
        ("Metrics", _metrics_section(evidence.metrics)),
        ("Gate configuration", _configuration_section(evidence.configuration)),
    ]
    if approval is not None:
        sections.append(("Human decision", _approval_section(approval)))
    artifacts = _artifacts_section(evidence.artifact_manifest)
    if artifacts:
        sections.append(("Artifacts", artifacts))

    lines: list[str] = [
        f"# Evidence pack · {evidence.change.ref}",
        "",
        "> This is `evidence.json` rendered for reading. Where the two disagree, the JSON "
        "and the ledger are the record.",
        "",
        *_overview(evidence, status, approval),
    ]
    for index, (title, body) in enumerate(sections, start=1):
        lines += ["", f"## {index}. {title}", "", *body]
    # One redaction pass at the boundary: everything above is free to quote the
    # change, and nothing below this line leaves with a credential in it.
    return redact_text("\n".join(lines).rstrip() + "\n")


#: A merge request is a conversation, not a report: the note has to be readable
#: in a comment box without scrolling past the part that matters.
_MAX_COMMENT_FINDINGS = 10
#: Every comment this deployment writes starts with one of these. They are what
#: lets a rerun update its own note instead of stacking a new one -- and what
#: keeps ``fetch_reviews`` from reading this product's own output back in as
#: "a human already raised this", which would be a machine agreeing with itself.
MARKER_PREFIX = "<!-- honest-ci:"
VERDICT_MARKER = f"{MARKER_PREFIX}verdict -->"
FINDING_MARKER_PREFIX = f"{MARKER_PREFIX}finding "


def finding_marker(finding_id: str, head_sha: str) -> str:
    """The de-duplication marker for one finding on one commit."""

    return f"{FINDING_MARKER_PREFIX}{finding_id}@{head_sha[:12]} -->"


#: The conclusion comment's second line, when the run read the platform's
#: approvals: the window they are read in (``ApprovalWindow``). It lives in the
#: comment because a run in CI has nowhere else to leave it for the next one.
APPROVAL_WINDOW_PREFIX = f"{MARKER_PREFIX}approvals "
_APPROVAL_WINDOW = re.compile(
    re.escape(APPROVAL_WINDOW_PREFIX)
    + r"fingerprint=(?P<fingerprint>[0-9A-Za-z._-]{1,128}) after=(?P<after>\S+) -->"
)


def approval_window_marker(window: ApprovalWindow) -> str:
    """The hidden line that records ``window`` in the conclusion comment."""

    after = window.after.astimezone(UTC).isoformat() if window.after is not None else "none"
    return f"{APPROVAL_WINDOW_PREFIX}fingerprint={window.fingerprint} after={after} -->"


def read_approval_window(comment: str) -> ApprovalWindow | None:
    """The window a conclusion comment recorded, or ``None`` when it recorded none.

    Read from the second line only, right under the verdict marker, where
    nothing but ``render_forge_comment`` writes: further down the comment
    quotes the change, and a line the change's author wrote must not be able to
    open a window. Anything this cannot read is no window at all, which the
    next run takes as the strict reading -- every review already given
    predates its evidence.
    """

    lines = comment.splitlines()
    if len(lines) < 2 or lines[0] != VERDICT_MARKER:
        return None
    match = _APPROVAL_WINDOW.fullmatch(lines[1])
    if match is None:
        return None
    after = match.group("after")
    if after == "none":
        return ApprovalWindow(fingerprint=match.group("fingerprint"))
    try:
        moment = datetime.fromisoformat(after)
    except ValueError:
        return None
    return ApprovalWindow(fingerprint=match.group("fingerprint"), after=moment)


def inline_findings(findings: Sequence[Finding]) -> list[Finding]:
    """Findings worth a comment on the line itself: blocks, plus every proposal.

    "Proposal" is anything this deployment's deterministic rules did not
    decide, not specifically a model's -- an external verdict is exactly as
    entitled to a line comment and exactly as unentitled to a block, and
    keying on ``== "model"`` would have dropped it out of the merge request
    without dropping it out of the pack, which is the one inconsistency a
    developer would read as the tool hiding something.
    """

    return [
        finding
        for finding in findings
        if finding.file
        and finding.line_start
        # A line in the base -- a removed test, a rewritten expected value --
        # is not a line in the head, and an inline comment is anchored to the
        # head side. Posted there it would sit on whatever line now has that
        # number. The comment's test-integrity
        # section and the evidence pack still carry it.
        and finding.side == "head"
        and (finding.blocking or finding.decided_by != "deterministic")
    ]


#: What an integrity rule's action does, as a reader of the change is told.
_INTEGRITY_ACTION_TEXT = {
    "block": "blocks the merge",
    "needs_human": "needs a human",
    "record": "recorded only",
}


def integrity_findings(reports: Sequence[GateReport]) -> list[tuple[Finding, str]]:
    """Every finding of a test-integrity rule, with that rule's action.

    Read from the gate report that raised the finding, which lists the rules
    it evaluated and what each one's hits do: that is the record, so a pack
    rendered again later names the same actions it was judged by.
    """

    found: list[tuple[Finding, str]] = []
    for report in reports:
        actions = _integrity_actions(report.verifier_report)
        found += [
            (finding, actions[finding.rule_id])
            for finding in report.findings
            if finding.rule_id in actions
        ]
    return found


def _integrity_actions(verifier_report: Mapping[str, object]) -> dict[str, str]:
    return {
        str(rule["id"]): str(rule["action"])
        for rule in _integrity_rules(verifier_report)
    }


def _integrity_rules(verifier_report: Mapping[str, object]) -> list[Mapping[str, object]]:
    rules = verifier_report.get("rules_evaluated")
    if not isinstance(rules, list):
        return []
    return [
        rule
        for rule in rules
        if isinstance(rule, Mapping)
        and rule.get("kind") == "integrity"
        and isinstance(rule.get("id"), str)
        and isinstance(rule.get("action"), str)
    ]


def render_forge_comment(
    evidence: EvidencePack,
    *,
    status: RunStatus,
    approvals: ApprovalReading | None = None,
    evidence_at: str | None = None,
    license_note: str | None = None,
) -> str:
    """Render the note a forge adapter posts back onto the change.

    Shorter than the evidence pack and ordered differently: a developer reading
    a merge request wants the terminal state, then what blocked them, then what
    nobody verified. Everything here is derived from the pack — the note never
    softens a verdict, and it never claims a gate ran that did not.

    ``approvals`` is passed when the run read the platform's approvals. The note
    then says how to approve and what did not count, or who approved; and it
    records the window the next run reads them in.

    ``evidence_at`` says, in Markdown, where the full evidence pack can be
    read. Left out, the note names ``honest-ci evidence``, which is right only
    where the ledger outlives the run -- a deployment's own machine, not a CI
    job that is thrown away with its ledger.

    ``license_note`` is what the license check says about the run, such as how
    many days its trial has left. It comes last, under the line, apart from
    everything the pack says: it is about the run's license, not the change.
    """

    blocking = [finding for finding in evidence.findings if finding.blocking]
    # Everything this deployment's deterministic rules did not decide: the
    # reading ``inline_findings`` takes, for the same reason. Keyed on
    # ``== "model"``, an external verdict would sink into the tally below, and
    # on a platform that posts no line comments the merge request would name
    # it nowhere. ``not blocking`` keeps the heading's "not blocking" true by
    # construction, and keeps a finding from being subtracted twice below.
    proposed = [
        finding
        for finding in evidence.findings
        if not finding.blocking and finding.decided_by != "deterministic"
    ]
    # ``record`` is the action that asks for the evidence pack and nothing
    # else, so its hits stay in the tally below.
    integrity = [
        (finding, action)
        for finding, action in integrity_findings(evidence.gate_reports)
        if action != "record"
    ]
    shown = len([finding for finding, _ in integrity if not finding.blocking])
    other = len(evidence.findings) - len(blocking) - len(proposed) - shown
    approved = status == "mergeable" and approvals is not None and approvals.approval is not None
    window = approvals.window if approvals is not None else None
    lines = [
        VERDICT_MARKER,
        *([approval_window_marker(window)] if window is not None else []),
        f"## honest-ci · {'approved' if approved else status}",
        "",
        f"**{_comment_hint(status, approvals, approved=approved)}**",
        "",
        *markdown_table(
            ("", ""),
            [
                ("Change", f"`{evidence.change.ref}`"),
                ("Commit / run", f"`{evidence.change.head_sha[:12]}` / `{evidence.run_id}`"),
                ("Worst gate verdict", _VERDICT_TEXT[evidence.verdict]),
                (
                    "Rule / policy versions",
                    f"rules {evidence.rule_version} · policy {evidence.policy_version}"
                    f"{_configuration_suffix(evidence.configuration)}",
                ),
                ("Evidence fingerprint", f"`{evidence.fingerprint[:16]}…`"),
                *_comment_tests_row(evidence.gate_reports),
                *_comment_reviews_row(evidence.change),
            ],
        ),
        "",
        "### Gates",
        "",
        *_gates_section(evidence.gate_reports),
        "",
        f"### Blocking findings ({len(blocking)})",
        "",
        *_comment_findings(blocking),
    ]
    if integrity:
        # Spelled out, detail and all, where every other finding gets a line:
        # "a test was deleted" is only something a person can act on once it
        # says which test, and a needs_human hit is why the change is waiting.
        # A blocking hit is listed above as well; this is where it says what
        # happened to the tests.
        lines += [
            "",
            f"### Test integrity ({len(integrity)})",
            "",
            *_integrity_items(integrity, _MAX_COMMENT_FINDINGS, "the evidence pack"),
        ]
    if proposed:
        # Its own heading rather than a line in the "other findings" tally: a
        # finding nobody here decided -- a judgment rule's doubt, a failed
        # generated test, another source's verdict -- is the reason this change
        # is waiting on a person, and burying it in the "Also N non-blocking
        # findings" line would hide the thing to act on. Named for what the
        # reader does with it rather than for one source, since it holds several.
        lines += [
            "",
            f"### For a person to judge ({len(proposed)}) · not blocking",
            "",
            *_comment_findings(proposed),
        ]
    if other:
        lines += [
            "",
            f"Also {_counted(other, 'non-blocking finding')}: recorded in the evidence pack "
            "and the metrics, and not blocking the merge.",
        ]
    lines += [
        "",
        "### What this run could not verify",
        "",
        *_limits_section(evidence),
    ]
    if approvals is not None:
        how = _comment_approval(evidence, approvals, approved=approved)
        lines += ["", "### Approval", "", *how]
    lines += [
        "",
        "---",
        "",
        "Verdicts are decided by deterministic code. A model may raise doubts and explain, "
        "but never changes a verdict. "
        f"Full evidence pack: {evidence_at or f'`honest-ci evidence {evidence.run_id}`'}.",
    ]
    if license_note is not None:
        lines += ["", license_note]
    # One redaction pass at the boundary, same as the evidence rendering: tool
    # output quoted above may carry a token, and this text leaves the process.
    return redact_text("\n".join(lines).rstrip() + "\n")


def _comment_hint(status: RunStatus, approvals: ApprovalReading | None, *, approved: bool) -> str:
    if approved:
        return APPROVED_HINT
    if status == "awaiting_human" and approvals is not None:
        return REVIEW_APPROVAL_HINT
    return STATUS_HINT.get(status, "")


#: Enough of a reviewer's reason to recognise it; the evidence pack has it whole.
_MAX_COMMENT_REASON = 300
#: Approvals listed as not counting, before the rest are only counted.
_MAX_COMMENT_APPROVALS = 10


def _comment_approval(
    evidence: EvidencePack, reading: ApprovalReading, *, approved: bool
) -> list[str]:
    """How this change is approved on the platform, and what became of the approvals so far."""

    commit = f"`{evidence.change.head_sha[:12]}`"
    review = reading.approval
    if approved and review is not None:
        return [
            f"Approved by {_reviewer(review)} on commit {commit}: {_reason_excerpt(review.reason)}",
            "",
            f"The approval is bound to evidence fingerprint `{evidence.fingerprint[:16]}…`. "
            "A new commit, or a change to the rules or the policy, needs a new approval.",
        ]
    lines = [
        "An Approve review counts when the reviewer can push to this repository, it is given "
        f"on commit {commit} after this conclusion was published, and its text gives the "
        "reason. A new commit, or a change to the rules or the policy, needs a new approval. "
        "A bot's approval never counts.",
    ]
    if reading.error is not None:
        lines += [
            "",
            "⚠ No approval was applied: the reviews on this pull request could not be read "
            f"or applied ({markdown_code(reading.error)}).",
        ]
    if reading.uncounted:
        lines += ["", "Approvals that do not count:", ""]
        lines += [
            f"- {_reviewer(item.review)}: {item.why}."
            for item in reading.uncounted[:_MAX_COMMENT_APPROVALS]
        ]
        if len(reading.uncounted) > _MAX_COMMENT_APPROVALS:
            lines.append(f"- (and {len(reading.uncounted) - _MAX_COMMENT_APPROVALS} more)")
    if reading.objections:
        lines += [
            "",
            "Requests for changes that stand. No approval counts until each one is withdrawn "
            "with a new review from the same person, or dismissed:",
            "",
        ]
        lines += [
            f"- {_reviewer(review)}" for review in reading.objections[:_MAX_COMMENT_APPROVALS]
        ]
        if len(reading.objections) > _MAX_COMMENT_APPROVALS:
            lines.append(f"- (and {len(reading.objections) - _MAX_COMMENT_APPROVALS} more)")
    return lines


#: A review's address as the platform gave it, before it goes into a Markdown
#: link: nothing that could end the link, start another construct, or read as
#: a ``key=value`` the final redaction pass would take the closing parenthesis with.
_REVIEW_URL = re.compile(r"https?://[A-Za-z0-9._~:/?#%-]+")


def _reviewer(review: PlatformApproval) -> str:
    """Who gave a review, quoted so that it cannot mention anyone, with a link to it."""

    who = markdown_code(review.reviewer or "(deleted account)")
    if review.url and _REVIEW_URL.fullmatch(review.url):
        return f"{who} ([review]({review.url}))"
    return who


def _reason_excerpt(reason: str) -> str:
    """Someone else's reason, quoted so that it renders as their words and nothing more."""

    text = " ".join(reason.split())
    if len(text) > _MAX_COMMENT_REASON:
        text = text[:_MAX_COMMENT_REASON] + "…"
    return markdown_code(text)


def _comment_findings(findings: Sequence[Finding]) -> list[str]:
    if not findings:
        return ["None."]
    lines: list[str] = []
    for finding in findings[:_MAX_COMMENT_FINDINGS]:
        rule = f"[{finding.rule_id}] " if finding.rule_id else ""
        location = _location(finding)
        confidence = (
            f" · confidence {finding.confidence:.2f}" if finding.decided_by == "model" else ""
        )
        lines.append(
            f"- **{rule}{finding.title}** · {finding.severity}"
            + (f" · {location}" if location else "")
            + confidence
        )
        if finding.suggested_fix:
            lines.append(f"  - Suggested fix: {_cell(finding.suggested_fix)}")
    if len(findings) > _MAX_COMMENT_FINDINGS:
        lines.append(
            f"- (only the first {_MAX_COMMENT_FINDINGS} of {len(findings)} are listed; "
            "the evidence pack has the rest)"
        )
    return lines


def _integrity_items(
    found: Sequence[tuple[Finding, str]], limit: int, rest: str
) -> list[str]:
    """Each integrity hit with what it found, as one list item.

    The detail is the whole point here: it is the line that names the removed
    test or the rewritten expected value, quoted as code where the change
    author wrote it.
    """

    lines: list[str] = []
    for finding, action in found[:limit]:
        rule = f"[{finding.rule_id}] " if finding.rule_id else ""
        location = _location(finding)
        lines.append(
            f"- **{rule}{finding.title}** · "
            + (_INTEGRITY_ACTION_TEXT.get(action) or markdown_code(action))
            + (f" · {location}" if location else "")
        )
        lines += [f"  - {row.strip()}" for row in finding.detail.splitlines() if row.strip()]
        if finding.suggested_fix:
            lines.append(f"  - Suggested fix: {_cell(finding.suggested_fix)}")
        if finding.false_positive:
            lines.append("  - Marked a false positive by a person; the verdict stands as it was.")
    if len(found) > limit:
        lines.append(f"- (only the first {limit} of {len(found)} are listed; {rest} has the rest)")
    return lines


# ── sections ─────────────────────────────────────────────────────────────────


def _overview(
    evidence: EvidencePack, status: RunStatus | None, approval: ApprovalDecision | None
) -> list[str]:
    rows = [("Run", f"`{evidence.run_id}`")]
    if status == "mergeable" and approval is not None and approval.approved:
        # Mergeable because a person signed, which the hint for mergeable --
        # every gate passed -- would misstate.
        rows.append(("Outcome", f"**approved** · {APPROVED_HINT}"))
    elif status is not None:
        rows.append(("Outcome", f"**{status}** · {STATUS_HINT.get(status, '')}"))
    rows += [
        ("Worst gate verdict", _VERDICT_TEXT[evidence.verdict]),
        ("Evidence fingerprint", f"`{evidence.fingerprint}`"),
        (
            "Policy / rule versions",
            f"{evidence.policy_version} / {evidence.rule_version}"
            f"{_configuration_suffix(evidence.configuration)}",
        ),
        ("Isolation", evidence.sandbox_isolation),
        ("Generated at", evidence.created_at.isoformat(timespec="seconds")),
    ]
    return markdown_table(("", ""), rows)


def _change_section(change: ChangeRequest) -> list[str]:
    ratio = (
        f"{change.ai_generated_ratio:.0%}"
        if change.ai_generated_ratio is not None
        # Not measured is not zero: a report that reads 0% would claim
        # something nobody checked.
        else "not measured (which is not 0%)"
    )
    # A patch that came in through ``--diff`` carries no git file list, so the
    # diff itself is the only inventory there is. Reading 0 files off an empty
    # list would understate a change the gates really did judge.
    from_diff = [] if change.files_changed else change.touched_paths
    # The count the size limits were held to, which takes a rename as two
    # names; the table shows it as one row, so the line says why they differ.
    file_count = len(change.touched_paths)
    renamed = any(item.previous_path for item in change.files_changed)
    # The line count the limits were held to leaves out a file moved without
    # edits, which the table still shows as its lines removed and added.
    moved = bool(moved_files(change.diff))
    lines = [
        # The one line here the change's author wrote freely; the rest are
        # names the platform constrains, or ours.
        f"- Title: {markdown_code(change.title) if change.title else '(none)'}",
        f"- Source: {change.forge} · {change.repo}"
        + (f" #{change.number}" if change.number is not None else ""),
        f"- Commit: `{change.head_sha[:12]}`"
        + (f" ← {change.source_branch}" if change.source_branch else "")
        + (f" → {change.target_branch}" if change.target_branch else ""),
        f"- Author: {change.author or '(unknown)'}",
        f"- Size: {_counted(file_count, 'file')}"
        + (" (a rename counts under both its names)" if renamed else "")
        + f" / {_counted(change.changed_lines, 'line')}"
        + (" (a file moved without edits counts no lines)" if moved else ""),
        f"- AI-generated share: {ratio}",
    ]
    if change.files_changed:
        rows = [
            (
                (
                    f"{markdown_path(item.previous_path)} → {markdown_path(item.path)}"
                    if item.previous_path
                    else markdown_path(item.path)
                ),
                item.change_type,
                f"+{item.added_lines} / -{item.removed_lines}",
            )
            for item in change.files_changed[:_MAX_ROWS]
        ]
        lines += ["", *markdown_table(("File", "Type", "Lines"), rows)]
        lines += _truncation_note(len(change.files_changed), "files")
    elif from_diff:
        lines += [
            "",
            "The file list is read off the diff itself: this change is a patch, which "
            "carries no git file list.",
            "",
            *[f"- {markdown_path(path)}" for path in from_diff[:_MAX_ROWS]],
        ]
        lines += _truncation_note(len(from_diff), "files")
    return lines


def _gates_section(reports: Sequence[GateReport]) -> list[str]:
    by_gate = {report.gate: report for report in reports}
    rows: list[tuple[str, ...]] = []
    for name in GATE_SEQUENCE:
        report = by_gate.get(name)
        if report is None:
            # Not in the pack at all: an upstream gate ended the run before this
            # one was asked. "Never asked" and "passed" must not look alike.
            rows.append(
                (
                    name,
                    "not run",
                    "An earlier gate reached a verdict first, so the run never got this far.",
                )
            )
            continue
        rows.append((name, _VERDICT_TEXT[report.verdict], report.summary or "—"))
    return markdown_table(("Gate", "Verdict", "Summary"), rows)


#: Failed tests listed in the evidence pack; the suite's own output has the rest.
_MAX_LISTED_FAILURES = 20

#: Files named in one limits line. The point is that the reader knows such
#: files exist; the whole list is in ``evidence.json``.
_MAX_LISTED_UNANALYSED = 10


def _tests_section(reports: Sequence[GateReport]) -> list[str]:
    """The suite the regression gate ran, or why it has nothing to show.

    Empty when no gate carried test results, so a pack from a deployment
    without the regression gate does not grow a section that says nothing.
    """

    lines: list[str] = []
    for report in reports:
        results = report.test_results
        if results is None:
            continue
        if not results.available:
            first = _first_nonblank(results.failures, results.stderr)
            lines += [
                # The first thing the run printed, which the change can decide.
                "The test suite did not start: "
                + ("(no output)" if first is None else markdown_code(first)),
                "",
                f"Command: `{results.command or 'pytest (not executed)'}`",
            ]
            continue
        threshold = report.metrics.get("coverage_threshold", 0.0)
        changed_bar = report.metrics.get("changed_line_coverage_threshold", 0.0)
        rows: list[tuple[str, ...]] = [
            ("Command", f"`{results.command or '—'}`"),
            (
                "Results",
                f"{results.passed} passed · {results.failed} failed"
                f" · {_counted(results.errors, 'error')} · {results.skipped} skipped",
            ),
            ("Exit code", "—" if results.exit_code is None else str(results.exit_code)),
            ("Line coverage", _coverage_cell(report.metrics.get("coverage"), threshold)),
        ]
        scope = _selection_cell(report.verifier_report)
        if scope is not None:
            rows.insert(1, ("Test selection", scope))
        if changed_bar > 0 or "changed_lines_total" in report.metrics:
            rows.append(
                ("Changed-line coverage", _changed_coverage_cell(report.metrics, changed_bar))
            )
        impact = _impact_cell(report.verifier_report)
        if impact is not None:
            rows.append(("Impact analysis", impact))
        generated = _generated_cell(report.verifier_report)
        if generated is not None:
            rows.append(("Generated regression tests", generated))
        rows.append(("Duration", _optional(results.duration_s, lambda v: f"{v:.1f}s")))
        lines += markdown_table(("", ""), rows)
        lines += _uncovered_lines(report)
        if results.failures:
            lines += ["", "Failing tests:"]
            # A test's id and its assertion message are the change's text, and
            # one backtick in either would close a one-backtick span.
            lines += [f"- {markdown_code(line)}" for line in results.failures[:_MAX_LISTED_FAILURES]]
            if len(results.failures) > _MAX_LISTED_FAILURES:
                lines.append(
                    f"- (only the first {_MAX_LISTED_FAILURES} of {len(results.failures)} "
                    "are listed; `evidence.json` has the full output)"
                )
    return _trimmed(lines)


def _coverage_cell(ratio: float | None, threshold: float) -> str:
    """The coverage figure next to the bar it was judged against.

    A bar of 0 is "not measured, by configuration"; a bar with no figure is
    "not measured, and the gate could not check it". The pack must not let
    those two read alike.
    """

    if threshold <= 0:
        return "not measured (the threshold is 0)"
    if ratio is None:
        return f"not measured (threshold {threshold:.0%}, so it could not be checked)"
    return f"{ratio:.0%} (threshold {threshold:.0%})"


def _selection_cell(verifier_report: Mapping[str, object]) -> str | None:
    """Which tests ran: the whole suite, or the ones the change reaches.

    ``None`` for a pack that carries no selection at all, so a report produced
    before test selection existed does not grow a row claiming it ran
    everything on evidence nobody recorded.
    """

    selection = verifier_report.get("test_selection")
    if not isinstance(selection, dict):
        return None
    requested = selection.get("requested")
    if selection.get("mode") == "impacted" and isinstance(requested, list):
        return (
            f"only the {_counted(len(requested), 'impacted test')}, "
            "weaker evidence than the full suite"
        )
    fallback = selection.get("fallback")
    if fallback:
        return f"{_FULL_SUITE} (only the impacted tests were asked for, but {fallback})"
    return _FULL_SUITE


def _changed_coverage_cell(metrics: Mapping[str, float], bar: float) -> str:
    """The bar on this change's own lines, never confused with the repository's.

    A repository at 90% can take an untested function without moving that
    number, which is the whole reason this row exists next to the other one.
    """

    against = f"threshold {bar:.0%}" if bar > 0 else "no threshold, recorded but not judged"
    total = metrics.get("changed_lines_total")
    if total is None:
        return f"not measured ({against})"
    if total <= 0:
        return f"this change adds no executable lines ({against})"
    covered = int(metrics.get("changed_lines_covered", 0.0))
    ratio = metrics.get("changed_line_coverage", 0.0)
    return f"{ratio:.0%} · {covered}/{int(total)} lines ({against})"


def _impact_cell(verifier_report: Mapping[str, object]) -> str | None:
    impact = verifier_report.get("impact")
    if not isinstance(impact, dict):
        return None
    if not impact.get("analysed"):
        language = impact.get("language")
        toolchain = f"the {language} toolchain" if language else "this toolchain"
        return f"{toolchain} has no impact analysis yet"
    symbols = impact.get("changed_symbols") or []
    tests = impact.get("impacted_tests") or []
    return (
        f"{_counted(len(symbols), 'changed symbol')}"
        f" · {_counted(len(tests), 'possibly impacted test')}"
    )


def _generated_cell(verifier_report: Mapping[str, object]) -> str | None:
    """What the model-written regression cases did, or why there were none.

    Always a row once the deployment has the feature, including "none": a
    reader who sees no row at all would be entitled to think the change was
    covered by generated cases that simply passed.
    """

    generated = verifier_report.get("generated_tests")
    if not isinstance(generated, dict):
        return None
    reason = generated.get("not_generated_reason")
    if not generated.get("ran"):
        return f"none generated ({reason or 'no reason recorded'})"
    passed = int(generated.get("passed") or 0)
    failed = int(generated.get("failed") or 0)
    tail = (
        " (the model's judgment: not blocking, only sent to a person)"
        if failed
        else " (not counted in the results or the coverage above)"
    )
    return f"{passed} passed · {failed} failed{tail}"


def _uncovered_lines(report: GateReport) -> list[str]:
    """The changed lines no test executed, by file and line number.

    The count comes from the metrics rather than from this list: the list is
    capped so the pack stays readable, and a reader must not mistake the cap
    for the number that decided the verdict.
    """

    coverage = report.verifier_report.get("coverage")
    if not isinstance(coverage, dict):
        return []
    changed = coverage.get("changed_lines")
    if not isinstance(changed, dict):
        return []
    listed = [str(item) for item in changed.get("uncovered") or []]
    if not listed:
        return []
    total = int(report.metrics.get("changed_lines_total", 0.0))
    covered = int(report.metrics.get("changed_lines_covered", 0.0))
    lines = ["", "Changed lines no test executed:"] + [f"- {markdown_path(item)}" for item in listed]
    if total - covered > len(listed):
        lines.append(
            f"- (only the first {len(listed)} of {total - covered} are listed; "
            "`evidence.json` has them all)"
        )
    return lines


def _comment_tests_row(reports: Sequence[GateReport]) -> list[tuple[str, ...]]:
    for report in reports:
        results = report.test_results
        if results is None:
            continue
        if not results.available:
            return [("Regression tests", "did not run, which is not a pass")]
        threshold = report.metrics.get("coverage_threshold", 0.0)
        changed_bar = report.metrics.get("changed_line_coverage_threshold", 0.0)
        cell = (
            f"{results.passed} passed · {results.failed} failed"
            f" · {_counted(results.errors, 'error')}"
            f" · coverage {_coverage_cell(report.metrics.get('coverage'), threshold)}"
        )
        if changed_bar > 0:
            # Only when it decided something: the platform note is one row, and
            # a number nobody judged does not earn space in it.
            cell += f" · changed lines {_changed_coverage_cell(report.metrics, changed_bar)}"
        scope = _selection_cell(report.verifier_report)
        if scope is not None and scope != _FULL_SUITE:
            cell += f" · selection: {scope}"
        generated = report.verifier_report.get("generated_tests")
        if isinstance(generated, dict) and generated.get("failed"):
            cell += (
                f" · {_counted(int(generated['failed']), 'generated test')} failed"
                " (not blocking, for a person to judge)"
            )
        return [("Regression tests", cell)]
    return []


#: Review comments quoted in the pack. The merge request itself has them all,
#: and a pack that reprints a 40-comment argument is one nobody finishes.
_MAX_LISTED_REVIEWS = 20


def _integrity_section(reports: Sequence[GateReport]) -> list[str]:
    """Which integrity checks ran, what their hits do, and every hit, detail and all.

    There whenever an integrity rule was evaluated, hits or none: a pack that
    named the checks only when they fired could not tell "ran and found
    nothing" from "never configured". The details are here and not only in the
    findings table because that table has no room for them, and without them
    a hit does not say which test went away.
    """

    rules = [rule for report in reports for rule in _integrity_rules(report.verifier_report)]
    if not rules:
        return []
    found = integrity_findings(reports)
    counts = Counter(finding.rule_id for finding, _ in found)
    rows = [
        (
            markdown_code(str(rule["id"])),
            markdown_code(str(rule.get("check"))),
            _INTEGRITY_ACTION_TEXT.get(str(rule["action"])) or markdown_code(str(rule["action"])),
            str(counts[str(rule["id"])]),
        )
        for rule in rules
    ]
    lines = [
        "Read off the diff alone: what this change did to the tests that judge it. "
        "A hit is a fact about the diff, not a conclusion about why it was made.",
        "",
        *markdown_table(("Rule", "Check", "Action", "Findings"), rows),
    ]
    if found:
        lines += ["", *_integrity_items(found, _MAX_ROWS, "`evidence.json`")]
    return lines


def _platform_reviews_section(change: ChangeRequest) -> list[str]:
    """What people had already said on this change before the gates ran.

    Empty for a local run and for a platform that hands over nothing, so a pack
    does not grow a section that says nothing. When the comments could not be
    read at all, this says so here as well as under "What this run could not
    verify" — the reader of this section is precisely the one who must not
    mistake silence for agreement.
    """

    if change.platform_reviews_error is not None:
        return [
            "The review comments already on the change **could not be read**: "
            + _cell(change.platform_reviews_error),
            "",
            "This verdict does not take them into account. Not having read them is not the "
            "same as nobody having raised a problem.",
        ]
    reviews = change.platform_reviews
    if not reviews:
        return []
    unresolved = change.unresolved_reviews
    rows = [
        (
            (review.author or "(unknown)") + (" (bot)" if review.is_bot else ""),
            _review_location(review),
            _RESOLVED_TEXT[review.resolved],
            _cell(_review_excerpt(review.body)),
        )
        for review in reviews[:_MAX_LISTED_REVIEWS]
    ]
    return [
        f"This change already has {_counted(len(reviews), 'review comment')}, "
        f"{len(unresolved)} of them unresolved. They are evidence that people (or the "
        "platform's own AI) have already read this change: they go into the evidence pack "
        "and into the model's context. A model's finding can still only cite a rule id and "
        "land on an added line; it cannot cite a review comment.",
        "",
        *markdown_table(("Author", "Location", "Status", "Excerpt"), rows),
        *_truncation_note_at(len(reviews), _MAX_LISTED_REVIEWS, "review comments"),
    ]


_RESOLVED_TEXT: dict[bool | None, str] = {
    True: "resolved",
    False: "**unresolved**",
    None: "plain comment (the platform does not track resolution)",
}


def _review_location(review: PlatformReview) -> str:
    if not review.path:
        return "the whole change"
    # The platform hands the path over as the commenter's client sent it.
    location = markdown_path(review.path)
    return f"{location}:{review.line}" if review.line else location


def _review_excerpt(body: str) -> str:
    text = " ".join(body.split())
    # Anyone who can comment wrote it, and cut short it may not even be the
    # Markdown they wrote -- only Markdown that renders, here under our name.
    return markdown_code(text[:160] + "…" if len(text) > 160 else text)


def _comment_reviews_row(change: ChangeRequest) -> list[tuple[str, ...]]:
    """One row about the comments already on this merge request, if there are any.

    Written back onto the very conversation it counts, which is the point: the
    reader can see that the run read their thread, and that an unanswered one
    is why this change is waiting on a person.
    """

    if change.platform_reviews_error is not None:
        return [
            ("Review comments", "could not be read, which does not mean nobody raised a problem")
        ]
    if not change.platform_reviews:
        return []
    unresolved = len(change.unresolved_reviews)
    return [
        (
            "Review comments",
            f"{len(change.platform_reviews)} on the change"
            + (f" · **{unresolved} unresolved**" if unresolved else " · all resolved or plain comments"),
        )
    ]


def _first_nonblank(candidates: Sequence[str], fallback: str) -> str | None:
    for text in (*candidates, fallback):
        stripped = text.strip()
        if stripped:
            return stripped.splitlines()[0]
    return None


def _blocking_section(blocking: Sequence[Finding]) -> list[str]:
    if not blocking:
        return ["None."]
    lines: list[str] = []
    for finding in blocking:
        lines += _finding_detail(finding)
    return _trimmed(lines)


def _finding_detail(finding: Finding) -> list[str]:
    rule = f"[{finding.rule_id}] " if finding.rule_id else ""
    lines = [
        f"### {rule}{finding.title}",
        "",
        f"- Gate / severity: {finding.gate} · **{finding.severity}**",
        f"- Location: {_location(finding) or 'the change as a whole (no specific line)'}",
        f"- Decided by: {_DECIDED_BY_TEXT.get(finding.decided_by, finding.decided_by)}"
        + (f" · confidence {finding.confidence:.2f}" if finding.decided_by == "model" else ""),
    ]
    if finding.detail:
        lines.append("- Why:")
        # A blank row stays in the quote, marked. Dropped, a code block in the
        # detail (``markdown_code_block``) would lose the blank lines of the
        # output it holds; left bare, it would end the quote and the block
        # with it, and the rest of that output would read as Markdown.
        rows = finding.detail.strip().splitlines()
        lines += [f"  > {row}" if row.strip() else "  >" for row in rows]
    if finding.suggested_fix:
        lines.append(f"- Suggested fix: {finding.suggested_fix}")
    if finding.cited_evidence_ids:
        lines.append("- Cited evidence: " + ", ".join(f"`{item}`" for item in finding.cited_evidence_ids))
    if finding.false_positive:
        lines.append(
            "- Marked a false positive by a person; the verdict stands as it was."
        )
    lines.append("")
    return lines


def _other_findings_section(other: Sequence[Finding]) -> list[str]:
    if not other:
        return ["None."]
    rows = [
        (
            finding.gate,
            finding.rule_id or "—",
            finding.severity,
            _location(finding) or "—",
            finding.title,
        )
        for finding in other[:_MAX_ROWS]
    ]
    return [
        "Not blocking, but recorded in the evidence pack and the metrics.",
        "",
        *markdown_table(("Gate", "Rule", "Severity", "Location", "Title"), rows),
        *_truncation_note(len(other), "findings"),
    ]


def _limits_section(evidence: EvidencePack) -> list[str]:
    """What this run did *not* verify. The section that keeps the rest honest."""

    limits = evidence_limits(evidence) or [NO_LIMITS]
    return [f"- {item}" for item in limits]


def evidence_limits(evidence: EvidencePack) -> list[str]:
    """One line per thing this run could not verify, in reading order."""

    limits: list[str] = []
    skipped = [
        report.gate for report in evidence.gate_reports if report.verdict is Verdict.SKIPPED
    ]
    if skipped:
        limits.append(
            "Gates recorded as skipped: "
            + ", ".join(skipped)
            + ". Not deployed or switched off, which is not a pass."
        )
    missing = unsatisfied_checks(
        evidence.policy_decision,
        {report.gate: report.verdict for report in evidence.gate_reports},
    )
    if missing:
        limits.append(
            "Gates the policy requires that reached no verdict: "
            + ", ".join(missing)
            + ". The outcome is held at needs_human for that reason."
        )
    limits += _tool_limits(evidence.gate_reports)
    limits += _regression_limits(evidence.gate_reports)
    limits += _judgment_limits(evidence.gate_reports)
    limits += _review_limits(evidence.change)
    limits += _policy_limits(evidence.policy_decision)
    limits += _unreviewable_files(evidence.change.files_changed)
    return limits


def _review_limits(change: ChangeRequest) -> list[str]:
    """The platform's own review comments, when nobody could read them.

    Reading them is fail-soft: the gates ran regardless. What must not happen
    is that the pack then reads like a change nobody had commented on, so the
    failure is named here rather than left as an empty list.
    """

    if change.platform_reviews_error is None:
        return []
    return [
        "The review comments already on the change were not read (the platform does not "
        "support it, or the request failed): "
        + change.platform_reviews_error
        + "; this verdict does not take them into account, and nothing checked for "
        "unresolved threads."
    ]


def _tool_limits(reports: Sequence[GateReport]) -> list[str]:
    limits: list[str] = []
    for report in reports:
        unavailable = [tool.name for tool in report.tool_reports if not tool.available]
        if unavailable:
            limits.append(
                f"{report.gate}: check tools that did not start: "
                + ", ".join(unavailable)
                + "; they gave no evidence about this change."
            )
        opaque = [
            tool.name
            for tool in report.tool_reports
            if tool.available and not tool.passed and not tool.diagnostics
        ]
        if opaque:
            limits.append(
                f"{report.gate}: check tools that failed without output that could be relied on: "
                + ", ".join(opaque)
                + "; whether the change meets the related rules cannot be told."
            )
    return limits


def _regression_limits(reports: Sequence[GateReport]) -> list[str]:
    """What a narrowed suite, an unreadable file and a missing report did not cover.

    Each line is about evidence that was *not* produced, which is exactly what
    a passing regression section cannot show on its own: the suite it ran says
    what passed, never what it never ran.
    """

    limits: list[str] = []
    for report in reports:
        selection = report.verifier_report.get("test_selection")
        if isinstance(selection, dict) and selection.get("mode") == "impacted":
            requested = selection.get("requested")
            count = len(requested) if isinstance(requested, list) else 0
            limits.append(
                f"{report.gate}: only the {_counted(count, 'impacted test')} ran "
                "(HONEST_CI_TEST_SELECTION=impacted); the rest of the suite did not run "
                "this time, which is weaker evidence than the full suite."
            )
        impact = report.verifier_report.get("impact")
        if isinstance(impact, dict):
            unanalysed = [str(item) for item in impact.get("unanalysed_files") or []]
            if unanalysed:
                limits.append(
                    f"{report.gate}: which symbols changed could not be read from these files, "
                    "so impact analysis is empty for them: "
                    + ", ".join(markdown_path(path) for path in unanalysed[:_MAX_LISTED_UNANALYSED])
                    + "."
                )
        generated = report.verifier_report.get("generated_tests")
        if isinstance(generated, dict) and generated.get("not_generated_reason"):
            limits.append(
                f"{report.gate}: no regression tests were generated: "
                f"{generated['not_generated_reason']}; no failing generated test does not "
                "mean this change was tested."
            )
        coverage = report.verifier_report.get("coverage")
        changed = coverage.get("changed_lines") if isinstance(coverage, dict) else None
        if isinstance(changed, dict):
            not_measured = [str(item) for item in changed.get("not_measured") or []]
            if not_measured:
                # Not a gap in the evidence -- the configuration says these are
                # not measured -- but still a part of the change the bar says
                # nothing about, which a reader of the bar needs to know.
                limits.append(
                    f"{report.gate}: these changed files are outside what the coverage "
                    "configuration measures, so changed-line coverage neither counts nor "
                    "checks them: "
                    + ", ".join(markdown_path(path) for path in not_measured[:_MAX_LISTED_UNANALYSED])
                    + "."
                )
            missing = [str(item) for item in changed.get("no_coverage_data") or []]
            if missing:
                limits.append(
                    f"{report.gate}: the coverage report has no data for these changed source "
                    "files, so changed-line coverage did not check them: "
                    + ", ".join(markdown_path(path) for path in missing[:_MAX_LISTED_UNANALYSED])
                    + "."
                )
    return limits


def _judgment_limits(reports: Sequence[GateReport]) -> list[str]:
    """Judgment-class rules that went unread, and doubts that stayed below the bar.

    Read out of ``verifier_report`` rather than out of the findings, because the
    whole point is what is *absent* from the findings: a rule nobody put to the
    model leaves no trace among them, and that silence is exactly what this
    section exists to break.
    """

    limits: list[str] = []
    for report in reports:
        judgment = report.verifier_report.get("judgment")
        if not isinstance(judgment, dict):
            continue
        rules = judgment.get("rules") or []
        names = ", ".join(str(rule.get("id")) for rule in rules if isinstance(rule, dict))
        if not judgment.get("judged"):
            limits.append(
                f"{report.gate}: judgment rules nobody judged: {names}; "
                f"reason: {judgment.get('not_judged_reason') or 'not recorded'}. "
                "Not asked is not the same as no problem, so the outcome goes to a person."
            )
            continue
        below = [str(item) for item in judgment.get("below_threshold") or []]
        if below:
            limits.append(
                f"{report.gate}: the model raised these doubts below the confidence the rule "
                "requires, so they are recorded and not counted as findings: "
                + "; ".join(below)
                + "."
            )
    return limits


def _policy_limits(decision: PolicyDecision | None) -> list[str]:
    if decision is None or decision.allowed:
        return []
    return [
        "The structural guardrails put this change beyond automated judgment "
        f"(risk {decision.risk_level}): "
        + "; ".join(decision.violations)
    ]


def _unreviewable_files(files: Iterable[ChangedFile]) -> list[str]:
    """Files carried by the change that no line-level check could read."""

    opaque = [
        item.path
        for item in files
        if item.change_type != "deleted"
        and not item.added_lines
        and not item.removed_lines
        # A name refused as a path is not unreadable content: the guardrails'
        # line already names it, with the reason, where this one would call it
        # binary or empty.
        and is_safe_relative_path(item.path)
    ]
    if not opaque:
        return []
    shown = ", ".join(markdown_path(path) for path in opaque[:10])
    more = f" ({len(opaque)} in all)" if len(opaque) > 10 else ""
    return [
        f"Files with no content to review line by line (binary, too large or empty): {shown}{more}; "
        "they are only recorded with the change, and the sensitive-path guardrails still apply."
    ]


def _model_section(
    briefs: Sequence[BlockBrief],
    assistant_report: ModelReviewReport | None,
    calls: Sequence[LLMCallRecord],
    reports: Sequence[GateReport] = (),
) -> list[str]:
    lines = [
        "**Advisory only: a model forms no verdict, and nothing below changes the verdicts "
        "above.**",
        "",
    ]
    if calls:
        last = calls[-1]
        failed = len([call for call in calls if not call.schema_valid])
        summary = (
            f"- Model: {last.provider} · {last.model} · prompt {last.prompt_version}; "
            f"{_counted(len(calls), 'call')}"
        )
        if failed:
            summary += f", {failed} of them without valid output"
        lines += [summary, ""]

    judgment = _judgment_detail(reports)
    if not briefs and assistant_report is None and not judgment:
        return [*lines, _no_model_output(calls)]

    lines += judgment
    for brief in briefs:
        lines += _brief_detail(brief)
    if assistant_report is not None:
        lines += _assistant_detail(assistant_report)
    return _trimmed(lines)


def _judgment_detail(reports: Sequence[GateReport]) -> list[str]:
    """What the model made of the judgment-class rules, and what it could not.

    The questions are given as much room as the findings on purpose: for a rule
    like "order status changes go through the state machine", "does this bypass
    the state machine?" handed to a reviewer is worth more than a confident
    answer that has to be re-checked anyway.
    """

    lines: list[str] = []
    for report in reports:
        judgment = report.verifier_report.get("judgment")
        if not isinstance(judgment, dict) or not judgment.get("judged"):
            # Unread rules are not a quiet model — they are a gap, and they are
            # reported under "What this run could not verify" instead.
            continue
        rules = [rule for rule in judgment.get("rules") or [] if isinstance(rule, dict)]
        named = ", ".join(f"`{rule.get('id')}` {rule.get('title')}" for rule in rules)
        lines += [
            f"### Judgment rules · {report.gate} ({len(rules)})",
            "",
            f"Rules put to the model: {named}",
            "",
        ]
        summary = str(judgment.get("summary") or "").strip()
        if summary:
            lines += [markdown_prose(summary), ""]
        findings = int(judgment.get("findings") or 0)
        lines += [
            f"The model raised {_counted(findings, 'doubt')}, listed with the findings above "
            "(**not blocking verdicts**: each is decided by a model's suggestion).",
            "",
        ]
        questions = [str(item) for item in judgment.get("questions") or []]
        if questions:
            lines += [
                "Questions the model was unsure about, left for the reviewer:",
                *_prose_items(questions),
                "",
            ]
    return lines


def _no_model_output(calls: Sequence[LLMCallRecord]) -> str:
    """Say which of the three "no model text" cases this run is in."""

    if not calls:
        return (
            "No model explanation this run: no model is enabled (`HONEST_CI_MODEL_PROVIDER`), "
            "its credential is missing, or `HONEST_CI_MODEL_EXPLAIN=0`. "
            "Deterministic verdicts and findings are unaffected."
        )
    if all(not call.schema_valid for call in calls):
        return (
            "The model call returned no valid output (see `llm-calls.json`), so its "
            "explanation was discarded whole. Deterministic verdicts and findings are unaffected."
        )
    return (
        "The model's output failed the citation check or the line check and was discarded "
        "whole: it cited evidence the gates did not produce, or pointed at a line this change "
        "did not add. Deterministic verdicts and findings are unaffected."
    )


def _brief_detail(brief: BlockBrief) -> list[str]:
    lines = [
        f"### Why gate {brief.gate} blocked (attempt {brief.attempt})",
        "",
        markdown_prose(brief.summary.strip()),
        "",
    ]
    if brief.root_causes:
        lines += ["Root causes:", *_prose_items(brief.root_causes), ""]
    if brief.recommended_actions:
        lines += ["Recommended actions:", *_prose_items(brief.recommended_actions), ""]
    lines += [
        "Gate evidence cited: " + ", ".join(f"`{item}`" for item in brief.cited_evidence_ids),
        "",
    ]
    return lines


def _assistant_detail(report: ModelReviewReport) -> list[str]:
    lines = [
        "### Reading aid for the human reviewer",
        "",
        f"Change summary: {markdown_prose(report.change_summary.strip())}",
        "",
        markdown_prose(report.summary.strip()),
        "",
    ]
    if report.review_questions:
        lines += ["Questions to confirm:", *_prose_items(report.review_questions), ""]
    if report.findings:
        rows = [
            (
                # Unlike a judgment's, this rule id is the model's to write:
                # nothing holds a reading aid to the rules that exist.
                markdown_prose(item.rule_id) if item.rule_id else "—",
                item.severity,
                f"{markdown_path(item.file)}:{item.line_start}",
                markdown_prose(item.title),
                f"{item.confidence:.2f}",
            )
            for item in report.findings[:_MAX_ROWS]
        ]
        lines += [
            "Doubts the model raised (**not gate verdicts**):",
            "",
            *markdown_table(("Rule", "Severity", "Location", "Title", "Confidence"), rows),
            "",
        ]
    return lines


def _metrics_section(metrics: RunMetrics) -> list[str]:
    severity = (
        ", ".join(f"{name} {count}" for name, count in sorted(metrics.findings_by_severity.items()))
        or "—"
    )
    tokens = (
        ", ".join(f"{name} {count}" for name, count in sorted(metrics.token_usage.items())) or "—"
    )
    rows = [
        ("Gates that ran", str(metrics.gates_run)),
        ("Gates that blocked", str(metrics.gates_blocked)),
        ("Findings / blocking", f"{metrics.findings_total} / {metrics.blocking_findings}"),
        ("By severity", severity),
        ("Reruns", str(metrics.rework_count)),
        ("Duration", _optional(metrics.cycle_time_s, lambda value: f"{value:.1f}s")),
        ("Human review", _optional(metrics.human_review_minutes, lambda value: f"{value:g} min")),
        ("AI-generated share", _optional(metrics.ai_generated_ratio, lambda value: f"{value:.0%}")),
        ("Marked false positive", _optional(metrics.false_positives, lambda value: f"{value:.0f}")),
        ("Tokens", tokens),
    ]
    return markdown_table(("Metric", "Value"), rows)


def _configuration_suffix(config: GateConfiguration | None) -> str:
    """The short config digest, appended to a version row.

    Two runs whose versions agree can still have been judged differently -- a
    coverage bar was lowered, a gate was switched off. This is the one string
    that says so at a glance; the section below says what differs.
    """

    return f" · config `{config.digest[:12]}`" if config is not None else ""


def _configuration_section(config: GateConfiguration | None) -> list[str]:
    """What this deployment was configured to do when the run happened.

    An old pack has no snapshot, and says which one it is rather than
    rendering an empty table: absent configuration is a fact about when the
    pack was written, not a deployment that configured nothing.
    """

    if config is None:
        return [
            "None: this run predates the gate configuration snapshot, so the pack does not "
            "record the configuration it ran under. Running the same commit again gives a "
            "pack that does."
        ]
    rows = [
        ("honest-ci version", f"`{config.product_version}`" if config.product_version else "—"),
        ("Configuration digest", f"`{config.digest}`"),
        ("Rules", _file_cell(config.rules_path, config.rules_sha256, config.rule_version)),
        ("Policy", _file_cell(config.policy_path, config.policy_sha256, config.policy_version)),
        ("Required by the policy", ", ".join(config.required_checks) or "—"),
        ("Deployed gates", ", ".join(config.deployed_gates) or "—"),
        ("Switched off", ", ".join(config.disabled_gates) or "none"),
        (
            "When a tool is unavailable",
            "fail closed (recorded as not verified)" if config.fail_closed else "fail open",
        ),
        ("Isolation / image", _isolation_cell(config)),
        ("Toolchain", config.language or "—"),
        (
            "Test command",
            f"`{config.test_command}`" if config.test_command else "(the toolchain's default)",
        ),
        ("Test report", f"`{config.test_report}`" if config.test_report else "—"),
        ("Coverage thresholds", _thresholds_cell(config)),
        ("Test selection", _FULL_SUITE if config.test_selection == "all" else config.test_selection),
        ("Model", _model_cell(config)),
        (
            "Model used for",
            f"explanations {_switch(config.model_explain)}"
            f", judgment rules {_switch(config.model_judgment)}"
            f", regression test generation {_switch(config.model_testgen)}"
            f" (minimum confidence {config.model_min_confidence:.2f})",
        ),
        ("Prompt version", config.prompt_version or "—"),
    ]
    return [
        *markdown_table(("Setting", "Value"), rows),
        "",
        "This section is a record and takes no part in the verdict: the evidence fingerprint "
        "binds the commit, the policy version and the rule version, not the thresholds and "
        "switches here. Changing a threshold does not void an existing human sign-off; to see "
        "whether two runs were judged alike, compare their configuration digests.",
    ]


def _thresholds_cell(config: GateConfiguration) -> str:
    """Both coverage bars in one row: they are two questions, not one number."""

    overall = f"{config.coverage_threshold:.0%}" if config.coverage_threshold else "off"
    changed = config.changed_line_coverage_threshold
    return f"repository {overall} · changed lines {f'{changed:.0%}' if changed else 'recorded, not judged'}"


def _isolation_cell(config: GateConfiguration) -> str:
    """Isolation and image in one cell, because the image only means something under docker.

    Rendering ``process / python:3.12-slim`` would read as if the tools had run
    in that image. Under process isolation they ran on the host, and a reader
    reproducing the verdict needs to know that before they pull the image.
    """

    if config.docker_image is None:
        return config.sandbox_isolation
    if config.sandbox_isolation == "docker":
        return f"docker / `{config.docker_image}`"
    return (
        f"{config.sandbox_isolation} (no container; "
        f"the configured image `{config.docker_image}` was not used this run)"
    )


def _model_cell(config: GateConfiguration) -> str:
    """Which model was asked; an empty provider means nobody switched one on."""

    if not config.model_provider:
        return "not enabled"
    return f"{config.model_provider} / {config.model or 'no credential configured'}"


def _file_cell(path: str | None, sha256: str | None, version: str) -> str:
    if path is None:
        # Not an empty digest: "no such file" and "the file is empty" explain a
        # quiet gate differently, and only one of them is a configuration mistake.
        return f"(no such file; built-in defaults) · version {version}"
    digest = f"`{sha256[:16]}…`" if sha256 else "(digest unreadable)"
    return f"{markdown_path(path)} · {digest} · version {version}"


def _switch(enabled: bool) -> str:
    return "on" if enabled else "off"


def _approval_section(approval: ApprovalDecision) -> list[str]:
    verdict = "approved" if approval.approved else "rejected"
    reason = " ".join((approval.reason or "").split())
    return [
        f"- Decision: **{verdict} by a person**",
        f"- Reviewer: {approval.reviewer or '(unknown)'}",
        # A review's text on the platform is the reviewer's own Markdown, and
        # here it would render under this product's name.
        f"- Reason: {markdown_code(reason) if reason else '(none given)'}",
        *([f"- Given in: {markdown_code(approval.url)}"] if approval.url else []),
        f"- Bound to fingerprint: `{approval.fingerprint}`",
        f"- Decided at: {approval.decided_at.isoformat(timespec='seconds')}",
        "",
        "A human decision changes no gate's verdict, and it does not travel with the change: "
        "once the fingerprint changes, this sign-off no longer applies.",
    ]


def _artifacts_section(manifest: ArtifactManifest | None) -> list[str]:
    if manifest is None or not manifest.artifacts:
        return []
    rows = [
        (f"`{record.name}`", record.kind, f"{record.size_bytes}", f"`{record.sha256[:16]}…`")
        for record in manifest.artifacts
    ]
    return [
        *markdown_table(("Artifact", "Kind", "Bytes", "SHA-256 (first 16)"), rows),
        "",
        # The pack renders its own manifest, so the one artifact it can never
        # list is itself: it is written after the digests are taken.
        "`manifest.json` in the same directory is the record of these digests. `evidence.md` "
        "is written after the manifest is fixed, so the copy written with the run does not "
        "list itself; one rendered later with `honest-ci evidence` does.",
    ]


# ── formatting ───────────────────────────────────────────────────────────────


def markdown_table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> list[str]:
    """A pipe table; cells are flattened so a multi-line value cannot break a row."""

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return lines


def _trimmed(lines: list[str]) -> list[str]:
    """Drop trailing blanks; the section loop owns the spacing between sections."""

    while lines and not lines[-1]:
        lines.pop()
    return lines


def _prose_items(items: Iterable[str]) -> list[str]:
    """Model-written list items, as ``markdown_prose``, each whole inside its bullet.

    An item's later lines are indented under its bullet: a code block that was
    not would close the list, and the rest of the item would stand outside it.
    A bullet opens a line of its own, so a block may follow it on that line;
    the line ``markdown_prose`` leaves empty ahead of one would make the bullet
    an empty item, and under a label a lone ``-`` underlines it as a heading.
    """

    written = (markdown_prose(item.strip()).removeprefix("\n") for item in items)
    return [f"- {text}".replace("\n", "\n  ") for text in written]


def _cell(value: str) -> str:
    """Keep a multi-line summary from breaking out of its table row.

    Its lines are joined with spaces, and the redaction every renderer runs
    over its finished text reads a value past a space. A line ending in
    ``token=`` would take the next line's first word as its value: the fence
    of a code block the model wrote, and the model's code after it would
    render. Such a line is redacted here, where its value ends with it.
    """

    return " ".join(_sealed(line) for line in value.replace("|", "\\|").split("\n"))


def _sealed(line: str) -> str:
    """``line``, redacted when the final pass would take its value from what follows it."""

    if redact_text(f"{line} `").endswith(" `"):
        return line
    # The backtick stands for the next line. Taken as the value here, it is
    # replaced as the value, and the line ends in the redaction marker, where
    # the final pass stops.
    return redact_text(f"{line}`").removesuffix("`")


def _counted(count: int, noun: str) -> str:
    """``count`` and ``noun``, the noun in the plural unless there is exactly one."""

    return f"{count} {noun}{'' if count == 1 else 's'}"


def _truncation_note(total: int, unit: str) -> list[str]:
    return _truncation_note_at(total, _MAX_ROWS, unit)


def _truncation_note_at(total: int, limit: int, unit: str) -> list[str]:
    if total <= limit:
        return []
    return ["", f"(only the first {limit} of {total} {unit} are listed; `evidence.json` has them all)"]


def _location(finding: Finding) -> str:
    if not finding.file:
        return ""
    # A path rule fires on a file's name alone, including a name refused as a
    # path, and this string lands in a table row and a platform comment.
    file = markdown_path(finding.file)
    # A reader looking the line up in the head would find a different line.
    side = " (base)" if finding.side == "base" else ""
    if finding.line_start is None:
        return f"{file}{side}"
    if finding.line_end and finding.line_end != finding.line_start:
        return f"{file}:{finding.line_start}-{finding.line_end}{side}"
    return f"{file}:{finding.line_start}{side}"


def _optional(value: float | None, render: Callable[[float], str]) -> str:
    """Render a metric that may never have been measured.

    "not recorded" rather than 0: a report that shows zero review minutes is
    making a claim nobody made.
    """

    return "not recorded" if value is None else render(value)


__all__ = [
    "APPROVAL_WINDOW_PREFIX",
    "APPROVED_HINT",
    "FINDING_MARKER_PREFIX",
    "MARKER_PREFIX",
    "NO_LIMITS",
    "REVIEW_APPROVAL_HINT",
    "STATUS_HINT",
    "VERDICT_MARKER",
    "approval_window_marker",
    "evidence_limits",
    "finding_marker",
    "integrity_findings",
    "markdown_table",
    "read_approval_window",
    "render_evidence",
    "render_forge_comment",
]
