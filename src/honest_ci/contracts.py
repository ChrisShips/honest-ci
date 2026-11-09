"""Data contracts for the verification layer.

The product shape is: a change someone else produced (a GitHub pull request,
or a local diff) enters, passes the gates -- spec, review, regression -- and
leaves with an evidence pack that explains both what was
blocked and what was let through.

Two rules shape everything below:

* **The model never owns a verdict.** Model output is a proposal that must cite
  deterministic evidence before it is retained (see ``grounding.py``). Every
  finding records ``decided_by`` so a report can always be re-read as "which of
  these did code decide, and which did a model suggest".
* **An approval is bound to a fingerprint.** The fingerprint covers the head
  commit, the policy version and the rule version, so a new push or a changed
  rule invalidates a prior approval instead of silently carrying it forward.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .diffs import count_changed_lines, file_patches, moved_files
from .redaction import redact_text

POLICY_VERSION = "v1"
RULE_VERSION = "v1"

#: The evidence format's own version, carried inside every pack. Separate from
#: the two above because they version a *deployment's* configuration, which the
#: team bumps, while this versions the contract they are written into,
#: which only this product may bump.
#:
#: The rule is add-only: a new field does not move it, a changed field meaning
#: does, and an old pack must still load and still say which version it is. The
#: shape stays ``vN`` -- one counter, no minor -- because "we added fields" is
#: precisely the change a reader is not supposed to have to care about.
EVIDENCE_SCHEMA_VERSION = "v1"

#: Every gate name a pack may carry, including ones no longer in the pipeline.
#: ``metrics`` is kept here purely so old ledgers, old evidence packs and old
#: checkpoints still deserialise -- the evidence format is a public contract
#: and dropping a value from this Literal would turn every pack written before
#: today into a ``ValidationError``.
GateName = Literal["spec", "review", "regression", "metrics"]

#: The pipeline itself: what the runner routes through, in fixed order. It is
#: the single driver -- routing, invalidation and the attempt budget all read
#: it -- so removing a name here removes the gate. ``metrics`` is deliberately
#: absent: measurement decides nothing, and a gate that decides nothing was
#: only ever there to make the count four. Each run's measurements are
#: written into its evidence pack instead.
GATE_SEQUENCE: tuple[GateName, ...] = ("spec", "review", "regression")

Severity = Literal["info", "low", "medium", "high", "critical"]
BLOCKING_SEVERITIES: frozenset[str] = frozenset({"high", "critical"})

#: Who decided a finding. ``model`` output is advisory until a human accepts it.
#:
#: Deliberately open rather than a ``Literal``. The three values this build
#: produces are ``deterministic``, ``model`` and ``human``; the next ones are
#: ``external:<source>``, for conclusions reached outside this deployment -- a
#: platform's own AI review, a CI status, SAST output -- and they will be
#: written by deployments newer than the verifier reading their packs. A closed
#: set here would put a closed ``enum`` in the published schema, and then a
#: verifier written against that schema would call a perfectly intact newer
#: pack malformed. Widening cannot widen who may block: that is
#: ``BLOCKING_SOURCES``, and it stays closed. An external conclusion goes into
#: the pack so an auditor can see what the platform said next to what this
#: deployment verified, never to sign for this deployment: it does not block,
#: the model may not cite it, and one that cannot be fetched is listed under
#: what the run could not verify rather than read as a pass.
DecisionSource = str

#: Which decision sources may block a merge -- see ``Finding.blocking``. A
#: closed set on purpose: anything this deployment did not decide itself is
#: advisory, and listing the blockers means a new source has to be argued in
#: rather than inherited by not being excluded.
BLOCKING_SOURCES: frozenset[str] = frozenset({"deterministic"})

#: ``gitlab`` and ``gitee`` stay although this edition has no adapter for
#: either: the evidence format is add-only, and a pack an earlier build wrote
#: for one must still load.
ForgeName = Literal["gitlab", "gitee", "github", "local"]

#: Which side of the diff a location is on: ``base`` is the file as it was
#: before the change, ``head`` the file as the change leaves it.
DiffSide = Literal["base", "head"]

#: Which tree a gate verified, as the evidence pack names it. Two words rather
#: than the absolute path: a pack is handed to an auditor who cannot check a
#: path on someone else's disk, and the distinction that carries the diagnostic
#: weight is whether the change was verified in a tree materialised for this
#: run (``checkout``) or in the deployment's own working copy
#: (``deployment-root``), whose state nobody pinned.
WorkspaceKind = Literal["checkout", "deployment-root"]


class Verdict(StrEnum):
    """A gate's conclusion about one change."""

    PASS = "pass"
    NEEDS_HUMAN = "needs_human"
    BLOCKED = "blocked"
    #: Explicitly configured off. Never implied by a missing tool -- an
    #: unavailable scanner blocks (fail closed) instead of silently skipping.
    SKIPPED = "skipped"


def utc_now() -> datetime:
    return datetime.now(UTC)


# ── The change under verification ────────────────────────────────────────────


class ChangedFile(BaseModel):
    path: str
    change_type: Literal["added", "modified", "deleted", "renamed"] = "modified"
    added_lines: int = 0
    removed_lines: int = 0
    previous_path: str | None = None


#: A review comment is someone else's prose, and one essay does not get to
#: decide how much every other review costs.
MAX_REVIEW_BODY_CHARS = 2000
#: Review comments one change may carry. A forge adapter that finds more must
#: refuse rather than return the first N: silently dropping the tail could drop
#: the unresolved thread that was going to route this change to a person, and
#: "not read in full" recorded as a limit is a far better answer than an
#: undercount nothing in the pack contradicts.
MAX_PLATFORM_REVIEWS = 200


class PlatformReview(BaseModel):
    """One review comment that was already on the change when this run started.

    A person -- or the platform's own assistant, which is why ``is_bot`` is
    recorded rather than filtered -- has read this change and written something
    down. That is evidence no gate produced and none can reproduce, so it
    enters as input: an unresolved thread is a question somebody asked and
    nobody answered, and merging past it automatically would be an automated
    verdict contradicting a human already on the record.

    ``body`` is redacted and bounded in the model rather than in each adapter,
    because this is the one place text from outside the deployment enters the
    contracts, and a credential pasted into a merge-request comment must not
    reach an artifact, a log or a model payload whichever platform it came from.
    """

    author: str = ""
    body: str = ""
    #: Where the comment sits, when the platform attached it to a line at all.
    path: str | None = None
    line: int | None = Field(default=None, ge=1)
    #: ``True`` resolved, ``False`` still open, ``None`` a plain comment the
    #: platform never tracked as resolvable. Only the middle one is an open
    #: question, so the three must not collapse into a boolean.
    resolved: bool | None = None
    is_bot: bool = False
    created_at: datetime | None = None

    @property
    def unresolved(self) -> bool:
        return self.resolved is False

    @field_validator("body")
    @classmethod
    def _bounded_body(cls, value: str) -> str:
        return _bounded_review_text(value)


def _bounded_review_text(value: str) -> str:
    redacted = redact_text(value)
    if len(redacted) <= MAX_REVIEW_BODY_CHARS:
        return redacted
    return redacted[:MAX_REVIEW_BODY_CHARS] + "…"


class ChangeRequest(BaseModel):
    """One incoming change: a merge/pull request, or a local diff under review."""

    forge: ForgeName = "local"
    repo: str
    number: int | None = None
    url: str | None = None
    title: str = ""
    description: str = ""
    author: str | None = None
    source_branch: str | None = None
    target_branch: str | None = None
    head_sha: str
    base_sha: str | None = None
    diff: str = ""
    files_changed: list[ChangedFile] = Field(default_factory=list)
    #: Share of the diff reported as AI-generated, when the forge or the client
    #: supplies it. ``None`` means "not measured", which is not the same as 0.
    ai_generated_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    labels: list[str] = Field(default_factory=list)
    #: Review comments the platform already carried, read once at intake.
    #: Empty means "there were none"; ``platform_reviews_error`` is what says
    #: "nobody could read them", and the two must never look alike.
    platform_reviews: list[PlatformReview] = Field(default_factory=list)
    #: Why those comments could not be read, when they could not. Reading them
    #: is fail-soft -- the gates still run -- so this is how the evidence pack
    #: names what the run did not get to see.
    platform_reviews_error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def unresolved_reviews(self) -> list[PlatformReview]:
        """Threads someone opened on this change and nobody has closed."""

        return [review for review in self.platform_reviews if review.unresolved]

    @property
    def ref(self) -> str:
        """Stable human-readable reference used in ledgers and reports."""

        number = f"#{self.number}" if self.number is not None else ""
        return f"{self.repo}{number}@{self.head_sha[:12]}"

    @property
    def changed_lines(self) -> int:
        """Lines the change adds and removes; a file moved without edits counts none.

        Local intake runs git without rename detection, so a moved file reaches
        it as every line removed under one name and added under another, while
        a forge reports the same move as a rename with no lines at all. Counted
        in full, one ``git mv`` of a 2000-line file crossed ``max_changed_lines``
        locally and counted nothing on a platform: the same change held or
        passed by where it came from. No line of such a file changed. What
        moved is its name, and ``touched_paths`` counts both names on either
        side.

        A move that also edits the file is not discounted. A forge counts only
        its edits; here it still counts in full, because telling which new file
        carries on which deleted one then takes a similarity guess, and a limit
        that shrinks on a guess is one a change can be shaped to pass. That
        difference only ever makes the local count the larger.

        A change with no line counts listed (a ``--diff`` patch) is counted
        from the diff itself.
        """

        moves = moved_files(self.diff)
        moved = {name for move in moves for name in (move.old_path, move.new_path)}
        listed = sum(
            item.added_lines + item.removed_lines
            for item in self.files_changed
            if item.path not in moved
        )
        return listed or count_changed_lines(self.diff) - 2 * sum(move.lines for move in moves)

    @property
    def renames(self) -> list[tuple[str, str]]:
        """Each file the forge reported as moved, as ``(old name, new name)``.

        What the diff cannot carry: a pure rename has no lines, so a forge
        sends no patch for it and the diff never names it. Local intake reads a
        move as a deletion and an addition, so a local change has none. A
        ``--diff`` patch has none either: git names its pure renames in the
        diff itself, in headers ``diffs.file_patches`` reads.
        """

        return [
            (item.previous_path, item.path) for item in self.files_changed if item.previous_path
        ]

    @property
    def touched_paths(self) -> list[str]:
        """Every name the change touches, sorted: a rename counts under both.

        Local intake runs git without rename detection, so a move reaches it
        as one name deleted and another added, and the guardrails, the path
        rules and the size limits all see both. A forge reports the same move
        as one renamed file. Read by its new name alone, moving
        ``.github/workflows/deploy.yml`` to ``docs/`` passed the sensitive-path
        guardrail on a platform while it was held locally, and one change was
        one file there and two here. The old name is part of the change:
        whatever looked for the file there -- a pipeline, an import, a glob --
        finds nothing now.

        A patch with no file list (``--diff``) is read from its headers, both
        sides, so a deleted file counts there as well: its new side is
        ``/dev/null``.
        """

        if not self.files_changed:
            return sorted(
                {
                    name
                    for patch in file_patches(self.diff)
                    for name in (patch.old_path, patch.new_path)
                    if name
                }
            )
        names = {item.path for item in self.files_changed}
        names.update(item.previous_path for item in self.files_changed if item.previous_path)
        return sorted(names)


# ── Deterministic tool execution ─────────────────────────────────────────────


class ToolDiagnostic(BaseModel):
    """One located message a scanner produced.

    Parsed by the scanner adapter, which owns each tool's output format. A gate
    needs the location to answer the only question that keeps a review gate
    usable: did this land on a line the change actually added?
    """

    tool: str
    file: str | None = None
    line: int | None = Field(default=None, ge=1)
    code: str | None = None
    message: str = ""


class ToolReport(BaseModel):
    """One command executed in the sandbox.

    ``available=False`` means the tool never ran. That is not evidence about
    the code: a gate that needs it must block rather than pass.
    """

    name: str
    command: str
    available: bool = True
    passed: bool = True
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float | None = None
    report_path: str | None = None
    findings: list[str] = Field(default_factory=list)
    #: Located diagnostics, when the adapter could parse the tool's output.
    #: Empty is not "clean": a tool that crashed also parses to nothing, which
    #: is why the gate reads ``passed`` as well.
    diagnostics: list[ToolDiagnostic] = Field(default_factory=list)


class TestResults(BaseModel):
    """The suite the regression gate ran, in counts a verdict can be read from.

    ``available=False`` mirrors ``ToolReport.available``: the suite never
    started, so the counts below are not evidence about the code. The gate
    turns that into a block (fail closed), never into a pass.
    """

    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    #: Tests generated by the regression gate, kept apart from the repo's own
    #: suite so a report can say which coverage the product added. They
    #: are never added to ``passed``/``failed``: a model-written case that
    #: fails is a question for a human, and mixing it into the suite's own
    #: counts would let it reach a block through arithmetic.
    generated: int = 0
    generated_failed: int = 0
    generated_failures: list[str] = Field(default_factory=list)
    #: What the counts were read from: ``pytest`` (its summary line), ``junit``
    #: (JUnit XML the command left behind, named by ``HONEST_CI_TEST_REPORT``) or
    #: ``exit-code`` (a custom command with no report configured, of which only
    #: the exit code is known). The gate reads runner-specific exit codes by it.
    runner: str = "pytest"
    #: The short-summary lines (``FAILED path::test - reason``), in order.
    failures: list[str] = Field(default_factory=list)
    available: bool = True
    exit_code: int | None = None
    command: str | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float | None = None
    report_path: str | None = None

    @property
    def collected(self) -> int:
        """Tests the run accounted for. Zero is "no evidence", not "no defects"."""

        return self.passed + self.failed + self.errors + self.skipped


class CoverageResult(BaseModel):
    """Line coverage of the suite that just ran, or why there is no figure.

    ``ratio`` and ``available`` are separate on purpose. An installed tool that
    found no data is a check that ran and produced no evidence; a tool that is
    not installed never ran. The gate reads the two differently and the
    evidence pack lists them under different lines, and neither is a pass.

    The ``changed_lines_*`` pair answers the narrower question a reviewer
    actually asks — did the tests reach the lines this change touched — and it
    is the one a repository can answer honestly while its whole-project figure
    stays where it has always been. It is only filled when the caller asked for it and the tool gave
    per-line data.
    """

    ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    available: bool = True
    command: str = ""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float | None = None
    #: Executable lines this change added, as the coverage tool counts them.
    #: ``None`` is "nobody could measure it"; ``0`` is "the change added no
    #: line this tool considers executable" (a comment, a docstring, a data
    #: file). Only the first is missing evidence, so the two never collapse.
    changed_lines_total: int | None = Field(default=None, ge=0)
    changed_lines_covered: int | None = Field(default=None, ge=0)
    #: ``(file, line)`` for each added line the suite never executed, bounded
    #: by the adapter. This is the list a developer acts on, so it carries the
    #: locations rather than only a count.
    uncovered: list[tuple[str, int]] = Field(default_factory=list)
    #: Sources the per-line data named that no file in the workspace matched
    #: (a JaCoCo package path with no module to resolve it against). Recorded
    #: rather than dropped: lines nobody could locate are lines nobody checked.
    unmapped_sources: list[str] = Field(default_factory=list)
    #: Changed files the coverage configuration never asked anyone to measure
    #: -- a test file under ``source = ["app"]``, a script outside the measured
    #: packages. Separate from ``unmapped_sources`` because they read in
    #: opposite directions: that one is evidence somebody expected and did not
    #: get, this one is the configuration doing what it says.
    out_of_scope_sources: list[str] = Field(default_factory=list)

    @property
    def changed_lines_ratio(self) -> float | None:
        """Coverage of the added lines, or ``None`` when there is nothing to divide."""

        total = self.changed_lines_total
        if not total or self.changed_lines_covered is None:
            return None
        return min(1.0, self.changed_lines_covered / total)


# ── Findings and gate reports ────────────────────────────────────────────────


class Finding(BaseModel):
    """Why a gate raised something, in a form a developer can act on."""

    id: str
    gate: GateName
    #: Team rule that fired (``R-03``, ``C-12`` ...). ``None`` for findings that
    #: come from a generic tool rather than the team's rule library.
    rule_id: str | None = None
    severity: Severity = "medium"
    title: str
    detail: str = ""
    file: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    #: Which version of ``file`` the lines count in. ``head`` for everything
    #: that reads the lines a change added, which was every finding before the
    #: test-integrity checks. ``base`` for a finding about what the change took
    #: away or rewrote -- a deleted test has no line left in the head to point
    #: at, so it points at the line it had in the base, under the name the
    #: file had there. A platform comment posted on such a line has to be
    #: posted on the old side of the diff, or it lands on whatever line now
    #: carries that number.
    side: DiffSide = "head"
    #: Deterministic evidence ids this finding is grounded in.
    cited_evidence_ids: list[str] = Field(default_factory=list)
    decided_by: DecisionSource = "deterministic"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    suggested_fix: str | None = None
    #: Set when a human marks the finding a false positive, which is where
    #: revising the rule starts. It never rewrites the original verdict.
    false_positive: bool | None = None

    @property
    def blocking(self) -> bool:
        """Whether this finding, on its own, stops the merge.

        Severity alone does not decide it: only this deployment's own
        deterministic rules block, whatever severity anyone else's rule
        carries. A block is terminal -- ``resume_approval`` reaches runs
        waiting on a human and not blocked ones -- so a judgment that reached
        it would be replacing an approval instead of informing one. Every
        other finding routes the change to a human, and its severity is what
        that human is told to weigh it at.

        Allow-list rather than "not a model", because the set that may block
        is the closed one and the set that may not is open: external verdicts
        (``external:<source>``) are next, and they must be non-blocking from
        the first line that produces one rather than from the line somebody
        remembers to add here. ``human`` is out for the same reason it is not
        a producer anywhere today -- a person who wants to stop a merge signs
        the approval, and a person who blocked via a finding would have locked
        themselves out of ``resume_approval``.

        Enforced on the finding rather than at each caller because "the model
        never owns a verdict" has to hold for the gate verdict, the metrics,
        the evidence rendering and the MR comment alike, and all of them read
        this one property.
        """

        return self.severity in BLOCKING_SEVERITIES and self.decided_by in BLOCKING_SOURCES


class GateReport(BaseModel):
    """One gate's conclusion plus everything it looked at."""

    gate: GateName
    verdict: Verdict
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    tool_reports: list[ToolReport] = Field(default_factory=list)
    test_results: TestResults | None = None
    #: Gate-specific numbers (coverage, rules checked, impacted symbols ...),
    #: kept in the evidence pack.
    metrics: dict[str, float] = Field(default_factory=dict)
    #: Deterministic detail behind the verdict, kept for the evidence pack.
    verifier_report: dict[str, Any] = Field(default_factory=dict)
    feedback: str | None = None
    rule_version: str = RULE_VERSION
    policy_version: str = POLICY_VERSION
    trace_id: str | None = None
    duration_s: float | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def blocking_findings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.blocking]


# ── Advisory model output (never a verdict) ──────────────────────────────────


class ModelFinding(BaseModel):
    """A model's proposed finding, before grounding accepts or discards it."""

    rule_id: str | None = None
    severity: Severity = "medium"
    title: str
    detail: str = ""
    file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    suggested_question: str | None = None


class ModelReviewReport(BaseModel):
    """A reading aid bound to exactly one evidence fingerprint."""

    evidence_fingerprint: str
    summary: str
    change_summary: str
    findings: list[ModelFinding] = Field(default_factory=list)
    review_questions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ProposedTest(BaseModel):
    """One regression case a model proposed for this change.

    ``targets`` names the changed symbols it claims to exercise, and it is the
    citation this artifact is checked against: a case aimed at something this
    change did not touch is not a regression case for this change, whatever
    else it may be. ``path`` and ``content`` are equally untrusted -- they
    become a file in the tree the suite runs in, so both pass the trust
    boundary before anything is written.
    """

    path: str
    content: str
    targets: list[str] = Field(default_factory=list)


class ProposedTests(BaseModel):
    """Regression cases a model proposed, bound to one evidence fingerprint.

    Bound for the same reason a reading aid is: the cases are about the symbols
    *this* change touched, and a proposal produced against a different state of
    the change is not evidence about this one.
    """

    evidence_fingerprint: str
    summary: str = ""
    tests: list[ProposedTest] = Field(default_factory=list)


class BlockBrief(BaseModel):
    """Model explanation of a deterministic block.

    The gate stays authoritative: the brief may only restate and prioritise
    evidence the gate already produced, and it is discarded unless every cited
    id came from that gate.
    """

    gate: GateName
    attempt: int = Field(ge=1)
    source_fingerprint: str
    summary: str
    root_causes: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ── Policy, metrics, evidence ────────────────────────────────────────────────


class PolicyDecision(BaseModel):
    """Structural guardrails evaluated before any gate spends model budget."""

    allowed: bool
    policy_version: str
    risk_level: Literal["low", "medium", "high"] = "low"
    violations: list[str] = Field(default_factory=list)
    required_checks: list[str] = Field(default_factory=list)
    #: Always empty: the policy has no reviewers setting any more. Kept because
    #: the evidence format is add-only -- a reader written against an earlier
    #: pack still finds the field it knew.
    required_reviewers: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    changed_lines: int = 0


class RunMetrics(BaseModel):
    """The minimum set of measurements a run records, in its evidence pack and the ledger."""

    gates_run: int = 0
    gates_blocked: int = 0
    findings_total: int = 0
    findings_by_severity: dict[str, int] = Field(default_factory=dict)
    blocking_findings: int = 0
    #: Minutes a human spent reviewing after the gates ran.
    human_review_minutes: float | None = None
    ai_generated_ratio: float | None = None
    rework_count: int = 0
    cycle_time_s: float | None = None
    #: Defects that reached production despite passing the gates. Nothing
    #: writes it yet, and ``None`` is how this file says "not measured"
    #: everywhere else: ``0`` in an evidence pack would be this product
    #: telling an auditor that it counted, and found none.
    escaped_defects: int | None = None
    false_positives: int | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)


class ArtifactRecord(BaseModel):
    name: str
    path: str
    sha256: str
    kind: str = "json"
    size_bytes: int = 0


class ArtifactManifest(BaseModel):
    run_id: str
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class GateConfiguration(BaseModel):
    """What this deployment was configured to do, at the moment of the run.

    An evidence pack answers "what was verified". Without this it cannot answer
    "under what", and the two runs someone will eventually put side by side
    differ precisely there: a rule was edited, a gate was switched off, a
    coverage bar was lowered, the judgment model was turned off, a different
    image ran the tools. Each of those is one line here instead of something a
    reader has to reconstruct from an environment that has since moved on.

    ``rules_sha256`` is ``None`` when no rule library was configured, never an
    empty string: "there was no file" and "the file was empty" are different
    facts, and only one of them explains a review gate that found nothing.

    Deliberately **not** part of the approval fingerprint. The fingerprint
    binds a signature to the commit, the policy version and the rule version;
    folding a threshold into it would invalidate every approval the moment an
    operator tuned one, which is a different decision from the one this
    snapshot makes. It records; the fingerprint decides.

    It carries no credential. The provider and the model name say which service
    was asked, never what it was asked with.
    """

    product_version: str = ""
    #: The rule library and the policy, by path and by content digest. The
    #: digest is over the file's bytes, so a reader can check it with
    #: ``sha256sum`` against the file they were handed.
    rules_path: str | None = None
    rules_sha256: str | None = None
    rule_version: str = RULE_VERSION
    policy_path: str | None = None
    policy_sha256: str | None = None
    policy_version: str = POLICY_VERSION
    required_checks: list[str] = Field(default_factory=list)
    deployed_gates: list[str] = Field(default_factory=list)
    disabled_gates: list[str] = Field(default_factory=list)
    fail_closed: bool = True
    sandbox_isolation: str = "process"
    #: The toolchain that judged the change, worded as ``doctor`` words it --
    #: including *how* it was chosen, because "the wrong toolchain ran" is a
    #: failure the evidence has to make visible.
    language: str = ""
    #: The image that toolchain runs its tools in; ``None`` when the toolchain
    #: names none. Read together with ``sandbox_isolation``: under ``process``
    #: no image was used at all.
    docker_image: str | None = None
    test_command: str | None = None
    test_report: str | None = None
    coverage_threshold: float = 0.0
    #: The bar on the lines this change added, and which tests were asked for.
    #: Both change what the regression gate's "passed" means, so a pack that
    #: does not carry them cannot be compared with one judged differently.
    changed_line_coverage_threshold: float = 0.0
    test_selection: str = "all"
    #: Empty when no provider was named, which is how the model is switched off.
    model_provider: str = ""
    #: The model a judgment call would really have used; ``None`` means no
    #: model was asked -- none was enabled, or its credential was missing --
    #: so judgment-class rules went unread.
    model: str | None = None
    model_explain: bool = False
    model_judgment: bool = False
    model_testgen: bool = False
    model_min_confidence: float = 0.0
    prompt_version: str = ""

    @property
    def digest(self) -> str:
        """One short string two runs can be compared by.

        A reader asking "was this judged the same way as that one" should not
        have to diff twenty fields; they compare this, and only diff when it
        differs.
        """

        material = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


class LLMCallRecord(BaseModel):
    provider: str = "deepseek"
    protocol: Literal["anthropic", "openai"] = "openai"
    model: str
    prompt_version: str = "v1"
    latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    attempts: int = 1
    schema_name: str | None = None
    schema_valid: bool = True
    error: str | None = None


class EvidencePack(BaseModel):
    """What ships with every decision: what ran, what it found, and why."""

    #: Which version of the public evidence contract this pack is written to.
    #: Defaulted rather than required so the one place that has to know "a pack
    #: without the field is v1" is this line, and not a branch every future
    #: verifier carries forever.
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    run_id: str
    change: ChangeRequest
    gate_reports: list[GateReport] = Field(default_factory=list)
    verdict: Verdict = Verdict.PASS
    fingerprint: str
    trace_id: str | None = None
    policy_version: str = POLICY_VERSION
    rule_version: str = RULE_VERSION
    policy_decision: PolicyDecision | None = None
    #: What this deployment was configured to do when the run happened.
    #: ``None`` for packs written before the snapshot existed, which is why it
    #: is optional: an old pack still renders, and it says which one it is.
    configuration: GateConfiguration | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    artifact_manifest: ArtifactManifest | None = None
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    briefs: list[BlockBrief] = Field(default_factory=list)
    sandbox_isolation: str = "process"
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def findings(self) -> list[Finding]:
        return [finding for report in self.gate_reports for finding in report.findings]


def evidence_schema() -> dict[str, Any]:
    """The JSON Schema of an evidence pack, as this build writes them.

    Exists so the contract can leave the building without ``honest_ci``:
    whatever reads a pack can check its structure against this file, and
    anyone running honest-ci gets it out of their own install
    (``honest-ci schema``) rather than out of a mail. ``by_alias`` is off and
    the mode is serialisation because what it has to describe is the JSON on
    disk, not the arguments a constructor takes.
    """

    schema = EvidencePack.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "honest-ci evidence pack"
    schema["x-schema-version"] = EVIDENCE_SCHEMA_VERSION
    return schema


def evidence_schema_json() -> str:
    """``evidence_schema`` as the bytes that go into the repository and the zip."""

    return json.dumps(evidence_schema(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


# ── Human approval ───────────────────────────────────────────────────────────


class ApprovalTicket(BaseModel):
    fingerprint: str
    evidence_summary: str = ""
    approved: bool | None = None
    reason: str | None = None
    reviewer: str | None = None
    forge: ForgeName | None = None
    repo: str | None = None
    number: int | None = None
    url: str | None = None
    assistant_report: ModelReviewReport | None = None


class ApprovalDecision(BaseModel):
    approved: bool
    fingerprint: str
    reason: str | None = None
    reviewer: str | None = None
    decided_at: datetime = Field(default_factory=utc_now)
    #: The review the decision was given in, when it was given on the platform,
    #: so a reader can hold the recorded words against the record.
    url: str | None = None


# ── Approvals given on the platform ──────────────────────────────────────────

#: The review states that say whether a person lets the change merge. A plain
#: comment is left out: it neither grants an approval nor withdraws one.
PlatformReviewState = Literal["approved", "changes_requested", "dismissed"]


def _aware(value: datetime | None) -> datetime | None:
    """A timestamp with its zone; one sent without is read as UTC, the platform's clock.

    The approval rules compare these, and Python refuses to order a naive
    timestamp against an aware one.
    """

    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


class PlatformApproval(BaseModel):
    """One decisive review on the change, as the platform reports it.

    An approval, a request for changes, or a review somebody dismissed. Every
    fact the approval rules need is recorded as the platform gave it and judged
    in ``approvals.py``, so what counts as an approval is decided in one place
    whichever platform it came from. ``reason`` is someone else's text, so it is
    redacted and bounded here, as ``PlatformReview.body`` is.
    """

    reviewer: str = ""
    state: PlatformReviewState
    #: The commit the review was given on; empty when the platform did not say.
    commit_sha: str = ""
    reason: str = ""
    #: Who edited ``reason`` last; empty when nobody has, or the platform did
    #: not say.
    edited_by: str = ""
    #: Whether the reviewer can push to the repository, as the platform says now.
    can_push: bool = False
    is_bot: bool = False
    submitted_at: datetime | None = None
    url: str | None = None

    @field_validator("reason")
    @classmethod
    def _bounded_reason(cls, value: str) -> str:
        return _bounded_review_text(value)

    @field_validator("submitted_at")
    @classmethod
    def _aware_submitted_at(cls, value: datetime | None) -> datetime | None:
        return _aware(value)


class ApprovalWindow(BaseModel):
    """From when approvals of one evidence fingerprint count.

    A review names the commit it was given on, which binds the head. It does
    not name the rule and policy versions the rest of the fingerprint covers,
    and a run in CI keeps no ledger to remember them by. So the conclusion
    comment records when each fingerprint's evidence first reached the change,
    in the platform's own clock: the latest decisive review already there at
    that point. A review given at or before ``after`` was given on an earlier
    conclusion -- another commit, other rules, another policy -- and does not
    approve this one.
    """

    fingerprint: str
    #: ``None`` when no decisive review was there yet, so none predates it.
    after: datetime | None = None

    @field_validator("after")
    @classmethod
    def _aware_after(cls, value: datetime | None) -> datetime | None:
        return _aware(value)


class PlatformApprovals(BaseModel):
    """What a platform holds about approving one change."""

    reviews: list[PlatformApproval] = Field(default_factory=list)
    #: The window this deployment recorded on the change with its last
    #: conclusion; ``None`` when it recorded none, or has written none yet.
    window: ApprovalWindow | None = None


class UncountedApproval(BaseModel):
    """An approval given on the platform that does not approve this evidence, and why."""

    review: PlatformApproval
    why: str


class ApprovalReading(BaseModel):
    """What the approvals on the platform amount to for one run's evidence.

    Decided by ``approvals.read_approvals`` and carried to the conclusion
    comment, which says what counted and what did not, and records ``window``
    for the next run to read back.
    """

    #: ``None`` records no window, and the next run then takes every review
    #: already given as predating its evidence. That is the strict reading, and
    #: the one a run that could not read the reviews has to leave behind.
    window: ApprovalWindow | None = None
    #: The approval that counts: the earliest one that meets every condition.
    approval: PlatformApproval | None = None
    uncounted: list[UncountedApproval] = Field(default_factory=list)
    #: Requests for changes that stand. While there is one, no approval counts.
    objections: list[PlatformApproval] = Field(default_factory=list)
    #: Why nothing was applied although the run waited on a person.
    error: str | None = None


class FindingAnnotation(BaseModel):
    """A human's reading of one finding, after the fact.

    It is bound to the evidence fingerprint the finding was produced under, for
    the same reason an approval is: once the rules change, "this was a false
    positive" is a statement about a finding that no longer exists. It never
    rewrites the verdict -- the finding's flag and the run's false-positive
    count follow from these, the block is not undone by them.
    """

    run_id: str
    finding_id: str
    fingerprint: str
    false_positive: bool
    reason: str | None = None
    annotator: str | None = None
    annotated_at: datetime = Field(default_factory=utc_now)


# ── Run accounting ───────────────────────────────────────────────────────────

RunOutcome = Literal["mergeable", "blocked", "needs_human", "failed", "running"]
Route = Literal["advance", "retry_current", "wait_human", "block", "fail"]


class GateAttempt(BaseModel):
    """An immutable record of one gate execution and the routing that followed."""

    gate: str
    attempt: int
    outcome: Literal["passed", "skipped", "blocked", "exception", "waiting"]
    route: Route
    feedback: str | None = None
    failure_class: str | None = None
    duration_s: float | None = None
    created_at: datetime = Field(default_factory=utc_now)


class RunRecord(BaseModel):
    run_id: str
    change_ref: str
    outcome: RunOutcome
    per_gate_timing: dict[str, float] = Field(default_factory=dict)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    cost_usd: float | None = None
    evidence_path: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    attempts: dict[str, int] = Field(default_factory=dict)
    failure_reasons: list[str] = Field(default_factory=list)
    sandbox_isolation: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


# ── Fingerprints ─────────────────────────────────────────────────────────────


def compute_fingerprint(
    head_sha: str,
    policy_version: str = POLICY_VERSION,
    rule_version: str = RULE_VERSION,
) -> str:
    """Identity of "this commit, judged by these rules".

    Any change to the commit, the policy or the rule library yields a different
    fingerprint, which is what makes a stale approval fail loudly.
    """

    material = f"{head_sha}|{policy_version}|{rule_version}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def compute_evidence_fingerprint(
    head_sha: str,
    policy_version: str,
    rule_version: str = RULE_VERSION,
    evidence_sha: str | None = None,
) -> str:
    material = "|".join(
        part for part in (head_sha, policy_version, rule_version, evidence_sha or "") if part
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
