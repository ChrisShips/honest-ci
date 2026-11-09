"""Ports. Everything replaceable sits behind one of these.

The gates are written against these interfaces only, so the same gate semantics
run against GitHub or a local diff, and against any model provider — or none
at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..contracts import (
    ApprovalReading,
    ApprovalTicket,
    ArtifactManifest,
    BlockBrief,
    ChangeRequest,
    CoverageResult,
    EvidencePack,
    Finding,
    FindingAnnotation,
    GateName,
    LLMCallRecord,
    ModelReviewReport,
    PlatformApprovals,
    PlatformReview,
    ProposedTests,
    RunRecord,
    TestResults,
    ToolReport,
    Verdict,
)
from ..license_check import Ownership
from ..state import RunState, RunStatus


class LLMAdapter(ABC):
    """Model-domain work only: reading, judging, explaining — never a verdict.

    Every method returns a schema-validated artifact that ``grounding.py`` still
    has to accept before anything reaches a report or a reviewer. ``judge_rules``
    is the one that comes closest to a verdict and is therefore the one to be
    clearest about: it proposes findings against rules the team wrote, and
    deterministic code decides their severity, whether they clear the confidence
    bar, and what they do to the gate's conclusion.
    """

    #: Per-call telemetry the runner copies into the evidence pack.
    call_records: list[LLMCallRecord]

    @abstractmethod
    def review_change(self, evidence: EvidencePack) -> ModelReviewReport:
        """Read the change and propose findings grounded in its added lines."""

        raise NotImplementedError

    @abstractmethod
    def judge_rules(
        self,
        *,
        change: ChangeRequest,
        fingerprint: str,
        rules: list[dict[str, str]],
        locations: list[dict[str, str]],
        reviews: Sequence[dict[str, str]] = (),
    ) -> ModelReviewReport:
        """Judge one change against judgment-class team rules.

        ``rules`` carries each rule's id, title and criterion; ``locations`` is
        the inventory of lines the change added within those rules' scope, which
        is exactly the set a finding may point at. Both are plain dicts rather
        than rule-library objects so this port stays independent of the file
        format the rules happen to be written in.

        ``reviews`` is what people already wrote on this change at the
        platform. It is context and never a citation: a finding still has to
        name a rule id and land on an added line, so a model cannot launder an
        invented claim through somebody else's sentence. Defaulted to empty
        because a forge that has no comments to hand over, and a local diff
        that has no platform at all, are the ordinary cases.

        Abstract rather than defaulting to "found nothing", because a model
        adapter that cannot do this must fail to exist rather than quietly turn
        every judgment rule into a clean bill of health.
        """

        raise NotImplementedError

    @abstractmethod
    def propose_tests(
        self,
        *,
        change: ChangeRequest,
        symbols: Sequence[dict[str, str]],
        fingerprint: str,
        language: str,
        test_directory: str,
    ) -> ProposedTests:
        """Propose regression cases for the symbols this change touched.

        ``symbols`` is the impact analysis's own output, flattened to plain
        dicts so this port does not depend on how that analysis is
        represented; ``test_directory`` is the only place a proposal may write,
        and it is given rather than assumed because it is a property of the
        toolchain, not of the model.

        What comes back is a **proposal**: files nobody has written yet.
        ``grounding.guarded_test_proposal`` decides whether any of it is
        allowed to become a file, and the gate decides what a failing generated
        case means -- which is never a block.

        Abstract for the same reason ``judge_rules`` is: an adapter that cannot
        do this should fail to exist rather than quietly hand back "no tests
        needed" for every change.
        """

        raise NotImplementedError

    @abstractmethod
    def explain_block(
        self,
        *,
        gate: GateName,
        attempt: int,
        source_fingerprint: str,
        summary: str,
        deterministic_feedback: str,
        evidence_items: list[dict[str, str]],
    ) -> BlockBrief:
        """Explain a deterministic block without changing its verdict."""

        raise NotImplementedError


class ForgeError(RuntimeError):
    """A platform call failed, or answered with something that cannot be judged.

    Declared on the port rather than per-implementation so the entry points can
    tell "the platform refused" from a bug in this codebase without importing a
    specific platform.
    """


class ForgeAdapter(ABC):
    """A code platform: where changes come from and where verdicts go back."""

    @abstractmethod
    def fetch_change(self, repo: str, number: int) -> ChangeRequest:
        """Load one merge/pull request, including its diff and file list."""

        raise NotImplementedError

    @abstractmethod
    def materialize(self, change: ChangeRequest, workspace: Path) -> Path:
        """Produce a working tree at the change's head commit for tool runs."""

        raise NotImplementedError

    @abstractmethod
    def publish_report(
        self,
        change: ChangeRequest,
        evidence: EvidencePack,
        status: RunStatus,
        approvals: ApprovalReading | None = None,
        evidence_at: str | None = None,
        license_note: str | None = None,
    ) -> None:
        """Write or update this deployment's conclusion comment on the change.

        The whole pack rather than one gate's report, because the two things a
        reader needs most are cross-gate: the terminal state (a run whose gates
        all passed can still be ``awaiting_human``) and what this run could
        *not* verify.

        ``approvals`` is what the approvals on the platform amounted to, when
        this run read them. The comment says what counted and what did not, and
        keeps the window the next run reads back through ``fetch_approvals``.

        ``evidence_at`` is where the comment says the full evidence pack is,
        when the caller knows better than the ledger (``render_forge_comment``).

        ``license_note`` is what the license check says about a run on its
        trial or on a key past its expiry date; the comment ends with it.
        """

        raise NotImplementedError

    @abstractmethod
    def publish_note(self, change: ChangeRequest, body: str) -> None:
        """Write ``body`` where ``publish_report`` writes the conclusion, in place of the last one.

        For a run that ends with no evidence pack to report, such as one the
        license check stopped before any gate. ``body`` starts with the marker
        ``render_forge_comment`` starts with, so the next run finds it as it
        finds a conclusion. It takes the place of the approval window the last
        conclusion recorded, so the next run reads approvals as strictly as a
        first run does.
        """

        raise NotImplementedError

    def fetch_reviews(self, change: ChangeRequest) -> list[PlatformReview]:
        """Read the review comments the platform already carries on this change.

        Concrete and empty by default: a forge with no discussion threads has
        nothing to hand over, and saying so honestly is not the same as
        claiming nobody has commented. The caller keeps the two apart -- an
        empty list here means "none found", and a raised ``ForgeError`` is
        recorded as "not read".
        """

        return []

    def fetch_approvals(self, change: ChangeRequest) -> PlatformApprovals | None:
        """Read the decisive reviews on this change, and the window last recorded there.

        ``None`` -- the default -- says this platform's approvals are not read
        here, so a change waiting on a person waits for ``honest-ci approve``.
        That is not "nobody approved", which is an empty list; and a raised
        ``ForgeError`` means they could not be read, so nothing is applied.
        Whether an approval counts is decided in ``approvals.py``, never here.
        """

        return None

    def ownership(self, change: ChangeRequest) -> Ownership | None:
        """Whose repository ``change`` merges into, whether it is public, and who opened it.

        For the license check alone, which reads it off what ``fetch_change``
        already received rather than asking the platform again. Not a field of
        ``ChangeRequest``: that goes into the evidence pack, and what the check
        finds must stay out of it.

        ``None`` -- the default -- says this platform doesn't tell, and the
        check, unable to tell, asks for no key. Raises
        ``license_check.Unclear`` when the platform's answer left out
        something the check needs.
        """

        return None

    def publish_progress(
        self,
        change: ChangeRequest,
        summary: str,
        url: str | None = None,
    ) -> None:
        """Publish an in-progress status when the platform supports it.

        ``url`` is the same page ``publish_status`` links to: until the verdict
        replaces this status, it is the only way from the change to its run.
        """

        return None

    def publish_inline(
        self,
        change: ChangeRequest,
        evidence: EvidencePack,
        findings: Sequence[Finding],
    ) -> list[str]:
        """Return only finding ids newly posted onto code lines; unsupported is empty."""

        return []

    @abstractmethod
    def publish_status(
        self,
        change: ChangeRequest,
        verdict: Verdict,
        summary: str,
        url: str | None = None,
    ) -> None:
        """Set the commit/merge-request status the team's merge rules read.

        Implementations must not map anything but ``PASS`` to their platform's
        success state: a change waiting on a human, or one whose gates were
        skipped, has not been verified, and a green check would say it was.
        """

        raise NotImplementedError

    def publish_error(
        self,
        change: ChangeRequest,
        summary: str,
        url: str | None = None,
    ) -> None:
        """Set the status for a run that fell over before reaching a verdict.

        Apart from ``publish_status`` because there is no ``Verdict`` to pass:
        "the gates blocked this change" and "nothing concluded anything" are
        different news, and a platform with a state for the second (GitHub's
        ``error``) should be able to say so. The default is the blocked status
        every forge already published for a failed run, so a platform that has
        no such state keeps behaving exactly as it did -- never green, because
        a merge rule must not be satisfied by a run that produced no verdict.
        """

        self.publish_status(change, Verdict.BLOCKED, summary, url)


class InlinePublishError(ForgeError):
    """Inline delivery was incomplete; preserve ids already posted before a refusal."""

    def __init__(self, message: str, posted: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.posted = list(posted)


class ScannerAdapter(ABC):
    """Deterministic checks executed in the sandbox, for one language's toolchain.

    One adapter is one toolchain: the scanners, the test runner and the
    coverage reader of one language, each invoked with arguments fixed in
    code. Which adapter judges a workspace is decided in ``adapters.scanners``
    (the registry and the language router), so a deployment that verifies both
    Java and Python repositories is a matter of configuration, not of a fork.
    """

    #: The language this toolchain serves. The router keys on it, and the
    #: evidence pack records it as the toolchain that judged the change.
    language: str = "any"
    #: Static tools a ``tool`` rule may bind to through this adapter. Declared
    #: in code and never read from configuration: a rule file is part of the
    #: tree under verification, and a tree that could name a new subprocess
    #: would hold an execution entry point into its own verifier.
    supported_tools: tuple[str, ...] = ()
    #: Files whose presence at a workspace root say this toolchain applies
    #: (``pom.xml``, ``pyproject.toml``). Read by the router's auto-detection.
    markers: tuple[str, ...] = ()

    def toolchain(self, workspace: Path) -> str:
        """Which toolchain judges ``workspace``, worded for the evidence pack.

        Concrete because a single-language adapter has one answer. The router
        overrides it to say how the choice was made -- forced by configuration,
        recognised from a marker, or a default nobody chose -- because "the
        wrong toolchain ran" is a failure the evidence has to make visible.
        """

        return self.language

    def language_at(self, workspace: Path) -> str:
        """The bare language name of the toolchain that judges ``workspace``.

        ``toolchain`` is prose for a reader; this is the same fact for code --
        impact analysis reads sources with one language's parser, and generated
        tests are written where that language keeps its tests. Splitting them
        keeps a caller from parsing "java (detected from pom.xml)" back into a key.
        """

        return self.language

    def image(self, workspace: Path) -> str | None:
        """The container image this toolchain runs its tools in, if it names one.

        Recorded in the evidence pack's configuration snapshot: a verdict is
        only reproducible in the environment that produced it, and for a
        containerised toolchain the image *is* that environment. The default is
        ``None`` rather than the deployment's own default image -- an adapter
        that does not know what it runs in must not put one in the evidence
        that nothing ever ran in.
        """

        return None

    def unselectable(self, workspace: Path) -> str | None:
        """Why ``run_tests`` here runs the whole suite whatever it is asked, or ``None``.

        A selection the toolchain silently ignores is worse than none: the
        evidence would say "only the impacted tests ran" over a full run, and
        a run of generated cases would credit them with the whole suite's
        counts. Callers ask first and, given a reason, do not select at all.
        The reason is a clause the evidence quotes.
        """

        return None

    @abstractmethod
    def run_static_checks(
        self,
        workspace: Path,
        change: ChangeRequest,
        tools: Sequence[str] | None = None,
        scopes: Mapping[str, Sequence[str]] | None = None,
    ) -> list[ToolReport]:
        """Run scanners over the change and report one result per tool.

        ``tools=None`` asks for the implementation's full set; an explicit
        sequence asks for exactly those. The caller selects, because which
        scanners a change must survive is the team's rule library talking,
        not a default buried in an adapter.

        ``scopes`` narrows one tool to the changed files the rules that bound
        it actually cover (``RuleSet.tool_scopes``). A tool absent from the
        mapping is asked about every changed file it can read, which is what a
        caller with no rule library to consult wants. A tool present with an
        empty scope has nothing to check -- and that is a tool reporting
        "nothing in scope", never a tool reporting a pass. An implementation
        that scans the repository rather than a file list (semgrep needs the
        whole tree to resolve cross-file rules) may ignore it, and says so.
        """

        raise NotImplementedError

    @abstractmethod
    def run_tests(self, workspace: Path, selection: list[str] | None = None) -> TestResults:
        """Run the repository's suite on ``workspace`` and read the counts.

        A suite that never started must come back with ``available=False``,
        not as zero failures: the gate reads that as missing evidence.
        """

        raise NotImplementedError

    @abstractmethod
    def measure_coverage(
        self,
        workspace: Path,
        lines: Mapping[str, set[int]] | None = None,
    ) -> CoverageResult:
        """Read the line coverage the suite left behind.

        Returns the figure in [0, 1] when there is one, and otherwise says
        which kind of "none" it is -- tool not installed, or tool ran and found
        no data -- because the gate blocks on both and the evidence pack has to
        name the right one.

        ``lines`` are the change's added lines per file. When given, the
        adapter also fills ``CoverageResult.changed_lines_*`` from whatever
        per-line data its tool produces, and leaves them ``None`` when the tool
        produced none -- an unmeasured bar is missing evidence like any other,
        never a bar that was met.
        """

        raise NotImplementedError


class LedgerStore(Protocol):
    """The durable side: checkpoints, evidence, approvals, run accounting."""

    def save_state(self, state: RunState) -> None: ...

    def load_state(self, run_id: str) -> RunState | None: ...

    def save_evidence(self, run_id: str, evidence: EvidencePack, fingerprint: str) -> None: ...

    def save_approval_ticket(self, run_id: str, ticket: ApprovalTicket) -> None: ...

    def resolve_approval(
        self,
        run_id: str,
        approved: bool,
        reason: str | None = ...,
        reviewer: str | None = ...,
    ) -> None: ...

    def clear_approval(self, run_id: str) -> None: ...

    def load_approval(self, run_id: str) -> dict[str, Any] | None: ...

    def load_evidence(self, run_id: str) -> EvidencePack | None: ...

    def save_annotation(self, annotation: FindingAnnotation) -> None: ...

    def list_annotations(self, run_id: str | None = ...) -> list[FindingAnnotation]: ...

    def save_run_record(self, record: RunRecord) -> None: ...

    def load_run_record(self, run_id: str) -> RunRecord | None: ...

    def save_artifact_manifest(self, run_id: str, manifest: ArtifactManifest) -> None: ...

    def load_artifact_manifest(self, run_id: str) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class AdapterSet:
    ledger: LedgerStore
    forge: ForgeAdapter | None = None
    scanner: ScannerAdapter | None = None
    #: ``None`` when no model credential is configured. Deterministic rules
    #: still run; judgment-class rules are recorded as unread, which routes the
    #: change to a human rather than passing it.
    llm: LLMAdapter | None = None
