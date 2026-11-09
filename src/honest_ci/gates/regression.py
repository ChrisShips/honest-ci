"""The regression gate: the repository's own suite, run on the change.

What the gate is for, in the product's terms: **once the change is in, does the
existing suite still pass, and has coverage dropped**. It runs the suite the
repository already has, in the sandbox, on the tree that holds the change, and
blocks when a test fails or line coverage sits below the configured bar.

Four decisions shape this file:

* **Tests run on the tree that holds the change, or they do not run.** A suite
  that passes on a tree without the change says nothing about the change.
  Before spending a sandbox the gate checks that the lines the diff added are
  where the diff says they are; when they are not, it hands the change to a
  human instead of reporting on the wrong code. A git working tree and a
  materialised merge request both pass this check; a patch judged against an
  unrelated tree does not.
* **No tests is no evidence.** A suite that collected nothing verified
  nothing, so the gate routes to a human and says so -- the reading the review
  gate gives an empty rule library. A suite that could not start is missing
  evidence of a different kind, and the gate template turns that into a block
  (fail closed).
* **Coverage is asked twice: of the repository, and of the change.** "Is the
  repository's line coverage at or above the bar" is the question ``make
  check`` asks here too; "were the lines this change added actually executed"
  is the one a reviewer asks, and a repository at 90% can take an untested
  function without moving that number. Each bar is configured on its own and a
  bar of 0 switches its check off, explicitly and auditably. A bar the
  deployment cannot measure -- no ``coverage`` module, a test command that
  leaves no data, a changed file absent from the report -- blocks rather than
  passes.
* **Selecting fewer tests never decides anything.** ``HONEST_CI_TEST_SELECTION=
  impacted`` asks ``impact.py`` which tests reach the change and runs those,
  and every way that can go wrong -- no reader for the language, a file that
  would not parse, an empty selection -- runs the whole suite instead. The
  evidence says which of the two happened, because "the impacted tests passed"
  is weaker evidence than "the suite passed" and a reader must not have to
  guess which they were given.

The gate does not know which language it is testing. The scanner port runs the
suite and reads the counts -- pytest's summary line, or the JUnit XML any other
runner leaves behind -- and ``TestResults.runner`` says which, because one exit
code (pytest's 5, "nothing collected") means something only for one runner. The
same holds for impacted selection: ``ScannerAdapter.language_at`` names the
toolchain's language so ``impact.py`` reads the right sources, and the selection
it produces is already in the vocabulary ``run_tests`` accepts.

Generated regression cases (``TestResults.generated``) are ``testgen.py``'s, in
the language ``impact.py`` was given. The gate asks for them only once the suite
and the coverage bars have passed, and a failing one sends the change to a
person, never to a block.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .. import impact
from ..contracts import (
    ChangeRequest,
    CoverageResult,
    Finding,
    GateName,
    GateReport,
    TestResults,
    ToolDiagnostic,
    ToolReport,
    Verdict,
    WorkspaceKind,
    compute_fingerprint,
)
from ..diffs import (
    AddedLine,
    added_lines,
    is_safe_relative_path,
    markdown_code,
    markdown_code_block,
    markdown_path,
    split_lines,
)
from ..impact import ImpactAnalysis
from ..settings import Settings
from ..state import RunState
from ..testgen import MAX_GENERATED_FAILURES, GeneratedTests, propose_and_run
from .base import Gate

#: Failing tests get one finding each up to here; the rest are counted in the
#: summary. Two hundred findings do not make a more actionable block.
_MAX_FAILURE_FINDINGS = 20

#: Uncovered changed lines listed in the evidence. The finding names the
#: first few so the reader has somewhere to start; the count is the number
#: that decides, and it is carried in the metrics whole.
_MAX_UNCOVERED_LISTED = 20

#: Files larger than this are not read for the tree check. A generated bundle
#: is not what a regression suite is about, and reading it proves nothing.
_MAX_CHECKED_BYTES = 512 * 1024

#: pytest's "no tests collected". Not a failure of the tool, and not evidence.
_PYTEST_NO_TESTS = 5

#: The one failure a generated run is given when its exit code says it failed
#: and it named no failing test (see ``_read_generated``).
_GENERATED_UNREADABLE = "ERROR (no test result could be read from the generated tests' run)"

#: The tool name the suite is filed under in the evidence, by what read it.
_SUITE_TOOL_NAMES = {"pytest": "pytest", "junit": "junit", "exit-code": "test-command"}

#: Why no regression cases were generated, per verdict that concluded before
#: generation was asked for. ``PASS`` is in here too: the passing paths that
#: stop early are the ones missing evidence, which the gate template turns into
#: a block afterwards -- nothing was generated because nothing was verifiable.
_UNREACHED_REASONS: dict[Verdict, str] = {
    Verdict.BLOCKED: "an earlier check already blocked the change, so generation was never reached",
    Verdict.NEEDS_HUMAN: (
        "an earlier check already sent the change to a person, so generation was never reached"
    ),
    Verdict.PASS: "an earlier check had no evidence, so generation was never reached",
    Verdict.SKIPPED: "this gate did not run, so generation was never reached",
}

TreeStatus = Literal["present", "absent", "unverifiable"]


@dataclass(frozen=True)
class TreeCheck:
    """Whether the workspace holds the change the gate is about to test.

    ``present``: every file the check could read has the diff's added lines at
    the positions the diff gives, and every deleted file is gone. ``absent``:
    at least one does not. ``unverifiable``: the change carries nothing the
    check can look for (no diff, binary files only), so the gate proceeds and
    the evidence says the check could not be made.
    """

    status: TreeStatus
    checked: int
    mismatched: tuple[str, ...] = ()


class RegressionGate(Gate):
    name: GateName = "regression"

    def evaluate(self, state: RunState) -> GateReport:
        workspace = self._workspace(state)
        tree = check_tree(workspace, state.change)
        settings = self.services.settings
        threshold = settings.coverage_threshold
        changed_bar = settings.changed_line_coverage_threshold
        frame = _Frame(
            workspace_kind=self._workspace_kind(state),
            tree=tree,
            threshold=threshold,
            changed_threshold=changed_bar,
        )
        if tree.status == "absent":
            return frame.report(Verdict.NEEDS_HUMAN, _absent_summary(tree))

        scanner = self.services.scanner
        if scanner is None:
            # No adapter at all. Reported as an unavailable tool so the gate
            # template turns it into a block, rather than a suite that never
            # existed being read as a suite that passed.
            frame.tools.append(_no_scanner("tests"))
            return frame.report(
                Verdict.PASS,
                "No scanner is configured in this deployment, so the regression tests did not run.",
            )

        frame.toolchain = scanner.toolchain(workspace)
        frame.impact = impact.analyse(workspace, state.change, scanner.language_at(workspace))
        frame.selection = _selection(settings, frame.impact, scanner.unselectable(workspace))
        results = scanner.run_tests(workspace, frame.selection.tests)
        frame.results = results
        frame.tools.append(_suite_tool_report(results))
        if not results.available:
            # Missing evidence; ``_enforce_fail_closed`` makes it a block.
            return frame.report(
                Verdict.PASS, f"The regression tests did not start: {_first_line(results)}."
            )
        if results.failed or results.errors:
            frame.findings = _failure_findings(results)
            return frame.report(Verdict.BLOCKED, _failed_summary(results))
        if not _ran_clean(results):
            frame.findings = [_opaque_finding(results)]
            return frame.report(Verdict.BLOCKED, _opaque_summary(results))
        if results.collected == 0:
            return frame.report(Verdict.NEEDS_HUMAN, _no_tests_summary(results))

        if not settings.coverage_measured:
            return self._passed(
                state,
                frame,
                f"{_passed_phrase(results)}; coverage not measured, because "
                "HONEST_CI_COVERAGE_THRESHOLD and HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD "
                "are both 0.",
            )
        # Asked for whenever anything is measured at all, and judged only when
        # a bar was configured: a deployment deciding whether to turn the bar on
        # should be able to read the number its own changes are producing first.
        coverage = scanner.measure_coverage(workspace, impact.changed_lines_for(state.change))
        frame.coverage = coverage
        frame.tools.append(_coverage_tool_report(coverage))
        bar = _pct(threshold)
        if not coverage.available:
            return frame.report(
                Verdict.PASS,
                f"{_passed_phrase(results)}; coverage is not available, so the coverage "
                f"threshold ({_bar_phrase(threshold, changed_bar)}) could not be checked.",
            )
        if threshold > 0:
            if coverage.ratio is None:
                frame.findings = [_unmeasurable_finding(coverage, threshold)]
                return frame.report(
                    Verdict.BLOCKED,
                    f"{_passed_phrase(results)}; coverage could not be measured, so the "
                    f"{bar} threshold could not be checked.",
                )
            if coverage.ratio < threshold:
                frame.findings = [_below_bar_finding(coverage.ratio, threshold)]
                return frame.report(
                    Verdict.BLOCKED,
                    f"{_passed_phrase(results)}; line coverage {_pct(coverage.ratio)}"
                    f" is below the {bar} threshold.",
                )
        if changed_bar > 0:
            verdict, clause, findings = _changed_line_verdict(coverage, changed_bar)
            frame.findings = findings
            if verdict is not Verdict.PASS:
                return frame.report(verdict, f"{_passed_phrase(results)}; {clause}")
        return self._passed(state, frame, _passed_summary(frame, results))

    def _passed(self, state: RunState, frame: _Frame, summary: str) -> GateReport:
        """The existing suite is green. Ask for regression cases, then conclude.

        Only from here: generation is about a change whose own suite already
        passes. A blocked change has a failure to fix first, and spending a
        model call and a second suite run to add a question to it would put
        noise on top of a conclusion the developer already has.
        """

        generated = _read_generated(self._generate(state, frame))
        frame.generated = generated
        if generated.results is not None and not generated.results.available:
            # Not a ToolReport: an advisory run that could not start must not
            # reach ``_enforce_fail_closed``, which would turn "we could not
            # generate" into a block. It is recorded as a limit instead.
            return frame.report(Verdict.PASS, _with_clause(summary, _generation_note(generated)))
        if generated.failed:
            frame.findings = [*frame.findings, _generated_failure_finding(generated)]
            # At most a hand-off, and only because a person has to decide which
            # of the two is wrong -- the generated case, or the code it fails
            # on. ``Finding.blocking`` is false for a model finding, so this
            # cannot become a block by any other route either.
            return frame.report(
                Verdict.NEEDS_HUMAN, _with_clause(summary, _generation_note(generated))
            )
        if generated.ran:
            return frame.report(Verdict.PASS, _with_clause(summary, _generation_note(generated)))
        return frame.report(Verdict.PASS, summary)

    def _generate(self, state: RunState, frame: _Frame) -> GeneratedTests:
        """Regression cases for what this change touched, run once and removed."""

        scanner = self.services.scanner
        settings = self.services.settings
        if scanner is None or frame.impact is None:
            return GeneratedTests(skipped_reason="there is no toolchain to run them")
        if frame.tree.status != "present":
            # The cases are written against the change; a tree nobody could
            # confirm holds it is a tree they would be written against blind.
            return GeneratedTests(
                skipped_reason="whether the workspace holds this change could not be checked"
            )
        return propose_and_run(
            change=state.change,
            symbols=frame.impact.changed_symbols,
            # The same fingerprint an approval binds to, so cases proposed
            # against another commit cannot be mistaken for cases about this one.
            fingerprint=compute_fingerprint(
                state.change.head_sha,
                settings.policy_version,
                self.services.rule_set.version,
            ),
            language=frame.impact.language,
            workspace=self._workspace(state),
            scanner=scanner,
            llm=self.services.llm,
            settings=settings,
            artifacts=self.services.artifact_store,
        )


# ── the tree check ───────────────────────────────────────────────────────────


def check_tree(workspace: Path, change: ChangeRequest) -> TreeCheck:
    """Does ``workspace`` hold ``change``? Read the added lines and look.

    Public because the answer matters beyond this gate: any check that runs
    over a tree rather than over the diff has the same question to ask first.
    """

    root = workspace.resolve()
    mismatched: list[str] = []
    checked = 0
    for path, lines in added_lines(change.diff).items():
        if not lines or not is_safe_relative_path(path):
            continue
        holds = _holds_lines(workspace / path, root, lines)
        if holds is None:
            continue
        checked += 1
        if not holds:
            mismatched.append(path)
    # A rename leaves its old name behind just as a deletion does, and a rename
    # that changed no line has nothing else here to look for: without this a
    # tree still on the base commit held it. Unless another file listed in
    # the change took that name over.
    listed = {item.path for item in change.files_changed}
    gone = [item.path for item in change.files_changed if item.change_type == "deleted"]
    gone += [
        item.previous_path
        for item in change.files_changed
        if item.previous_path and item.previous_path not in listed
    ]
    for path in gone:
        if not is_safe_relative_path(path):
            continue
        checked += 1
        if (workspace / path).exists():
            mismatched.append(path)
    if not checked:
        return TreeCheck(status="unverifiable", checked=0)
    return TreeCheck(
        status="absent" if mismatched else "present",
        checked=checked,
        mismatched=tuple(mismatched),
    )


def _holds_lines(target: Path, root: Path, lines: list[AddedLine]) -> bool | None:
    """``None`` when the file cannot be read for the check; else the answer."""

    try:
        resolved = target.resolve()
        if not resolved.is_relative_to(root):
            return False
        if not resolved.is_file():
            return False
        if resolved.stat().st_size > _MAX_CHECKED_BYTES:
            return None
        # Bytes, then git's line breaks: ``read_text`` would turn a lone "\r"
        # into a line break and ``splitlines`` a form feed, and from there on
        # the file's line numbers stop matching its own diff's.
        content = split_lines(resolved.read_bytes().decode("utf-8", errors="replace"))
    except OSError:
        return False
    for item in lines:
        if item.line > len(content) or content[item.line - 1].rstrip() != item.text.rstrip():
            return False
    return True


# ── report assembly ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Selection:
    """Which tests the gate asked the toolchain to run, and why those.

    ``tests is None`` is the whole suite. The distinction from an empty list
    matters more than it looks: a selection nobody could narrow must run
    everything, because "0 tests were selected, so nothing failed" is not a
    conclusion about the change.
    """

    mode: str = "all"
    tests: list[str] | None = None
    #: Why an ``impacted`` request ended up running everything anyway. Empty
    #: when nothing had to fall back.
    fallback: str = ""

    @property
    def note(self) -> str:
        """The one clause every verdict of this run carries about its scope."""

        if self.mode == "impacted" and self.tests is not None:
            return (
                f"only the {_counted(len(self.tests), 'impacted test')} ran"
                " (HONEST_CI_TEST_SELECTION=impacted), which is weaker evidence than the full suite"
            )
        if self.fallback:
            return (
                "only the impacted tests were asked for (HONEST_CI_TEST_SELECTION=impacted), "
                f"but {self.fallback}, so the full suite ran"
            )
        return ""


@dataclass
class _Frame:
    """Everything one evaluation has gathered so far, in the order it happened."""

    #: Which tree the suite ran in, as the pack names it. The path itself is
    #: host-specific and the pack leaves the host, so only the kind travels;
    #: the tree the gate is working on lives in ``evaluate``'s own scope.
    workspace_kind: WorkspaceKind
    tree: TreeCheck
    threshold: float
    changed_threshold: float = 0.0
    #: Which toolchain ran the suite and how it was chosen, from the port.
    toolchain: str | None = None
    impact: ImpactAnalysis | None = None
    selection: _Selection = field(default_factory=_Selection)
    results: TestResults | None = None
    coverage: CoverageResult | None = None
    #: What came of asking for regression cases -- ``None`` until somebody
    #: asks. Not an empty ``GeneratedTests``, because that reads as "asked,
    #: got nothing, no reason given": every path that concludes before
    #: generation would then render as "none generated (no reason recorded)",
    #: which is a gap in the evidence rather than what happened.
    generated: GeneratedTests | None = None
    tools: list[ToolReport] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def report(self, verdict: Verdict, summary: str) -> GateReport:
        generated = (
            self.generated if self.generated is not None else _unreached_generation(verdict)
        )
        results = _with_generated(self.results, generated)
        # Scope travels with every verdict, not only the passing ones: "the
        # impacted tests failed" and "the whole suite failed" are different
        # facts, and the reader of a block needs to know which one this is.
        note = self.selection.note
        metrics: dict[str, float] = {
            "tests_passed": float(results.passed),
            "tests_failed": float(results.failed),
            "tests_errors": float(results.errors),
            "tests_skipped": float(results.skipped),
            "tests_collected": float(results.collected),
            "coverage_threshold": float(self.threshold),
            "changed_line_coverage_threshold": float(self.changed_threshold),
            "findings_total": float(len(self.findings)),
            "findings_blocking": float(len([f for f in self.findings if f.blocking])),
            "tools_run": float(len([t for t in self.tools if t.available])),
            "tools_unavailable": float(len([t for t in self.tools if not t.available])),
        }
        if self.impact is not None:
            metrics["changed_symbols"] = float(len(self.impact.changed_symbols))
            metrics["impacted_tests"] = float(len(self.impact.impacted_tests))
        if self.coverage is not None and self.coverage.ratio is not None:
            metrics["coverage"] = self.coverage.ratio
        if generated.ran:
            metrics["generated_tests"] = float(results.generated)
            metrics["generated_tests_failed"] = float(results.generated_failed)
        if self.coverage is not None and self.coverage.changed_lines_total is not None:
            metrics["changed_lines_total"] = float(self.coverage.changed_lines_total)
            metrics["changed_lines_covered"] = float(self.coverage.changed_lines_covered or 0)
            ratio = self.coverage.changed_lines_ratio
            if ratio is not None:
                metrics["changed_line_coverage"] = ratio
        return GateReport(
            gate="regression",
            verdict=verdict,
            summary=_with_clause(summary, note) if note else summary,
            findings=self.findings,
            tool_reports=self.tools,
            test_results=results if self.results is not None else None,
            metrics=metrics,
            verifier_report=self._verifier_report(generated),
        )

    def _verifier_report(self, generated: GeneratedTests) -> dict[str, Any]:
        coverage: dict[str, Any] | None = None
        if self.coverage is not None:
            coverage = {
                "available": self.coverage.available,
                "ratio": self.coverage.ratio,
                "exit_code": self.coverage.exit_code,
                "command": self.coverage.command,
            }
            if self.coverage.changed_lines_total is not None or self.changed_threshold > 0:
                coverage["changed_lines"] = {
                    # 0 is "recorded, not judged": the numbers are here either
                    # way, and this says which of the two this run was.
                    "threshold": self.changed_threshold,
                    "total": self.coverage.changed_lines_total,
                    "covered": self.coverage.changed_lines_covered,
                    "ratio": self.coverage.changed_lines_ratio,
                    "uncovered": [
                        f"{path}:{line}"
                        for path, line in self.coverage.uncovered[:_MAX_UNCOVERED_LISTED]
                    ],
                    "no_coverage_data": list(self.coverage.unmapped_sources),
                    "not_measured": list(self.coverage.out_of_scope_sources),
                }
        report: dict[str, Any] = {
            "workspace_kind": self.workspace_kind,
            "toolchain": self.toolchain,
            "tree_check": {
                "status": self.tree.status,
                "checked": self.tree.checked,
                "mismatched": list(self.tree.mismatched),
            },
            "test_command": self.results.command if self.results else None,
            "test_runner": self.results.runner if self.results else None,
            "exit_code": self.results.exit_code if self.results else None,
            "test_selection": {
                "mode": self.selection.mode,
                "requested": self.selection.tests,
                "fallback": self.selection.fallback or None,
            },
            "coverage_threshold": self.threshold,
            "coverage": coverage,
        }
        if self.impact is not None:
            report["impact"] = self.impact.as_dict()
        report["generated_tests"] = generated.as_dict()
        return report


def _selection(
    settings: Settings, analysis: ImpactAnalysis, unselectable: str | None
) -> _Selection:
    """Which tests to ask for, and every reason a narrowed run runs everything.

    Narrowing the suite is an optimisation, never a verdict. Five things put
    the whole suite back, and each of them is recorded rather than assumed
    away: a toolchain this build has no impact reader for, a changed file the
    reader could not parse -- its tests are unknown, not absent -- a
    selection that came out empty, a test command the toolchain cannot narrow,
    and an overall coverage bar. "0 tests were selected, so nothing failed"
    is not a statement about the change.

    The overall bar asks about the repository, and a narrowed run measures
    only what its own tests reached: with coverage's ``source`` set, every
    file they never import counts as missed; without it, those files are not
    counted at all, and a repository at 45% reads 100%. Neither number is the
    one the bar is about. The changed-line bar has no such problem -- fewer
    tests can only execute fewer of the changed lines -- so it does not
    force the whole suite.
    """

    if settings.test_selection != "impacted":
        return _Selection()
    if not analysis.analysed:
        return _Selection(fallback=f"the {analysis.language} toolchain has no impact analysis yet")
    if analysis.unanalysed_files:
        return _Selection(
            fallback=f"the changed symbols in {_named(analysis.unanalysed_files)} could not be "
            "read, which would leave the impacted tests incomplete"
        )
    if not analysis.impacted_tests:
        return _Selection(fallback="no impacted test was selected")
    if unselectable:
        return _Selection(fallback=unselectable)
    if settings.coverage_threshold > 0:
        return _Selection(
            fallback="the overall coverage threshold (HONEST_CI_COVERAGE_THRESHOLD) is a figure "
            "about the whole suite, which a run of the impacted tests cannot measure"
        )
    return _Selection(mode="impacted", tests=list(analysis.impacted_tests))


def _changed_line_verdict(
    coverage: CoverageResult, bar: float
) -> tuple[Verdict, str, list[Finding]]:
    """Hold the changed-line bar against what the toolchain could measure.

    Returns ``PASS`` with an empty clause when the bar is met or does not
    apply; the passing wording is assembled once, in ``_passed_summary``. Every
    other answer is a block, including the two shapes of "nobody measured it":
    a bar that cannot be checked is not a bar that passed.
    """

    pct = _pct(bar)
    if coverage.changed_lines_total is None or coverage.changed_lines_covered is None:
        return (
            Verdict.BLOCKED,
            f"changed-line coverage was not measured, so the {pct} threshold could not be checked.",
            [_changed_unmeasurable_finding(coverage, bar)],
        )
    if coverage.unmapped_sources:
        verb = "has" if len(coverage.unmapped_sources) == 1 else "have"
        return (
            Verdict.BLOCKED,
            f"{_named(coverage.unmapped_sources)} {verb} no coverage data, so the {pct} "
            "changed-line coverage threshold could not be checked.",
            [_no_coverage_data_finding(coverage, bar)],
        )
    ratio = coverage.changed_lines_ratio
    if ratio is None or ratio >= bar:
        # ``None`` here is a change that added no executable line at all --
        # documentation, comments, a moved import. There is nothing to cover.
        return (Verdict.PASS, "", [])
    return (
        Verdict.BLOCKED,
        f"changed-line coverage {_pct(ratio)}"
        f" ({coverage.changed_lines_covered}/{coverage.changed_lines_total} lines)"
        f" is below the {pct} threshold.",
        [_changed_below_bar_finding(coverage, bar)],
    )


def _read_generated(generated: GeneratedTests) -> GeneratedTests:
    """The generated run, read by its exit code as well as by its counts.

    A model-written Java case that does not compile stops Maven before
    Surefire: exit code 1, no report, nothing counted. Read by its counts
    alone that is "0 ran, none failed". The exit code says the run failed, so
    it is counted as one error -- the reading pytest already gives a
    generated file that does not import -- and every number downstream
    agrees with the hand-off it causes.
    """

    results = generated.results
    if results is None or not results.available or generated.failed or _ran_clean(results):
        return generated
    return replace(
        generated,
        results=results.model_copy(update={"errors": 1, "failures": [_GENERATED_UNREADABLE]}),
    )


def _with_generated(results: TestResults | None, generated: GeneratedTests) -> TestResults:
    """The suite's own counts, with the generated ones alongside — never inside.

    ``passed`` and ``failed`` stay exactly what the repository's suite
    produced. A model-written case that fails is counted in ``generated_failed``
    and nowhere else, so no reader and no arithmetic can turn it into a failing
    suite.
    """

    base = results or TestResults()
    if not generated.ran or generated.results is None:
        return base
    ran = generated.results
    return base.model_copy(
        update={
            "generated": ran.collected,
            "generated_failed": ran.failed + ran.errors,
            "generated_failures": list(ran.failures[:MAX_GENERATED_FAILURES]),
        }
    )


def _ran_clean(results: TestResults) -> bool:
    """Whether the command's exit code says it did its job.

    pytest's 5 (nothing collected) is not the tool failing: the command did
    what it was asked and had nothing to run, which the gate reads as no
    evidence. No other runner gets that reading -- an unfamiliar non-zero code
    from a custom command is opaque, and opaque blocks.
    """

    if results.exit_code in (0, None):
        return True
    return results.runner == "pytest" and results.exit_code == _PYTEST_NO_TESTS


def _suite_tool_report(results: TestResults) -> ToolReport:
    name = _SUITE_TOOL_NAMES.get(results.runner, results.runner)
    return ToolReport(
        name=name,
        command=results.command or f"{name} (not executed)",
        available=results.available,
        passed=results.available and _ran_clean(results),
        exit_code=results.exit_code,
        stdout=results.stdout,
        stderr=results.stderr,
        duration_s=results.duration_s,
        report_path=results.report_path,
        findings=list(results.failures[:_MAX_FAILURE_FINDINGS]),
        diagnostics=[
            ToolDiagnostic(tool=name, file=_failure_file(line), message=line)
            for line in results.failures
        ],
    )


def _coverage_tool_report(coverage: CoverageResult) -> ToolReport:
    measured = coverage.available and coverage.ratio is not None
    note = coverage.stderr or coverage.stdout or "coverage produced no figure"
    return ToolReport(
        name="coverage",
        command=coverage.command or "coverage (not executed)",
        available=coverage.available,
        passed=measured,
        exit_code=coverage.exit_code,
        stdout=coverage.stdout,
        stderr=coverage.stderr,
        duration_s=coverage.duration_s,
        findings=[] if measured else [note.strip()],
    )


def _no_scanner(name: str) -> ToolReport:
    reason = "No ScannerAdapter is configured in this deployment."
    return ToolReport(
        name=name,
        command=f"{name} (no scanner adapter configured)",
        available=False,
        passed=False,
        stderr=reason,
        findings=[reason],
    )


# ── findings ─────────────────────────────────────────────────────────────────


def _failure_findings(results: TestResults) -> list[Finding]:
    findings: list[Finding] = []
    for index, line in enumerate(results.failures[:_MAX_FAILURE_FINDINGS], start=1):
        kind, _, rest = line.partition(" ")
        location, _, _reason = rest.partition(" - ")
        _, _, test = location.partition("::")
        # A parametrised test's id and its assertion message are the change's
        # own text, and the title is Markdown in the merge request note.
        title = (
            f"Regression test failed: {markdown_code(test or location)}"
            if kind == "FAILED"
            else f"Error collecting or running tests: {markdown_code(location)}"
        )
        findings.append(
            Finding(
                id=f"regression.test.{index}",
                gate="regression",
                severity="high",
                title=title,
                detail=markdown_code(line),
                file=_failure_file(line),
                decided_by="deterministic",
                suggested_fix=(
                    "Fix the failing test or the code it covers. If the test itself is out "
                    "of date, update it too and say why in the commit message."
                ),
            )
        )
    if not findings:
        # The counts say something failed but no summary line parsed (a test
        # command that reports counts differently). The block still names what
        # it knows, and the tail of the output is where the reason will be.
        findings.append(
            Finding(
                id="regression.suite",
                gate="regression",
                severity="high",
                title="Regression tests failed",
                detail=(
                    f"{results.failed} failed · {_counted(results.errors, 'error')}, "
                    "but no failing test could be read from the output. "
                    "The end of the output:\n" + _tail_lines(results)
                ),
                decided_by="deterministic",
            )
        )
    return findings


def _opaque_finding(results: TestResults) -> Finding:
    return Finding(
        id="regression.suite",
        gate="regression",
        severity="high",
        title="The test command failed, but no failing test could be read",
        detail=(
            f"The test command ended with exit code {results.exit_code} and no failing test "
            "could be read from its output, so whether the regression tests passed is unknown. "
            "The end of the output:\n" + _tail_lines(results)
        ),
        decided_by="deterministic",
        suggested_fix=(
            "Get the test command to run cleanly on your machine first. If the command is not "
            "pytest, point HONEST_CI_TEST_REPORT at the JUnit XML it leaves, so the gate can "
            "read the tests."
        ),
    )


#: Where a Python command of a team's own has to write its coverage data. The
#: run points COVERAGE_FILE at a new file, so a command that writes elsewhere --
#: under a wrapper that drops the variable, say -- leaves nothing to read.
_COVERAGE_FILE_NOTE = (
    ", writing to the COVERAGE_FILE the run sets, which a wrapper that filters the "
    "environment has to pass through"
)


def _unmeasurable_finding(coverage: CoverageResult, threshold: float) -> Finding:
    return Finding(
        id="regression.coverage",
        gate="regression",
        severity="high",
        title="Coverage could not be measured",
        detail=(
            f"coverage ended with exit code {coverage.exit_code} and gave no coverage figure, "
            f"so the {_pct(threshold)} threshold could not be checked." + _coverage_tail(coverage)
        ),
        decided_by="deterministic",
        suggested_fix=(
            "Make the test command collect coverage (the default command runs under "
            f"coverage run; a custom HONEST_CI_TEST_COMMAND has to add it{_COVERAGE_FILE_NOTE}), "
            "or set HONEST_CI_COVERAGE_THRESHOLD to 0 to switch the coverage check off "
            "explicitly."
        ),
    )


def _coverage_tail(coverage: CoverageResult) -> str:
    """The end of what coverage printed, as a code block on a line of its own.

    Coverage runs the suite, so what it prints is the change's tests talking,
    and a failing one prints what it likes.
    """

    tail = (coverage.stderr or coverage.stdout or "").strip()[-600:]
    return "\n" + markdown_code_block(tail) if tail else ""


def _below_bar_finding(ratio: float, threshold: float) -> Finding:
    return Finding(
        id="regression.coverage",
        gate="regression",
        severity="high",
        title="Coverage is below the threshold",
        detail=(
            f"Line coverage is {_pct(ratio)}; the threshold is {_pct(threshold)} "
            "(HONEST_CI_COVERAGE_THRESHOLD)."
        ),
        decided_by="deterministic",
        suggested_fix=(
            "Add tests for this change. Changing the threshold itself is a change to the "
            "configuration, and an auditable one."
        ),
    )


def _generated_failure_finding(generated: GeneratedTests) -> Finding:
    """A model-written case failed. Which side is wrong is a person's call."""

    results = generated.results
    failures = list(results.failures[:MAX_GENERATED_FAILURES]) if results else []
    if results is not None and _GENERATED_UNREADABLE in failures:
        # Nothing names a failing test, so the reason is in what the run
        # printed: for a case that did not compile, the compiler's message.
        outcome = (
            f"their run ended with exit code {results.exit_code} without a test result anyone "
            "could read, so whether they pass is unknown"
        )
        listed = "The end of the output:\n" + _output_tail(results)
    else:
        outcome = f"{generated.failed} did not pass"
        # The model named these cases, after reading the diff.
        listed = (
            "\n".join(f"- {markdown_code(line)}" for line in failures)
            or "- (no test name could be read)"
        )
    return Finding(
        id="regression.generated",
        gate="regression",
        severity="medium",
        title="A model-written regression test failed: the test may be wrong, or the code",
        detail=(
            f"This change got {_counted(generated.proposed, 'generated regression test')}, and "
            f"{outcome}. The existing suite passed, so this is not "
            "evidence of a regression but a question for a person: first check whether the "
            "behaviour the test asserts is right.\n"
            f"{listed}\nThe generated test text is kept in the artifacts: "
            + ", ".join(generated.artifacts)
        ),
        decided_by="model",
        confidence=0.5,
        suggested_fix=(
            "If the test is wrong, drop it (it is not in the repository; this run already "
            "deleted it). If the code is wrong, fix the code and add the test to the test "
            "directory for good."
        ),
    )


def _changed_unmeasurable_finding(coverage: CoverageResult, bar: float) -> Finding:
    return Finding(
        id="regression.changed-coverage",
        gate="regression",
        severity="high",
        title="Changed-line coverage could not be measured",
        detail=(
            f"The coverage tool gave no per-line data, so the {_pct(bar)} changed-line coverage "
            "threshold (HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD) could not be checked."
            + _coverage_tail(coverage)
        ),
        decided_by="deterministic",
        suggested_fix=(
            "Make sure the test command collects coverage (Python through coverage"
            f"{_COVERAGE_FILE_NOTE}; Java needs JaCoCo's prepare-agent and report goals), or set "
            "HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD to 0 to switch this check off explicitly."
        ),
    )


def _no_coverage_data_finding(coverage: CoverageResult, bar: float) -> Finding:
    shown = "\n".join(
        f"- {markdown_path(path)}" for path in coverage.unmapped_sources[:_MAX_UNCOVERED_LISTED]
    )
    return Finding(
        id="regression.changed-coverage",
        gate="regression",
        severity="high",
        title="Changed source files have no coverage data",
        detail=(
            "These files changed, but the coverage report does not include them, so whether "
            f"their changed lines were tested is unknown (threshold {_pct(bar)}):\n{shown}"
        ),
        file=coverage.unmapped_sources[0],
        decided_by="deterministic",
        suggested_fix=(
            "Bring these files into what coverage measures (for Python, coverage's "
            "source/include settings; for Java, whether the module takes part in the JaCoCo "
            "report), or add tests for them."
        ),
    )


def _changed_below_bar_finding(coverage: CoverageResult, bar: float) -> Finding:
    shown = "\n".join(
        f"- {markdown_path(path)}:{line}"
        for path, line in coverage.uncovered[:_MAX_UNCOVERED_LISTED]
    )
    more = (
        f"\n(only the first {_MAX_UNCOVERED_LISTED} of {len(coverage.uncovered)} "
        "uncovered lines are listed)"
        if len(coverage.uncovered) > _MAX_UNCOVERED_LISTED
        else ""
    )
    return Finding(
        id="regression.changed-coverage",
        gate="regression",
        severity="high",
        title="Changed-line coverage is below the threshold",
        detail=(
            f"Tests executed {coverage.changed_lines_covered} of the "
            f"{coverage.changed_lines_total} executable lines this change adds; the threshold "
            f"is {_pct(bar)} (HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD). "
            f"The lines no test executed:\n{shown}{more}"
        ),
        file=coverage.uncovered[0][0] if coverage.uncovered else None,
        line_start=coverage.uncovered[0][1] if coverage.uncovered else None,
        decided_by="deterministic",
        suggested_fix=(
            "Add tests that execute the lines above. Enough coverage across the repository "
            "does not mean the code this change touched was tested."
        ),
    )


# ── wording ──────────────────────────────────────────────────────────────────


def _absent_summary(tree: TreeCheck) -> str:
    return (
        f"The workspace does not hold this change ({_named(tree.mismatched)}), so a test run "
        "here says nothing about it; the verdict goes to a person. Run on a checkout that "
        "holds the change: a git working tree, or the checkout run-pr makes."
    )


def _failed_summary(results: TestResults) -> str:
    text = (
        f"Regression tests: {results.passed} passed · {results.failed} failed"
        f" · {_counted(results.errors, 'error')} · {results.skipped} skipped; any failure blocks."
    )
    if len(results.failures) > _MAX_FAILURE_FINDINGS:
        text += (
            f" Only the first {_MAX_FAILURE_FINDINGS} of {len(results.failures)} failures "
            "are listed."
        )
    return text


def _opaque_summary(results: TestResults) -> str:
    return (
        f"The test command ended with exit code {results.exit_code}, but no failing test "
        f"could be read ({results.passed} passed); whether the regression tests passed is "
        "unknown."
    )


def _no_tests_summary(results: TestResults) -> str:
    text = (
        "The test run collected 0 tests, so this change has no regression evidence; "
        "the verdict goes to a person."
    )
    if results.runner == "exit-code":
        text += (
            " The custom test command has no HONEST_CI_TEST_REPORT, so the gate cannot read "
            "how many tests ran: point it at the JUnit XML the command leaves."
        )
    elif results.runner == "junit":
        text += " HONEST_CI_TEST_REPORT matched no JUnit XML, or the report holds no tests."
    return text


def _unreached_generation(verdict: Verdict) -> GeneratedTests:
    """What the pack says about generation when the gate concluded before asking.

    Generation only ever happens on a change whose own suite is already green,
    so most verdicts stop upstream of it. One reason per verdict rather than one
    per return: the summary beside it already names the check that concluded,
    and the reader of "none generated" is asking the narrower question -- was
    generation asked for and declined, or never reached at all.
    """

    return GeneratedTests(skipped_reason=_UNREACHED_REASONS[verdict])


def _generation_note(generated: GeneratedTests) -> str:
    """One clause about the generated cases, for the verdict that carries them."""

    results = generated.results
    if results is None:
        return f"no regression tests were generated ({generated.skipped_reason})"
    if not results.available:
        return (
            f"the generated regression tests did not start ({_first_line(results)}), "
            "so this run has no evidence from them"
        )
    if _GENERATED_UNREADABLE in results.failures:
        return (
            f"the generated regression tests' run ended with exit code {results.exit_code} and "
            "no test result could be read from it; a model-written test does not block, so a "
            "person decides"
        )
    if generated.failed:
        return (
            f"{_counted(results.collected, 'generated regression test')} also ran and "
            f"{generated.failed} did not pass; a model-written test does not block, "
            "so a person decides"
        )
    if not results.collected:
        # A clean exit and nothing counted: a module that never compiled the
        # file, a file with no test in it. "0 ran and none failed" would be
        # true and read as a pass.
        return (
            "the generated regression tests ran, but no test result could be read from them, "
            "so this run has no evidence from them"
        )
    return (
        f"{_counted(results.collected, 'generated regression test')} also ran and none "
        "failed (not counted in the coverage or in the suite's own results)"
    )


def _passed_summary(frame: _Frame, results: TestResults) -> str:
    """The passing summary, one clause per bar that was actually held."""

    coverage = frame.coverage
    parts = [_passed_phrase(results)]
    if frame.threshold > 0 and coverage is not None and coverage.ratio is not None:
        parts.append(
            f"line coverage {_pct(coverage.ratio)} meets the {_pct(frame.threshold)} threshold"
        )
    if frame.changed_threshold > 0 and coverage is not None:
        parts.append(_changed_passed_clause(coverage, frame.changed_threshold))
    return "; ".join(parts) + "."


def _changed_passed_clause(coverage: CoverageResult, bar: float) -> str:
    ratio = coverage.changed_lines_ratio
    if ratio is None:
        return (
            f"this change adds no executable lines, so the {_pct(bar)} changed-line coverage "
            "threshold does not apply"
        )
    return (
        f"changed-line coverage {_pct(ratio)}"
        f" ({coverage.changed_lines_covered}/{coverage.changed_lines_total} lines)"
        f" meets the {_pct(bar)} threshold"
    )


def _passed_phrase(results: TestResults) -> str:
    skipped = f" ({results.skipped} skipped)" if results.skipped else ""
    return f"Regression tests: {results.passed} passed{skipped}"


def _with_clause(summary: str, clause: str) -> str:
    """``summary`` with one more clause, still ending in a single full stop."""

    return f"{summary.removesuffix('.')}; {clause}."


def _first_line(results: TestResults) -> str:
    """The first line the run printed, as code: the summary it goes into is Markdown."""

    for text in (*results.failures, results.stderr, results.stdout):
        stripped = text.strip()
        if stripped:
            return markdown_code(stripped.splitlines()[0])
    return "(no output)"


def _tail_lines(results: TestResults, count: int = 12) -> str:
    """The end of the run's output, as a code block: a failing test prints what it likes."""

    output = results.stderr or results.stdout
    return markdown_code_block("\n".join(output.strip().splitlines()[-count:]))


def _output_tail(results: TestResults, count: int = 20) -> str:
    """The end of both streams, as a code block.

    Both, unlike ``_tail_lines``: Maven prints a compile error on stdout, and
    the JUnit reader's note that no report matched lands on stderr, where on
    its own it would hide the line that says why.
    """

    output = "\n".join(text.strip() for text in (results.stdout, results.stderr) if text.strip())
    return markdown_code_block("\n".join(output.splitlines()[-count:]))


def _failure_file(line: str) -> str | None:
    """The file a failure line names: pytest's ``path::test``, or a bare path.

    A JUnit report names a class instead (``com.acme.OrderTest#testSplit``),
    and a class is not a path in the tree: a finding filed under it would
    point a reader at a file that does not exist. A bare test name with no
    directory and no suffix is left alone for the same reason.
    """

    _, _, rest = line.partition(" ")
    location, _, _ = rest.partition(" - ")
    if "#" in location:
        return None
    file, _, _ = location.partition("::")
    if not file or ("/" not in file and "." not in file):
        return None
    return file if is_safe_relative_path(file) else None


def _bar_phrase(threshold: float, changed_bar: float) -> str:
    """The bars that are switched on, for a message about not checking them."""

    return ", ".join(
        part
        for part in (
            f"{_pct(threshold)} overall" if threshold > 0 else "",
            f"{_pct(changed_bar)} on changed lines" if changed_bar > 0 else "",
        )
        if part
    )


def _named(paths: Sequence[str], shown: int = 3) -> str:
    """The first few paths, as Markdown, and how many more there are."""

    named = ", ".join(markdown_path(path) for path in paths[:shown])
    if len(paths) > shown:
        named += f" and {len(paths) - shown} more"
    return named


def _counted(count: int, noun: str) -> str:
    """``count`` and ``noun``, the noun in the plural unless there is exactly one."""

    return f"{count} {noun}{'' if count == 1 else 's'}"


def _pct(value: float) -> str:
    return f"{value:.0%}"


__all__ = ["RegressionGate", "TreeCheck", "check_tree"]
