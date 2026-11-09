"""The regression gate: the repository's own suite, run on the tree that holds the change."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from honest_ci.adapters.base import AdapterSet
from honest_ci.contracts import (
    ChangedFile,
    ChangeRequest,
    CoverageResult,
    EvidencePack,
    GateReport,
    ProposedTest,
    Verdict,
)

# Aliased so pytest does not try to collect a contract as a test class.
from honest_ci.contracts import TestResults as SuiteResults
from honest_ci.gates import GATE_REGISTRY
from honest_ci.gates.base import Gate
from honest_ci.gates.regression import check_tree
from honest_ci.report import render_evidence, render_forge_comment
from honest_ci.runner import GateRunner
from honest_ci.settings import MODEL_NOT_ENABLED, Settings
from honest_ci.state import RunState, RunStatus
from honest_ci.storage import SQLiteLedger

from .factories import (
    DIFF,
    HOSTILE_OUTPUT,
    HOSTILE_TEXT,
    ScriptedLLM,
    StubScanner,
    cjk_lines,
    make_change,
    markdown_tokens,
    rendered_as_markdown,
)

#: ``src/app/refund.py`` after the factory diff has been applied: the three
#: added lines sit at 11-13, between the context lines the hunk quotes.
REFUND_WITH_CHANGE = "\n".join(
    [
        "from decimal import Decimal",
        "",
        "",
        "class Order:",
        "    items: list",
        "",
        "",
        "def split_refund(order, amount):",
        '    """Split a refund across the order lines."""',
        "    total = 0",
        "    share = float(amount) / len(order.items)",
        "    for item in order.items:",
        "        item.refund = share",
        "    return total",
        "",
    ]
)

PASSING = SuiteResults(passed=3, exit_code=0, command="python -m pytest -q")
MEASURED = CoverageResult(ratio=0.8, command="python -m coverage report --format=total")


def tree(tmp_path: Path, content: str = REFUND_WITH_CHANGE) -> Path:
    target = tmp_path / "src" / "app" / "refund.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return tmp_path


def build(
    tmp_path: Path,
    scanner: StubScanner | None,
    *,
    threshold: float = 0.7,
    changed_threshold: float = 0.0,
    selection: str = "all",
    fail_closed: bool = True,
    llm: ScriptedLLM | None = None,
    testgen: bool = True,
    provider: str | None = None,
) -> Gate:
    runner = GateRunner(
        workspace_root=tmp_path,
        settings=Settings(
            sandbox_mode="process",
            fail_closed=fail_closed,
            coverage_threshold=threshold,
            changed_line_coverage_threshold=changed_threshold,
            test_selection=selection,
            model_testgen=testgen,
            model_provider=provider,
        ),
        adapters=AdapterSet(
            ledger=SQLiteLedger(tmp_path / "ledger.sqlite3"), scanner=scanner, llm=llm
        ),
    )
    return runner.gates["regression"]


def state() -> RunState:
    return RunState(run_id="r1", change=make_change())


# ── deployment ───────────────────────────────────────────────────────────────


def test_the_regression_gate_is_deployed() -> None:
    assert "regression" in GATE_REGISTRY


# ── the tree check ───────────────────────────────────────────────────────────


def test_a_tree_without_the_change_goes_to_a_human_and_runs_nothing(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    gate = build(tree(tmp_path, REFUND_WITH_CHANGE.replace("share = float", "share = int")), scanner)

    report = gate.run(state())

    # A green suite on the wrong tree would be evidence about nothing. The gate
    # does not spend the sandbox, and it says which file gave it away.
    assert report.verdict is Verdict.NEEDS_HUMAN
    assert scanner.tests_run == []
    assert "`src/app/refund.py`" in report.summary
    # And where to run instead: the commands that make a checkout holding it.
    assert "run-pr" in report.summary
    assert report.verifier_report["tree_check"] == {
        "status": "absent",
        "checked": 1,
        "mismatched": ["src/app/refund.py"],
    }
    assert report.test_results is None


def test_a_tree_that_holds_the_change_is_tested(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    gate = build(tree(tmp_path), scanner)

    report = gate.run(state())

    assert scanner.tests_run == [tmp_path]
    assert report.verifier_report["tree_check"]["status"] == "present"


def test_the_check_reads_the_workspace_the_run_recorded(tmp_path: Path) -> None:
    checkout = tree(tmp_path / "checkout")
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    gate = build(tmp_path, scanner)

    gate.run(RunState(run_id="r1", change=make_change(), workspace_path=str(checkout)))

    # ``run-pr`` materialises the head commit somewhere else; that is the tree
    # to test, not the directory the process was started from.
    assert scanner.tests_run == [checkout]


def test_the_pack_names_the_kind_of_tree_that_was_tested_not_its_path(
    tmp_path: Path,
) -> None:
    checkout = tree(tmp_path / "checkout")
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    gate = build(tmp_path, scanner)

    report = gate.run(
        RunState(run_id="r1", change=make_change(), workspace_path=str(checkout))
    )

    # The host path is the operator's infrastructure, and the pack is handed to
    # an auditor who cannot check it. What the tree check needs to be readable
    # survives without it: which kind of tree was tested, whether it held the
    # change, and -- repo-relative already -- which files said otherwise.
    assert report.verifier_report["workspace_kind"] == "checkout"
    assert report.verifier_report["tree_check"]["status"] == "present"
    assert "workspace" not in report.verifier_report


def test_a_run_in_the_deployments_own_tree_says_so(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    gate = build(tree(tmp_path), scanner)

    report = gate.run(state())

    # ``honest-ci verify`` tests the directory it was pointed at, whose state
    # nobody pinned. Reading that off the pack is the difference the path was
    # there to make, so it must survive as a word of its own.
    assert report.verifier_report["workspace_kind"] == "deployment-root"


def test_the_deployment_root_is_recognised_through_a_symlinked_temp_dir(
    tmp_path: Path,
) -> None:
    link = tmp_path / "link"
    link.symlink_to(tree(tmp_path / "real"), target_is_directory=True)
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    gate = build(tmp_path / "real", scanner)

    report = gate.run(
        RunState(run_id="r1", change=make_change(), workspace_path=str(link))
    )

    # The local entry point records the workspace resolved while the runner
    # keeps the spelling it was given. On macOS those differ (``/var`` ->
    # ``/private/var``), and two spellings of one directory must not read as
    # two different trees.
    assert report.verifier_report["workspace_kind"] == "deployment-root"


def test_check_tree_answers_present_absent_or_unverifiable(tmp_path: Path) -> None:
    change = make_change()
    assert check_tree(tree(tmp_path / "present"), change).status == "present"
    assert check_tree(tmp_path / "missing", change).status == "absent"

    # Trailing whitespace is not a difference a test suite would notice.
    padded = tree(tmp_path / "padded", REFUND_WITH_CHANGE.replace("item.refund = share", "item.refund = share  "))
    assert check_tree(padded, change).status == "present"

    # Nothing to look for: no diff at all. The gate proceeds and the evidence
    # records that the check could not be made.
    nothing = check_tree(tmp_path / "present", make_change(diff="", files_changed=[]))
    assert nothing.status == "unverifiable"
    assert nothing.checked == 0


def test_a_file_the_change_deleted_must_be_gone(tmp_path: Path) -> None:
    root = tree(tmp_path)
    (root / "src" / "app" / "legacy.py").write_text("x = 1\n", encoding="utf-8")
    change = make_change(
        files_changed=[
            ChangedFile(path="src/app/refund.py", change_type="modified", added_lines=3),
            ChangedFile(path="src/app/legacy.py", change_type="deleted"),
        ]
    )

    still_there = check_tree(root, change)
    assert still_there.status == "absent"
    assert still_there.mismatched == ("src/app/legacy.py",)

    (root / "src" / "app" / "legacy.py").unlink()
    assert check_tree(root, change).status == "present"


def test_a_file_the_change_renamed_must_be_gone_from_its_old_name(tmp_path: Path) -> None:
    root = tree(tmp_path)
    legacy = root / "src" / "app" / "legacy.py"
    legacy.write_text("x = 1\n", encoding="utf-8")
    renamed = ChangedFile(
        path="src/app/current.py", change_type="renamed", previous_path="src/app/legacy.py"
    )
    change = make_change(
        files_changed=[
            ChangedFile(path="src/app/refund.py", change_type="modified", added_lines=3),
            renamed,
        ]
    )

    # A rename that changed no line leaves nothing added to look for; the old
    # name still on disk is how a tree that never took the move is told apart.
    stale = check_tree(root, change)
    assert stale.status == "absent"
    assert stale.mismatched == ("src/app/legacy.py",)

    legacy.rename(root / "src" / "app" / "current.py")
    assert check_tree(root, change).status == "present"

    # Unless the change put another file under the old name.
    legacy.write_text("x = 2\n", encoding="utf-8")
    reused = make_change(
        files_changed=[
            *change.files_changed,
            ChangedFile(path="src/app/legacy.py", change_type="added", added_lines=1),
        ]
    )
    assert check_tree(root, reused).status == "present"


def test_a_path_that_escapes_the_workspace_is_never_read(tmp_path: Path) -> None:
    diff = DIFF.replace("src/app/refund.py", "../outside.py")
    (tmp_path / "outside.py").write_text("whatever\n", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()

    # An unsafe path is not checkable, so the change has nothing to look for.
    assert check_tree(root, make_change(diff=diff, files_changed=[])).status == "unverifiable"


@pytest.mark.parametrize(
    "separator", ["\r", "\x0c", " "], ids=["carriage-return", "form-feed", "line-sep"]
)
def test_a_line_python_would_break_is_one_line_in_the_file_too(
    tmp_path: Path, separator: str
) -> None:
    # Git numbers the file with this line as one line. Read with Python's line
    # breaks, every line after it moves down one and the tree seems to lack
    # the change it holds.
    diff = (
        "--- a/src/app/refund.py\n"
        "+++ b/src/app/refund.py\n"
        "@@ -1 +1,3 @@\n"
        " keep = 1\n"
        f"+x = 1{separator}y = 2\n"
        "+z = 3\n"
    )
    root = tree(tmp_path, f"keep = 1\nx = 1{separator}y = 2\nz = 3\n")

    assert check_tree(root, make_change(diff=diff, files_changed=[])).status == "present"


# ── the suite ────────────────────────────────────────────────────────────────


def test_a_suite_that_never_started_blocks(tmp_path: Path) -> None:
    gate = build(tree(tmp_path), StubScanner(tests=None))

    report = gate.run(state())

    # ``available=False`` is missing evidence; fail-closed makes it a block
    # with the tool's own reason attached, never a pass with zero failures.
    assert report.verdict is Verdict.BLOCKED
    assert [finding.id for finding in report.findings] == ["regression.unavailable"]
    assert "pytest is not installed" in report.summary
    assert report.test_results is not None and report.test_results.available is False


def test_a_failing_test_blocks_and_names_the_test(tmp_path: Path) -> None:
    scanner = StubScanner(
        tests=SuiteResults(
            passed=3,
            failed=1,
            exit_code=1,
            command="python -m pytest -q",
            failures=["FAILED tests/test_refund.py::test_split - AssertionError: 0 != 1"],
        ),
        coverage=MEASURED,
    )
    gate = build(tree(tmp_path), scanner)

    report = gate.run(state())

    assert report.verdict is Verdict.BLOCKED
    (finding,) = report.findings
    assert finding.id == "regression.test.1"
    assert finding.blocking
    assert "`test_split`" in finding.title
    assert finding.file == "tests/test_refund.py"
    assert "AssertionError: 0 != 1" in finding.detail
    assert "3 passed · 1 failed" in report.summary
    # Coverage of a red suite is not a number anyone should act on.
    assert scanner.coverage_measured == []
    assert "coverage" not in report.metrics
    assert report.metrics["tests_failed"] == 1.0


def test_a_failing_tests_own_id_and_message_never_render_as_markdown(tmp_path: Path) -> None:
    line = f"FAILED tests/test_refund.py::test_split[{HOSTILE_TEXT}] - AssertionError: {HOSTILE_TEXT}"
    scanner = StubScanner(
        tests=SuiteResults(passed=3, failed=1, exit_code=1, failures=[line]), coverage=MEASURED
    )
    report = build(tree(tmp_path), scanner).run(state())
    evidence = EvidencePack(
        run_id="r1",
        change=make_change(),
        gate_reports=[report],
        verdict=report.verdict,
        fingerprint="f" * 64,
    )

    # A parametrised test is named by whatever id the change gave it, and the
    # title leads the merge request note.
    (finding,) = report.findings
    assert "Approved by security" in finding.title
    for text in (render_forge_comment(evidence, status="blocked"), render_evidence(evidence)):
        assert rendered_as_markdown(text, "Approved by security") == []


def test_why_a_suite_never_ran_is_quoted_as_what_it_printed(tmp_path: Path) -> None:
    suite = SuiteResults(available=False, exit_code=4, stderr=f"{HOSTILE_TEXT}\nusage: pytest")
    report = build(tree(tmp_path), StubScanner(tests=suite)).run(state())
    evidence = EvidencePack(
        run_id="r1",
        change=make_change(),
        gate_reports=[report],
        verdict=report.verdict,
        fingerprint="f" * 64,
    )

    # A conftest the change added can print anything before the suite starts,
    # and its first line is the gate's summary: a row of the note's table.
    assert "Approved by security" in report.summary
    for text in (render_forge_comment(evidence, status="blocked"), render_evidence(evidence)):
        assert rendered_as_markdown(text, "Approved by security") == []


def test_a_collection_error_blocks_too(tmp_path: Path) -> None:
    scanner = StubScanner(
        tests=SuiteResults(errors=1, exit_code=2, failures=["ERROR tests/test_broken.py"])
    )
    gate = build(tree(tmp_path), scanner)

    report = gate.run(state())

    assert report.verdict is Verdict.BLOCKED
    assert "`tests/test_broken.py`" in report.findings[0].title
    assert report.findings[0].file == "tests/test_broken.py"


def test_failures_beyond_the_cap_are_counted_not_listed(tmp_path: Path) -> None:
    failures = [f"FAILED tests/test_many.py::test_{i} - boom" for i in range(25)]
    scanner = StubScanner(tests=SuiteResults(failed=25, exit_code=1, failures=failures))
    gate = build(tree(tmp_path), scanner)

    report = gate.run(state())

    assert len(report.findings) == 20
    assert "20 of 25" in report.summary
    assert report.metrics["tests_failed"] == 25.0


def test_a_red_suite_without_a_parsable_case_still_blocks(tmp_path: Path) -> None:
    scanner = StubScanner(
        tests=SuiteResults(failed=2, exit_code=1, command="make test", stdout="2 failed")
    )
    gate = build(tree(tmp_path), scanner)

    report = gate.run(state())

    assert report.verdict is Verdict.BLOCKED
    assert [finding.id for finding in report.findings] == ["regression.suite"]
    assert "2 failed" in report.findings[0].detail


def test_an_opaque_exit_code_blocks_rather_than_passes(tmp_path: Path) -> None:
    # Counts say nothing failed, the exit code says something did (an internal
    # pytest error, a command that is not pytest at all): not a pass.
    scanner = StubScanner(tests=SuiteResults(passed=3, exit_code=3, stderr="INTERNALERROR> boom"))
    gate = build(tree(tmp_path), scanner)

    report = gate.run(state())

    assert report.verdict is Verdict.BLOCKED
    assert [finding.id for finding in report.findings] == ["regression.suite"]
    assert "exit code 3" in report.summary
    assert "INTERNALERROR> boom" in report.findings[0].detail


@pytest.mark.parametrize(
    ("suite", "coverage", "changed_threshold"),
    [
        (SuiteResults(failed=2, exit_code=1, stdout=HOSTILE_OUTPUT), MEASURED, 0.0),
        (SuiteResults(passed=3, exit_code=3, stderr=HOSTILE_OUTPUT), MEASURED, 0.0),
        (PASSING, CoverageResult(ratio=None, exit_code=1, stderr=HOSTILE_OUTPUT), 0.0),
        (
            PASSING,
            CoverageResult(ratio=0.8, command="python -m coverage json -o -", stderr=HOSTILE_OUTPUT),
            0.8,
        ),
    ],
    ids=["counted-without-cases", "opaque-exit-code", "coverage", "changed-line-coverage"],
)
def test_the_tail_of_what_the_suite_printed_is_quoted_as_a_block_it_cannot_close(
    tmp_path: Path, suite: SuiteResults, coverage: CoverageResult, changed_threshold: float
) -> None:
    scanner = StubScanner(tests=suite, coverage=coverage)
    gate = build(tree(tmp_path), scanner, changed_threshold=changed_threshold)
    report = gate.run(state())
    evidence = EvidencePack(
        run_id="r1",
        change=make_change(),
        gate_reports=[report],
        verdict=report.verdict,
        fingerprint="f" * 64,
    )

    # The change decides what its failing tests print, and coverage runs them
    # too. The tail is where the reason is, so it is shown -- every line of
    # it, as output.
    text = render_evidence(evidence)
    assert rendered_as_markdown(text, "Approved by security") == []
    blocks = [
        token.content
        for token in markdown_tokens(text)
        if token.type == "fence" and "Approved by security" in token.content
    ]
    assert len(blocks) == 1
    assert blocks[0].count("Approved by security") == 4
    assert "\n\n" in blocks[0]


def test_a_suite_that_collected_nothing_is_no_evidence(tmp_path: Path) -> None:
    scanner = StubScanner(tests=SuiteResults(exit_code=5, command="python -m pytest -q"))
    gate = build(tree(tmp_path), scanner)

    report = gate.run(state())

    # Like an empty rule library: nothing verified, so nobody can say "pass".
    assert report.verdict is Verdict.NEEDS_HUMAN
    assert "0 tests" in report.summary
    assert report.findings == []
    assert scanner.coverage_measured == []


# ── coverage ─────────────────────────────────────────────────────────────────


def test_coverage_below_the_bar_blocks(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=CoverageResult(ratio=0.5))
    gate = build(tree(tmp_path), scanner, threshold=0.7)

    report = gate.run(state())

    assert report.verdict is Verdict.BLOCKED
    (finding,) = report.findings
    assert finding.id == "regression.coverage"
    assert "50%" in finding.detail and "70%" in finding.detail
    assert "HONEST_CI_COVERAGE_THRESHOLD" in finding.detail
    assert report.metrics["coverage"] == 0.5


def test_coverage_at_the_bar_passes(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=CoverageResult(ratio=0.7))
    gate = build(tree(tmp_path), scanner, threshold=0.7)

    report = gate.run(state())

    assert report.verdict is Verdict.PASS
    assert report.findings == []
    assert "70%" in report.summary
    assert report.metrics["coverage"] == 0.7
    assert report.metrics["tests_passed"] == 3.0
    assert [tool.name for tool in report.tool_reports] == ["pytest", "coverage"]
    assert all(tool.available and tool.passed for tool in report.tool_reports)
    assert report.test_results == PASSING


def test_coverage_the_tool_could_not_measure_blocks(tmp_path: Path) -> None:
    scanner = StubScanner(
        tests=PASSING,
        coverage=CoverageResult(ratio=None, exit_code=1, stderr="No data to report."),
    )
    gate = build(tree(tmp_path), scanner)

    report = gate.run(state())

    # The suite is green but the bar cannot be checked. Passing here would make
    # the threshold decorative; the finding says how to make it measurable.
    assert report.verdict is Verdict.BLOCKED
    (finding,) = report.findings
    assert finding.id == "regression.coverage"
    assert "70%" in finding.detail
    assert "No data to report." in finding.detail
    assert finding.suggested_fix and "HONEST_CI_COVERAGE_THRESHOLD" in finding.suggested_fix
    assert "coverage" not in report.metrics


def test_a_missing_coverage_tool_blocks_through_fail_closed(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=None)
    gate = build(tree(tmp_path), scanner)

    report = gate.run(state())

    assert report.verdict is Verdict.BLOCKED
    assert [finding.id for finding in report.findings] == ["regression.unavailable"]
    assert [tool.name for tool in report.tool_reports if not tool.available] == ["coverage"]
    # The summary names the bar nobody could check.
    assert "70%" in report.summary


def test_a_zero_bar_switches_coverage_off_explicitly(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=None)
    gate = build(tree(tmp_path), scanner, threshold=0.0)

    report = gate.run(state())

    # Nothing was measured, and the pack says so as configuration, not as a
    # tool that failed: no coverage tool report, no unavailable finding.
    assert report.verdict is Verdict.PASS
    assert scanner.coverage_measured == []
    assert "HONEST_CI_COVERAGE_THRESHOLD" in report.summary
    assert [tool.name for tool in report.tool_reports] == ["pytest"]
    assert "coverage" not in report.metrics
    assert report.metrics["coverage_threshold"] == 0.0


# ── deployment without a scanner ─────────────────────────────────────────────


def test_no_scanner_means_no_evidence_and_a_block(tmp_path: Path) -> None:
    gate = build(tree(tmp_path), None)

    report = gate.run(state())

    assert report.verdict is Verdict.BLOCKED
    assert [finding.id for finding in report.findings] == ["regression.unavailable"]
    assert [(tool.name, tool.available) for tool in report.tool_reports] == [("tests", False)]


def test_the_report_is_bound_to_the_policy_version(tmp_path: Path) -> None:
    gate = build(tree(tmp_path), StubScanner(tests=PASSING, coverage=MEASURED))

    report = gate.run(state())

    assert report.policy_version == gate.services.settings.policy_version
    assert report.verifier_report["coverage"] == {
        "available": True,
        "ratio": 0.8,
        "exit_code": None,
        "command": "python -m coverage report --format=total",
    }


# ── suites that are not pytest ───────────────────────────────────────────────


def test_pytests_no_tests_exit_code_means_nothing_for_another_runner(tmp_path: Path) -> None:
    scanner = StubScanner(tests=SuiteResults(exit_code=5, command="make test", runner="exit-code"))

    report = build(tree(tmp_path), scanner).run(state())

    # 5 is "nothing collected" only when pytest says it. From a custom command
    # it is an exit code nobody can read, and opaque blocks.
    assert report.verdict is Verdict.BLOCKED
    assert report.findings[0].id == "regression.suite"
    assert "HONEST_CI_TEST_REPORT" in (report.findings[0].suggested_fix or "")


def test_a_junit_read_suite_is_filed_under_its_reader(tmp_path: Path) -> None:
    scanner = StubScanner(
        tests=SuiteResults(
            passed=2,
            failed=1,
            exit_code=1,
            command="mvn -B test",
            runner="junit",
            failures=["FAILED com.acme.OrderTest#rejectsNegative - expected 0"],
        )
    )

    report = build(tree(tmp_path), scanner).run(state())

    assert report.verdict is Verdict.BLOCKED
    assert report.tool_reports[0].name == "junit"
    finding = report.findings[0]
    assert "`com.acme.OrderTest#rejectsNegative`" in finding.title
    # A Java class is not a path in the tree; the finding does not pretend it is.
    assert finding.file is None
    assert report.verifier_report["test_runner"] == "junit"


def test_a_custom_command_with_no_cases_says_how_to_make_them_readable(tmp_path: Path) -> None:
    scanner = StubScanner(tests=SuiteResults(exit_code=0, command="make test", runner="exit-code"))

    report = build(tree(tmp_path), scanner).run(state())

    assert report.verdict is Verdict.NEEDS_HUMAN
    assert "HONEST_CI_TEST_REPORT" in report.summary


def test_the_evidence_records_which_toolchain_ran_the_suite(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"

    report = build(tree(tmp_path), scanner).run(state())

    assert report.verifier_report["toolchain"] == "python"


# ── changed-line coverage ────────────────────────────────────────────────────


def covering(**fields: object) -> CoverageResult:
    payload: dict[str, object] = {
        "ratio": 0.8,
        "command": "python -m coverage json -o -",
        "changed_lines_total": 3,
        "changed_lines_covered": 3,
    }
    payload.update(fields)
    return CoverageResult.model_validate(payload)


def test_changed_lines_the_suite_never_ran_block(tmp_path: Path) -> None:
    scanner = StubScanner(
        tests=PASSING,
        coverage=covering(
            changed_lines_covered=1,
            uncovered=[("src/app/refund.py", 12), ("src/app/refund.py", 13)],
        ),
    )
    gate = build(tree(tmp_path), scanner, changed_threshold=0.8)

    report = gate.run(state())

    # The repository sits at 80% and the whole suite is green; two of the three
    # lines this change added were never executed. That is the gap the overall
    # bar cannot see, and it is what this bar exists for.
    assert report.verdict is Verdict.BLOCKED
    (finding,) = report.findings
    assert finding.id == "regression.changed-coverage"
    assert "HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD" in finding.detail
    assert finding.file == "src/app/refund.py"
    assert finding.line_start == 12
    assert "`src/app/refund.py`:13" in finding.detail
    assert all(fact in report.summary for fact in ("33%", "1/3", "80%"))
    assert report.metrics["changed_line_coverage"] == 1 / 3
    assert report.metrics["changed_lines_total"] == 3.0


def test_changed_lines_the_suite_ran_pass_next_to_the_overall_bar(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=covering())
    gate = build(tree(tmp_path), scanner, threshold=0.7, changed_threshold=0.8)

    report = gate.run(state())

    assert report.verdict is Verdict.PASS
    assert report.findings == []
    # Both bars, each with the figure it was held against.
    assert all(fact in report.summary for fact in ("80%", "70%", "100%", "3/3"))
    assert scanner.coverage_lines == [{"src/app/refund.py": {11, 12, 13}}]


def test_a_change_with_no_executable_line_is_not_held_to_the_changed_bar(
    tmp_path: Path,
) -> None:
    scanner = StubScanner(
        tests=PASSING, coverage=covering(changed_lines_total=0, changed_lines_covered=0)
    )
    gate = build(tree(tmp_path), scanner, changed_threshold=0.8)

    report = gate.run(state())

    # A comment and a blank line are not something a test can execute. The bar
    # is not met by dividing by zero; it simply does not apply.
    assert report.verdict is Verdict.PASS
    assert "0/0" not in report.summary


def test_a_changed_file_with_no_coverage_data_blocks_rather_than_passes(
    tmp_path: Path,
) -> None:
    scanner = StubScanner(
        tests=PASSING,
        coverage=covering(
            changed_lines_total=0,
            changed_lines_covered=0,
            unmapped_sources=["src/app/refund.py"],
        ),
    )
    gate = build(tree(tmp_path), scanner, changed_threshold=0.8)

    report = gate.run(state())

    # "Not in the report" is not "nothing to cover": the bar was never checked
    # for this file, and a bar nobody checked has not been met.
    assert report.verdict is Verdict.BLOCKED
    (finding,) = report.findings
    assert finding.file == "src/app/refund.py"
    assert "`src/app/refund.py`" in report.summary


def test_a_file_name_holding_a_backtick_is_named_whole(tmp_path: Path) -> None:
    name = "src/app/a`b.py"
    unmapped = build(
        tree(tmp_path / "unmapped"),
        StubScanner(
            tests=PASSING,
            coverage=covering(changed_lines_total=0, changed_lines_covered=0, unmapped_sources=[name]),
        ),
        changed_threshold=0.8,
    ).run(state())
    below = build(
        tree(tmp_path / "below"),
        StubScanner(tests=PASSING, coverage=covering(changed_lines_covered=1, uncovered=[(name, 12)])),
        changed_threshold=0.8,
    ).run(state())

    # The summary and the detail are Markdown in the evidence pack and the
    # platform comment. A span opened with one backtick would close at the
    # name's own and leave the rest of the name to render.
    assert "``src/app/a`b.py``" in unmapped.summary
    assert "- ``src/app/a`b.py``\n" in unmapped.findings[0].detail + "\n"
    assert "- ``src/app/a`b.py``:12" in below.findings[0].detail


def test_a_changed_file_nobody_configured_coverage_to_measure_does_not_block(
    tmp_path: Path,
) -> None:
    scanner = StubScanner(
        tests=PASSING,
        coverage=covering(out_of_scope_sources=["tests/test_refund.py"]),
    )
    gate = build(tree(tmp_path), scanner, changed_threshold=0.8)

    report = gate.run(state())

    # The opposite reading from the test above, and the distinction is the
    # whole point: "the report never named it" is missing evidence, while "the
    # configuration never measures it" is the configuration working. A bar on
    # files a repository deliberately does not measure is a bar nobody can meet.
    assert report.verdict is Verdict.PASS
    changed = report.verifier_report["coverage"]["changed_lines"]
    assert changed["not_measured"] == ["tests/test_refund.py"]


def test_per_line_data_the_tool_could_not_produce_blocks(tmp_path: Path) -> None:
    scanner = StubScanner(
        tests=PASSING, coverage=CoverageResult(ratio=0.8, command="python -m coverage json -o -")
    )
    gate = build(tree(tmp_path), scanner, threshold=0.0, changed_threshold=0.8)

    report = gate.run(state())

    assert report.verdict is Verdict.BLOCKED
    (finding,) = report.findings
    assert finding.id == "regression.changed-coverage"
    assert finding.suggested_fix and "HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD" in finding.suggested_fix


def test_a_zero_changed_bar_records_the_number_and_judges_nothing(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=covering(changed_lines_covered=0))
    gate = build(tree(tmp_path), scanner, threshold=0.7, changed_threshold=0.0)

    report = gate.run(state())

    # None of this change's lines were executed, and the bar is off: a
    # deployment deciding whether to turn it on can read the number its own
    # changes produce, and nothing is blocked on a bar nobody configured.
    assert report.verdict is Verdict.PASS
    assert report.findings == []
    assert scanner.coverage_lines == [{"src/app/refund.py": {11, 12, 13}}]
    assert report.metrics["changed_line_coverage"] == 0.0
    assert report.metrics["changed_line_coverage_threshold"] == 0.0
    assert "0/3" not in report.summary


# ── test selection ───────────────────────────────────────────────────────────


def with_tests(tmp_path: Path) -> Path:
    """A workspace holding the change and a test file that names what changed."""

    workspace = tree(tmp_path)
    target = workspace / "tests" / "test_refund.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("from src.app.refund import split_refund\n", encoding="utf-8")
    (workspace / "tests" / "test_users.py").write_text("def test_users():\n    pass\n", "utf-8")
    return workspace


def test_the_full_suite_is_the_default(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"

    report = build(with_tests(tmp_path), scanner).run(state())

    assert scanner.selections == [None]
    assert report.verifier_report["test_selection"]["mode"] == "all"
    assert "HONEST_CI_TEST_SELECTION" not in report.summary


def test_impacted_selection_runs_those_tests_and_says_the_evidence_is_weaker(
    tmp_path: Path,
) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"

    # The overall bar off: it is a figure about the whole suite, so with it on
    # the whole suite runs (see the test about narrowed runs below).
    report = build(with_tests(tmp_path), scanner, selection="impacted", threshold=0.0).run(
        state()
    )

    assert scanner.selections == [["tests/test_refund.py"]]
    assert report.verdict is Verdict.PASS
    assert "HONEST_CI_TEST_SELECTION=impacted" in report.summary
    assert report.verifier_report["test_selection"] == {
        "mode": "impacted",
        "requested": ["tests/test_refund.py"],
        "fallback": None,
    }


def test_an_empty_selection_falls_back_to_the_whole_suite(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"

    # No test file mentions the change, so impact analysis selects nothing.
    # "0 tests were selected, so nothing failed" is not a conclusion.
    report = build(tree(tmp_path), scanner, selection="impacted").run(state())

    assert scanner.selections == [None]
    assert report.verdict is Verdict.PASS
    assert "HONEST_CI_TEST_SELECTION=impacted" in report.summary
    assert report.verifier_report["test_selection"]["mode"] == "all"
    assert report.verifier_report["test_selection"]["fallback"]


def test_an_overall_bar_is_never_held_against_a_narrowed_run(tmp_path: Path) -> None:
    # What a run of the impacted tests alone reports as the repository's line
    # coverage. Measured on a real repository with coverage's `source` unset,
    # this is 100% where the whole suite gives 45%: the narrowed run never
    # imports the files only the other tests reach.
    scanner = StubScanner(tests=PASSING, coverage=CoverageResult(ratio=1.0))
    scanner.language = "python"

    report = build(with_tests(tmp_path), scanner, selection="impacted").run(state())

    assert scanner.selections == [None]
    selection = report.verifier_report["test_selection"]
    assert selection["mode"] == "all"
    assert "HONEST_CI_COVERAGE_THRESHOLD" in selection["fallback"]
    assert "the full suite ran" in report.summary


def test_the_changed_line_bar_is_held_on_a_narrowed_run(tmp_path: Path) -> None:
    # Unlike the overall figure, this one can only be lower on a narrowed
    # run, never higher: fewer tests execute fewer of the lines. A narrowed
    # run can block where the full suite would have passed, but cannot pass
    # where it would have blocked.
    scanner = StubScanner(tests=PASSING, coverage=covering())
    scanner.language = "python"

    report = build(
        with_tests(tmp_path), scanner, selection="impacted", threshold=0.0, changed_threshold=0.8
    ).run(state())

    assert scanner.selections == [["tests/test_refund.py"]]
    assert report.verdict is Verdict.PASS


def test_a_language_without_impact_analysis_runs_the_whole_suite(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "rust"

    report = build(with_tests(tmp_path), scanner, selection="impacted").run(state())

    assert scanner.selections == [None]
    assert "rust" in report.summary
    assert report.verifier_report["impact"]["analysed"] is False


def test_a_file_nobody_could_parse_runs_the_whole_suite(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"
    workspace = with_tests(tmp_path)
    # The tree check reads the added lines literally, so a file that still holds
    # them but does not parse reaches impact analysis and defeats it there.
    (workspace / "src" / "app" / "refund.py").write_text(
        REFUND_WITH_CHANGE + "\ndef broken(:\n", encoding="utf-8"
    )

    report = build(workspace, scanner, selection="impacted").run(state())

    assert scanner.selections == [None]
    assert "`src/app/refund.py`" in report.summary
    assert report.verifier_report["impact"]["unanalysed_files"] == ["src/app/refund.py"]


def test_the_scope_travels_with_a_block_too(tmp_path: Path) -> None:
    scanner = StubScanner(tests=SuiteResults(passed=1, failed=1, exit_code=1, failures=["FAILED a"]))
    scanner.language = "python"

    report = build(with_tests(tmp_path), scanner, selection="impacted", threshold=0.0).run(
        state()
    )

    # "The impacted tests failed" and "the whole suite failed" are different
    # facts; a reader of the block must not have to guess which one this is.
    assert scanner.selections == [["tests/test_refund.py"]]
    assert report.verdict is Verdict.BLOCKED
    assert "HONEST_CI_TEST_SELECTION=impacted" in report.summary


def test_the_impact_analysis_travels_in_the_evidence(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"

    report = build(with_tests(tmp_path), scanner).run(state())

    impact = report.verifier_report["impact"]
    assert impact["language"] == "python"
    assert impact["changed_symbols"] == [
        {
            "file": "src/app/refund.py",
            "symbol": "split_refund",
            "kind": "function",
            "line": 8,
        }
    ]
    assert impact["impacted_tests"] == ["tests/test_refund.py"]
    assert report.metrics["changed_symbols"] == 1.0
    assert report.metrics["impacted_tests"] == 1.0


# ── generated regression cases ───────────────────────────────────────────────

GENERATED_CASE = ProposedTest(
    path="tests/honest_ci_generated/test_split_refund.py",
    content="def test_split_refund_divides_evenly():\n    assert True\n",
    targets=["split_refund"],
)
GENERATED_PASSED = SuiteResults(passed=1, exit_code=0, command="python -m pytest -q")


def generating(tmp_path: Path, **kwargs: object) -> tuple[StubScanner, ScriptedLLM, Gate]:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED, **kwargs)  # type: ignore[arg-type]
    scanner.language = "python"
    llm = ScriptedLLM(tests=[GENERATED_CASE])
    return scanner, llm, build(tree(tmp_path), scanner, llm=llm)


def test_generated_cases_are_written_run_and_then_removed(tmp_path: Path) -> None:
    scanner, llm, gate = generating(tmp_path, generated=GENERATED_PASSED)

    report = gate.run(state())

    # Written before the run, gone after it. A verification pass that leaves
    # files in the tree has changed the thing it was verifying.
    assert scanner.selections == [None, ["tests/honest_ci_generated/test_split_refund.py"]]
    assert scanner.generated_files_seen == [[], ["tests/honest_ci_generated/test_split_refund.py"]]
    assert scanner.generated_files(tmp_path) == []
    assert not (tmp_path / "tests" / "honest_ci_generated").exists()
    assert llm.testgen_symbols == [("split_refund",)]
    assert report.verdict is Verdict.PASS


JAVA_DIFF = """--- a/src/main/java/com/acme/Refund.java
+++ b/src/main/java/com/acme/Refund.java
@@ -1,6 +1,7 @@
 package com.acme;
 
 public class Refund {
     public int split(int amount) {
+        return amount / 2;
     }
 }
"""

JAVA_SOURCE = """package com.acme;

public class Refund {
    public int split(int amount) {
        return amount / 2;
    }
}
"""

JAVA_CASE = ProposedTest(
    path="src/test/java/honest_ci/generated/RefundSplitTest.java",
    content="package honest_ci.generated;\n\nclass RefundSplitTest {\n    void split() {}\n}\n",
    targets=["split"],
)


def test_maven_is_handed_the_class_name_it_selects_tests_by(tmp_path: Path) -> None:
    source = tmp_path / "src" / "main" / "java" / "com" / "acme" / "Refund.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(JAVA_SOURCE, encoding="utf-8")
    scanner = StubScanner(tests=PASSING, coverage=MEASURED, generated=GENERATED_PASSED)
    scanner.language = "java"
    llm = ScriptedLLM(tests=[JAVA_CASE])
    gate = build(tmp_path, scanner, llm=llm)

    report = gate.run(RunState(run_id="r1", change=make_change(diff=JAVA_DIFF)))

    # A path is what pytest takes; Maven takes `-Dtest=RefundSplitTest`. Handing
    # it a path selects nothing at all, and a suite that ran no case would
    # report as a clean pass.
    assert scanner.selections == [None, ["RefundSplitTest"]]
    assert report.verdict is Verdict.PASS


def test_generated_cases_are_counted_apart_from_the_repositorys_own_suite(
    tmp_path: Path,
) -> None:
    _, _, gate = generating(tmp_path, generated=GENERATED_PASSED)

    report = gate.run(state())

    results = report.test_results
    assert results is not None
    # The suite's own counts are untouched; no arithmetic can let a generated
    # case move them.
    assert (results.passed, results.failed) == (3, 0)
    assert (results.generated, results.generated_failed) == (1, 0)
    assert report.metrics["generated_tests"] == 1.0
    assert "1 generated regression test" in report.summary


def test_a_failing_generated_case_goes_to_a_human_and_never_blocks(
    tmp_path: Path,
) -> None:
    failing = SuiteResults(
        passed=0,
        failed=1,
        exit_code=1,
        command="python -m pytest -q",
        failures=["FAILED tests/honest_ci_generated/test_split_refund.py::test_split_refund_divides_evenly"],
    )
    _, _, gate = generating(tmp_path, generated=failing)

    report = gate.run(state())

    # The repository's own suite is green. A model-written case that fails is a
    # question (is the case wrong, or the code?), and a question is a hand-off.
    assert report.verdict is Verdict.NEEDS_HUMAN
    (finding,) = report.findings
    assert finding.id == "regression.generated"
    assert finding.decided_by == "model"
    assert not finding.blocking
    assert report.test_results is not None
    assert report.test_results.generated_failed == 1
    assert "1 generated regression test" in report.summary


def test_a_failing_generated_case_is_named_as_code(tmp_path: Path) -> None:
    name = f"FAILED tests/honest_ci_generated/test_split_refund.py::test_split - {HOSTILE_TEXT}"
    failing = SuiteResults(failed=1, exit_code=1, command="python -m pytest -q", failures=[name])
    _, _, gate = generating(tmp_path, generated=failing)

    (finding,) = gate.run(state()).findings

    # The model named the case, and wrote its assertion, after reading the diff.
    assert "Approved by security" in finding.detail
    assert rendered_as_markdown(finding.detail, "Approved by security") == []


def test_a_failing_suite_blocks_and_nothing_is_generated(tmp_path: Path) -> None:
    scanner = StubScanner(tests=SuiteResults(passed=1, failed=1, exit_code=1, failures=["FAILED a"]))
    scanner.language = "python"
    llm = ScriptedLLM(tests=[GENERATED_CASE])

    report = build(tree(tmp_path), scanner, llm=llm).run(state())

    # There is a failure to fix already; a generated case would only add noise
    # on top of a conclusion the developer already has.
    assert report.verdict is Verdict.BLOCKED
    assert llm.testgen_symbols == []
    assert len(scanner.tests_run) == 1


def test_a_verdict_reached_before_generation_says_so_instead_of_reading_blank(
    tmp_path: Path,
) -> None:
    scanner = StubScanner(tests=PASSING, coverage=CoverageResult(ratio=0.5))
    scanner.language = "python"
    llm = ScriptedLLM(tests=[GENERATED_CASE])

    report = build(tree(tmp_path), scanner, threshold=0.7, llm=llm).run(state())

    # Coverage blocked first, so nobody ever asked for a case. Dogfooding read
    # the evidence as "none generated (no reason recorded)" — a gap in the
    # pack, when the truth is simply that the run concluded upstream of
    # generation.
    assert report.verdict is Verdict.BLOCKED
    assert llm.testgen_symbols == []
    assert report.verifier_report["generated_tests"]["not_generated_reason"]


def test_a_hand_off_reached_before_generation_says_which_way_it_concluded(
    tmp_path: Path,
) -> None:
    scanner = StubScanner(tests=SuiteResults(passed=0, exit_code=5, command="pytest -q"))
    scanner.language = "python"
    blocking = StubScanner(tests=PASSING, coverage=CoverageResult(ratio=0.5))
    blocking.language = "python"

    report = build(tree(tmp_path / "handed-off"), scanner).run(state())
    blocked = build(tree(tmp_path / "blocked"), blocking).run(state())

    # "Blocked upstream" and "handed to a person upstream" are different
    # reasons for the same empty row, and the pack carries the one that happened.
    assert report.verdict is Verdict.NEEDS_HUMAN
    assert blocked.verdict is Verdict.BLOCKED
    handed_off = report.verifier_report["generated_tests"]["not_generated_reason"]
    assert handed_off
    assert handed_off != blocked.verifier_report["generated_tests"]["not_generated_reason"]


def test_a_language_with_nowhere_to_put_a_case_generates_nothing(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "rust"
    llm = ScriptedLLM(tests=[GENERATED_CASE])

    report = build(tree(tmp_path), scanner, llm=llm).run(state())

    # No generated test directory for this toolchain means no answer to "where
    # would this file even go" — better said out loud than guessed at.
    assert llm.testgen_symbols == []
    assert "rust" in report.verifier_report["generated_tests"]["not_generated_reason"]


def test_a_model_that_proposed_no_case_is_not_a_change_that_was_covered(
    tmp_path: Path,
) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"
    llm = ScriptedLLM(tests=[])

    report = build(tree(tmp_path), scanner, llm=llm).run(state())

    assert llm.testgen_symbols == [("split_refund",)]
    generated = report.verifier_report["generated_tests"]
    assert generated["ran"] is False
    assert generated["not_generated_reason"]


def test_generation_without_a_model_says_so_rather_than_saying_nothing(
    tmp_path: Path,
) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"

    report = build(tree(tmp_path), scanner, llm=None).run(state())

    # Recorded rather than summarised: a deployment with no model would
    # otherwise repeat the same clause on every merge request forever. What
    # must not happen is that the pack reads like a change with generated
    # cases that all passed.
    assert report.verdict is Verdict.PASS
    assert "generated" not in report.summary
    assert report.verifier_report["generated_tests"]["not_generated_reason"] == MODEL_NOT_ENABLED


def test_generation_with_a_named_provider_but_no_key_names_the_key(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"

    report = build(tree(tmp_path), scanner, llm=None, provider="qwen").run(state())

    reason = report.verifier_report["generated_tests"]["not_generated_reason"]
    assert "DASHSCOPE_API_KEY" in reason


def test_generation_can_be_switched_off_explicitly(tmp_path: Path) -> None:
    scanner, llm, _ = generating(tmp_path)
    gate = build(tree(tmp_path), scanner, llm=llm, testgen=False)

    report = gate.run(state())

    assert llm.testgen_symbols == []
    assert "HONEST_CI_MODEL_TESTGEN=0" in report.verifier_report["generated_tests"]["not_generated_reason"]


def test_a_model_call_that_failed_is_not_a_change_that_was_covered(
    tmp_path: Path,
) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"
    llm = ScriptedLLM(tests=[GENERATED_CASE], error=RuntimeError("provider down"))

    report = build(tree(tmp_path), scanner, llm=llm).run(state())

    assert report.verdict is Verdict.PASS
    assert report.findings == []
    assert report.verifier_report["generated_tests"]["not_generated_reason"]
    assert scanner.generated_files(tmp_path) == []


def test_a_proposal_the_trust_boundary_threw_away_leaves_nothing_behind(
    tmp_path: Path,
) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"
    outside = ProposedTest(path="src/app/refund.py", content="x = 1\n", targets=["split_refund"])
    llm = ScriptedLLM(tests=[outside])

    report = build(tree(tmp_path), scanner, llm=llm).run(state())

    # The proposal was discarded whole, so the file it named still holds what
    # the change put there.
    assert report.verifier_report["generated_tests"]["not_generated_reason"]
    assert "share = float" in (tmp_path / "src" / "app" / "refund.py").read_text("utf-8")
    assert len(scanner.tests_run) == 1


def test_a_tree_nobody_could_check_generates_nothing(tmp_path: Path) -> None:
    scanner = StubScanner(tests=PASSING, coverage=MEASURED)
    scanner.language = "python"
    llm = ScriptedLLM(tests=[GENERATED_CASE])
    gate = build(tree(tmp_path), scanner, llm=llm)

    # A change carrying nothing the tree check can look for: nobody could
    # confirm this workspace holds it, so a case written against it is blind.
    report = gate.run(RunState(run_id="r1", change=make_change(diff="", files_changed=[])))

    assert report.verifier_report["tree_check"]["status"] == "unverifiable"
    assert llm.testgen_symbols == []
    assert report.verifier_report["generated_tests"]["not_generated_reason"]


def test_a_generated_run_that_never_started_is_a_limit_not_a_block(
    tmp_path: Path,
) -> None:
    _, _, gate = generating(
        tmp_path,
        generated=SuiteResults(available=False, errors=1, failures=["pytest: not found"]),
    )

    report = gate.run(state())

    # It must not reach fail-closed as an unavailable tool: "we could not
    # generate" is missing advisory evidence, not a check the change failed.
    assert report.verdict is Verdict.PASS
    # The summary quotes what the generated run printed instead of starting.
    assert "pytest: not found" in report.summary
    assert [tool.name for tool in report.tool_reports] == ["pytest", "coverage"]


#: A model-written Java case that does not compile: Maven stops before
#: Surefire, exits 1, and leaves no report to count.
GENERATED_DID_NOT_COMPILE = SuiteResults(
    exit_code=1,
    runner="junit",
    command="mvn -B -o -q test -Dtest=RefundSplitTest",
    stdout=(
        "[ERROR] COMPILATION ERROR :\n"
        "[ERROR] /workspace/src/test/java/honest_ci/generated/RefundSplitTest.java:[5,9] "
        "cannot find symbol\n"
    ),
    stderr="[honest-ci] no JUnit XML matched HONEST_CI_TEST_REPORT=**/target/surefire-reports/"
    "TEST-*.xml; test counts unknown",
)


def test_a_generated_run_that_failed_with_nothing_readable_is_not_none_failed(
    tmp_path: Path,
) -> None:
    _, _, gate = generating(tmp_path, generated=GENERATED_DID_NOT_COMPILE)

    report = gate.run(state())

    # Read by its counts alone this run was "0 ran, none failed" and a pass.
    # Its exit code says it failed, so it is read the way a failing generated
    # case is: a question for a person, never a block.
    assert report.verdict is Verdict.NEEDS_HUMAN
    assert "none failed" not in report.summary
    assert report.test_results is not None
    assert report.test_results.generated_failed == 1
    assert report.verifier_report["generated_tests"]["failed"] == 1
    finding = next(item for item in report.findings if item.id == "regression.generated")
    assert not finding.blocking
    # The reason is in what Maven printed, so the person is shown it.
    assert "cannot find symbol" in finding.detail


def test_a_generated_run_that_counted_nothing_is_no_evidence_rather_than_none_failed(
    tmp_path: Path,
) -> None:
    # Clean exit, no case counted: a Maven module that never compiled the
    # generated file, or a pytest file with no test in it.
    _, _, gate = generating(
        tmp_path, generated=SuiteResults(exit_code=0, runner="junit", command="mvn -B test")
    )

    report = gate.run(state())

    assert report.verdict is Verdict.PASS
    assert "none failed" not in report.summary
    assert "no test result could be read" in report.summary


# ── every way out, in English ────────────────────────────────────────────────

#: More files than a summary names one by one.
MANY_FILES = [f"src/app/module_{index}.py" for index in range(5)]

#: A change the tree check can find in a file no impact reader parses, so the
#: gate reaches generation with no changed symbol to aim a case at.
NOTES_DIFF = "--- a/docs/notes.txt\n+++ b/docs/notes.txt\n@@ -1 +1,2 @@\n notes\n+one more\n"

#: More uncovered changed lines than the finding lists.
UNCOVERED = [("src/app/refund.py", line) for line in range(1, 30)]

_STATUS: dict[Verdict, RunStatus] = {
    Verdict.PASS: "mergeable",
    Verdict.SKIPPED: "mergeable",
    Verdict.BLOCKED: "blocked",
    Verdict.NEEDS_HUMAN: "awaiting_human",
}


def with_deleted_files(tmp_path: Path) -> Path:
    workspace = tree(tmp_path)
    for name in MANY_FILES:
        (workspace / name).write_text("x = 1\n", encoding="utf-8")
    return workspace


def with_notes(tmp_path: Path) -> Path:
    notes = tmp_path / "docs" / "notes.txt"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("notes\none more\n", encoding="utf-8")
    return tmp_path


def with_an_unparsable_file(tmp_path: Path) -> Path:
    workspace = with_tests(tmp_path)
    (workspace / "src" / "app" / "refund.py").write_text(
        REFUND_WITH_CHANGE + "\ndef broken(:\n", encoding="utf-8"
    )
    return workspace


def proposing(*tests: ProposedTest, error: Exception | None = None) -> Callable[[], ScriptedLLM]:
    return lambda: ScriptedLLM(tests=list(tests), error=error)


def run_path(
    tmp_path: Path,
    *,
    tests: SuiteResults | None = PASSING,
    coverage: CoverageResult | None = MEASURED,
    generated: SuiteResults | None = None,
    language: str = "python",
    scanner: bool = True,
    workspace: Callable[[Path], Path] = tree,
    change: ChangeRequest | None = None,
    llm: Callable[[], ScriptedLLM] | None = None,
    **settings: Any,
) -> GateReport:
    stub = StubScanner(tests=tests, coverage=coverage, generated=generated)
    stub.language = language
    gate = build(
        workspace(tmp_path), stub if scanner else None, llm=llm() if llm else None, **settings
    )
    return gate.run(RunState(run_id="r1", change=change or make_change()))


#: One entry per way the gate can conclude, and per clause it can add on the
#: way: each summary, finding and not-generated reason the gate can write.
EVERY_PATH: dict[str, dict[str, Any]] = {
    "tree-without-the-change": {
        "workspace": with_deleted_files,
        "change": make_change(
            files_changed=[ChangedFile(path=name, change_type="deleted") for name in MANY_FILES]
        ),
    },
    "no-scanner": {"scanner": False},
    "suite-never-started": {"tests": None},
    "failures-and-errors-beyond-the-cap": {
        "tests": SuiteResults(
            failed=24,
            errors=1,
            exit_code=1,
            failures=[
                "ERROR tests/test_broken.py",
                *(f"FAILED tests/test_many.py::test_{index} - boom" for index in range(24)),
            ],
        )
    },
    "failures-without-a-readable-case": {
        "tests": SuiteResults(failed=2, exit_code=1, stdout="2 failed")
    },
    "opaque-exit-code": {"tests": SuiteResults(passed=3, exit_code=3, stderr="INTERNALERROR> boom")},
    "nothing-collected": {"tests": SuiteResults(exit_code=5, command="python -m pytest -q")},
    "nothing-counted-by-a-custom-command": {
        "tests": SuiteResults(exit_code=0, command="make test", runner="exit-code")
    },
    "nothing-in-the-junit-reports": {
        "tests": SuiteResults(exit_code=0, command="mvn -B test", runner="junit")
    },
    "coverage-switched-off": {"threshold": 0.0},
    "coverage-tool-missing": {"coverage": None, "changed_threshold": 0.8},
    "coverage-unmeasurable": {
        "coverage": CoverageResult(ratio=None, exit_code=1, stderr="No data to report.")
    },
    "coverage-below-the-bar": {"coverage": CoverageResult(ratio=0.5)},
    "changed-lines-unmeasurable": {"threshold": 0.0, "changed_threshold": 0.8},
    "changed-files-without-coverage-data": {
        "coverage": covering(
            changed_lines_total=0, changed_lines_covered=0, unmapped_sources=MANY_FILES
        ),
        "changed_threshold": 0.8,
    },
    "changed-lines-below-the-bar": {
        "coverage": covering(changed_lines_total=30, changed_lines_covered=1, uncovered=UNCOVERED),
        "changed_threshold": 0.8,
    },
    "no-executable-changed-line": {
        "coverage": covering(changed_lines_total=0, changed_lines_covered=0),
        "changed_threshold": 0.8,
    },
    "both-bars-met": {"coverage": covering(), "changed_threshold": 0.8},
    "impacted-tests-only": {"workspace": with_tests, "selection": "impacted", "threshold": 0.0},
    "impacted-tests-failed": {
        "workspace": with_tests,
        "selection": "impacted",
        "threshold": 0.0,
        "tests": SuiteResults(passed=1, failed=1, exit_code=1, failures=["FAILED a"]),
    },
    "selection-under-an-overall-bar": {"workspace": with_tests, "selection": "impacted"},
    "selection-without-impact-analysis": {
        "workspace": with_tests,
        "selection": "impacted",
        "language": "rust",
    },
    "selection-with-an-unparsable-file": {
        "workspace": with_an_unparsable_file,
        "selection": "impacted",
    },
    "selection-that-came-out-empty": {"selection": "impacted"},
    "generated-cases-passed": {"llm": proposing(GENERATED_CASE), "generated": GENERATED_PASSED},
    "generated-case-failed": {
        "llm": proposing(GENERATED_CASE),
        "generated": SuiteResults(
            failed=1,
            exit_code=1,
            failures=["FAILED tests/honest_ci_generated/test_split_refund.py::test_split"],
        ),
    },
    "generated-cases-never-started": {
        "llm": proposing(GENERATED_CASE),
        "generated": SuiteResults(available=False, errors=1, failures=["pytest: not found"]),
    },
    "generated-cases-failed-with-nothing-readable": {
        "llm": proposing(GENERATED_CASE),
        "generated": GENERATED_DID_NOT_COMPILE,
    },
    "generated-cases-counted-nothing": {
        "llm": proposing(GENERATED_CASE),
        "generated": SuiteResults(exit_code=0, runner="junit"),
    },
    "generation-switched-off": {"llm": proposing(GENERATED_CASE), "testgen": False},
    "generation-without-a-model": {},
    "generation-without-a-key": {"provider": "qwen"},
    "generation-with-nowhere-to-put-a-case": {"llm": proposing(GENERATED_CASE), "language": "rust"},
    "generation-without-a-changed-symbol": {
        "llm": proposing(GENERATED_CASE),
        "workspace": with_notes,
        "change": make_change(
            diff=NOTES_DIFF, files_changed=[ChangedFile(path="docs/notes.txt", added_lines=1)]
        ),
    },
    "generation-call-failed": {
        "llm": proposing(GENERATED_CASE, error=RuntimeError("provider down"))
    },
    "generation-discarded-at-the-trust-boundary": {
        "llm": proposing(
            ProposedTest(path="src/app/refund.py", content="x = 1\n", targets=["split_refund"])
        )
    },
    "generation-proposed-nothing": {"llm": proposing()},
    "generation-on-a-tree-nobody-could-check": {
        "llm": proposing(GENERATED_CASE),
        "change": make_change(diff="", files_changed=[]),
    },
}


@pytest.mark.parametrize("path", list(EVERY_PATH))
def test_every_way_the_gate_concludes_is_written_in_english(tmp_path: Path, path: str) -> None:
    report = run_path(tmp_path, **EVERY_PATH[path])
    evidence = EvidencePack(
        run_id="r1",
        # The factory's title is Chinese on purpose elsewhere; here it would
        # be the only Chinese on the page, and not the gate's.
        change=make_change(title="feat: split refunds by line"),
        gate_reports=[report],
        verdict=report.verdict,
        fingerprint="f" * 64,
    )

    # What a reader sees of the verdict -- the summary, every finding, the
    # evidence the gate records -- and the pages it reaches them on.
    written = [report.summary, json.dumps(report.verifier_report, ensure_ascii=False)]
    for finding in report.findings:
        written += [finding.title, finding.detail, finding.suggested_fix or ""]
    written += [
        render_evidence(evidence),
        render_forge_comment(evidence, status=_STATUS[report.verdict]),
    ]
    assert cjk_lines("\n".join(written)) == []
