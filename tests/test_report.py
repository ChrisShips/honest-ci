from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import pytest

from honest_ci.contracts import (
    ApprovalDecision,
    ApprovalReading,
    ApprovalWindow,
    ArtifactManifest,
    ArtifactRecord,
    BlockBrief,
    ChangedFile,
    EvidencePack,
    Finding,
    GateConfiguration,
    GateName,
    GateReport,
    LLMCallRecord,
    ModelFinding,
    ModelReviewReport,
    PlatformApproval,
    PlatformReview,
    PolicyDecision,
    ToolReport,
    UncountedApproval,
    Verdict,
)

# Aliased so pytest does not try to collect a contract as a test class.
from honest_ci.contracts import TestResults as SuiteResults
from honest_ci.diffs import markdown_code
from honest_ci.policy import PolicyConfig, PolicyEngine
from honest_ci.report import (
    APPROVED_HINT,
    NO_LIMITS,
    REVIEW_APPROVAL_HINT,
    STATUS_HINT,
    VERDICT_MARKER,
    approval_window_marker,
    inline_findings,
    read_approval_window,
    render_evidence,
    render_forge_comment,
)
from honest_ci.rules import Rule, RuleSet, evaluate_rules
from honest_ci.state import RunStatus

from .factories import (
    DIFF,
    HOSTILE_DIFF,
    HOSTILE_LINES,
    HOSTILE_NAME,
    HOSTILE_PROSE,
    HOSTILE_TEXT,
    headings,
    hostile_judgment_finding,
    make_change,
    make_weakened_change,
    markdown_outside_code,
    markdown_tokens,
    reaching_markdown,
    rendered_as_markdown,
    section,
    table_row,
    unverified,
)

BLOCKING = Finding(
    id="review.R-01",
    gate="review",
    rule_id="R-01",
    severity="high",
    title="Money is never computed in float",
    detail="Floating point leaves remainders nobody can explain in a split or a "
    "reconciliation.\nMatched: share = float(amount)",
    file="src/app/refund.py",
    line_start=11,
    line_end=11,
    suggested_fix="Use decimal.Decimal.",
)


def review(**overrides: Any) -> GateReport:
    payload: dict[str, Any] = {
        "gate": "review",
        "verdict": Verdict.BLOCKED,
        "summary": "Reviewed against 2 rules (v1); 1 finding, 1 blocking.",
        "findings": [BLOCKING],
    }
    payload.update(overrides)
    return GateReport.model_validate(payload)


def undeployed(gate: GateName) -> GateReport:
    return GateReport(
        gate=gate,
        verdict=Verdict.SKIPPED,
        summary=f"This deployment has no {gate} gate; recorded as skipped, which is not a pass.",
    )


def pack(**overrides: Any) -> EvidencePack:
    payload: dict[str, Any] = {
        "run_id": "r1",
        "change": make_change(),
        "gate_reports": [undeployed("spec"), review()],
        "verdict": Verdict.BLOCKED,
        "fingerprint": "fp-1",
        "policy_decision": PolicyDecision(
            allowed=True, policy_version="v1", required_checks=["spec", "review"]
        ),
    }
    payload.update(overrides)
    return EvidencePack.model_validate(payload)


def test_the_note_says_where_the_full_evidence_pack_can_be_read() -> None:
    """By the command that renders it from a kept ledger, or where the caller says.

    A CI job's ledger goes with its runner, and a note that sent a reader to
    ``honest-ci evidence`` from there would send them to a command with
    nothing to read.
    """

    kept = render_forge_comment(pack(), status="blocked")
    uploaded = render_forge_comment(
        pack(), status="blocked", evidence_at="the `honest-ci-evidence-7-1` artifact of this run"
    )

    assert "Full evidence pack: `honest-ci evidence r1`." in kept
    assert "Full evidence pack: the `honest-ci-evidence-7-1` artifact of this run." in uploaded
    assert "honest-ci evidence r1" not in uploaded


def test_what_the_license_check_says_comes_last_apart_from_what_the_pack_says() -> None:
    note = "honest-ci is on its 14-day trial here: 3 days left, through 2026-10-11."

    without = render_forge_comment(pack(), status="blocked")
    with_note = render_forge_comment(pack(), status="blocked", license_note=note)

    # The run's license is no part of the verdict: everything above it is as it was.
    assert with_note == f"{without}\n{note}\n"


def test_a_skipped_gate_is_never_rendered_as_a_pass() -> None:
    for text in (
        render_evidence(pack(), status="awaiting_human"),
        render_forge_comment(pack(), status="awaiting_human"),
    ):
        verdict, _ = table_row(text, "spec")
        assert verdict != "pass"
        assert "not a pass" in verdict
        assert [line for line in unverified(text) if "skipped" in line and "spec" in line]


def test_a_required_gate_without_a_conclusion_explains_the_handoff() -> None:
    text = render_evidence(pack(), status="awaiting_human")

    # spec is required, and a skip is not a verdict: that is why a person is asked.
    (handoff,) = [line for line in unverified(text) if "requires" in line]
    assert "spec" in handoff
    assert "needs_human" in handoff


def test_a_block_names_the_rule_the_line_the_reason_and_the_fix() -> None:
    text = render_evidence(pack())

    blocks = section(text, "Blocking findings (1)")
    assert "### [R-01] Money is never computed in float" in blocks
    block = section(blocks, "[R-01]")
    assert "`src/app/refund.py`:11" in block
    assert "remainders nobody can explain" in block
    assert "Use decimal.Decimal." in block
    assert "deterministic code" in block


def test_a_tool_that_never_ran_is_listed_as_something_nobody_verified() -> None:
    report = review(
        tool_reports=[ToolReport(name="semgrep", command="semgrep", available=False, passed=False)]
    )
    text = render_evidence(pack(gate_reports=[report]))

    (line,) = [line for line in unverified(text) if "semgrep" in line]
    assert line.startswith("review:")
    assert "did not start" in line
    assert "no evidence" in line


def test_a_tool_that_failed_without_output_is_listed_too() -> None:
    report = review(
        tool_reports=[ToolReport(name="ruff", command="ruff check", passed=False, exit_code=2)]
    )
    text = render_evidence(pack(gate_reports=[report]))

    (line,) = [line for line in unverified(text) if "ruff" in line]
    assert line.startswith("review:")
    assert "failed without output" in line


def regression(**overrides: Any) -> GateReport:
    payload: dict[str, Any] = {
        "gate": "regression",
        "verdict": Verdict.BLOCKED,
        "summary": "Regression tests: 3 passed, 1 failed, 0 errors, 0 skipped; a failure blocks.",
        "test_results": SuiteResults(
            passed=3,
            failed=1,
            exit_code=1,
            command="python -m coverage run -m pytest -q",
            duration_s=2.5,
            failures=["FAILED tests/test_refund.py::test_split - AssertionError"],
        ),
        "metrics": {"coverage_threshold": 0.7},
    }
    payload.update(overrides)
    return GateReport.model_validate(payload)


def test_the_suite_the_regression_gate_ran_is_part_of_the_reading() -> None:
    text = render_evidence(pack(gate_reports=[review(verdict=Verdict.PASS), regression()]))

    suite = section(text, "Regression tests")
    assert table_row(suite, "Results") == ["3 passed · 1 failed · 0 errors · 0 skipped"]
    assert table_row(suite, "Command") == ["`python -m coverage run -m pytest -q`"]
    assert "FAILED tests/test_refund.py::test_split - AssertionError" in suite
    # The suite passed nothing to coverage, so the bar could not be checked --
    # and the pack says that instead of showing a blank.
    (coverage,) = table_row(suite, "Line coverage")
    assert coverage.startswith("not measured")
    assert "70%" in coverage
    # The sections after it still number without a gap.
    assert "## 4. Blocking findings (1)" in text


def test_a_measured_coverage_is_shown_against_its_bar() -> None:
    report = regression(
        verdict=Verdict.PASS,
        test_results=SuiteResults(passed=4, exit_code=0, command="python -m pytest -q"),
        metrics={"coverage_threshold": 0.7, "coverage": 0.82},
    )
    text = render_evidence(pack(gate_reports=[report]))

    assert table_row(text, "Line coverage") == ["82% (threshold 70%)"]


def test_a_zero_bar_reads_as_switched_off_not_as_missing() -> None:
    def coverage(threshold: float) -> str:
        report = regression(
            verdict=Verdict.PASS,
            test_results=SuiteResults(passed=4, exit_code=0, command="python -m pytest -q"),
            metrics={"coverage_threshold": threshold},
        )
        (cell,) = table_row(render_evidence(pack(gate_reports=[report])), "Line coverage")
        return cell

    switched_off, missing = coverage(0.0), coverage(0.7)

    # Nobody asked for the number, or the number asked for never came: both
    # are "not measured", and only one of them is something to go and fix.
    assert switched_off.startswith("not measured")
    assert missing.startswith("not measured")
    assert "could not be checked" in missing
    assert "could not be checked" not in switched_off


def test_a_suite_that_never_ran_is_not_rendered_as_counts() -> None:
    report = regression(
        verdict=Verdict.BLOCKED,
        test_results=SuiteResults(
            available=False, errors=1, failures=["pytest is not installed in the sandbox."]
        ),
    )
    text = render_evidence(pack(gate_reports=[report]))

    suite = section(text, "Regression tests")
    first = suite.splitlines()[0]
    assert "did not start" in first
    assert first.endswith("`pytest is not installed in the sandbox.`")
    assert "passed" not in suite


def test_why_a_suite_never_ran_is_quoted_as_what_it_printed() -> None:
    # The first line the run printed: the change's conftest can decide it.
    report = regression(
        verdict=Verdict.BLOCKED,
        test_results=SuiteResults(available=False, errors=1, stderr=f"{HOSTILE_TEXT}\nmore"),
    )
    text = render_evidence(pack(gate_reports=[report]))

    assert rendered_as_markdown(text, "Approved by security") == []
    first = section(text, "Regression tests").splitlines()[0]
    assert first.endswith(markdown_code(HOSTILE_TEXT))
    assert "more" not in first


def narrowed(**overrides: Any) -> GateReport:
    """A regression report from a run that narrowed the suite and held both bars."""

    payload: dict[str, Any] = {
        "verdict": Verdict.BLOCKED,
        "summary": "Regression tests: 3 passed; changed-line coverage 33% (1/3) is below "
        "the 80% threshold, so it blocks.",
        "test_results": SuiteResults(passed=3, exit_code=0, command="python -m pytest -q"),
        "metrics": {
            "coverage": 0.9,
            "coverage_threshold": 0.7,
            "changed_line_coverage_threshold": 0.8,
            "changed_lines_total": 3.0,
            "changed_lines_covered": 1.0,
            "changed_line_coverage": 1 / 3,
        },
        "verifier_report": {
            "test_selection": {
                "mode": "impacted",
                "requested": ["tests/test_refund.py"],
                "fallback": None,
            },
            "impact": {
                "language": "python",
                "analysed": True,
                "changed_symbols": [{"symbol": "split_refund"}],
                "impacted_tests": ["tests/test_refund.py"],
                "unanalysed_files": ["src/app/legacy.py"],
            },
            "coverage": {
                "available": True,
                "ratio": 0.9,
                "changed_lines": {
                    "threshold": 0.8,
                    "total": 3,
                    "covered": 1,
                    "uncovered": ["src/app/refund.py:12"],
                    "no_coverage_data": ["src/app/new.py"],
                    "not_measured": ["tests/test_refund.py"],
                },
            },
        },
    }
    payload.update(overrides)
    return regression(**payload)


def test_the_change_s_own_coverage_is_read_next_to_the_repository_s() -> None:
    text = render_evidence(pack(gate_reports=[narrowed()]))

    # A repository at 90% that covered one of this change's three lines: the
    # two numbers must not be able to stand in for each other.
    assert table_row(text, "Line coverage") == ["90% (threshold 70%)"]
    assert table_row(text, "Changed-line coverage") == ["33% · 1/3 lines (threshold 80%)"]
    assert "- `src/app/refund.py:12`" in section(text, "Regression tests")


def test_a_narrowed_run_says_so_in_the_table_and_in_the_limits() -> None:
    text = render_evidence(pack(gate_reports=[narrowed()]))

    (selection,) = table_row(text, "Test selection")
    assert "1 impacted test" in selection
    assert "weaker evidence than the full suite" in selection
    (line,) = [line for line in unverified(text) if "HONEST_CI_TEST_SELECTION" in line]
    assert line.startswith("regression:")
    assert "1 impacted test" in line
    assert "the rest of the suite did not run" in line


def test_files_the_analysis_and_the_coverage_report_missed_are_named() -> None:
    lines = unverified(render_evidence(pack(gate_reports=[narrowed()])))

    (unanalysed,) = [line for line in lines if "`src/app/legacy.py`" in line]
    assert "impact analysis is empty" in unanalysed
    (uncovered,) = [line for line in lines if "`src/app/new.py`" in line]
    assert "the coverage report has no data" in uncovered


def test_files_coverage_never_measured_are_named_apart_from_the_ones_it_missed() -> None:
    lines = unverified(render_evidence(pack(gate_reports=[narrowed()])))

    # Not a gap in the evidence -- the configuration says these are not
    # measured -- but still a part of the change the bar says nothing about.
    (line,) = [line for line in lines if "`tests/test_refund.py`" in line]
    assert "outside what the coverage configuration measures" in line
    assert "`src/app/new.py`" not in line


def test_a_full_run_is_not_dressed_up_as_a_narrowed_one() -> None:
    report = narrowed(
        verifier_report={
            "test_selection": {
                "mode": "all",
                "requested": None,
                "fallback": "no test was found to be impacted",
            }
        }
    )
    text = render_evidence(pack(gate_reports=[report]))

    (selection,) = table_row(text, "Test selection")
    assert selection.startswith("full suite")
    assert "no test was found to be impacted" in selection
    assert not [line for line in unverified(text) if "impacted" in line]


def generated(**payload: Any) -> GateReport:
    """A regression report whose run also wrote regression cases of its own."""

    return regression(
        summary="Regression tests: 3 passed; line coverage 90%, at or above the 70% threshold.",
        test_results=SuiteResults(passed=3, exit_code=0, command="python -m pytest -q"),
        metrics={"coverage": 0.9, "coverage_threshold": 0.7},
        verifier_report={"generated_tests": {"proposed": 2, "paths": [], "targets": [], **payload}},
    )


def test_generated_cases_are_counted_apart_from_the_suite_that_decided() -> None:
    text = render_evidence(pack(gate_reports=[generated(ran=True, passed=2, failed=0)]))

    # The verdict rests on the 3 cases the repository already had. Adding the
    # generated ones into that number would inflate the evidence with cases
    # a model wrote minutes ago.
    (cases,) = table_row(text, "Generated regression tests")
    assert cases.startswith("2 passed · 0 failed")
    assert "not counted" in cases
    assert table_row(text, "Results") == ["3 passed · 0 failed · 0 errors · 0 skipped"]


def test_a_generated_case_that_failed_is_marked_as_a_model_s_word() -> None:
    report = generated(ran=True, passed=1, failed=1)
    text = render_evidence(pack(gate_reports=[report]))
    note = render_forge_comment(pack(gate_reports=[report]), status="awaiting_human")

    (cases,) = table_row(text, "Generated regression tests")
    assert cases.startswith("1 passed · 1 failed")
    assert "model's judgment" in cases
    assert "not blocking" in cases
    (suite,) = table_row(note, "Regression tests")
    assert "1 generated test failed (not blocking" in suite


def test_nothing_generated_is_a_row_and_a_limit_rather_than_a_silence() -> None:
    report = generated(
        proposed=0,
        ran=False,
        not_generated_reason="no model credential is configured (DEEPSEEK_API_KEY)",
    )
    text = render_evidence(pack(gate_reports=[report]))

    # Without the row, a reader who knows the deployment generates cases would
    # read the silence as "generated cases ran and none of them failed".
    (cases,) = table_row(text, "Generated regression tests")
    assert cases.startswith("none generated")
    assert "DEEPSEEK_API_KEY" in cases
    (line,) = [line for line in unverified(text) if "DEEPSEEK_API_KEY" in line]
    assert line.startswith("regression:")
    assert "does not mean this change was tested" in line


def test_a_pack_without_test_results_has_no_test_section() -> None:
    text = render_evidence(pack())

    assert "Regression tests" not in headings(text)
    assert "## 3. Blocking findings (1)" in text


def test_the_forge_note_carries_the_suite_in_one_row() -> None:
    text = render_forge_comment(pack(gate_reports=[regression()]), status="blocked")

    (suite,) = table_row(text, "Regression tests")
    assert suite.startswith("3 passed · 1 failed · 0 errors · coverage not measured")
    assert "70%" in suite


def test_a_run_with_nothing_left_unverified_says_so_plainly() -> None:
    clean = GateReport(
        gate="review", verdict=Verdict.PASS, summary="Reviewed against 2 rules; no findings."
    )
    text = render_evidence(
        pack(
            gate_reports=[clean],
            verdict=Verdict.PASS,
            policy_decision=PolicyDecision(
                allowed=True, policy_version="v1", required_checks=["review"]
            ),
        ),
        status="mergeable",
    )

    assert unverified(text) == [NO_LIMITS]


def test_a_structural_guardrail_hit_is_part_of_the_reading() -> None:
    decision = PolicyDecision(
        allowed=False,
        policy_version="v1",
        risk_level="high",
        violations=["sensitive paths require human review: src/auth/login.py"],
        required_checks=["review"],
    )
    text = render_evidence(pack(policy_decision=decision), status="awaiting_human")

    (line,) = [line for line in unverified(text) if "src/auth/login.py" in line]
    assert "guardrails" in line
    assert "risk high" in line


def test_a_file_with_no_reviewable_content_is_called_out() -> None:
    change = make_change(
        files_changed=[
            ChangedFile(path="src/app/refund.py", added_lines=3),
            ChangedFile(path="assets/logo.png", change_type="added"),
        ]
    )
    text = render_evidence(pack(change=change))

    (line,) = [line for line in unverified(text) if "`assets/logo.png`" in line]
    assert "no content to review line by line" in line
    assert "`src/app/refund.py`" not in line


def test_a_name_refused_as_a_path_is_shown_escaped_and_not_as_unreadable_content() -> None:
    # An escape sequence that erases a line, and a newline that opens a heading.
    hostile = "src/app/refund\x1b[2K\n## Approved.py"
    change = make_change(
        files_changed=[
            ChangedFile(path="src/app/refund.py", added_lines=3),
            ChangedFile(path=hostile, change_type="added"),
        ]
    )
    # A path rule fires on a name alone.
    finding = BLOCKING.model_copy(update={"file": hostile, "line_start": None, "line_end": None})
    evidence = pack(change=change, gate_reports=[undeployed("spec"), review(findings=[finding])])

    for text in (render_evidence(evidence), render_forge_comment(evidence, status="blocked")):
        assert "\x1b" not in text
        assert "\n## Approved" not in text
        assert repr(hostile) in text
    # What it is, the guardrails say; this line would call it binary or empty.
    assert not [line for line in unverified(render_evidence(evidence)) if "no content" in line]


def test_a_file_name_cannot_close_its_code_span_and_render_as_markdown() -> None:
    change = make_change(
        files_changed=[
            ChangedFile(path="src/app/refund.py", added_lines=3),
            # With no line counts it is also listed as content nobody could read.
            ChangedFile(path=HOSTILE_NAME, change_type="added"),
        ],
        platform_reviews=[OPEN_THREAD.model_copy(update={"path": HOSTILE_NAME})],
    )
    decision = PolicyEngine(PolicyConfig(version="v1", sensitive_paths=["src/**"])).evaluate_change(
        change
    )
    measured = narrowed()
    report = measured.verifier_report
    changed_lines = {
        **report["coverage"]["changed_lines"],
        "uncovered": [f"{HOSTILE_NAME}:12"],
        "no_coverage_data": [HOSTILE_NAME],
        "not_measured": [HOSTILE_NAME],
    }
    measured = measured.model_copy(
        update={
            "verifier_report": {
                **report,
                "impact": {**report["impact"], "unanalysed_files": [HOSTILE_NAME]},
                "coverage": {**report["coverage"], "changed_lines": changed_lines},
            }
        }
    )
    finding = BLOCKING.model_copy(update={"file": HOSTILE_NAME})
    evidence = pack(
        change=change,
        gate_reports=[undeployed("spec"), review(findings=[finding]), measured],
        policy_decision=decision,
        configuration=configuration(rules_path=HOSTILE_NAME),
    )
    # A patch has no file list; its files are read off the diff.
    patch = make_change(files_changed=[], diff=f"+++ b/{HOSTILE_NAME}\n@@ -0,0 +1 @@\n+x = 1\n")
    aid = ModelReviewReport(
        evidence_fingerprint="fp-1",
        summary="Look closely at the precision.",
        change_summary="Refunds are split by line item",
        findings=[
            ModelFinding(
                title="The split's remainder is not handled",
                file=HOSTILE_NAME,
                line_start=1,
                line_end=1,
            )
        ],
    )

    for text in (
        render_evidence(evidence, assistant_report=aid),
        render_forge_comment(evidence, status="blocked"),
        render_evidence(pack(change=patch)),
    ):
        tokens = markdown_outside_code(text)
        assert HOSTILE_NAME in [token.content for token in tokens if token.type == "code_inline"]
        assert not [
            token
            for token in tokens
            if token.type != "code_inline" and "Approved by security" in token.content
        ]
        assert not [token for token in tokens if token.type in {"image", "link_open"}]


#: A rule the hostile lines trip, so each of them is quoted in a finding.
SELF_APPROVAL = Rule(
    id="R-07",
    title="Code must not vouch for its own review",
    kind="regex",
    severity="high",
    pattern=re.compile("Approved by security"),
)


def test_a_line_that_trips_a_rule_is_quoted_as_code_and_never_renders_as_markdown() -> None:
    change = make_change(diff=HOSTILE_DIFF)
    findings = evaluate_rules(RuleSet(version="v1", rules=(SELF_APPROVAL,)), change)
    evidence = pack(change=change, gate_reports=[undeployed("spec"), review(findings=findings)])

    assert len(findings) == len(HOSTILE_LINES)
    for text in (render_evidence(evidence), render_forge_comment(evidence, status="blocked")):
        assert rendered_as_markdown(text, "Approved by security") == []
    # Inert, and still the line: every one of them is quoted whole.
    quoted = [
        token.content
        for token in markdown_outside_code(render_evidence(evidence))
        if token.type == "code_inline" and "Approved by security" in token.content
    ]
    assert len(quoted) == len(HOSTILE_LINES)
    assert "\\r**Approved by security**" in quoted[2]
    assert quoted[3].endswith("# token=[REDACTED]")


def test_what_the_author_the_suite_or_a_commenter_wrote_never_renders_in_the_pack() -> None:
    change = make_change(
        title=HOSTILE_TEXT,
        platform_reviews=[OPEN_THREAD.model_copy(update={"body": HOSTILE_TEXT})],
    )
    suite = regression().test_results
    assert suite is not None
    failing = f"FAILED tests/test_refund.py::test_split[{HOSTILE_TEXT}] - AssertionError"
    red = regression(test_results=suite.model_copy(update={"failures": [failing]}))

    text = render_evidence(pack(change=change, gate_reports=[undeployed("spec"), review(), red]))

    assert rendered_as_markdown(text, "Approved by security") == []
    # The title, the comment and the failing test are each still there to read.
    quoted = [
        token
        for token in markdown_outside_code(text)
        if token.type == "code_inline" and "Approved by security" in token.content
    ]
    assert len(quoted) == 3


def test_a_patch_without_a_git_file_list_still_reports_its_files() -> None:
    change = section(render_evidence(pack(change=make_change(files_changed=[]))), "The change")

    # The gates judged this diff; rendering "0 files" would understate it.
    assert "- Size: 1 file / " in change
    assert "read off the diff" in change
    assert "- `src/app/refund.py`" in change


def test_a_rename_is_shown_under_both_names_and_counted_under_both() -> None:
    change = make_change(
        files_changed=[
            ChangedFile(path="src/app/refund.py", added_lines=3),
            ChangedFile(
                path="docs/deploy.yml",
                change_type="renamed",
                previous_path=".github/workflows/deploy.yml",
            ),
        ]
    )

    text = render_evidence(pack(change=change))

    # The same count the size limits were held to, and the table says which
    # two names the one row stands for.
    (size,) = [line for line in text.splitlines() if line.startswith("- Size:")]
    assert size.startswith("- Size: 3 files")
    assert "rename" in size
    assert table_row(text, "`.github/workflows/deploy.yml` → `docs/deploy.yml`")[0] == "renamed"


def test_a_file_moved_without_edits_is_counted_as_the_limits_counted_it() -> None:
    move = (
        "--- a/tests/test_ledger.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-def test_a():\n-    pass\n"
        "--- /dev/null\n+++ b/tests/ledger/test_ledger.py\n@@ -0,0 +1,2 @@\n+def test_a():\n+    pass\n"
    )
    change = make_change(
        diff=DIFF + move,
        files_changed=[
            ChangedFile(path="src/app/refund.py", added_lines=3),
            ChangedFile(path="tests/test_ledger.py", change_type="deleted", removed_lines=2),
            ChangedFile(path="tests/ledger/test_ledger.py", change_type="added", added_lines=2),
        ],
    )

    text = render_evidence(pack(change=change))

    # The table still shows the move's lines going and coming; the size line
    # is the count the limits were held to, and says why it is smaller.
    (size,) = [line for line in text.splitlines() if line.startswith("- Size:")]
    assert size.startswith("- Size: 3 files / 3 lines")
    assert "moved without edits" in size
    assert table_row(text, "`tests/test_ledger.py`") == ["deleted", "+0 / -2"]


def test_model_text_is_advisory_and_carries_the_evidence_it_cited() -> None:
    brief = BlockBrief(
        gate="review",
        attempt=1,
        source_fingerprint="src-1",
        summary="The refund is split in float.",
        root_causes=["Decimal is not used"],
        recommended_actions=["Use decimal.Decimal"],
        cited_evidence_ids=["review.R-01"],
    )
    text = render_evidence(pack(briefs=[brief], llm_calls=[LLMCallRecord(model="scripted-1")]))

    said = section(text, "What the model said")
    # The label comes before anything the model wrote.
    assert said.startswith("**Advisory only")
    assert "### Why gate review blocked (attempt 1)" in said
    assert "`review.R-01`" in section(said, "Why gate review blocked")


def test_a_reading_aid_is_rendered_apart_from_the_gate_findings() -> None:
    aid = ModelReviewReport(
        evidence_fingerprint="fp-1",
        summary="Look closely at the precision.",
        change_summary="Refunds are split by line item",
        review_questions=["Who absorbs the remainder?"],
        findings=[
            ModelFinding(
                title="The split's remainder is not handled",
                file="src/app/refund.py",
                line_start=11,
                line_end=11,
                confidence=0.4,
            )
        ],
    )
    text = render_evidence(pack(), assistant_report=aid)

    reading_aid = section(section(text, "What the model said"), "Reading aid")
    assert "Who absorbs the remainder?" in reading_aid
    assert "**not gate verdicts**" in reading_aid
    # Its doubt is there and nowhere else: not among the findings the gates raised.
    assert "The split's remainder is not handled" in reading_aid
    assert text.count("The split's remainder is not handled") == 1


def test_missing_model_text_says_which_kind_of_missing_it_is() -> None:
    def said(evidence: EvidencePack) -> str:
        return section(render_evidence(evidence), "What the model said")

    not_asked = said(pack())
    # Names the switch a reader would flip, not only "there was no model".
    assert "no model is enabled (`HONEST_CI_MODEL_PROVIDER`)" in not_asked
    assert "its credential is missing" in not_asked

    failed = said(
        pack(llm_calls=[LLMCallRecord(model="scripted-1", schema_valid=False, error="timeout")])
    )
    assert "no valid output" in failed
    assert "`llm-calls.json`" in failed

    discarded = said(pack(llm_calls=[LLMCallRecord(model="scripted-1")]))
    assert "failed the citation check or the line check" in discarded


def test_an_unmeasured_metric_is_never_rendered_as_zero() -> None:
    text = render_evidence(pack())

    assert table_row(text, "Human review") == ["not recorded"]
    # Measured, so it is reported as measured.
    assert "- AI-generated share: 71%" in section(text, "The change")


def test_an_approval_is_rendered_next_to_the_evidence_it_signed() -> None:
    decision = ApprovalDecision(
        approved=True, fingerprint="fp-1", reason="Product confirmed the rounding", reviewer="chris"
    )
    text = render_evidence(pack(), status="mergeable", approval=decision)

    human = section(text, "Human decision")
    assert "- Decision: **approved by a person**" in human
    assert "- Reviewer: chris" in human
    assert "- Bound to fingerprint: `fp-1`" in human
    assert "once the fingerprint changes, this sign-off no longer applies" in human


def test_sections_are_numbered_without_gaps_whether_or_not_a_human_signed() -> None:
    unsigned = re.findall(r"^## (\d+)\.", render_evidence(pack()), flags=re.MULTILINE)
    assert [int(number) for number in unsigned] == list(range(1, len(unsigned) + 1))

    signed = render_evidence(
        pack(), approval=ApprovalDecision(approved=False, fingerprint="fp-1")
    )
    numbers = re.findall(r"^## (\d+)\.", signed, flags=re.MULTILINE)
    assert [int(number) for number in numbers] == list(range(1, len(numbers) + 1))
    assert len(numbers) == len(unsigned) + 1


# ── approvals ────────────────────────────────────────────────────────────────

WINDOW = ApprovalWindow(fingerprint="fp-1", after=datetime(2026, 9, 1, 10, 0, tzinfo=UTC))
REVIEW_URL = "https://github.com/octo-org/orders/pull/7#pullrequestreview-1"


def waiting() -> EvidencePack:
    """Every gate that ran passed; the policy requires a spec gate this deployment lacks."""

    return pack(
        gate_reports=[undeployed("spec"), review(verdict=Verdict.PASS, findings=[])],
        verdict=Verdict.NEEDS_HUMAN,
    )


def approval_section(status: RunStatus, reading: ApprovalReading) -> str:
    """The comment's approval section, for a run that ended at ``status``."""

    return section(render_forge_comment(waiting(), status=status, approvals=reading), "Approval")


def approval(**overrides: Any) -> PlatformApproval:
    payload: dict[str, Any] = {
        "reviewer": "octo-lead",
        "state": "approved",
        "commit_sha": "a" * 40,
        "reason": "Checked the rounding with finance.",
        "can_push": True,
        "submitted_at": datetime(2026, 9, 1, 10, 5, tzinfo=UTC),
        "url": REVIEW_URL,
    }
    payload.update(overrides)
    return PlatformApproval.model_validate(payload)


def test_a_run_a_person_approved_is_not_described_as_one_every_gate_passed() -> None:
    decision = ApprovalDecision(
        approved=True,
        fingerprint="fp-1",
        reason="Checked the rounding with finance.",
        reviewer="octo-lead",
        url=REVIEW_URL,
    )

    text = render_evidence(waiting(), status="mergeable", approval=decision)

    assert table_row(text, "Outcome") == [f"**approved** · {APPROVED_HINT}"]
    assert STATUS_HINT["mergeable"] not in text
    human = section(text, "Human decision")
    assert "- Reason: `Checked the rounding with finance.`" in human
    assert f"- Given in: `{REVIEW_URL}`" in human


def test_a_reason_in_the_pack_is_quoted_as_the_reviewers_words() -> None:
    decision = ApprovalDecision(
        approved=True, fingerprint="fp-1", reason=HOSTILE_TEXT, reviewer="octo-lead"
    )

    human = section(
        render_evidence(waiting(), status="mergeable", approval=decision), "Human decision"
    )

    assert rendered_as_markdown(human, "example.invalid") == []


@pytest.mark.parametrize(
    "window",
    [WINDOW, ApprovalWindow(fingerprint="f" * 64)],
    ids=["after-a-review", "nothing-before-it"],
)
def test_the_window_rides_on_the_line_under_the_verdict_marker_and_reads_back(
    window: ApprovalWindow,
) -> None:
    comment = render_forge_comment(
        waiting(), status="awaiting_human", approvals=ApprovalReading(window=window)
    )

    assert comment.splitlines()[:2] == [VERDICT_MARKER, approval_window_marker(window)]
    assert read_approval_window(comment) == window


def test_a_run_that_read_no_approvals_records_no_window() -> None:
    for approvals in (None, ApprovalReading(error="GitHub answered 502")):
        comment = render_forge_comment(waiting(), status="awaiting_human", approvals=approvals)

        # The next run then takes every review already given as given before.
        assert read_approval_window(comment) is None
        assert approval_window_marker(WINDOW).split("fingerprint=")[0] not in comment


def test_a_window_line_the_change_wrote_further_down_opens_no_window() -> None:
    planted = BLOCKING.model_copy(
        update={"detail": f"Matched:\n{approval_window_marker(WINDOW)}\nend"}
    )
    comment = render_forge_comment(
        pack(gate_reports=[review(findings=[planted])]), status="blocked"
    )

    assert read_approval_window(comment) is None
    # Written on the second line by hand, under someone else's first line, it
    # is no window either: only the verdict marker's own comment carries one.
    assert read_approval_window(f"LGTM\n{approval_window_marker(WINDOW)}") is None


@pytest.mark.parametrize(
    "line",
    [
        "<!-- honest-ci:approvals fingerprint=fp-1 after=yesterday -->",
        "<!-- honest-ci:approvals fingerprint=fp-1 after=2026-13-01T10:00:00+00:00 -->",
        "<!-- honest-ci:approvals fingerprint=fp 1 after=none -->",
        "<!-- honest-ci:approvals fingerprint=fp-1 after=none --> trailing",
        "<!-- honest-ci:approvals fingerprint= after=none -->",
    ],
    ids=["words", "no-such-month", "space-in-fingerprint", "trailing-text", "no-fingerprint"],
)
def test_a_window_line_that_does_not_read_cleanly_is_no_window(line: str) -> None:
    assert read_approval_window(f"{VERDICT_MARKER}\n{line}\n## honest-ci") is None


def test_a_window_recorded_without_a_zone_is_read_as_utc() -> None:
    comment = (
        f"{VERDICT_MARKER}\n"
        "<!-- honest-ci:approvals fingerprint=fp-1 after=2026-09-01T10:00:00 -->\n"
    )

    assert read_approval_window(comment) == WINDOW


def test_a_run_waiting_on_a_person_says_how_to_approve_it_on_the_pull_request() -> None:
    comment = render_forge_comment(
        waiting(), status="awaiting_human", approvals=ApprovalReading(window=WINDOW)
    )

    assert "## honest-ci · awaiting_human" in comment
    assert f"**{REVIEW_APPROVAL_HINT}**" in comment
    how = section(comment, "Approval")
    assert "can push to this repository" in how
    assert "on commit `aaaaaaaaaaaa` after this conclusion was published" in how
    assert "its text gives the reason" in how
    assert "A new commit, or a change to the rules or the policy, needs a new approval." in how


def test_where_approvals_are_not_read_the_comment_does_not_ask_for_one() -> None:
    # A fork's pull request, whose approvals are not read: an Approve review
    # there changes nothing.
    comment = render_forge_comment(waiting(), status="awaiting_human")

    assert f"**{STATUS_HINT['awaiting_human']}**" in comment
    assert REVIEW_APPROVAL_HINT not in comment
    assert "Approval" not in headings(comment)


def test_approvals_that_do_not_count_are_listed_with_the_reason_each_does_not() -> None:
    reading = ApprovalReading(
        window=WINDOW,
        uncounted=[
            UncountedApproval(review=approval(reason=""), why="it gives no reason"),
            UncountedApproval(
                review=approval(reviewer="", url=None),
                why="the account that gave it no longer exists",
            ),
        ],
        objections=[approval(reviewer="octo-qa", state="changes_requested")],
    )

    how = approval_section("awaiting_human", reading)

    assert f"- `octo-lead` ([review]({REVIEW_URL})): it gives no reason." in how
    assert "- `(deleted account)`: the account that gave it no longer exists." in how
    assert "Requests for changes that stand." in how
    assert f"- `octo-qa` ([review]({REVIEW_URL}))" in how


def test_long_lists_of_approvals_and_objections_are_cut_and_counted() -> None:
    uncounted = [
        UncountedApproval(review=approval(reviewer=f"octo-{index}"), why="it gives no reason")
        for index in range(13)
    ]
    objections = [
        approval(reviewer=f"octo-qa-{index}", state="changes_requested") for index in range(12)
    ]
    reading = ApprovalReading(window=WINDOW, uncounted=uncounted, objections=objections)

    how = approval_section("awaiting_human", reading)

    assert "`octo-9`" in how and "`octo-10`" not in how
    assert "- (and 3 more)" in how
    assert "`octo-qa-9`" in how and "`octo-qa-10`" not in how
    assert "- (and 2 more)" in how


def test_approvals_that_could_not_be_read_are_said_to_have_changed_nothing() -> None:
    reading = ApprovalReading(error="GitHub answered 502")

    comment = render_forge_comment(waiting(), status="awaiting_human", approvals=reading)

    assert (
        "⚠ No approval was applied: the reviews on this pull request could not be read "
        "or applied (`GitHub answered 502`)."
    ) in section(comment, "Approval")


def test_an_approved_run_is_headed_approved_and_names_who_signed_on_what_and_why() -> None:
    reading = ApprovalReading(window=WINDOW, approval=approval())

    comment = render_forge_comment(waiting(), status="mergeable", approvals=reading)

    assert "## honest-ci · approved" in comment
    assert f"**{APPROVED_HINT}**" in comment
    # "Every gate passed" is exactly what an approval does not mean.
    assert STATUS_HINT["mergeable"] not in comment
    how = section(comment, "Approval")
    assert (
        f"Approved by `octo-lead` ([review]({REVIEW_URL})) on commit `aaaaaaaaaaaa`: "
        "`Checked the rounding with finance.`"
    ) in how
    assert "bound to evidence fingerprint `fp-1…`" in how


def test_an_approval_on_a_run_that_did_not_end_mergeable_is_not_announced() -> None:
    # The runner refused it, say: the reading still holds the approval it found.
    reading = ApprovalReading(window=WINDOW, approval=approval())

    comment = render_forge_comment(waiting(), status="awaiting_human", approvals=reading)

    assert "## honest-ci · awaiting_human" in comment
    assert "Approved by" not in comment


def test_a_reviewers_words_never_render_as_markdown_or_reach_anyone() -> None:
    hostile = approval(reason=f"{HOSTILE_TEXT} @octo-org/everyone", url=None)
    reading = ApprovalReading(
        window=WINDOW,
        approval=hostile,
        uncounted=[UncountedApproval(review=hostile, why="it gives no reason")],
    )

    for status in ("mergeable", "awaiting_human"):
        how = section(
            render_forge_comment(waiting(), status=status, approvals=reading), "Approval"
        )

        assert reaching_markdown(how) == []
        assert rendered_as_markdown(how, "example.invalid") == []


@pytest.mark.parametrize(
    "url",
    [
        f"{REVIEW_URL}) [more](https://example.invalid",
        "javascript:alert(1)",
        "https://github.com/octo-org/orders/pull/7?token=abc",
        "//example.invalid/review",
    ],
    ids=["closes-the-link", "script", "key-value", "no-scheme"],
)
def test_a_review_is_only_linked_from_an_address_that_cannot_break_out_of_the_link(
    url: str,
) -> None:
    reading = ApprovalReading(window=WINDOW, approval=approval(url=url))

    how = approval_section("mergeable", reading)

    assert "[review](" not in how
    assert how.startswith("Approved by `octo-lead` on commit")


def test_a_long_reason_is_cut_in_the_comment() -> None:
    reason = "Checked every rounding case with finance. " * 30
    reading = ApprovalReading(window=WINDOW, approval=approval(reason=reason))

    how = approval_section("mergeable", reading)

    [quoted] = re.findall(r": `([^`]*)`$", how.splitlines()[0])
    assert quoted.endswith("…")
    assert len(quoted) == 301


def test_artifacts_are_rendered_with_the_digests_they_can_be_checked_against() -> None:
    manifest = ArtifactManifest(
        run_id="r1",
        artifacts=[
            ArtifactRecord(name="evidence.json", path="r1/evidence.json", sha256="a" * 64, size_bytes=12)
        ],
    )
    text = render_evidence(pack(artifact_manifest=manifest))

    assert "`evidence.json`" in text
    assert "a" * 16 in text
    assert "manifest.json" in text


def test_a_credential_in_a_finding_never_reaches_the_rendering() -> None:
    leaked = BLOCKING.model_copy(
        update={"detail": "Matched: token = sk-abcdefghijklmnopqrstuvwx"}
    )
    text = render_evidence(pack(gate_reports=[review(findings=[leaked])]))

    assert "sk-abcdefghijklmnopqrstuvwx" not in text
    assert "[REDACTED]" in text


def test_evidence_limits_is_empty_when_everything_ran_and_the_rendering_still_says_so() -> None:
    from honest_ci.report import evidence_limits

    clean = pack(
        gate_reports=[review(verdict=Verdict.PASS, findings=[])],
        verdict=Verdict.PASS,
        policy_decision=PolicyDecision(allowed=True, policy_version="v1", required_checks=["review"]),
    )

    assert evidence_limits(clean) == []
    assert f"- {NO_LIMITS}" in render_evidence(clean, status="mergeable")
    first = evidence_limits(pack())[0]
    assert "skipped" in first and "spec" in first


# ── judgment-class rules ─────────────────────────────────────────────────────


def judged(**overrides: Any) -> GateReport:
    judgment: dict[str, Any] = {
        "rules": [
            {"id": "J-01", "title": "Order status changes go through the state machine",
             "severity": "high", "min_confidence": 0.7}
        ],
        "judged": True,
        "not_judged_reason": None,
        "findings": 1,
        "below_threshold": [],
        "questions": [],
        "summary": "Read this change against the judgment rules.",
    }
    judgment.update(overrides)
    return GateReport(
        gate="review",
        verdict=Verdict.NEEDS_HUMAN,
        summary="1 judgment rule put to a model.",
        verifier_report={"judgment": judgment},
    )


def test_a_judgment_rule_nobody_put_to_a_model_is_named_as_a_limit() -> None:
    from honest_ci.report import evidence_limits

    limits = evidence_limits(
        pack(
            gate_reports=[judged(judged=False, not_judged_reason="no model credential")],
            verdict=Verdict.NEEDS_HUMAN,
        )
    )

    # The silence this section exists to break: an unread rule leaves no
    # finding behind, so the only trace of it is this line.
    assert any("J-01" in item and "no model credential" in item for item in limits)


def test_doubt_below_the_threshold_is_reported_as_a_limit_not_a_finding() -> None:
    from honest_ci.report import evidence_limits

    limits = evidence_limits(
        pack(
            gate_reports=[
                judged(
                    findings=0,
                    below_threshold=["[J-01] State machine · a.py:3 (confidence 0.40 < 0.70)"],
                )
            ]
        )
    )

    assert any("confidence 0.40 < 0.70" in item for item in limits)


def test_the_model_section_separates_judgment_output_from_the_verdict() -> None:
    text = render_evidence(
        pack(
            gate_reports=[judged(questions=["Which state does a failed refund roll back to?"])],
            verdict=Verdict.NEEDS_HUMAN,
            llm_calls=[LLMCallRecord(model="m", schema_name="ModelReviewReport")],
        )
    )

    judgment = section(section(text, "What the model said"), "Judgment rules · review")
    assert "Which state does a failed refund roll back to?" in judgment
    assert "**not blocking verdicts**" in judgment


def test_a_model_finding_is_rendered_as_advisory_and_not_under_the_blocks() -> None:
    finding = Finding(
        id="review.J-01.model.1",
        gate="review",
        rule_id="J-01",
        severity="high",
        title="Order status changes go through the state machine",
        file="src/app/refund.py",
        line_start=13,
        line_end=13,
        decided_by="model",
        confidence=0.9,
    )
    text = render_evidence(
        pack(gate_reports=[judged(findings=1).model_copy(update={"findings": [finding]})],
             verdict=Verdict.NEEDS_HUMAN)
    )

    # High severity, yet it must not appear as something that blocked: the
    # count in the heading is what a reader trusts.
    assert "Blocking findings (0)" in headings(text)
    assert "Other findings (1)" in headings(text)


def test_the_mr_comment_gives_judgment_findings_their_own_heading() -> None:
    finding = Finding(
        id="review.J-01.model.1",
        gate="review",
        rule_id="J-01",
        severity="high",
        title="Order status changes go through the state machine",
        file="src/app/refund.py",
        line_start=13,
        line_end=13,
        decided_by="model",
        confidence=0.85,
        suggested_fix="Change the status through OrderState.transition.",
    )
    text = render_forge_comment(
        pack(gate_reports=[judged().model_copy(update={"findings": [finding]})],
             verdict=Verdict.NEEDS_HUMAN),
        status="awaiting_human",
    )

    (heading,) = [title for title in headings(text) if title.startswith("For a person to judge")]
    assert "(1)" in heading
    assert "not blocking" in heading
    doubts = section(text, heading)
    assert "confidence 0.85" in doubts
    assert "Change the status through OrderState.transition." in doubts
    # It is the reason the change is waiting, so it must not be folded into the
    # "Also N non-blocking findings" tally that a reader skips.
    assert "non-blocking finding" not in text


def test_the_mr_comment_names_a_verdict_nobody_here_decided_for_a_person() -> None:
    """The note reads "who decided it" the same way the line comments do.

    An external verdict is as much a person's call as a model's doubt. Keyed
    on ``decided_by == "model"``, the note would have counted it in the "Also
    N non-blocking findings" tally, and on a platform that posts no line
    comments it would be named nowhere in the merge request.
    """

    external = BLOCKING.model_copy(
        update={"id": "review.EXT-1", "decided_by": "external:gitlab-duo"}
    )
    text = render_forge_comment(
        pack(gate_reports=[judged().model_copy(update={"findings": [external]})],
             verdict=Verdict.NEEDS_HUMAN),
        status="awaiting_human",
    )

    (heading,) = [title for title in headings(text) if title.startswith("For a person to judge")]
    assert "(1)" in heading
    assert external.title in section(text, heading)
    # High severity, and still not a block: nobody here decided it.
    assert "Blocking findings (0)" in headings(text)
    assert "non-blocking finding" not in text


def test_model_prose_formats_the_pack_and_the_note_but_never_reaches_out() -> None:
    brief = BlockBrief(
        gate="review",
        attempt=1,
        source_fingerprint="src-1",
        summary=HOSTILE_PROSE,
        root_causes=[HOSTILE_PROSE],
        recommended_actions=[HOSTILE_PROSE],
        cited_evidence_ids=["review.R-01"],
    )
    aid = ModelReviewReport(
        evidence_fingerprint="fp-1",
        summary=HOSTILE_PROSE,
        change_summary=HOSTILE_PROSE,
        review_questions=[HOSTILE_PROSE],
        findings=[
            ModelFinding(
                rule_id=HOSTILE_PROSE,
                title=HOSTILE_PROSE,
                file="src/app/refund.py",
                line_start=11,
                line_end=11,
            )
        ],
    )
    gate = judged(summary=HOSTILE_PROSE, questions=[HOSTILE_PROSE]).model_copy(
        update={"findings": [hostile_judgment_finding()]}
    )
    evidence = pack(gate_reports=[gate], briefs=[brief], verdict=Verdict.NEEDS_HUMAN)

    text = render_evidence(evidence, assistant_report=aid)
    note = render_forge_comment(evidence, status="awaiting_human")

    assert reaching_markdown(text) == []
    assert reaching_markdown(note) == []
    # None of it dropped either. In the pack: the judgment's summary and
    # question, the brief's summary, cause and action, and the reading aid's
    # change summary, summary and question, and its finding's rule and title.
    mentions = [token for token in markdown_tokens(text) if token.content == "@reviewer"]
    assert len(mentions) == 10
    # In the note, the judgment finding's question, as what to do about it.
    assert "@reviewer" in [token.content for token in markdown_tokens(note)]


def test_a_model_written_item_keeps_its_code_block_under_its_bullet() -> None:
    brief = BlockBrief(
        gate="review",
        attempt=1,
        source_fingerprint="src-1",
        summary="The refund is split with float.",
        root_causes=[
            "```\nitem.refund = share\n```",
            "The share is computed as:\n```python\nshare = float(amount)\n```",
        ],
        recommended_actions=["Use decimal.Decimal."],
        cited_evidence_ids=["review.R-01"],
    )

    tokens = markdown_tokens(render_evidence(pack(briefs=[brief])))

    # Inside the list item (list, item, block), not after the list it ended --
    # and a block that opens an item opens it on the bullet's own line.
    assert [(token.content, token.level) for token in tokens if token.type == "fence"] == [
        ("item.refund = share\n", 2),
        ("share = float(amount)\n", 2),
    ]


# ── review comments the platform already carried ─────────────────────────────

OPEN_THREAD = PlatformReview(
    author="wangwei",
    body="Shouldn't this split go through Decimal?",
    path="src/app/refund.py",
    line=11,
    resolved=False,
)
BOT_COMMENT = PlatformReview(
    author="gitlab-duo", body="Consider asserting on the remainder too.", is_bot=True
)


def reviewed(*reviews: PlatformReview, **change_overrides: Any) -> EvidencePack:
    return pack(change=make_change(platform_reviews=list(reviews), **change_overrides))


def test_the_pack_lists_what_people_already_said_on_the_change() -> None:
    text = render_evidence(reviewed(OPEN_THREAD, BOT_COMMENT), status="awaiting_human")

    reviews = section(text, "Review comments already on the change")
    assert table_row(reviews, "wangwei")[:2] == ["`src/app/refund.py`:11", "**unresolved**"]
    assert table_row(reviews, "gitlab-duo (bot)")[0] == "the whole change"
    # The reader has to be able to tell context from authority.
    assert "cannot cite a review comment" in reviews


def test_a_change_nobody_commented_on_grows_no_review_section() -> None:
    text = render_evidence(pack(), status="blocked")

    assert not [title for title in headings(text) if title.startswith("Review comments")]


def test_reviews_nobody_could_read_are_named_rather_than_left_blank() -> None:
    text = render_evidence(
        reviewed(platform_reviews_error="GitLab returned 403"), status="awaiting_human"
    )

    # An empty section and a failed read must not look alike: the reader of
    # this page is exactly the one who must not mistake silence for agreement.
    reviews = section(text, "Review comments already on the change")
    assert "**could not be read**" in reviews
    assert "GitLab returned 403" in reviews
    assert "not the same as nobody having raised a problem" in reviews
    assert [line for line in unverified(text) if "GitLab returned 403" in line]


def test_the_note_on_the_merge_request_counts_its_own_open_threads() -> None:
    text = render_forge_comment(reviewed(OPEN_THREAD, BOT_COMMENT), status="awaiting_human")

    assert table_row(text, "Review comments") == ["2 on the change · **1 unresolved**"]


def test_the_note_says_when_the_discussion_could_not_be_read() -> None:
    text = render_forge_comment(
        reviewed(platform_reviews_error="GitLab returned 403"), status="blocked"
    )

    (reviews,) = table_row(text, "Review comments")
    assert reviews.startswith("could not be read")
    assert "does not mean nobody raised a problem" in reviews


def test_a_long_discussion_is_trimmed_in_the_pack_rather_than_reprinted() -> None:
    many = [OPEN_THREAD.model_copy(update={"author": f"dev{index}"}) for index in range(25)]
    text = render_evidence(reviewed(*many), status="awaiting_human")

    reviews = section(text, "Review comments already on the change")
    assert "only the first 20 of 25 review comments are listed" in reviews
    assert "dev19" in reviews and "dev20" not in reviews


def test_a_credential_quoted_in_a_comment_never_reaches_the_pack() -> None:
    leaked = PlatformReview(author="wangwei", body="token=glpat-abcdefghijklmnopqrst keep it")
    text = render_evidence(reviewed(leaked), status="blocked")

    assert "glpat-abcdefghijklmnopqrst" not in text and "[REDACTED]" in text


# ── the configuration snapshot ───────────────────────────────────────────────


def configuration(**overrides: Any) -> GateConfiguration:
    payload: dict[str, Any] = {
        "product_version": "0.4.0",
        "rules_path": "honest-ci-rules.toml",
        "rules_sha256": "a" * 64,
        "rule_version": "v3",
        "policy_path": "honest-ci-policy.toml",
        "policy_sha256": "b" * 64,
        "policy_version": "v1",
        "required_checks": ["spec", "review", "regression"],
        "deployed_gates": ["spec", "review", "regression", "metrics"],
        "disabled_gates": ["metrics"],
        "sandbox_isolation": "docker",
        "language": "python (detected from pyproject.toml)",
        "docker_image": "python:3.12-slim",
        "test_command": "pytest -q",
        "coverage_threshold": 0.75,
        "model_provider": "deepseek",
        "model": "deepseek-chat",
        "model_explain": True,
        "model_judgment": False,
        "model_testgen": True,
        "model_min_confidence": 0.7,
        "prompt_version": "v1",
    }
    payload.update(overrides)
    return GateConfiguration.model_validate(payload)


def test_the_configuration_section_names_the_files_switches_and_thresholds() -> None:
    config = section(render_evidence(pack(configuration=configuration())), "Gate configuration")

    assert table_row(config, "Rules") == [f"`honest-ci-rules.toml` · `{'a' * 16}…` · version v3"]
    assert table_row(config, "Isolation / image") == ["docker / `python:3.12-slim`"]
    assert table_row(config, "Test command") == ["`pytest -q`"]
    # The changed-line bar is unset: its number is kept, and judges nothing.
    assert table_row(config, "Coverage thresholds") == [
        "repository 75% · changed lines recorded, not judged"
    ]
    (uses,) = table_row(config, "Model used for")
    assert "judgment rules off" in uses
    assert "regression test generation on" in uses
    assert table_row(config, "Switched off") == ["metrics"]
    assert table_row(config, "Model") == ["deepseek / deepseek-chat"]


def test_the_snapshot_tells_a_model_nobody_enabled_from_one_without_a_key() -> None:
    """Two different fixes: name a provider, or give the named one its key."""

    not_enabled = render_evidence(
        pack(configuration=configuration(model_provider="", model=None))
    )
    no_key = render_evidence(pack(configuration=configuration(model=None)))

    assert table_row(not_enabled, "Model") == ["not enabled"]
    assert table_row(no_key, "Model") == ["deepseek / no credential configured"]


def test_the_snapshot_separates_the_two_coverage_bars_and_the_scope() -> None:
    """A repository bar and a changed-line bar answer different questions."""

    text = render_evidence(
        pack(
            configuration=configuration(
                changed_line_coverage_threshold=0.8, test_selection="impacted"
            )
        )
    )

    assert table_row(text, "Coverage thresholds") == ["repository 75% · changed lines 80%"]
    assert table_row(text, "Test selection") == ["impacted"]


def test_an_image_that_did_not_run_is_not_rendered_as_if_it_had() -> None:
    """Under process isolation the tools ran on the host, not in the image."""

    text = render_evidence(pack(configuration=configuration(sandbox_isolation="process")))

    (isolation,) = table_row(text, "Isolation / image")
    assert isolation.startswith("process (no container")
    assert "was not used" in isolation


def test_a_missing_rule_file_reads_as_absent_not_as_an_empty_one() -> None:
    text = render_evidence(pack(configuration=configuration(rules_path=None, rules_sha256=None)))

    (rules,) = table_row(text, "Rules")
    assert rules.startswith("(no such file; built-in defaults)")
    assert "digest" not in rules


def test_the_section_says_it_does_not_take_part_in_the_verdict() -> None:
    """A reader must not mistake the snapshot for something the fingerprint binds."""

    config = section(render_evidence(pack(configuration=configuration())), "Gate configuration")

    assert "takes no part in the verdict" in config
    assert "not the thresholds and switches here" in config


def test_a_pack_written_before_the_snapshot_existed_says_so() -> None:
    text = render_evidence(pack())

    assert "predates the gate configuration snapshot" in section(text, "Gate configuration")
    assert "config `" not in text


def test_the_version_row_carries_the_configuration_digest() -> None:
    config = configuration()
    text = render_evidence(pack(configuration=config), status="blocked")

    (versions,) = table_row(text, "Policy / rule versions")
    assert versions.endswith(f"config `{config.digest[:12]}`")


def test_the_platform_comment_names_rule_policy_and_configuration() -> None:
    """Versions stay the pack's own; the snapshot only adds the digest beside them."""

    config = configuration()
    text = render_forge_comment(
        pack(configuration=config, rule_version="v3", policy_version="v1"),
        status="blocked",
    )

    assert table_row(text, "Rule / policy versions") == [
        f"rules v3 · policy v1 · config `{config.digest[:12]}`"
    ]


def test_two_deployments_that_differ_only_in_a_threshold_have_different_digests() -> None:
    assert configuration().digest != configuration(coverage_threshold=0.9).digest


def test_the_snapshot_never_carries_a_credential() -> None:
    """It says which service was asked, never what it was asked with."""

    assert "api_key" not in GateConfiguration.model_fields
    text = render_evidence(pack(configuration=configuration()))
    assert "deepseek / deepseek-chat" in text


def test_a_verdict_nobody_here_decided_still_gets_its_line_comment() -> None:
    """Inline comments follow "who proposed it", not "was it the model".

    An external verdict (a platform's AI review, CI, SAST, recorded as
    ``external:<source>``) is exactly as entitled to a comment on the line as a
    model's own suggestion, and exactly as unentitled to a block. Keying the
    inline set on ``decided_by == "model"`` would have dropped it out of the
    merge request while keeping it in the pack — which a developer reads as
    the tool hiding something.
    """

    external = BLOCKING.model_copy(
        update={"id": "review.EXT-1", "decided_by": "external:gitlab-duo"}
    )

    selected = inline_findings([BLOCKING, external])

    assert [finding.id for finding in selected] == ["review.R-01", "review.EXT-1"]
    # And it reached the comment without becoming a block on the way.
    assert not external.blocking


# ── test integrity ───────────────────────────────────────────────────────────

INTEGRITY = (
    Rule(
        id="I-01",
        title="Tests do not disappear",
        kind="integrity",
        check="deleted_test",
        action="needs_human",
        fix="Keep the test, or say in the change why it no longer applies.",
    ),
    Rule(
        id="I-02",
        title="No new skipped tests",
        kind="integrity",
        severity="low",
        check="skip_or_focus",
        action="record",
    ),
    Rule(
        id="I-04",
        title="Expected values change for a reason",
        kind="integrity",
        check="changed_expectation",
        action="needs_human",
    ),
    Rule(
        id="I-05",
        title="The coverage bar does not move down",
        kind="integrity",
        severity="high",
        check="loosened_threshold",
        action="block",
    ),
)


def integrity_review(
    rules: tuple[Rule, ...] = INTEGRITY, change: Any = None, verdict: Verdict = Verdict.NEEDS_HUMAN
) -> GateReport:
    """The review gate's report on ``change``: its findings, and each rule's action."""

    findings = evaluate_rules(
        RuleSet(version="v1", rules=rules), change or make_weakened_change()
    )
    return review(
        verdict=verdict,
        findings=findings,
        verifier_report={
            "rules_evaluated": [
                {"id": rule.id, "kind": rule.kind, "check": rule.check, "action": rule.action}
                for rule in rules
            ]
        },
    )


def integrity_pack(report: GateReport | None = None, **overrides: Any) -> EvidencePack:
    report = report or integrity_review()
    payload: dict[str, Any] = {
        "change": make_weakened_change(),
        "gate_reports": [undeployed("spec"), report],
        "verdict": report.verdict,
    }
    payload.update(overrides)
    return pack(**payload)


def test_the_comment_names_the_deleted_test_and_the_rewritten_expectation() -> None:
    text = render_forge_comment(integrity_pack(), status="awaiting_human")

    integrity = section(text, "Test integrity (2)")
    assert (
        "- **[I-01] Tests do not disappear** · needs a human · `tests/test_pricing.py`:3 (base)"
        in integrity
    )
    assert "  - Test `test_total_doubles` is removed" in integrity
    assert "  - Removed: `def test_total_doubles():`" in integrity
    assert "  - Suggested fix: Keep the test" in integrity
    assert "  - Was: `assert total(3) == 6`" in integrity
    assert "  - Now: `assert total(3) == 9`" in integrity


def test_a_hit_that_is_only_recorded_stays_out_of_the_comment_but_is_counted() -> None:
    text = render_forge_comment(integrity_pack(), status="awaiting_human")

    # ``record`` asks for the evidence pack and nothing else; the two that
    # were spelled out are not counted again in the tally.
    assert "No new skipped tests" not in text
    assert "Also 1 non-blocking finding:" in text


def test_a_blocking_integrity_hit_is_listed_as_a_block_and_says_what_happened() -> None:
    change = make_change(
        diff=(
            "--- a/pyproject.toml\n+++ b/pyproject.toml\n@@ -1,2 +1,2 @@\n"
            " [tool.coverage.report]\n-fail_under = 90\n+fail_under = 60\n"
        ),
        files_changed=[ChangedFile(path="pyproject.toml", added_lines=1, removed_lines=1)],
    )
    report = integrity_review(change=change, verdict=Verdict.BLOCKED)

    text = render_forge_comment(integrity_pack(report), status="blocked")

    blocks = section(text, "Blocking findings (1)")
    assert "[I-05] The coverage bar does not move down" in blocks
    integrity = section(text, "Test integrity (1)")
    assert "· blocks the merge · `pyproject.toml`:2 (base)" in integrity
    assert "`fail_under` 90 → 60" in integrity
    assert "non-blocking finding" not in text


def test_the_pack_lists_every_integrity_rule_that_ran_hits_or_none() -> None:
    text = render_evidence(integrity_pack(), status="awaiting_human")

    integrity = section(text, "Test integrity")
    # "Ran and found nothing" has to read differently from "never configured".
    assert "| `I-05` | `loosened_threshold` | blocks the merge | 0 |" in integrity
    assert "| `I-01` | `deleted_test` | needs a human | 1 |" in integrity
    assert "| `I-02` | `skip_or_focus` | recorded only | 1 |" in integrity
    # Recorded hits are what the pack is for, detail and all.
    assert 'Added: `@pytest.mark.skip(reason="flaky")`' in integrity
    assert "Removed: `def test_total_doubles():`" in integrity


def test_a_pack_without_integrity_rules_grows_no_integrity_section() -> None:
    assert "Test integrity" not in render_evidence(pack())
    assert "Test integrity" not in render_forge_comment(pack(), status="blocked")


def test_a_location_in_the_base_says_so_wherever_it_is_shown() -> None:
    text = render_evidence(integrity_pack(), status="awaiting_human")

    # The findings table has no room for the detail, only for the location.
    row = next(line for line in text.splitlines() if line.startswith("| review | I-01 |"))
    assert "`tests/test_pricing.py`:3 (base)" in row
    head_row = next(line for line in text.splitlines() if line.startswith("| review | I-02 |"))
    assert "(base)" not in head_row


def test_a_line_the_change_removed_never_gets_a_comment_on_the_new_side() -> None:
    removed = BLOCKING.model_copy(update={"id": "review.I-05.1", "side": "base"})

    # Anchored to the head, it would sit on whatever line now has that number.
    assert [finding.id for finding in inline_findings([BLOCKING, removed])] == ["review.R-01"]


def test_a_removed_or_rewritten_test_line_is_quoted_as_code_and_never_renders() -> None:
    hostile = HOSTILE_TEXT.replace('"', "'")
    change = make_weakened_change(
        diff=(
            "--- a/src/app/render.py\n+++ b/src/app/render.py\n@@ -1,2 +1,2 @@\n"
            " def render():\n-    return 1\n+    return 2\n"
            "--- a/tests/test_render.py\n+++ b/tests/test_render.py\n@@ -1,4 +1,2 @@\n"
            f'-@pytest.mark.parametrize("value", [pytest.param(1, id="{hostile}")])\n'
            "-def test_render_case(value):\n"
            " def test_render():\n"
            f'-    assert render() == "{hostile}"\n'
            '+    assert render() == "plain"\n'
        )
    )
    report = integrity_review(change=change)
    evidence = integrity_pack(report, change=change)

    assert {finding.rule_id for finding in report.findings} == {"I-01", "I-04"}
    for text in (
        render_evidence(evidence, status="awaiting_human"),
        render_forge_comment(evidence, status="awaiting_human"),
    ):
        assert "Approved by security" in text
        assert rendered_as_markdown(text, "Approved by security") == []
        assert reaching_markdown(text) == []


def test_a_long_list_of_deleted_tests_is_cut_in_the_comment_and_says_where_the_rest_is() -> None:
    count = 12
    removed = "".join(f"-def test_case_{index}():\n-    assert run({index})\n" for index in range(count))
    change = make_change(
        diff=f"--- a/tests/test_wide.py\n+++ /dev/null\n@@ -1,{2 * count} +0,0 @@\n{removed}",
        files_changed=[
            ChangedFile(path="tests/test_wide.py", change_type="deleted", removed_lines=2 * count)
        ],
    )
    report = integrity_review(change=change)

    text = render_forge_comment(integrity_pack(report, change=change), status="awaiting_human")

    assert f"### Test integrity ({count})" in text
    assert f"- (only the first 10 of {count} are listed; the evidence pack has the rest)" in text
    assert "`test_case_9`" in text and "`test_case_10`" not in text


def test_a_hit_a_person_marked_a_false_positive_says_so_and_keeps_its_verdict() -> None:
    report = integrity_review()
    marked = [
        finding.model_copy(update={"false_positive": True}) if finding.rule_id == "I-01" else finding
        for finding in report.findings
    ]

    text = render_evidence(
        integrity_pack(report.model_copy(update={"findings": marked})), status="awaiting_human"
    )

    integrity = section(text, "Test integrity")
    assert "Marked a false positive by a person; the verdict stands as it was." in integrity
