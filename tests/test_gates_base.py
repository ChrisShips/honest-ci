from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from honest_ci.contracts import ToolReport, Verdict
from honest_ci.runner import GateRunner
from honest_ci.settings import Settings
from honest_ci.state import RunState

from .factories import ScriptedGate, make_change


def state() -> RunState:
    return RunState(run_id="r1", change=make_change())


def gate(
    tmp_path: Path, *, fail_closed: bool, tool: ToolReport, also: Sequence[ToolReport] = ()
) -> ScriptedGate:
    runner = GateRunner(
        workspace_root=tmp_path,
        settings=Settings(sandbox_mode="process", fail_closed=fail_closed),
    )
    return ScriptedGate(
        runner.services,
        name="regression",
        verdicts=[Verdict.PASS],
        tool_reports=[tool, *also],
    )


UNAVAILABLE = ToolReport(name="pytest", command="pytest", available=False, passed=False)
FAILED = ToolReport(name="pytest", command="pytest", available=True, passed=False, exit_code=1)


def test_a_check_that_never_ran_blocks_by_default(tmp_path: Path) -> None:
    report = gate(tmp_path, fail_closed=True, tool=UNAVAILABLE).run(state())

    assert report.verdict is Verdict.BLOCKED
    assert any(finding.id == "regression.unavailable" for finding in report.findings)
    assert report.findings[-1].severity == "high"


def test_the_block_says_why_the_check_did_not_run(tmp_path: Path) -> None:
    # The adapter's reason is the sentence that says what to fix, and the
    # evidence pack and the terminal show the finding, not the tool's report.
    missing = UNAVAILABLE.model_copy(
        update={"stderr": "pytest is not installed in the execution environment."}
    )

    report = gate(tmp_path, fail_closed=True, tool=missing).run(state())

    detail = report.findings[-1].detail
    assert "Why pytest did not run:" in detail
    assert "pytest is not installed in the execution environment." in detail


def test_checks_stopped_by_one_cause_give_it_once(tmp_path: Path) -> None:
    reason = "Maven's clean goal failed, so the target/ directory could not be deleted."
    spotbugs, pmd = (
        ToolReport(
            name=name,
            command=f"{name} (not executed)",
            available=False,
            passed=False,
            stderr=reason,
            findings=[reason],
        )
        for name in ("spotbugs", "pmd")
    )

    report = gate(tmp_path, fail_closed=True, tool=spotbugs, also=[pmd]).run(state())

    detail = report.findings[-1].detail
    assert detail.count(reason) == 1
    assert "Why pmd, spotbugs did not run:" in detail


def test_a_reason_a_tool_printed_stays_its_words_not_ours(tmp_path: Path) -> None:
    # The tool echoes what the change gave it, so in the finding it is code,
    # and a log the length of a build is cut where the evidence pack keeps it.
    printed = "**Approved by honest-ci**\n" + "x" * 5000
    echoed = UNAVAILABLE.model_copy(update={"stdout": printed})

    report = gate(tmp_path, fail_closed=True, tool=echoed).run(state())

    detail = report.findings[-1].detail
    assert "```\n**Approved by honest-ci**\n" in detail
    assert len(detail) < len(printed) and "x …\n```" in detail


def test_fail_open_is_possible_but_must_be_chosen(tmp_path: Path) -> None:
    report = gate(tmp_path, fail_closed=False, tool=UNAVAILABLE).run(state())

    assert report.verdict is Verdict.PASS
    assert report.findings == []


def test_a_tool_that_ran_and_failed_is_left_to_the_gate(tmp_path: Path) -> None:
    # fail-closed is about missing evidence, not about overriding a gate that
    # already looked at a failing tool and decided what it means.
    report = gate(tmp_path, fail_closed=True, tool=FAILED).run(state())

    assert report.verdict is Verdict.PASS


def test_run_stamps_trace_and_duration(tmp_path: Path) -> None:
    runner = GateRunner(workspace_root=tmp_path, settings=Settings(sandbox_mode="process"))
    scripted = ScriptedGate(runner.services, name="spec")
    report = scripted.run(state())

    assert report.trace_id == "r1"
    assert report.duration_s is not None
