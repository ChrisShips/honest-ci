from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import coverage
import pytest

from honest_ci.adapters.base import AdapterSet
from honest_ci.adapters.python_scanner import PythonScannerAdapter
from honest_ci.adapters.scanners import LanguageRouter
from honest_ci.contracts import ChangedFile, ProposedTest, ToolReport
from honest_ci.runner import GateRunner
from honest_ci.settings import Settings
from honest_ci.state import RunState
from honest_ci.storage import SQLiteLedger
from honest_ci.tools import ToolRunner

from .factories import ScriptedLLM, make_change, make_settings, recorded_commands

RUFF_DIFF = """--- a/pkg/app.py
+++ b/pkg/app.py
@@ -0,0 +1,2 @@
+import os
+VALUE = 1
"""


def adapter(tmp_path: Path) -> PythonScannerAdapter:
    return PythonScannerAdapter(make_settings(), tmp_path / "artifacts")


def workspace_with(tmp_path: Path, relative: str, content: str) -> Path:
    workspace = tmp_path / "repo"
    target = workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return workspace


def canned(
    monkeypatch: pytest.MonkeyPatch, stdout: str, exit_code: int = 1
) -> list[tuple[list[str], dict[str, str]]]:
    """Run the adapter against a prepared tool output instead of a real tool.

    Returns every command it was handed, with the environment it was given.
    """

    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(
        self: ToolRunner,
        name: str,
        command: list[str],
        report_name: str | None = None,
        allow_network: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> ToolReport:
        calls.append((command, dict(env or {})))
        return ToolReport(
            name=name,
            command=" ".join(command),
            available=True,
            passed=exit_code == 0,
            exit_code=exit_code,
            stdout=stdout,
        )

    monkeypatch.setattr(ToolRunner, "module_available", lambda *args, **kwargs: True)
    monkeypatch.setattr(ToolRunner, "run", run)
    return calls


def test_ruff_output_becomes_located_diagnostics(tmp_path: Path) -> None:
    pytest.importorskip("ruff")
    workspace = workspace_with(tmp_path, "pkg/app.py", "import os\nVALUE = 1\n")
    change = make_change(diff=RUFF_DIFF, files_changed=[ChangedFile(path="pkg/app.py")])

    reports = adapter(tmp_path).run_static_checks(workspace, change, tools=["ruff"])

    assert len(reports) == 1
    report = reports[0]
    assert report.available and not report.passed
    unused_import = [item for item in report.diagnostics if item.code == "F401"]
    assert unused_import, report.stdout
    assert unused_import[0].file == "pkg/app.py"
    assert unused_import[0].line == 1


def test_a_tool_that_is_not_installed_never_reports_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ToolRunner, "module_available", lambda *args, **kwargs: False)
    workspace = workspace_with(tmp_path, "pkg/app.py", "VALUE = 1\n")
    change = make_change(diff=RUFF_DIFF, files_changed=[ChangedFile(path="pkg/app.py")])

    report = adapter(tmp_path).run_static_checks(workspace, change, tools=["ruff"])[0]

    assert not report.available
    assert not report.passed
    assert "not installed" in report.stderr


def test_an_unsupported_scanner_is_unavailable_rather_than_ignored(tmp_path: Path) -> None:
    report = adapter(tmp_path).run_static_checks(tmp_path, make_change(), tools=["pip-audit"])[0]

    assert not report.available
    assert "Unsupported scanner" in report.stderr


def test_paths_from_a_diff_cannot_become_tool_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = recorded_commands(monkeypatch)
    workspace = workspace_with(tmp_path, "pkg/app.py", "VALUE = 1\n")
    change = make_change(
        diff=RUFF_DIFF,
        files_changed=[
            ChangedFile(path="pkg/app.py"),
            ChangedFile(path="--fix.py"),
            ChangedFile(path="../../etc/passwd.py"),
            ChangedFile(path="pkg/deleted.py"),
        ],
    )

    adapter(tmp_path).run_static_checks(workspace, change, tools=["ruff"])

    # A diff is written by whoever opened the merge request. Only real files
    # inside the tree reach argv, and they reach it after "--".
    assert commands[0][-2:] == ["--", "pkg/app.py"]


def test_a_tools_scope_decides_which_files_reach_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = recorded_commands(monkeypatch)
    workspace = workspace_with(tmp_path, "pkg/app.py", "VALUE = 1\n")
    (workspace / "tests").mkdir(exist_ok=True)
    (workspace / "tests" / "test_app.py").write_text("def test_x(): pass\n", encoding="utf-8")
    change = make_change(
        files_changed=[ChangedFile(path="pkg/app.py"), ChangedFile(path="tests/test_app.py")]
    )

    adapter(tmp_path).run_static_checks(
        workspace, change, tools=["ruff"], scopes={"ruff": ["pkg/app.py"]}
    )

    # The rule asked about pkg only, so that is what the tool is asked about.
    assert commands[0][-2:] == ["--", "pkg/app.py"]


def test_a_tool_asked_about_nothing_reports_nothing_to_check_not_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ToolRunner, "module_available", lambda *args, **kwargs: True)
    workspace = workspace_with(tmp_path, "pkg/app.py", "VALUE = 1\n")

    report = adapter(tmp_path).run_static_checks(
        workspace, make_change(), tools=["ruff"], scopes={"ruff": []}
    )[0]

    # Which of the two emptied the list, because they read differently.
    assert "the rules bound to ruff cover no" in report.stdout


def test_a_change_with_no_files_in_scope_is_reported_as_scanning_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ToolRunner, "module_available", lambda *args, **kwargs: True)
    workspace = workspace_with(tmp_path, "docs/readme.md", "# hi\n")
    change = make_change(files_changed=[ChangedFile(path="docs/readme.md")], diff="")

    report = adapter(tmp_path).run_static_checks(workspace, change, tools=["ruff"])[0]

    assert report.available and report.passed
    assert "0 applicable files" in report.stdout


def test_semgrep_without_an_offline_ruleset_cannot_report_a_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ToolRunner, "module_available", lambda *args, **kwargs: True)
    workspace = workspace_with(tmp_path, "pkg/app.py", "VALUE = 1\n")

    report = adapter(tmp_path).run_static_checks(workspace, make_change(), tools=["semgrep"])[0]

    # --config auto downloads a ruleset, and a scanner is never given network.
    assert not report.available
    assert ".semgrep.yml" in report.stderr


def test_mypy_notes_are_context_not_defects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(
        monkeypatch,
        "pkg/app.py:11: error: Incompatible return value type\n"
        "pkg/app.py:11: note: Consider annotating the function\n",
    )
    workspace = workspace_with(tmp_path, "pkg/app.py", "VALUE = 1\n")
    change = make_change(diff=RUFF_DIFF, files_changed=[ChangedFile(path="pkg/app.py")])

    report = adapter(tmp_path).run_static_checks(workspace, change, tools=["mypy"])[0]

    assert [item.message for item in report.diagnostics] == [
        "error: Incompatible return value type"
    ]


def test_semgrep_json_is_read_as_located_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "results": [
            {
                "path": "pkg/app.py",
                "check_id": "python.lang.security.audit.eval",
                "start": {"line": 7},
                "extra": {"message": "eval on user input"},
            }
        ]
    }
    canned(monkeypatch, json.dumps(payload))
    workspace = workspace_with(tmp_path, ".semgrep.yml", "rules: []\n")
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    change = make_change(diff=RUFF_DIFF, files_changed=[ChangedFile(path="pkg/app.py")])

    report = adapter(tmp_path).run_static_checks(workspace, change, tools=["semgrep"])[0]

    assert len(report.diagnostics) == 1
    assert report.diagnostics[0].line == 7
    code = report.diagnostics[0].code
    assert code is not None and code.endswith("eval")


def test_a_tool_that_crashed_keeps_its_raw_output_as_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, "Traceback (most recent call last): ...", exit_code=2)
    workspace = workspace_with(tmp_path, "pkg/app.py", "VALUE = 1\n")
    change = make_change(diff=RUFF_DIFF, files_changed=[ChangedFile(path="pkg/app.py")])

    report = adapter(tmp_path).run_static_checks(workspace, change, tools=["ruff"])[0]

    assert report.diagnostics == []
    assert not report.passed


def test_asking_for_no_scanners_runs_none(tmp_path: Path) -> None:
    assert adapter(tmp_path).run_static_checks(tmp_path, make_change(), tools=[]) == []


def test_test_counts_are_read_from_the_suite_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, "FAILED tests/test_refund.py::test_split\n3 passed, 1 failed in 0.4s\n")

    results = adapter(tmp_path).run_tests(tmp_path)

    assert (results.passed, results.failed) == (3, 1)
    assert results.failures == ["FAILED tests/test_refund.py::test_split"]
    # The gate reads these two, not the counts, to tell "ran and failed" from
    # "never ran" and from "ran and had nothing to say".
    assert results.available is True
    assert results.exit_code == 1


def test_collection_errors_are_listed_alongside_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(
        monkeypatch,
        "ERROR tests/test_broken.py\nFAILED tests/test_refund.py::test_split - boom\n"
        "1 failed, 1 error in 0.2s\n",
        exit_code=2,
    )

    results = adapter(tmp_path).run_tests(tmp_path)

    assert (results.failed, results.errors) == (1, 1)
    assert results.failures == [
        "ERROR tests/test_broken.py",
        "FAILED tests/test_refund.py::test_split - boom",
    ]


def test_the_default_suite_collects_coverage_when_a_bar_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, "3 passed in 0.1s\n", exit_code=0)

    results = adapter(tmp_path).run_tests(tmp_path)

    # ``coverage run`` rewrites the data file, so the figure the gate reads
    # afterwards is about this run, not about whatever ran here last week.
    assert results.command is not None
    assert "-m coverage run -m pytest -q" in results.command


def test_a_changed_line_bar_alone_still_collects_coverage_from_this_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, "3 passed in 0.1s\n", exit_code=0)
    settings = Settings(
        sandbox_mode="process", coverage_threshold=0.0, changed_line_coverage_threshold=0.8
    )

    results = PythonScannerAdapter(settings, tmp_path / "artifacts").run_tests(tmp_path)

    # The gate asks for per-line data whenever either bar is on. Without
    # `coverage run` there is none, or there is a .coverage file an earlier run
    # left -- and with one, a real run passed a change whose new `raise` no
    # test executed, at "100% (2/2 lines)".
    assert results.command is not None
    assert "-m coverage run -m pytest -q" in results.command


def test_a_zero_bar_runs_pytest_without_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, "3 passed in 0.1s\n", exit_code=0)
    settings = Settings(sandbox_mode="process", coverage_threshold=0.0)

    results = PythonScannerAdapter(settings, tmp_path / "artifacts").run_tests(tmp_path)

    assert results.command is not None
    assert "coverage" not in results.command
    assert results.command.endswith("-m pytest -q")


def test_a_custom_test_command_runs_exactly_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, "2 passed in 0.1s\n", exit_code=0)
    # Not even the pytest probe: the operator's command is theirs to get right.
    monkeypatch.setattr(ToolRunner, "module_available", lambda *args, **kwargs: False)
    settings = Settings(sandbox_mode="process", test_command="make test")

    results = PythonScannerAdapter(settings, tmp_path / "artifacts").run_tests(
        tmp_path, ["tests/test_refund.py"]
    )

    # A selection included: appended to `make test`, a path is a make target.
    assert results.command == "make test"
    assert results.available is True


def test_a_missing_pytest_is_a_suite_that_never_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ToolRunner, "module_available", lambda *args, **kwargs: False)

    def never(*args: object, **kwargs: object) -> ToolReport:
        raise AssertionError("nothing should be executed without pytest")

    monkeypatch.setattr(ToolRunner, "run", never)

    results = adapter(tmp_path).run_tests(tmp_path)

    assert results.available is False
    assert results.errors == 1
    assert "pytest is not installed" in results.failures[0]


def test_a_suite_that_never_ran_is_an_error_not_an_empty_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ToolRunner, "module_available", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        ToolRunner,
        "run",
        lambda self, name, command, report_name=None, allow_network=False, env=None: ToolReport(
            name=name,
            command=" ".join(command),
            available=False,
            passed=False,
            stderr="pytest: not found",
        ),
    )

    results = adapter(tmp_path).run_tests(tmp_path)

    assert results.passed == 0
    assert results.errors >= 1
    assert results.failures == ["pytest: not found"]


def test_coverage_is_a_ratio_or_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    canned(monkeypatch, "82\n", exit_code=0)
    measured = adapter(tmp_path).measure_coverage(tmp_path)
    assert measured.ratio == 0.82
    assert measured.available is True

    canned(monkeypatch, "No data to report.", exit_code=1)
    unmeasured = adapter(tmp_path).measure_coverage(tmp_path)
    # The tool ran and had no figure: available, but nothing to judge by. The
    # gate blocks on it, and the evidence has the tool's own words.
    assert unmeasured.ratio is None
    assert unmeasured.available is True
    assert unmeasured.stdout == "No data to report."


COVERAGE_JSON = json.dumps(
    {
        "totals": {"percent_covered": 82.0},
        "files": {
            "pkg/refund.py": {"executed_lines": [11, 12], "missing_lines": [13]},
            "pkg/other.py": {"executed_lines": [1], "missing_lines": []},
        },
    }
)


def test_changed_lines_are_judged_against_the_json_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, COVERAGE_JSON, exit_code=0)

    result = adapter(tmp_path).measure_coverage(tmp_path, {"pkg/refund.py": {11, 12, 13, 99}})

    # 99 is in neither set: not an executable line, so it is in neither the
    # total nor the covered count. The overall ratio still comes from the same
    # run, so one command answers both questions.
    assert result.ratio == 0.82
    assert (result.changed_lines_total, result.changed_lines_covered) == (3, 2)
    assert result.uncovered == [("pkg/refund.py", 13)]
    assert result.unmapped_sources == []
    assert "coverage json -o -" in result.command


def test_a_changed_file_the_report_never_named_is_reported_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, COVERAGE_JSON, exit_code=0)

    result = adapter(tmp_path).measure_coverage(tmp_path, {"pkg/new.py": {1}})

    assert result.unmapped_sources == ["pkg/new.py"]
    assert (result.changed_lines_total, result.changed_lines_covered) == (0, 0)


def test_non_source_changes_are_not_held_to_a_line_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, COVERAGE_JSON, exit_code=0)

    result = adapter(tmp_path).measure_coverage(tmp_path, {"README.md": {1, 2}})

    assert result.unmapped_sources == []
    assert result.changed_lines_total == 0


def test_a_json_report_nobody_could_read_measures_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, "Traceback (most recent call last):", exit_code=1)

    result = adapter(tmp_path).measure_coverage(tmp_path, {"pkg/refund.py": {11}})

    # Available, with no figure and no per-line data: the gate blocks on both
    # bars rather than reading a crash as a change that covers nothing.
    assert result.available is True
    assert result.ratio is None
    assert result.changed_lines_total is None


def test_asking_for_no_lines_keeps_the_cheaper_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, "82\n", exit_code=0)

    result = adapter(tmp_path).measure_coverage(tmp_path)

    assert result.ratio == 0.82
    assert "--format=total" in result.command


def test_coverage_is_unmeasurable_without_the_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ToolRunner, "module_available", lambda *args, **kwargs: False)

    result = adapter(tmp_path).measure_coverage(tmp_path)

    assert result.ratio is None
    assert result.available is False
    assert "coverage is not installed" in result.stderr


# ── what coverage was configured to measure ──────────────────────────────────


def _configured(tmp_path: Path, name: str, body: str) -> Path:
    workspace = tmp_path / "repo"
    (workspace / "src" / "app").mkdir(parents=True, exist_ok=True)
    (workspace / "tests").mkdir(parents=True, exist_ok=True)
    (workspace / name).write_text(body, encoding="utf-8")
    return workspace


def test_a_changed_test_file_is_out_of_scope_rather_than_missing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With ``source = ["app"]`` a test file is never in the coverage report.
    # Before this, its absence read as "nobody measured this" and blocked every
    # change that touched a test.
    workspace = _configured(
        tmp_path, "pyproject.toml", '[tool.coverage.run]\nsource = ["app"]\n'
    )
    canned(monkeypatch, COVERAGE_JSON, exit_code=0)

    result = adapter(tmp_path).measure_coverage(
        workspace, {"src/app/refund.py": {1}, "tests/test_refund.py": {1, 2}}
    )

    assert result.out_of_scope_sources == ["tests/test_refund.py"]
    # Inside the measured scope, a file the report never named still blocks.
    assert result.unmapped_sources == ["src/app/refund.py"]


def test_a_repository_with_no_coverage_configuration_judges_every_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, COVERAGE_JSON, exit_code=0)

    result = adapter(tmp_path).measure_coverage(tmp_path, {"tests/test_refund.py": {1}})

    # Nothing says what is measured, so nothing is excused: fail closed.
    assert result.unmapped_sources == ["tests/test_refund.py"]
    assert result.out_of_scope_sources == []


@pytest.mark.parametrize(
    ("name", "body"),
    [
        (".coveragerc", "[run]\nsource = app\n"),
        ("setup.cfg", "[coverage:run]\nsource =\n    app\n"),
        ("tox.ini", "[coverage:run]\nsource = app, other\n"),
        ("pyproject.toml", '[tool.coverage.run]\nsource_pkgs = ["app"]\n'),
    ],
)
def test_every_place_coverage_reads_its_configuration_is_read_here_too(
    name: str, body: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _configured(tmp_path, name, body)
    canned(monkeypatch, COVERAGE_JSON, exit_code=0)

    result = adapter(tmp_path).measure_coverage(workspace, {"tests/test_refund.py": {1}})

    assert result.out_of_scope_sources == ["tests/test_refund.py"]


def test_a_configuration_nobody_can_parse_falls_back_to_judging_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _configured(tmp_path, "pyproject.toml", "[tool.coverage.run\nsource = broken")
    canned(monkeypatch, COVERAGE_JSON, exit_code=0)

    result = adapter(tmp_path).measure_coverage(workspace, {"tests/test_refund.py": {1}})

    # A typo must not retire the bar, so an unreadable config says nothing
    # about scope and every source file is judged.
    assert result.unmapped_sources == ["tests/test_refund.py"]


def test_a_source_naming_a_package_nothing_resolves_to_is_not_a_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _configured(
        tmp_path, "pyproject.toml", '[tool.coverage.run]\nsource = ["nowhere"]\n'
    )
    canned(monkeypatch, COVERAGE_JSON, exit_code=0)

    result = adapter(tmp_path).measure_coverage(workspace, {"tests/test_refund.py": {1}})

    assert result.unmapped_sources == ["tests/test_refund.py"]


# ── a suite that is not pytest ───────────────────────────────────────────────

JUNIT = """<testsuite name="go">
  <testcase classname="orders" name="TestSplit"/>
  <testcase classname="orders" name="TestTail"><failure message="want 1, got 2"/></testcase>
</testsuite>
"""


def test_a_custom_command_is_read_through_the_junit_xml_it_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def leave_report(
        self: ToolRunner, name: str, command: list[str], report_name: str | None = None, allow_network: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> ToolReport:
        (tmp_path / "reports").mkdir(exist_ok=True)
        (tmp_path / "reports" / "junit.xml").write_text(JUNIT, encoding="utf-8")
        return ToolReport(name=name, command=" ".join(command), passed=False, exit_code=1)

    monkeypatch.setattr(ToolRunner, "run", leave_report)
    settings = Settings(sandbox_mode="process", test_command="go test ./...", test_report="reports/*.xml")

    results = PythonScannerAdapter(settings, tmp_path / "artifacts").run_tests(tmp_path)

    assert results.runner == "junit"
    assert (results.passed, results.failed) == (1, 1)
    assert results.failures == ["FAILED orders#TestTail - want 1, got 2"]
    assert results.available is True and results.exit_code == 1


def test_a_stale_report_is_cleared_before_the_command_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "junit.xml").write_text(JUNIT, encoding="utf-8")
    seen_during_run: list[bool] = []

    def runs_without_writing(
        self: ToolRunner, name: str, command: list[str], report_name: str | None = None, allow_network: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> ToolReport:
        seen_during_run.append((tmp_path / "junit.xml").exists())
        return ToolReport(name=name, command=" ".join(command), exit_code=0)

    monkeypatch.setattr(ToolRunner, "run", runs_without_writing)
    settings = Settings(sandbox_mode="process", test_command="make test", test_report="junit.xml")

    results = PythonScannerAdapter(settings, tmp_path / "artifacts").run_tests(tmp_path)

    # Last week's report was gone before the suite started, so a command that
    # writes nothing reports nothing -- never the old counts.
    assert seen_during_run == [False]
    assert results.collected == 0
    assert "no JUnit XML matched HONEST_CI_TEST_REPORT=junit.xml" in results.stderr


def test_an_unreadable_report_counts_as_an_error_not_as_a_clean_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def leave_garbage(
        self: ToolRunner, name: str, command: list[str], report_name: str | None = None, allow_network: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> ToolReport:
        (tmp_path / "junit.xml").write_text("<testsuite><testcase", encoding="utf-8")
        return ToolReport(name=name, command=" ".join(command), exit_code=0)

    monkeypatch.setattr(ToolRunner, "run", leave_garbage)
    settings = Settings(sandbox_mode="process", test_command="make test", test_report="junit.xml")

    results = PythonScannerAdapter(settings, tmp_path / "artifacts").run_tests(tmp_path)

    assert results.errors == 1
    assert results.failures[0].startswith("ERROR junit.xml: not well-formed")


def test_a_custom_command_without_a_report_is_only_its_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, "ok\n", exit_code=0)
    settings = Settings(sandbox_mode="process", test_command="make test")

    results = PythonScannerAdapter(settings, tmp_path / "artifacts").run_tests(tmp_path)

    assert results.runner == "exit-code"
    assert results.collected == 0


def test_a_custom_pytest_command_is_still_read_as_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, "2 passed in 0.1s\n", exit_code=0)
    settings = Settings(sandbox_mode="process", test_command="python -m pytest -x tests")

    results = PythonScannerAdapter(settings, tmp_path / "artifacts").run_tests(tmp_path)

    assert results.runner == "pytest"
    assert results.passed == 2


# ── a team's own command, behind the regression gate ──────────────────────

REFUND = ["def split_refund(total):", "    return total / 2"]
REFUND_TEST = "from refund import split_refund\n\n\ndef test_split():\n    assert split_refund(4) == 2\n"
GENERATED_REFUND_CASE = ProposedTest(
    path="tests/honest_ci_generated/test_split_refund_regression.py",
    content="from refund import split_refund\n\n\ndef test_split_refund_halves():\n"
    "    assert split_refund(4) == 2\n",
    targets=["split_refund"],
)


def refund_repo(tmp_path: Path) -> Path:
    workspace = tmp_path / "repo"
    files = {
        "pyproject.toml": '[project]\nname = "refund"\n',
        "refund.py": "\n".join(REFUND) + "\n",
        "tests/test_refund.py": REFUND_TEST,
        "tests/test_users.py": "def test_users():\n    assert True\n",
    }
    for relative, content in files.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return workspace


def regression_gate(
    tmp_path: Path, llm: ScriptedLLM | None = None, **settings: Any
) -> tuple[GateRunner, RunState]:
    """The real Python toolchain behind the router, a change to ``split_refund``, and a test naming it."""

    added = "".join(f"+{line}\n" for line in REFUND)
    change = make_change(
        diff=f"--- /dev/null\n+++ b/refund.py\n@@ -0,0 +1,{len(REFUND)} @@\n{added}",
        files_changed=[],
    )
    configured = replace(make_settings(), coverage_threshold=0.0, **settings)
    runner = GateRunner(
        refund_repo(tmp_path),
        settings=configured,
        adapters=AdapterSet(
            ledger=SQLiteLedger(tmp_path / "ledger.sqlite3"),
            scanner=LanguageRouter(configured, tmp_path / "artifacts"),
            llm=llm,
        ),
    )
    return runner, RunState(run_id="r1", change=change)


def test_a_custom_command_is_never_reported_as_running_only_the_impacted_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = canned(monkeypatch, "2 passed in 0.1s\n", exit_code=0)
    runner, state = regression_gate(
        tmp_path, test_command="python -m pytest -q tests", test_selection="impacted"
    )

    report = runner.gates["regression"].run(state)

    # Handed `tests` and then a file inside it, pytest runs all of `tests`:
    # the selection narrowed nothing, so the evidence must not say it did.
    assert [command for command, _ in calls] == [["python", "-m", "pytest", "-q", "tests"]]
    selection = report.verifier_report["test_selection"]
    assert selection["mode"] == "all"
    assert "HONEST_CI_TEST_COMMAND" in selection["fallback"]
    assert "only the 1 impacted test ran" not in report.summary


def test_generated_cases_are_not_credited_with_a_custom_commands_whole_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = canned(monkeypatch, "2 passed in 0.1s\n", exit_code=0)
    llm = ScriptedLLM(tests=[GENERATED_REFUND_CASE])
    runner, state = regression_gate(tmp_path, llm=llm, test_command="python -m pytest -q tests")

    report = runner.gates["regression"].run(state)

    # The generated file, appended to a command that already names `tests`,
    # ran the whole suite again: its two passes read as two generated
    # regression tests that passed.
    assert report.test_results is not None
    assert report.test_results.generated == 0
    generated = report.verifier_report["generated_tests"]
    assert generated["ran"] is False
    assert "HONEST_CI_TEST_COMMAND" in generated["not_generated_reason"]
    assert len(calls) == 1
    # Nothing was going to run what the model wrote, so it was not asked.
    assert llm.testgen_symbols == []


def test_coverage_is_read_from_the_data_this_runs_suite_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = canned(monkeypatch, "2 passed in 0.1s\n", exit_code=0)
    workspace = refund_repo(tmp_path)
    scanner = PythonScannerAdapter(
        Settings(sandbox_mode="process", test_command="make test"), tmp_path / "artifacts"
    )

    scanner.run_tests(workspace)
    scanner.measure_coverage(workspace, {"refund.py": {1}})
    scanner.run_tests(workspace)

    first, measured, second = (env["COVERAGE_FILE"] for _, env in calls)
    # The coverage command reads what the suite before it wrote; the next
    # suite writes somewhere new, so nothing it fails to write is read as its.
    assert measured == first
    assert second != first
    # Not in the tree: nothing there is overwritten, and a symlink the change
    # planted cannot redirect the write.
    assert Path(first).is_relative_to((tmp_path / "artifacts").resolve())


def test_coverage_an_earlier_run_left_is_never_read_as_this_runs(tmp_path: Path) -> None:
    workspace = refund_repo(tmp_path)
    # An earlier run's data, where coverage.py looks by default, saying every
    # line of the changed file ran.
    stale = coverage.CoverageData(basename=str(workspace / ".coverage"))
    stale.add_lines({str((workspace / "refund.py").resolve()): [1, 2]})
    stale.write()
    # The repository's own suite, under a command that collects no coverage.
    command = f"{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider"
    scanner = PythonScannerAdapter(
        Settings(sandbox_mode="process", test_command=command), tmp_path / "artifacts"
    )

    results = scanner.run_tests(workspace)
    measured = scanner.measure_coverage(workspace, {"refund.py": {1, 2}})

    assert results.passed == 2
    # Read from the old file this was 100% of the changed lines, from a run
    # that measured none. Nothing measured is no figure, and the gate blocks.
    assert measured.ratio is None
    assert measured.changed_lines_covered is None
    assert "No data to report" in measured.stdout
    # The developer's own file is left as it was.
    assert (workspace / ".coverage").exists()


def test_the_report_setting_is_ignored_for_the_default_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, "3 passed in 0.1s\n", exit_code=0)
    settings = Settings(sandbox_mode="process", test_report="junit.xml")

    results = PythonScannerAdapter(settings, tmp_path / "artifacts").run_tests(tmp_path)

    assert results.runner == "pytest"
    assert results.passed == 3
