"""The scanner registry and the language router: which toolchain judges a change."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from honest_ci.adapters.base import ScannerAdapter
from honest_ci.adapters.python_scanner import PythonScannerAdapter
from honest_ci.adapters.scanners import (
    SCANNER_ADAPTERS,
    LanguageRouter,
    ScannerConfigError,
    build_scanner_adapter,
    detect_languages,
    supported_tools,
    tool_languages,
)
from honest_ci.contracts import ChangeRequest, CoverageResult, ToolReport
from honest_ci.contracts import TestResults as SuiteResults
from honest_ci.rules import RuleError, RuleSet
from honest_ci.runner import GateRunner
from honest_ci.settings import Settings

from .factories import make_change, make_settings


class FakeToolchain(ScannerAdapter):
    """A toolchain that records what it was asked and answers in its own name."""

    def __init__(self, language: str, tools: Sequence[str], markers: Sequence[str] = ()) -> None:
        self.language = language
        self.supported_tools = tuple(tools)
        self.markers = tuple(markers)
        self.static_calls: list[list[str]] = []
        self.scopes_seen: list[dict[str, Sequence[str]] | None] = []
        self.suites_run: list[Path] = []
        self.coverage_measured: list[Path] = []
        self.coverage_lines: list[Mapping[str, set[int]] | None] = []

    def run_static_checks(
        self,
        workspace: Path,
        change: ChangeRequest,
        tools: Sequence[str] | None = None,
        scopes: Mapping[str, Sequence[str]] | None = None,
    ) -> list[ToolReport]:
        names = list(self.supported_tools) if tools is None else list(tools)
        self.static_calls.append(names)
        self.scopes_seen.append(None if scopes is None else dict(scopes))
        return [ToolReport(name=name, command=f"{self.language} {name}") for name in names]

    def run_tests(self, workspace: Path, selection: list[str] | None = None) -> SuiteResults:
        self.suites_run.append(workspace)
        return SuiteResults(passed=1, exit_code=0, command=f"{self.language} tests")

    def measure_coverage(
        self,
        workspace: Path,
        lines: Mapping[str, set[int]] | None = None,
    ) -> CoverageResult:
        self.coverage_measured.append(workspace)
        self.coverage_lines.append(lines)
        return CoverageResult(ratio=0.9, command=f"{self.language} coverage")


def router(language: str = "auto", **adapters: FakeToolchain) -> LanguageRouter:
    return LanguageRouter(Settings(language=language), Path("artifacts"), adapters)


def two_toolchains() -> tuple[FakeToolchain, FakeToolchain]:
    python = FakeToolchain("python", ["ruff", "mypy", "semgrep"], ["pyproject.toml"])
    java = FakeToolchain("java", ["checkstyle", "semgrep"], ["pom.xml"])
    return python, java


# ── the registry ─────────────────────────────────────────────────────────────


def test_the_registry_ships_the_python_toolchain() -> None:
    assert SCANNER_ADAPTERS["python"] is PythonScannerAdapter
    assert PythonScannerAdapter.language == "python"
    assert "pyproject.toml" in PythonScannerAdapter.markers


def test_the_tool_whitelist_is_the_union_of_what_toolchains_declare() -> None:
    # The rule file may only name what some registered toolchain runs with
    # arguments fixed in code; nothing in configuration can extend this.
    assert supported_tools() == ("ruff", "mypy", "semgrep", "checkstyle", "pmd", "spotbugs")
    assert tool_languages("ruff") == ("python",)
    assert tool_languages("pip-audit") == ()


def test_a_rule_bound_to_a_tool_no_toolchain_runs_names_the_whitelist(tmp_path: Path) -> None:
    (tmp_path / "honest-ci-rules.toml").write_text(
        '[[rule]]\nid = "T-1"\ntitle = "x"\nkind = "tool"\ntool = "unknown-scanner"\n',
        encoding="utf-8",
    )

    with pytest.raises(RuleError, match="ruff, mypy, semgrep"):
        RuleSet.load(tmp_path, Path("honest-ci-rules.toml"), "v1")


def test_a_workspace_is_recognised_by_its_build_markers(tmp_path: Path) -> None:
    assert detect_languages(tmp_path) == []
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    assert detect_languages(tmp_path) == [("python", "pyproject.toml")]


def test_the_runner_builds_its_scanner_from_the_registry(tmp_path: Path) -> None:
    runner = GateRunner(workspace_root=tmp_path, settings=make_settings())

    assert isinstance(runner.adapters.scanner, LanguageRouter)
    assert runner.adapters.scanner.supported_tools == supported_tools()


# ── choosing a toolchain ─────────────────────────────────────────────────────


def test_the_configured_language_wins_over_the_markers(tmp_path: Path) -> None:
    python, java = two_toolchains()
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

    choice = router("java", python=python, java=java).choose(tmp_path)

    assert (choice.language, choice.basis) == ("java", "configured")
    assert "HONEST_CI_LANGUAGE=java" in choice.description


def test_a_marker_picks_the_toolchain_and_the_evidence_says_which(tmp_path: Path) -> None:
    python, java = two_toolchains()
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")

    choice = router(python=python, java=java).choose(tmp_path)

    assert (choice.language, choice.basis) == ("java", "detected")
    assert "detected from pom.xml" in choice.description


def test_two_build_systems_take_the_first_and_say_how_to_override(tmp_path: Path) -> None:
    python, java = two_toolchains()
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")

    choice = router(python=python, java=java).choose(tmp_path)

    assert choice.language == "python"
    assert "pom.xml" in choice.reason
    assert "HONEST_CI_LANGUAGE" in choice.reason


def test_a_workspace_without_markers_falls_back_and_says_so(tmp_path: Path) -> None:
    python, java = two_toolchains()

    choice = router(python=python, java=java).choose(tmp_path)

    # Never a silent guess: the evidence carries "default" and the markers
    # that would have made it a recognition.
    assert (choice.language, choice.basis) == ("python", "default")
    assert "pyproject.toml" in choice.reason and "pom.xml" in choice.reason


def test_an_unknown_configured_language_is_refused_at_build_time(tmp_path: Path) -> None:
    with pytest.raises(ScannerConfigError, match="klingon"):
        build_scanner_adapter(Settings(language="klingon"), tmp_path)


def test_language_configuration_is_case_insensitive(tmp_path: Path) -> None:
    python, _ = two_toolchains()
    assert router(" Python ", python=python).choose(tmp_path).basis == "configured"


# ── routing through the port ─────────────────────────────────────────────────


def test_static_tools_go_to_the_toolchain_that_owns_them(tmp_path: Path) -> None:
    python, java = two_toolchains()
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")

    reports = router(python=python, java=java).run_static_checks(
        tmp_path, make_change(), tools=["ruff", "checkstyle", "semgrep"]
    )

    # semgrep is declared by both; the workspace's own toolchain runs it.
    assert java.static_calls == [["checkstyle", "semgrep"]]
    assert python.static_calls == [["ruff"]]
    # One report per requested tool, in the order the rules asked.
    assert [(report.name, report.command) for report in reports] == [
        ("ruff", "python ruff"),
        ("checkstyle", "java checkstyle"),
        ("semgrep", "java semgrep"),
    ]


def test_a_tool_no_toolchain_knows_is_unavailable_not_ignored(tmp_path: Path) -> None:
    python, _ = two_toolchains()

    report = router(python=python).run_static_checks(tmp_path, make_change(), tools=["pmd"])[0]

    assert report.name == "pmd"
    assert not report.available and not report.passed
    assert "No registered toolchain runs 'pmd'" in report.stderr
    assert python.static_calls == []


def test_asking_for_nothing_runs_nothing(tmp_path: Path) -> None:
    python, _ = two_toolchains()
    assert router(python=python).run_static_checks(tmp_path, make_change(), tools=[]) == []
    assert python.static_calls == []


def test_asking_for_everything_means_every_registered_tool(tmp_path: Path) -> None:
    python, java = two_toolchains()

    reports = router(python=python, java=java).run_static_checks(tmp_path, make_change())

    assert [report.name for report in reports] == ["ruff", "mypy", "semgrep", "checkstyle"]


def test_a_toolchain_that_stays_silent_on_a_tool_is_missing_evidence(tmp_path: Path) -> None:
    class Silent(FakeToolchain):
        def run_static_checks(
            self,
            workspace: Path,
            change: ChangeRequest,
            tools: Sequence[str] | None = None,
            scopes: Mapping[str, Sequence[str]] | None = None,
        ) -> list[ToolReport]:
            return []

    silent = Silent("python", ["ruff"])

    report = router(python=silent).run_static_checks(tmp_path, make_change(), tools=["ruff"])[0]

    assert not report.available
    assert "returned no report for 'ruff'" in report.stderr


def test_tests_and_coverage_go_to_the_workspace_toolchain(tmp_path: Path) -> None:
    python, java = two_toolchains()
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    routed = router(python=python, java=java)

    results = routed.run_tests(tmp_path)
    coverage = routed.measure_coverage(tmp_path)

    assert results.command == "java tests"
    assert coverage.command == "java coverage"
    assert python.suites_run == [] and python.coverage_measured == []
    assert routed.toolchain(tmp_path) == "java (detected from pom.xml)"
