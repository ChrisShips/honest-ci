"""Maven checks with fixed offline commands and fresh, located evidence."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from .. import testreports
from ..contracts import ChangeRequest, CoverageResult, TestResults, ToolReport
from ..impact import changed_line_coverage
from ..java_reports import read_diagnostics, read_jacoco_line_data, read_jacoco_totals
from ..maven import (
    JACOCO_REPORTS,
    MAVEN_PLUGINS,
    SUREFIRE_REPORTS,
    maven_command,
    maven_env,
    unprovisioned_artifacts,
    unprovisioned_hint,
    unreadable_class_files,
    unreadable_class_files_hint,
)
from ..redaction import redact_text
from ..sandbox import DependencyProvisionError, JavaSandboxBuilder
from ..settings import Settings
from ..tools import ToolRunner
from .base import ScannerAdapter
from .python_scanner import SEMGREP_CONFIG_NAMES, _from_junit, _parse_semgrep_json

_REPORTS = {
    "checkstyle": "**/target/checkstyle-result.xml",
    "pmd": "**/target/pmd.xml",
    "spotbugs": "**/target/spotbugsXml.xml",
}
_TEST_CLASS = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*")


class JavaScannerAdapter(ScannerAdapter):
    language = "java"
    markers = ("pom.xml",)
    supported_tools = ("checkstyle", "pmd", "spotbugs", "semgrep")

    def __init__(self, settings: Settings, artifacts_path: Path) -> None:
        self.settings = settings
        self.artifacts_path = artifacts_path
        self._test_runs: dict[Path, ToolReport] = {}
        #: Workspaces whose previous build output this run has already dropped.
        #: One adapter is built per run, so this is "once per run".
        self._cleaned: set[Path] = set()

    def image(self, workspace: Path) -> str | None:
        return self.settings.java_docker_image

    def unselectable(self, workspace: Path) -> str | None:
        # Maven selects with `-Dtest=`, added to a command this adapter
        # composes. A team's own command may be Gradle, a wrapper script, or
        # Maven with a profile that fixes the tests; it runs as written.
        if self.settings.test_command:
            return (
                "HONEST_CI_TEST_COMMAND runs as written, and the Java toolchain cannot narrow a "
                "command it did not compose"
            )
        return None

    def _tool_runner(self, workspace: Path, *, tests: bool = False) -> ToolRunner:
        return ToolRunner(
            workspace,
            self.artifacts_path,
            mode=self.settings.effective_sandbox_isolation(),
            docker_image=self.settings.java_docker_image,
            timeout_seconds=self.settings.test_timeout_seconds
            if tests
            else self.settings.command_timeout_seconds,
            docker_cpus=self.settings.docker_cpus,
            docker_memory=self.settings.docker_memory,
            docker_pids_limit=self.settings.docker_pids_limit,
        )

    def _prepare(
        self,
        workspace: Path,
        runner: ToolRunner,
        *,
        needs_maven: bool = True,
    ) -> str | None:
        # A custom command can use a vendored wrapper without a system mvn.
        # Provisioning still needs mvn when an isolated cache must be built.
        provisioning = self._provisioning_skipped(runner.mode) is None
        if (needs_maven or provisioning) and not runner.available("mvn"):
            return "Maven (mvn) is not installed in the execution environment; check the Java image."
        try:
            JavaSandboxBuilder(self.settings).ensure_deps(
                workspace, self.artifacts_path, runner.mode
            )
        except DependencyProvisionError as exc:
            return redact_text(str(exc))
        return self._drop_stale_build(workspace, runner) if needs_maven else None

    def _provisioning_skipped(self, mode: str) -> str | None:
        """Why nothing filled the offline repository before this run, or ``None``."""

        if mode != "docker":
            return (
                "it ran in process mode (HONEST_CI_SANDBOX_MODE=process, or no Docker daemon "
                "was reachable)"
            )
        if not self.settings.sandbox_build_deps:
            return "HONEST_CI_SANDBOX_BUILD_DEPS=0 switches provisioning off"
        return None

    def _host_repository(self, mode: str) -> Path | None:
        """The machine's own repository a run reads behind its own, or ``None``.

        Process mode only: there is nothing provisioned to resolve from, and a
        container could not see the host's directory anyway.
        """

        return None if mode == "docker" else self.settings.maven_host_repository

    def _maven(self, workspace: Path, mode: str) -> list[str]:
        host = self._host_repository(mode)
        # Only one that is there: what Maven makes of a tail that does not
        # exist is a question this run does not need to ask.
        return maven_command(
            workspace, mode, tail=host if host is not None and host.is_dir() else None
        )

    def _unprovisioned(
        self, report: ToolReport, mode: str, *, as_written: bool = False
    ) -> str | None:
        """The remedy when ``mvn -o`` stopped on a download, else ``None``.

        This is the ``available`` / ``passed`` line, at the one place the Java
        toolchain can cross it by accident. Maven resolves a few artifacts
        while a goal runs -- Surefire's test provider, a report plugin's skin
        -- so an offline run can stop before it has looked at a single source
        file. Read as a failure, that becomes "this change violates the rule";
        read as what it is, it is a tool that never ran, which fail-closed
        blocks anyway, but with the coordinate to pre-fetch instead of a rule
        to argue about.

        ``as_written`` is a HONEST_CI_TEST_COMMAND, which was handed no
        ``maven.repo.local.tail``: naming this machine's repository to it
        would send the operator after a fix for a command that never ran.
        """

        if report.passed or not report.available:
            return None
        artifacts = unprovisioned_artifacts(report.stdout, report.stderr)
        if artifacts is None:
            return None
        host = None if as_written else self._host_repository(mode)
        return unprovisioned_hint(artifacts, self._provisioning_skipped(mode), host)

    def _drop_stale_build(self, workspace: Path, runner: ToolRunner) -> str | None:
        """Delete the previous build's ``target/`` before this run compiles.

        ``_clear`` already refuses to let an old report stand in for this run's
        evidence. Class files are the same thing one step earlier, and Maven
        does not remove them for a source that disappeared: delete a test and
        ``target/test-classes`` keeps its ``.class``, Surefire keeps running
        it, and the suite reports passes for a file no reviewer can open. On a
        fresh checkout there is nothing to delete; on a developer's own
        workspace, which is where ``honest-ci verify`` runs, there always is.

        The cost is a full recompile per run. It buys the property the whole
        product rests on: every number in the evidence pack came from the tree
        that was fingerprinted.

        A custom test command is left alone (``needs_maven`` is false for it):
        it owns its build, including whether an incremental one is honest.
        """

        if workspace in self._cleaned:
            return None
        report = runner.run(
            "clean",
            [*self._maven(workspace, runner.mode), f"{MAVEN_PLUGINS['clean']}:clean"],
            "maven-clean.log",
            env=maven_env(workspace, runner.mode),
        )
        missing = self._unprovisioned(report, runner.mode)
        if missing is not None:
            return missing
        if not (report.available and report.passed):
            # The cause comes first and Maven's own words on a line of their
            # own: what failed is Maven, often before it touched target/ at
            # all, and the first line is what a summary of this run shows.
            return redact_text(
                "Maven's clean goal failed, so the target/ directory the previous build left "
                "could not be deleted, and classes compiled into it cannot be evidence for "
                "this run.\n"
                + (report.stderr or report.stdout or f"exit code {report.exit_code}")
            )
        self._cleaned.add(workspace)
        return None

    def run_static_checks(
        self,
        workspace: Path,
        change: ChangeRequest,
        tools: Sequence[str] | None = None,
        scopes: Mapping[str, Sequence[str]] | None = None,
    ) -> list[ToolReport]:
        """``scopes`` is accepted and ignored: every tool here runs over the module.

        Checkstyle, PMD and SpotBugs are Maven goals over the whole build --
        SpotBugs reads compiled classes, not source paths -- so there is no
        target list to narrow. Which of their diagnostics can become a finding
        is still decided by the review gate, on the lines the change added.
        """

        requested = list(self.supported_tools) if tools is None else list(dict.fromkeys(tools))
        if not requested:
            return []
        runner = self._tool_runner(workspace)
        reason = (
            self._prepare(workspace, runner)
            if any(name in _REPORTS for name in requested)
            else None
        )
        reports: list[ToolReport] = []
        for name in requested:
            if name == "semgrep":
                reports.append(self._semgrep(workspace, runner))
            elif name not in _REPORTS:
                reports.append(_unavailable(name, f"The Java toolchain does not support {name}."))
            elif reason:
                reports.append(_unavailable(name, reason))
            else:
                reports.append(self._scan(workspace, runner, name))
        return reports

    def _scan(self, workspace: Path, runner: ToolRunner, name: str) -> ToolReport:
        pattern = _REPORTS[name]
        try:
            _clear(workspace, pattern)
        except testreports.ReportError as exc:
            return _unavailable(name, str(exc))
        plugin = MAVEN_PLUGINS[name]
        goals = {
            "checkstyle": [f"{plugin}:check", "-Dcheckstyle.failOnViolation=false"],
            "pmd": [f"{plugin}:check", "-Dpmd.failOnViolation=false"],
            "spotbugs": ["compile", f"{plugin}:spotbugs", "-Dspotbugs.xmlOutput=true"],
        }
        report = runner.run(
            name,
            [*self._maven(workspace, runner.mode), *goals[name]],
            f"{name}.log",
            env=maven_env(workspace, runner.mode),
        )
        missing = self._unprovisioned(report, runner.mode)
        if missing is not None:
            return _unavailable(name, missing)
        unreadable = unreadable_class_files(report.stdout, report.stderr)
        if unreadable is not None:
            # Before the report is read, because a report may be there: SpotBugs
            # stops with none, but PMD logs each class it could not parse and
            # writes a whole-looking one anyway, and a clean report from a tool
            # that resolved half the types is no evidence the change is clean.
            # Unavailable, as a plugin that was never downloaded is, rather
            # than a failed run under the tool's rule: either way the tool
            # never examined the change, and a failed run blocks only where
            # the rule is high, so PMD's medium rule let an unchecked change
            # through. The reason still reaches the terminal and the evidence
            # pack, on the fail-closed finding and in the gate's summary.
            return _unavailable(name, unreadable_class_files_hint(name, unreadable, runner.mode))
        try:
            paths = testreports.find_reports(workspace, pattern, strict=True)
            if not paths:
                raise testreports.ReportError(
                    f"This run produced no {name} XML; check that the plugin is provisioned "
                    "and the report path is right."
                )
            diagnostics = [
                item for path in paths for item in read_diagnostics(path, workspace, name)
            ]
        except testreports.ReportError as exc:
            return _opaque(report, str(exc))
        findings = [
            f"{item.file or name}:{item.line or '?'} [{item.code or name}] {item.message}"
            for item in diagnostics[:20]
        ]
        return report.model_copy(
            update={
                "passed": report.available and report.passed and not diagnostics,
                "diagnostics": diagnostics,
                "findings": findings or report.findings,
            }
        )

    def _semgrep(self, workspace: Path, runner: ToolRunner) -> ToolReport:
        if not runner.available("semgrep"):
            return _unavailable("semgrep", "semgrep is not installed in the Java image.")
        config = next(
            (
                name
                for name in SEMGREP_CONFIG_NAMES
                if (workspace / name).exists()
                and (workspace / name).resolve().is_relative_to(workspace.resolve())
            ),
            None,
        )
        if config is None:
            return _unavailable(
                "semgrep",
                "semgrep needs a rule file in the repo; --config auto is unavailable to a "
                "scanner denied the network.",
            )
        report = runner.run(
            "semgrep",
            [
                "semgrep",
                "scan",
                "--json",
                "--quiet",
                "--disable-version-check",
                "--metrics=off",
                "--config",
                config,
            ],
        )
        try:
            payload = json.loads(report.stdout)
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("results"), list)
                or payload.get("errors")
            ):
                raise ValueError
        except ValueError:
            return _opaque(report, "semgrep did not return a complete, readable JSON report.")
        diagnostics = _parse_semgrep_json("semgrep", report.stdout)
        return report.model_copy(
            update={
                "passed": report.available and report.passed and not diagnostics,
                "diagnostics": diagnostics,
                "findings": [f"{item.file}:{item.line} {item.message}" for item in diagnostics[:20]]
                or report.findings,
            }
        )

    def run_tests(self, workspace: Path, selection: list[str] | None = None) -> TestResults:
        runner = self._tool_runner(workspace, tests=True)
        custom = self.settings.test_command
        reason = self._prepare(workspace, runner, needs_maven=not bool(custom))
        if reason:
            return self._suite_unavailable(workspace, reason)
        pattern = self.settings.test_report or SUREFIRE_REPORTS
        try:
            if custom:
                command = shlex.split(custom)
                if not command:
                    raise ValueError("HONEST_CI_TEST_COMMAND is empty.")
            else:
                command = self._maven(workspace, runner.mode)
                jacoco = MAVEN_PLUGINS["jacoco"]
                if self.settings.coverage_measured:
                    command += [f"{jacoco}:prepare-agent"]
                command += ["test"]
                if self.settings.coverage_measured:
                    command += [f"{jacoco}:report"]
                if selection:
                    if not all(_TEST_CLASS.fullmatch(item) for item in selection):
                        raise ValueError(
                            "A Java test selection must be class names, not command arguments."
                        )
                    # Surefire fails every module that runs none of the named
                    # classes, and in a reactor most modules hold none: the
                    # build stops at the first, with no failing test to name.
                    # Whether the named tests ran at all is read from the
                    # reports, which count zero when none did.
                    command += [
                        f"-Dtest={','.join(dict.fromkeys(selection))}",
                        "-Dsurefire.failIfNoSpecifiedTests=false",
                    ]
            for output in (pattern, JACOCO_REPORTS, "**/target/jacoco.exec"):
                _clear(workspace, output)
        except (ValueError, testreports.ReportError) as exc:
            return self._suite_unavailable(workspace, str(exc))
        report = runner.run(
            "tests", command, "maven-tests.log", env=maven_env(workspace, runner.mode)
        )
        missing = self._unprovisioned(report, runner.mode, as_written=bool(custom))
        if missing is not None:
            return self._suite_unavailable(workspace, missing)
        self._test_runs[workspace] = report
        try:
            summary = testreports.read_reports(
                testreports.find_reports(workspace, pattern, strict=True), workspace
            )
        except testreports.ReportError as exc:
            summary = testreports.SuiteSummary(problems=(str(exc),))
        results = _from_junit(report, summary, pattern)
        if not report.available:
            return results.model_copy(
                update={"errors": max(1, results.errors), "failures": [report.stderr]}
            )
        return results

    def _suite_unavailable(self, workspace: Path, reason: str) -> TestResults:
        report = _unavailable("tests", reason)
        self._test_runs[workspace] = report
        return TestResults(
            available=False,
            errors=1,
            failures=[reason],
            stderr=reason,
            command=report.command,
            runner="junit",
        )

    def measure_coverage(
        self,
        workspace: Path,
        lines: Mapping[str, set[int]] | None = None,
    ) -> CoverageResult:
        previous = self._test_runs.get(workspace)
        if previous is not None and not previous.available:
            return CoverageResult(available=False, stderr=previous.stderr)
        try:
            paths = testreports.find_reports(workspace, JACOCO_REPORTS, strict=True)
            if not paths:
                raise testreports.ReportError(
                    "No JaCoCo XML; run the prepare-agent and report goals in the test command."
                )
            counts = [read_jacoco_totals(path) for path in paths]
            covered = sum(item[0] for item in counts)
            total = sum(sum(item) for item in counts)
            if total == 0:
                raise testreports.ReportError(
                    "JaCoCo counted no executable lines, so there is no coverage figure."
                )
        except testreports.ReportError as exc:
            return CoverageResult(
                available=True, ratio=None, stderr=str(exc), command=JACOCO_REPORTS
            )
        result = CoverageResult(
            available=True,
            ratio=covered / total,
            command=JACOCO_REPORTS,
            stdout=(
                f"JaCoCo: {covered}/{total} lines across {len(paths)} "
                f"{'module' if len(paths) == 1 else 'modules'}."
            ),
        )
        if lines is None:
            return result
        try:
            # The same reports, read a second way. Its own failure, because the
            # two numbers answer different questions: a per-line pass nobody
            # could read leaves the changed-line fields empty and the overall
            # ratio standing, and it is the gate that decides whether a bar was
            # thereby left unchecked.
            changed = changed_line_coverage(
                read_jacoco_line_data(paths, workspace), lines, (".java",)
            )
        except testreports.ReportError as exc:
            return result.model_copy(update={"stderr": str(exc)})
        return result.model_copy(
            update={
                "changed_lines_total": changed.total,
                "changed_lines_covered": changed.covered,
                "uncovered": list(changed.uncovered),
                "unmapped_sources": list(changed.unmapped),
            }
        )


def _clear(workspace: Path, pattern: str) -> None:
    paths = testreports.find_reports(workspace, pattern, strict=True)
    testreports.clear_reports(workspace, pattern)
    if any(path.exists() for path in paths):
        raise testreports.ReportError(
            "An old report could not be deleted, so this run cannot use old results as evidence."
        )


def _unavailable(name: str, reason: str) -> ToolReport:
    return ToolReport(
        name=name,
        command=f"{name} (not executed)",
        available=False,
        passed=False,
        stderr=reason,
        findings=[reason],
    )


def _opaque(report: ToolReport, reason: str) -> ToolReport:
    return report.model_copy(
        update={
            "passed": False,
            "diagnostics": [],
            "findings": [*report.findings, report.stderr or report.stdout, reason],
            "stderr": "\n".join(part for part in (report.stderr, reason) if part),
        }
    )
