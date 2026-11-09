"""Deterministic Python checks, run under the sandbox.

This is the Python toolchain: one ``ScannerAdapter`` among those registered in
``adapters.scanners``, which is where a workspace is matched to a toolchain.
Three properties are the whole point of routing every check through here
rather than calling a linter inline:

* **A tool that is not installed reports ``available=False``.** The gate template
  turns that into a block. "Did not run" never becomes "passed".
* **Scanners never get the network.** Every command below reaches a verdict
  offline, which is why ``pip-audit`` (an advisory-feed client) is not here and
  why semgrep requires a rule file inside the repo instead of ``--config auto``.
* **Output is parsed into located diagnostics.** Without a file and a line, the
  review gate cannot tell a problem this change introduced from one it merely
  stood next to, and a gate that cannot tell those apart is a noise generator.

Arguments are fixed in code. Paths from the change are validated before they
become argv: a diff is written by whoever opened the merge request, so a file
named ``--fix`` must never reach a tool as a flag.
"""

from __future__ import annotations

import configparser
import json
import re
import shlex
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .. import testreports
from ..contracts import ChangeRequest, CoverageResult, TestResults, ToolDiagnostic, ToolReport
from ..diffs import extract_changed_files, is_safe_relative_path
from ..impact import ChangedLineCoverage, changed_line_coverage
from ..sandbox import DependencyProvisionError, SandboxBuilder
from ..settings import Settings
from ..tools import ToolRunner
from ..utils import recorded_command
from .base import ScannerAdapter

#: Where a repo declares semgrep rules it can run offline. ``--config auto``
#: downloads a registry ruleset, so it cannot be used by a scanner that is
#: denied the network; a repo without one of these has no semgrep evidence.
SEMGREP_CONFIG_NAMES = (".semgrep.yml", ".semgrep.yaml", "semgrep.yml", ".semgrep")

#: ``path:line:col: message`` — the shape ruff (concise) and mypy both emit.
_TEXT_DIAGNOSTIC_RE = re.compile(
    r"^(?P<file>[^\s:][^:]*):(?P<line>\d+)(?::\d+)?:\s*(?P<message>.+)$"
)
_RULE_CODE_RE = re.compile(r"^(?P<code>[A-Z]+[0-9]+)\s+(?P<message>.*)$")
_PYTEST_COUNT_RE = re.compile(r"(?P<count>\d+)\s+(?P<outcome>passed|failed|errors?|skipped)")
#: pytest's short summary, which ``-r fE`` (the default) prints even under
#: ``-q``: one line per failed test and per collection/setup error.
_PYTEST_SUMMARY_PREFIXES = ("FAILED ", "ERROR ")
_MAX_LISTED_DIAGNOSTICS = 20
#: A failing suite prints tracebacks; the evidence pack keeps the tail, where
#: the summary is, and the full text is what the terminal already showed.
_MAX_SUITE_OUTPUT_CHARS = 32_000

#: Where coverage.py reads its own configuration, in its own precedence order.
#: Read here so the changed-line bar judges the files this repository asked to
#: measure -- and only those.
_COVERAGE_CONFIG_FILES = (".coveragerc", "setup.cfg", "tox.ini", "pyproject.toml")

#: Directories a package named in ``source`` is looked for under, when it is a
#: package name rather than a path. coverage.py resolves it by importing; the
#: two layouts that reach here are flat and ``src``.
_PACKAGE_ROOTS = ("", "src", "lib")


@dataclass(frozen=True)
class _ToolSpec:
    """How one scanner is invoked and how its output is read."""

    module: str
    args: tuple[str, ...] = ()
    #: File extensions the tool accepts. Empty means "whole project".
    suffixes: tuple[str, ...] = ()
    per_file: bool = True
    output: str = "text"
    #: Prefixes that are context, not defects (mypy's ``note:`` lines).
    ignore_prefixes: tuple[str, ...] = field(default=())


_SPECS: dict[str, _ToolSpec] = {
    # --no-cache / --cache-dir keep scanner state out of the tree under
    # verification: a check must not modify what it is judging.
    "ruff": _ToolSpec(
        module="ruff",
        args=("check", "--no-cache", "--output-format=concise", "--quiet"),
        suffixes=(".py", ".pyi"),
    ),
    "mypy": _ToolSpec(
        module="mypy",
        args=("--no-color-output", "--no-error-summary", "--cache-dir=/dev/null"),
        suffixes=(".py", ".pyi"),
        ignore_prefixes=("note:",),
    ),
    "semgrep": _ToolSpec(
        module="semgrep",
        args=("scan", "--quiet", "--json", "--disable-version-check", "--metrics=off"),
        output="json",
    ),
}


class PythonScannerAdapter(ScannerAdapter):
    language = "python"
    supported_tools = tuple(_SPECS)
    #: What says "this is a Python project" at a repository root. The build
    #: manifests first: they are also what ``sandbox.py`` installs from.
    markers = (
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "requirements-dev.txt",
        "Pipfile",
    )

    def __init__(self, settings: Settings, artifacts_path: Path) -> None:
        self.settings = settings
        self.artifacts_path = artifacts_path
        self._interpreters: dict[Path, tuple[str, str | None]] = {}
        #: The artifacts subdirectory holding each workspace's latest suite run's
        #: coverage data. See ``_coverage_env``.
        self._coverage_runs: dict[Path, str] = {}

    # ── static checks ────────────────────────────────────────────────────

    def run_static_checks(
        self,
        workspace: Path,
        change: ChangeRequest,
        tools: Sequence[str] | None = None,
        scopes: Mapping[str, Sequence[str]] | None = None,
    ) -> list[ToolReport]:
        """Run the requested scanners over the change's files.

        ``tools=None`` means "every supported scanner"; an empty sequence means
        "none", which is what a rule library with no tool rules asks for. The
        caller chooses, because which scanners are mandatory is the team's
        configuration, not a default hidden in this file.

        ``scopes`` narrows a tool to the files the rules that bound it cover.
        semgrep is the exception and ignores it: it is run over the tree rather
        than over a file list, because a rule that matches across files cannot
        be answered from a list of the ones that changed.
        """

        requested = list(_SPECS) if tools is None else list(dict.fromkeys(tools))
        if not requested:
            return []
        runner = self._tool_runner(workspace)
        interpreter, provision_error = self._interpreter(workspace)
        return [
            self._run_tool(
                name,
                runner,
                interpreter,
                provision_error,
                workspace,
                change,
                None if scopes is None or name not in scopes else tuple(scopes[name]),
            )
            for name in requested
        ]

    def _run_tool(
        self,
        name: str,
        runner: ToolRunner,
        interpreter: str,
        provision_error: str | None,
        workspace: Path,
        change: ChangeRequest,
        scope: tuple[str, ...] | None = None,
    ) -> ToolReport:
        spec = _SPECS.get(name)
        if spec is None:
            return _unavailable(
                name,
                f"Unsupported scanner {name!r}. Supported: {', '.join(_SPECS)}.",
            )
        if not runner.module_available(interpreter, spec.module):
            reason = f"{name} is not installed in the execution environment."
            if provision_error:
                reason += f" Dependency provisioning failed first: {provision_error}"
            return _unavailable(name, reason)

        args = list(spec.args)
        if name == "semgrep":
            config = _semgrep_config(workspace)
            if config is None:
                return _unavailable(
                    name,
                    "semgrep needs a rule file in the repo ("
                    + ", ".join(SEMGREP_CONFIG_NAMES)
                    + "); --config auto is unavailable to a scanner denied the network.",
                )
            args += ["--config", config]

        targets: list[str] = []
        if spec.per_file:
            targets = _applicable_files(workspace, change, spec.suffixes, scope)
            if not targets:
                return _nothing_to_check(name, interpreter, spec, scope)
            # "--" keeps a file named like a flag from being read as one.
            args += ["--", *targets]

        report = runner.run(name, [interpreter, "-m", spec.module, *args])
        return _with_diagnostics(report, name, spec, len(targets))

    # ── tests & coverage (consumed by the regression gate) ───────────────

    def unselectable(self, workspace: Path) -> str | None:
        # pytest takes paths, but a team's command may not be pytest, may be
        # `make test` (a path becomes a make target), or may already name a
        # directory that holds every selected file, and pytest then runs all
        # of it. Nothing here can tell those apart from a command that narrows.
        if self.settings.test_command:
            return (
                "HONEST_CI_TEST_COMMAND runs as written, and the Python toolchain cannot narrow a "
                "command it did not compose"
            )
        return None

    def run_tests(self, workspace: Path, selection: list[str] | None = None) -> TestResults:
        """Run the repository's own suite and read the counts out of it.

        The default command is pytest, wrapped in ``coverage run`` when the
        gate will need a figure and the module is importable by the interpreter
        that runs the tests. A team's own ``HONEST_CI_TEST_COMMAND`` runs
        exactly as written, a selection included (``unselectable`` says so to
        whoever asks first): this code does not know how to add coverage or
        paths to a command it did not compose. Either way the coverage data
        goes to a file no earlier run wrote (``_coverage_env``), so a figure
        read afterwards was collected by this command or there is none.

        A custom command's counts come from the JUnit XML it leaves behind when
        ``HONEST_CI_TEST_REPORT`` names it, from pytest's summary line when the
        command is still pytest, and otherwise from nothing: only the exit code
        is known, and ``runner="exit-code"`` tells the gate so.
        """

        runner = self._tool_runner(workspace, timeout=self.settings.test_timeout_seconds)
        interpreter, provision_error = self._interpreter(workspace)
        custom = self.settings.test_command
        if custom:
            command = shlex.split(custom)
        else:
            if not runner.module_available(interpreter, "pytest"):
                return _suite_unavailable(
                    _missing_module("pytest", provision_error), interpreter, "pytest"
                )
            command = [interpreter, "-m", "pytest", "-q"]
            if self.settings.coverage_measured and runner.module_available(
                interpreter, "coverage"
            ):
                command = [interpreter, "-m", "coverage", "run", "-m", "pytest", "-q"]
        if selection and not custom:
            command += [item for item in selection if is_safe_relative_path(item)]
        pattern = self.settings.test_report if custom else None
        if pattern:
            # A report left by an earlier run must not be read as this run's
            # evidence, for the same reason coverage data goes to a new file:
            # the data has to be about the tree as it is now.
            testreports.clear_reports(workspace, pattern)
        report = runner.run(
            "tests" if custom else "pytest",
            command,
            env=self._coverage_env(workspace, runner.mode, fresh=True),
        )
        if pattern:
            results = _from_junit(
                report,
                testreports.read_reports(testreports.find_reports(workspace, pattern), workspace),
                pattern,
            )
        else:
            results = _parse_pytest(report)
            if custom and not any("pytest" in part for part in command):
                results = results.model_copy(update={"runner": "exit-code"})
        if not report.available:
            # No suite ran, so there are no counts to report. The caller must
            # read this as missing evidence, never as a green suite.
            return results.model_copy(
                update={"errors": max(1, results.errors), "failures": [report.stderr.strip()]}
            )
        return results

    def measure_coverage(
        self,
        workspace: Path,
        lines: Mapping[str, set[int]] | None = None,
    ) -> CoverageResult:
        """The suite's line coverage, and — when asked — the change's own lines.

        Two commands rather than one because they answer different questions
        and cost differently: ``report --format=total`` prints a single number,
        while ``json -o -`` carries every file's executed and missing lines.
        Both read the data the suite already left behind, so neither re-runs
        anything, and a caller with no changed lines to ask about is not made
        to carry a whole-repository report to learn one percentage.
        """

        runner = self._tool_runner(workspace)
        interpreter, provision_error = self._interpreter(workspace)
        if not runner.module_available(interpreter, "coverage"):
            reason = _missing_module("coverage", provision_error)
            return CoverageResult(
                available=False, command="coverage (not executed)", stderr=reason
            )
        env = self._coverage_env(workspace, runner.mode)
        if lines is None:
            report = runner.run(
                "coverage", [interpreter, "-m", "coverage", "report", "--format=total"], env=env
            )
            return _coverage_result(report, _parse_coverage_total(report))
        report = runner.run(
            "coverage", [interpreter, "-m", "coverage", "json", "-o", "-"], env=env
        )
        payload = _parse_coverage_json(report)
        if payload is None:
            return _coverage_result(report, None)
        ratio, per_file = payload
        # The JSON is the tool's working data, not evidence a person reads, and
        # it is megabytes on a real repository. The evidence keeps a line that
        # says what was read; the numbers it produced are in the result itself.
        summarised = report.model_copy(
            update={
                "stdout": f"coverage json: per-line data for {len(per_file)} "
                f"{'file' if len(per_file) == 1 else 'files'}."
            }
        )
        # Only this toolchain's own sources: a changed .md file has no
        # lines anyone could cover, and counting it would put a bar on it.
        # And only the sources the repository configured coverage to measure:
        # a test file is never in the report, so judging it would block every
        # change that touched one.
        return _coverage_result(
            summarised,
            ratio,
            changed_line_coverage(per_file, lines, (".py",), _measured_scope(workspace)),
        )

    # ── execution environment ────────────────────────────────────────────

    def image(self, workspace: Path) -> str | None:
        return self.settings.docker_image

    def _coverage_env(self, workspace: Path, mode: str, *, fresh: bool = False) -> dict[str, str]:
        """``COVERAGE_FILE`` for ``workspace``'s suite, in a directory no earlier run wrote.

        ``fresh`` is a suite about to start, and gets a new directory; the
        coverage commands after it read the same one. A team's own command
        cannot be made to rewrite its data file, so before this a ``.coverage``
        an earlier run left was read as this run's, and a stale 100% passed a
        bar. The variable outranks a ``data_file`` in the repository's coverage
        configuration, and pytest-cov reads it too.

        Under the artifacts directory rather than the tree: this run owns it,
        so nothing in the tree is overwritten or deleted, and a symlink the
        change planted cannot point the write somewhere else.
        """

        name = self._coverage_runs.get(workspace)
        if fresh or name is None:
            name = f"coverage-{uuid4().hex}"
            self._coverage_runs[workspace] = name
            (self.artifacts_path / name).mkdir(parents=True, exist_ok=True)
        root = "/artifacts" if mode == "docker" else str(self.artifacts_path.resolve())
        return {"COVERAGE_FILE": f"{root}/{name}/.coverage"}

    def _tool_runner(self, workspace: Path, timeout: int | None = None) -> ToolRunner:
        return ToolRunner(
            sandbox_path=workspace,
            artifacts_path=self.artifacts_path,
            mode=self.settings.effective_sandbox_isolation(),
            docker_image=self.settings.docker_image,
            timeout_seconds=timeout or self.settings.command_timeout_seconds,
            docker_cpus=self.settings.docker_cpus,
            docker_memory=self.settings.docker_memory,
            docker_pids_limit=self.settings.docker_pids_limit,
        )

    def _interpreter(self, workspace: Path) -> tuple[str, str | None]:
        """Resolve the interpreter that will run the tools, once per workspace.

        It has to be the same one availability is probed with: asking the host
        whether it has mypy says nothing about a container image or a sandbox
        virtualenv.
        """

        cached = self._interpreters.get(workspace)
        if cached is not None:
            return cached
        mode = self.settings.effective_sandbox_isolation()
        resolved: tuple[str, str | None] = (sys.executable, None)
        if mode == "docker":
            try:
                venv = SandboxBuilder(self.settings).ensure_deps(
                    workspace, self.artifacts_path, mode
                )
                resolved = (venv or "python", None)
            except DependencyProvisionError as exc:
                # Not fatal here: the tools are reported unavailable instead,
                # and fail-closed turns that into a block with the reason
                # attached, rather than a run that dies without evidence.
                resolved = ("python", str(exc))
        self._interpreters[workspace] = resolved
        return resolved


# ── report assembly ──────────────────────────────────────────────────────────


def _unavailable(name: str, reason: str) -> ToolReport:
    return ToolReport(
        name=name,
        command=f"{name} (not executed)",
        available=False,
        passed=False,
        stderr=reason,
        findings=[reason],
    )


def _missing_module(module: str, provision_error: str | None) -> str:
    reason = f"{module} is not installed in the execution environment."
    if provision_error:
        reason += f" Dependency provisioning failed first: {provision_error}"
    return reason


def _suite_unavailable(reason: str, interpreter: str, module: str) -> TestResults:
    """The suite never started. One error and the reason, never zero failures."""

    return TestResults(
        errors=1,
        failures=[reason],
        available=False,
        command=recorded_command([interpreter, "-m", module]) + " (not executed)",
        stderr=reason,
    )


def _nothing_to_check(
    name: str, interpreter: str, spec: _ToolSpec, scope: tuple[str, ...] | None = None
) -> ToolReport:
    """The tool is installed, and this change gave it nothing in its scope.

    Which of the two emptied the list is in the note, because they read
    differently: a change with no Python in it is a fact about the change,
    while a rule that covers none of its files is a fact about the rule.
    """

    kinds = "/".join(spec.suffixes)
    note = (
        f"0 applicable files: the rules bound to {name} cover no {kinds} file this change touched."
        if scope is not None
        else f"0 applicable files: this change touches no {kinds} file."
    )
    return ToolReport(
        name=name,
        command=recorded_command([interpreter, "-m", spec.module, *spec.args]),
        available=True,
        passed=True,
        stdout=note,
    )


def _with_diagnostics(
    report: ToolReport, name: str, spec: _ToolSpec, scanned_files: int
) -> ToolReport:
    if spec.output == "json":
        diagnostics = _parse_semgrep_json(name, report.stdout)
    else:
        diagnostics = _parse_text_diagnostics(name, report.stdout, spec.ignore_prefixes)
    findings = [_format_diagnostic(item) for item in diagnostics[:_MAX_LISTED_DIAGNOSTICS]]
    if not findings and not report.passed:
        # Exit code says it failed but nothing parsed: a crash, a bad config, a
        # tool version we do not understand. Keep the raw output as evidence.
        findings = list(report.findings)
    return report.model_copy(
        update={
            "diagnostics": diagnostics,
            "findings": findings,
            "stdout": report.stdout
            + f"\n[honest-ci] scanned {scanned_files} file(s), {len(diagnostics)} diagnostic(s)",
        }
    )


def _format_diagnostic(item: ToolDiagnostic) -> str:
    location = f"{item.file}:{item.line}" if item.file and item.line else item.file or item.tool
    code = f"[{item.code}] " if item.code else ""
    return f"{location}: {code}{item.message}"


# ── output parsing ───────────────────────────────────────────────────────────


def _parse_text_diagnostics(
    tool: str, stdout: str, ignore_prefixes: tuple[str, ...]
) -> list[ToolDiagnostic]:
    diagnostics: list[ToolDiagnostic] = []
    for line in stdout.splitlines():
        match = _TEXT_DIAGNOSTIC_RE.match(line.strip())
        if not match:
            continue
        message = match.group("message").strip()
        if message.startswith(ignore_prefixes):
            continue
        code = None
        coded = _RULE_CODE_RE.match(message)
        if coded:
            code = coded.group("code")
            message = coded.group("message").strip()
        diagnostics.append(
            ToolDiagnostic(
                tool=tool,
                file=match.group("file"),
                line=max(1, int(match.group("line"))),
                code=code,
                message=message,
            )
        )
    return diagnostics


def _parse_semgrep_json(tool: str, stdout: str) -> list[ToolDiagnostic]:
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return []
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    diagnostics: list[ToolDiagnostic] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        extra = item.get("extra")
        line = start.get("line") if isinstance(start, dict) else None
        message = extra.get("message", "") if isinstance(extra, dict) else ""
        diagnostics.append(
            ToolDiagnostic(
                tool=tool,
                file=str(item.get("path")) if item.get("path") else None,
                line=max(1, int(line)) if isinstance(line, int) else None,
                code=str(item.get("check_id")) if item.get("check_id") else None,
                message=str(message).strip(),
            )
        )
    return diagnostics


def _parse_pytest(report: ToolReport) -> TestResults:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for match in _PYTEST_COUNT_RE.finditer(report.stdout):
        outcome = match.group("outcome")
        key = "errors" if outcome.startswith("error") else outcome
        counts[key] = int(match.group("count"))
    failures = [
        line.strip()
        for line in report.stdout.splitlines()
        if line.startswith(_PYTEST_SUMMARY_PREFIXES)
    ]
    return TestResults(
        passed=counts["passed"],
        failed=counts["failed"],
        errors=counts["errors"],
        skipped=counts["skipped"],
        failures=failures,
        available=report.available,
        exit_code=report.exit_code,
        command=report.command,
        stdout=_tail(report.stdout),
        stderr=_tail(report.stderr),
        duration_s=report.duration_s,
        report_path=report.report_path,
    )


def _from_junit(report: ToolReport, summary: testreports.SuiteSummary, pattern: str) -> TestResults:
    """Counts from the JUnit XML a custom command left, with every gap named.

    A report nobody could read counts as an error rather than as nothing: the
    command may have exited 0, but a suite whose results cannot be read has
    not been shown to pass. No report at all is left at zero cases, which the
    gate reads as no evidence and hands to a person -- with the note below
    saying which glob matched nothing.
    """

    notes = [f"[honest-ci] {problem}" for problem in summary.problems]
    if not summary.reports and not summary.problems:
        notes.append(f"[honest-ci] no JUnit XML matched HONEST_CI_TEST_REPORT={pattern}; test counts unknown")
    failures = list(summary.failures) + [f"ERROR {problem}" for problem in summary.problems]
    stderr = "\n".join(part for part in (report.stderr.rstrip(), *notes) if part)
    return TestResults(
        passed=summary.passed,
        failed=summary.failed,
        errors=summary.errors + len(summary.problems),
        skipped=summary.skipped,
        failures=failures,
        available=report.available,
        exit_code=report.exit_code,
        command=report.command,
        stdout=_tail(report.stdout),
        stderr=_tail(stderr),
        duration_s=report.duration_s,
        report_path=report.report_path,
        runner="junit",
    )


def _parse_coverage_total(report: ToolReport) -> float | None:
    """``coverage report --format=total`` prints one number: the percentage."""

    if not report.passed:
        return None
    try:
        return min(1.0, max(0.0, float(report.stdout.strip()) / 100.0))
    except ValueError:
        return None


def _parse_coverage_json(
    report: ToolReport,
) -> tuple[float | None, dict[str, tuple[set[int], set[int]]]] | None:
    """``coverage json -o -``: the overall ratio plus each file's line sets.

    ``None`` when the output is not the report we asked for — a crash, a
    version that writes something else — so the caller reports a figure that
    could not be measured rather than a figure derived from a guess.
    """

    if not report.passed:
        return None
    try:
        payload = json.loads(report.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
        return None
    totals = payload.get("totals")
    percent = totals.get("percent_covered") if isinstance(totals, dict) else None
    ratio = (
        min(1.0, max(0.0, float(percent) / 100.0))
        if isinstance(percent, int | float)
        else None
    )
    per_file: dict[str, tuple[set[int], set[int]]] = {}
    for path, data in payload["files"].items():
        if not isinstance(data, dict):
            continue
        per_file[_normalize_report_path(str(path))] = (
            _line_set(data.get("executed_lines")),
            _line_set(data.get("missing_lines")),
        )
    return ratio, per_file


def _line_set(value: object) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, int) and item > 0}


def _normalize_report_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def _coverage_result(
    report: ToolReport,
    ratio: float | None,
    changed: ChangedLineCoverage | None = None,
) -> CoverageResult:
    """One ``CoverageResult`` from a tool run, with or without per-line data."""

    result = CoverageResult(
        ratio=ratio,
        available=report.available,
        command=report.command,
        exit_code=report.exit_code,
        stdout=report.stdout,
        stderr=report.stderr,
        duration_s=report.duration_s,
    )
    if changed is None:
        return result
    return result.model_copy(
        update={
            "changed_lines_total": changed.total,
            "changed_lines_covered": changed.covered,
            "uncovered": list(changed.uncovered),
            "unmapped_sources": list(changed.unmapped),
            "out_of_scope_sources": list(changed.out_of_scope),
        }
    )


# ── what coverage was configured to measure ──────────────────────────────────


def _measured_scope(workspace: Path) -> tuple[str, ...] | None:
    """Path prefixes ``[coverage:run] source`` asked to measure, or ``None``.

    ``None`` means the configuration does not say — no config file, no
    ``source``, a name nothing in the tree resolves to — and the caller then
    judges every Python file the change touched. That is the fail-closed
    reading: an unconfigured deployment keeps blocking on a changed source
    the report never named, which is the property the bar exists for.

    What it stops is the opposite mistake, and the one dogfooding hit: with
    ``source = ["honest_ci"]`` a test file is *never* in the report, so
    reading its absence as "nobody measured this" blocked every change that
    touched a test — a bar on files the repository deliberately does not
    measure is not a bar anybody can meet.
    """

    prefixes: list[str] = []
    for entry in _configured_sources(workspace):
        resolved = _resolve_source(workspace, entry)
        if resolved is not None and resolved not in prefixes:
            prefixes.append(resolved)
    return tuple(prefixes) or None


def _configured_sources(workspace: Path) -> list[str]:
    """``source`` / ``source_pkgs`` from the first coverage config that has one."""

    for name in _COVERAGE_CONFIG_FILES:
        path = workspace / name
        if not path.is_file():
            continue
        try:
            found = (
                _pyproject_sources(path) if name == "pyproject.toml" else _ini_sources(path, name)
            )
        except (OSError, ValueError, configparser.Error):
            # A config file nobody can parse says nothing about scope, and
            # saying nothing here means "judge everything" -- never "judge
            # nothing", which would retire the bar on a typo.
            continue
        if found:
            return found
    return []


def _pyproject_sources(path: Path) -> list[str]:
    run = tomllib.loads(path.read_text(encoding="utf-8")).get("tool", {})
    if not isinstance(run, dict):
        return []
    section = run.get("coverage", {}).get("run", {}) if isinstance(run.get("coverage"), dict) else {}
    if not isinstance(section, dict):
        return []
    return [
        str(item)
        for key in ("source", "source_pkgs")
        for item in section.get(key, [])
        if isinstance(item, str)
    ]


def _ini_sources(path: Path, name: str) -> list[str]:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    # ``.coveragerc`` uses ``[run]``; the shared files use ``[coverage:run]``.
    section = "run" if name == ".coveragerc" else "coverage:run"
    if not parser.has_section(section):
        return []
    return [
        line.strip()
        for key in ("source", "source_pkgs")
        for line in parser.get(section, key, fallback="").replace(",", "\n").splitlines()
        if line.strip()
    ]


def _resolve_source(workspace: Path, entry: str) -> str | None:
    """One ``source`` entry as a path relative to the workspace, or ``None``.

    An entry is either a directory (``src/app``) or an importable package
    name (``honest_ci``). coverage.py tells them apart by importing; here the
    tree is looked at instead, under the layouts a Python repository uses.
    """

    candidate = entry.strip().strip("/")
    if not candidate or not is_safe_relative_path(candidate):
        return None
    if (workspace / candidate).is_dir():
        return candidate
    as_path = candidate.replace(".", "/")
    for root in _PACKAGE_ROOTS:
        relative = f"{root}/{as_path}" if root else as_path
        if (workspace / relative).is_dir():
            return relative
    return None


def _tail(text: str) -> str:
    if len(text) <= _MAX_SUITE_OUTPUT_CHARS:
        return text
    return "[honest-ci] … earlier output truncated …\n" + text[-_MAX_SUITE_OUTPUT_CHARS:]


# ── inputs ───────────────────────────────────────────────────────────────────


def _applicable_files(
    workspace: Path,
    change: ChangeRequest,
    suffixes: tuple[str, ...],
    scope: tuple[str, ...] | None = None,
) -> list[str]:
    """Changed files that exist in the workspace and are safe to pass as argv.

    ``scope`` is the subset the rules asked this tool about; ``None`` is every
    changed file, which is what a caller with no rule library to consult means.
    """

    declared = [item.path for item in change.files_changed] or extract_changed_files(change.diff)
    root = workspace.resolve()
    keep: list[str] = []
    for path in sorted(set(declared)):
        if not is_safe_relative_path(path):
            continue
        if scope is not None and path not in scope:
            continue
        if suffixes and not path.endswith(suffixes):
            continue
        resolved = (workspace / path).resolve()
        # A deleted file has nothing to scan; a symlink out of the tree must not
        # drag an unrelated file into the evidence.
        if not resolved.is_file() or not resolved.is_relative_to(root):
            continue
        keep.append(path)
    return keep


def _semgrep_config(workspace: Path) -> str | None:
    for name in SEMGREP_CONFIG_NAMES:
        if (workspace / name).exists():
            return name
    return None


__all__ = ["SEMGREP_CONFIG_NAMES", "PythonScannerAdapter"]
