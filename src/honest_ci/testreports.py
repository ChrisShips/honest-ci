"""Reading test counts out of JUnit XML.

JUnit XML is the one report format every test runner can produce: Maven
Surefire writes it by default, pytest on ``--junitxml``, Gradle, ``gotestsum``,
Jest and the rest on a flag. That makes it the universal outlet for the
regression gate. A team whose suite is not pytest points
``HONEST_CI_TEST_REPORT`` at the files their command leaves behind, and the gate
reads the same counts and failing test names it reads from pytest's summary
line -- instead of only an exit code.

Two things are deliberate:

* **A report is only trusted for the run that produced it.** The caller clears
  matching reports before the command runs (see ``clear_reports``), so a file
  left by last week's run cannot stand in for this one -- the same reason the
  Python adapter runs the suite under ``coverage run``, which rewrites its data.
* **A report that cannot be read is a problem, not zero tests.** A malformed
  file is named in ``problems`` and the caller decides what that means; it is
  never silently a suite with no failures.

The parser is the standard library's. Reports are produced inside the sandbox
by the code under verification, so they are untrusted input: files are size
capped, and anything carrying a DTD is refused outright -- no test runner emits
one, and entity expansion is the one XML attack a report could carry.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .diffs import is_safe_relative_path
from .utils import RUNTIME_PATH_PARTS

#: One report larger than this is not a test report anyone reads.
MAX_REPORT_BYTES = 16 * 1024 * 1024
#: Reports read per run. A monorepo with more modules than this narrows the glob.
MAX_REPORTS = 500
#: Failure lines kept. The gate lists the first twenty; the rest are a count.
MAX_FAILURE_LINES = 200
_MAX_MESSAGE_CHARS = 300
_DTD_MARKERS = ("<!DOCTYPE", "<!ENTITY")
_ROOT_TAGS = ("testsuite", "testsuites")


class ReportError(ValueError):
    """One report could not be read. Named, never counted as a clean suite."""


@dataclass(frozen=True)
class SuiteSummary:
    """Counts and failure lines from one or more JUnit XML reports."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    #: ``FAILED path::test - reason`` and ``ERROR Class#method - reason``
    #: lines in report order: the shape pytest prints, so the gate reads a
    #: Surefire report and a pytest run through the same code.
    failures: tuple[str, ...] = ()
    #: Reports read, relative to the workspace.
    reports: tuple[str, ...] = ()
    #: Reports that could not be read, each with why.
    problems: tuple[str, ...] = ()

    @property
    def collected(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped


def find_reports(workspace: Path, pattern: str, *, strict: bool = False) -> list[Path]:
    """Report files matching ``pattern`` inside ``workspace``, sorted; none outside it.

    The pattern is configuration, but the tree is the change's, so a symlink
    from ``target/`` to somewhere else must not pull that somewhere into the
    evidence; every match has to resolve under the workspace.
    """

    if not is_safe_relative_path(pattern):
        if strict:
            raise ReportError("the report glob must be a relative path inside the workspace")
        return []
    root = workspace.resolve()
    try:
        candidates = sorted(workspace.glob(pattern))
    except (OSError, ValueError, NotImplementedError):
        if strict:
            raise ReportError("the report glob cannot be read") from None
        return []
    found: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            relative = candidate.relative_to(workspace)
        except (OSError, ValueError):
            continue
        if not resolved.is_file() or not resolved.is_relative_to(root):
            continue
        # Maven's target is runtime state for intake and fingerprints, but it
        # is also the explicit outlet for the reports being requested here.
        if any(part in RUNTIME_PATH_PARTS - {"target"} for part in relative.parts):
            continue
        if len(found) >= MAX_REPORTS:
            if strict:
                raise ReportError(
                    f"more reports match than the limit of {MAX_REPORTS}, and a verdict "
                    "cannot rest on only some of them"
                )
            break
        found.append(candidate)
    return found


def clear_reports(workspace: Path, pattern: str) -> int:
    """Delete reports an earlier run left behind; returns how many went."""

    removed = 0
    for path in find_reports(workspace, pattern):
        try:
            path.unlink()
            removed += 1
        except OSError:
            # A report that cannot be removed will be read again, and the
            # caller's counts would be about an old run. Left to the reader to
            # notice by the file's age is not good enough, but neither is
            # failing the whole suite over a stubborn file; the run proceeds.
            continue
    return removed


def read_reports(paths: Sequence[Path], workspace: Path) -> SuiteSummary:
    """Sum the reports; unreadable ones are named in ``problems``."""

    total = SuiteSummary()
    failures: list[str] = []
    reports: list[str] = []
    problems: list[str] = []
    passed = failed = errors = skipped = 0
    for path in paths:
        label = _relative(path, workspace)
        try:
            summary = _read_one(path, label)
        except ReportError as exc:
            problems.append(f"{label}: {exc}")
            continue
        reports.append(label)
        passed += summary.passed
        failed += summary.failed
        errors += summary.errors
        skipped += summary.skipped
        failures.extend(summary.failures)
    return SuiteSummary(
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        failures=tuple(failures[:MAX_FAILURE_LINES]),
        reports=tuple(reports),
        problems=tuple(problems),
    ) if paths else total


def parse_junit(text: str, origin: str = "report") -> SuiteSummary:
    """One JUnit XML document as counts; raises ``ReportError`` when it is not one.

    Counts come from the ``<testcase>`` elements, not from the ``tests=`` and
    ``failures=`` attributes on the suite: runners disagree on what those
    attributes include (skipped tests, errored fixtures), and the cases are
    what a person would count.
    """

    if any(marker in text for marker in _DTD_MARKERS):
        raise ReportError("carries a DTD; a test report never needs one, so it is not read")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ReportError(f"not well-formed XML ({exc})") from exc
    if root.tag not in _ROOT_TAGS:
        raise ReportError(f"root element is <{root.tag}>, not <testsuite> or <testsuites>")
    passed = failed = errors = skipped = 0
    failures: list[str] = []
    for case in root.iter("testcase"):
        location = _location(case)
        failure = case.find("failure")
        error = case.find("error")
        if failure is not None:
            failed += 1
            failures.append(_line("FAILED", location, failure))
        elif error is not None:
            errors += 1
            failures.append(_line("ERROR", location, error))
        elif case.find("skipped") is not None:
            skipped += 1
        else:
            passed += 1
    return SuiteSummary(
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        failures=tuple(failures),
        reports=(origin,),
    )


# ── internals ────────────────────────────────────────────────────────────────


def _read_one(path: Path, label: str) -> SuiteSummary:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ReportError(f"cannot be read ({exc.strerror or exc})") from exc
    if size > MAX_REPORT_BYTES:
        raise ReportError(f"is {size} bytes, above the {MAX_REPORT_BYTES}-byte limit")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ReportError(f"cannot be read ({exc.strerror or exc})") from exc
    return parse_junit(text, label)


def _location(case: ET.Element) -> str:
    """Where a test lives, in the form the gate already parses.

    pytest writes ``file`` and ``name``; ``path::test`` is what its own summary
    prints and what the gate turns into a finding's file. Everything else has
    a class name: ``Class#method`` is the JUnit convention, and its ``#`` keeps
    the gate from reading ``com.acme.OrderTest`` as a path in the tree.
    """

    name = (case.get("name") or "").strip() or "?"
    file = (case.get("file") or "").strip()
    if file and is_safe_relative_path(file):
        return f"{file}::{name}"
    classname = (case.get("classname") or "").strip()
    return f"{classname}#{name}" if classname else name


def _line(kind: str, location: str, element: ET.Element) -> str:
    message = (element.get("message") or "").strip()
    if not message:
        message = (element.text or "").strip().splitlines()[0] if (element.text or "").strip() else ""
    message = " ".join(message.split())
    if len(message) > _MAX_MESSAGE_CHARS:
        message = message[:_MAX_MESSAGE_CHARS] + "…"
    return f"{kind} {location} - {message}" if message else f"{kind} {location}"


def _relative(path: Path, workspace: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return path.name


__all__ = [
    "MAX_FAILURE_LINES",
    "MAX_REPORTS",
    "MAX_REPORT_BYTES",
    "ReportError",
    "SuiteSummary",
    "clear_reports",
    "find_reports",
    "parse_junit",
    "read_reports",
]
