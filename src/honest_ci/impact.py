"""Impact analysis: what this change touched, and which tests reach it.

The regression gate's v0 question was "does the suite still pass". v1 adds the
one a reviewer asks next: **what changed, and what does it affect**. Both
answers here are deterministic — this module reads the diff and the files in
the workspace, and nothing else decides anything.

Two properties are the whole reason it is a module of its own:

* **A file nobody could analyse is named, not dropped.** A syntax error, a
  language this build has no reader for, a file the workspace does not hold:
  each lands in ``unanalysed_files`` and travels into the evidence pack. An
  impact analysis that silently shrinks is worse than none, because the gate
  that selects tests from it would then run a smaller suite for a reason
  nobody recorded.
* **Impacted tests are named the way the toolchain selects them.** Python
  selects by file path, Maven by class name (``-Dtest=``). Producing the
  toolchain's own vocabulary here keeps the gate from learning either one, and
  keeps ``run_tests(selection=...)`` receiving something its adapter already
  validates.

What this deliberately is *not* is a call graph. A symbol index with real
resolution is a different project; what a regression gate needs to select a
narrower suite is "which test files mention what changed", and the cost of
being wrong is bounded by the gate's own rule that an empty selection always
falls back to the whole suite.
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import intake
from .contracts import ChangeRequest
from .diffs import changed_line_locations, is_safe_relative_path
from .utils import RUNTIME_PATH_PARTS, is_runtime_path

#: Files read for symbol mapping, and bytes read from each. A generated bundle
#: is not a symbol anybody reviews, and reading it would only slow the gate.
MAX_ANALYSED_FILES = 200
MAX_ANALYSED_BYTES = 512 * 1024

#: Candidate test files scanned for a mention of a changed symbol. Bounded for
#: the same reason: a monorepo's test tree is not something to walk per change.
MAX_SCANNED_TESTS = 2000

#: Symbols reported, and tests selected. A change that touches more than this
#: is a change whose blast radius is the whole suite anyway.
MAX_SYMBOLS = 200
MAX_IMPACTED_TESTS = 100

#: Java declarations, read with a regex rather than a parser. A real Java
#: parser is a dependency this product does not want in a private deployment
#: (see AGENTS.md), and the question here — "which class and which method does
#: this added line sit in" — is answerable from the declaration lines alone.
_JAVA_TYPE_RE = re.compile(
    r"^\s*(?:@\w+[^\n]*\s+)?(?:public|protected|private|abstract|final|static|sealed|non-sealed|\s)*"
    r"\b(?:class|interface|enum|record)\s+(?P<name>[A-Za-z_$][\w$]*)"
)
_JAVA_METHOD_RE = re.compile(
    r"^\s*(?:public|protected|private|abstract|final|static|synchronized|native|default|\s)*"
    r"(?:<[^>]+>\s*)?[\w$.<>\[\],?\s]+\s+(?P<name>[A-Za-z_$][\w$]*)\s*\([^;{]*\)\s*"
    r"(?:throws\s[\w$.,\s]+)?\{"
)
_JAVA_KEYWORDS = frozenset(
    {"if", "for", "while", "switch", "catch", "synchronized", "return", "new", "do", "try"}
)

#: Where each toolchain keeps its tests, and how a test is named to it.
_PYTHON_TEST_DIRS = ("tests", "test")
_JAVA_TEST_ROOT = "src/test/java"
_JAVA_TEST_PARTS = _JAVA_TEST_ROOT.split("/")
_JAVA_TEST_SUFFIXES = ("Test.java", "Tests.java", "IT.java")


@dataclass(frozen=True)
class ChangedSymbol:
    """One function, class or method this change added lines inside.

    ``kind`` is ``module`` for a top-level statement: a change to an import or
    a module constant has no enclosing symbol, and calling it one would invent
    a name no test can be matched against.
    """

    file: str
    name: str
    kind: str
    line: int

    @property
    def simple_name(self) -> str:
        """The last segment — what a test file would actually mention."""

        return self.name.rsplit(".", maxsplit=1)[-1]


@dataclass(frozen=True)
class ImpactAnalysis:
    """What changed, what reaches it, and what could not be read at all."""

    language: str
    changed_symbols: tuple[ChangedSymbol, ...] = ()
    #: Selection items in the toolchain's own vocabulary: file paths for
    #: Python, class names for Maven's ``-Dtest=``.
    impacted_tests: tuple[str, ...] = ()
    unanalysed_files: tuple[str, ...] = ()

    @property
    def analysed(self) -> bool:
        """Whether this build can read this language's sources at all."""

        return self.language in _READERS

    def as_dict(self) -> dict[str, Any]:
        """The shape the evidence pack carries, JSON-serialisable as it is."""

        return {
            "language": self.language,
            "analysed": self.analysed,
            "changed_symbols": [
                {"file": item.file, "symbol": item.name, "kind": item.kind, "line": item.line}
                for item in self.changed_symbols
            ],
            "impacted_tests": list(self.impacted_tests),
            "unanalysed_files": list(self.unanalysed_files),
        }


def analyse(workspace: Path, change: ChangeRequest, language: str) -> ImpactAnalysis:
    """Map the change's added lines to symbols, then find the tests that reach them.

    ``language`` comes from the scanner port (``ScannerAdapter.language_at``),
    so the toolchain that will run the suite is the one whose sources are read
    here. A language this build has no reader for returns an analysis that says
    so — ``analysed`` is false — rather than an empty one that reads like a
    change touching nothing.
    """

    reader = _READERS.get(language)
    if reader is None:
        return ImpactAnalysis(language=language)
    symbols: list[ChangedSymbol] = []
    unanalysed: list[str] = []
    root = workspace.resolve()
    added = changed_line_locations(change.diff)
    for path in sorted(added)[:MAX_ANALYSED_FILES]:
        lines = added[path]
        if not lines or not is_safe_relative_path(path) or not reader.owns(path):
            continue
        source = _read(workspace / path, root)
        if source is None:
            unanalysed.append(path)
            continue
        found = reader.symbols(path, source, lines)
        if found is None:
            unanalysed.append(path)
            continue
        symbols.extend(found)
    symbols = _dedupe(symbols)[:MAX_SYMBOLS]
    # Enumerated once, here, rather than per reader: which files belong to this
    # repository is not a per-language question, and asking git twice for the
    # same answer would only be slower.
    files = _repository_files(workspace)
    return ImpactAnalysis(
        language=language,
        changed_symbols=tuple(symbols),
        impacted_tests=tuple(reader.tests(workspace, symbols, set(added), files)),
        unanalysed_files=tuple(unanalysed),
    )


# ── per-language readers ─────────────────────────────────────────────────────


class _Reader:
    """How one toolchain reads its own sources and names its own tests."""

    suffixes: tuple[str, ...] = ()

    def owns(self, path: str) -> bool:
        return path.endswith(self.suffixes)

    def symbols(
        self, path: str, source: str, lines: Iterable[int]
    ) -> list[ChangedSymbol] | None:
        """``None`` when the file could not be read — never an empty list for it."""

        raise NotImplementedError

    def tests(
        self,
        workspace: Path,
        symbols: list[ChangedSymbol],
        changed: set[str],
        files: Sequence[str],
    ) -> list[str]:
        """Selection items, chosen out of ``files`` — never out of the tree.

        ``files`` is what the repository holds (see ``_repository_files``), as
        posix paths relative to ``workspace``. A reader that walked the
        workspace itself instead would pick up whatever else is lying in it.
        """

        raise NotImplementedError


class _PythonReader(_Reader):
    suffixes = (".py",)

    def symbols(
        self, path: str, source: str, lines: Iterable[int]
    ) -> list[ChangedSymbol] | None:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            # A file the interpreter itself cannot read. Reported as
            # unanalysable rather than as a file with no symbols: the second
            # would be a claim about the code, and this is a claim about us.
            return None
        spans = _python_spans(tree)
        found: list[ChangedSymbol] = []
        for line in sorted(set(lines)):
            enclosing = [span for span in spans if span[0] <= line <= span[1]]
            if not enclosing:
                found.append(ChangedSymbol(path, _module_name(path), "module", line))
                continue
            # Innermost wins: a method inside a class is what a test names.
            start, _end, name, kind = max(enclosing, key=lambda span: span[0])
            found.append(ChangedSymbol(path, name, kind, start))
        return found

    def tests(
        self,
        workspace: Path,
        symbols: list[ChangedSymbol],
        changed: set[str],
        files: Sequence[str],
    ) -> list[str]:
        wanted = {symbol.simple_name for symbol in symbols}
        wanted |= {_module_name(symbol.file).rsplit(".", maxsplit=1)[-1] for symbol in symbols}
        if not wanted:
            return []
        root = workspace.resolve()
        selected: list[str] = []
        for candidate in _python_test_files(files):
            if candidate in changed:
                # A test the change itself edited is impacted by definition,
                # and reading it for a mention would be beside the point.
                selected.append(candidate)
                continue
            text = _read(workspace / candidate, root)
            if text is not None and any(name in text for name in wanted):
                selected.append(candidate)
            if len(selected) >= MAX_IMPACTED_TESTS:
                break
        return sorted(set(selected))


class _JavaReader(_Reader):
    suffixes = (".java",)

    def symbols(
        self, path: str, source: str, lines: Iterable[int]
    ) -> list[ChangedSymbol] | None:
        spans = _java_spans(source)
        if not spans:
            # No declaration at all in a file the diff added lines to: this
            # reader did not understand it, which is not the same as a file
            # whose change sits outside every method.
            return None
        found: list[ChangedSymbol] = []
        for line in sorted(set(lines)):
            enclosing = [span for span in spans if span[0] <= line]
            start, name, kind = max(enclosing, key=lambda span: span[0], default=spans[0])
            found.append(ChangedSymbol(path, name, kind, start))
        return found

    def tests(
        self,
        workspace: Path,
        symbols: list[ChangedSymbol],
        changed: set[str],
        files: Sequence[str],
    ) -> list[str]:
        wanted = {symbol.name.split(".", maxsplit=1)[0] for symbol in symbols}
        wanted |= {symbol.simple_name for symbol in symbols}
        if not wanted:
            return []
        root = workspace.resolve()
        selected: list[str] = []
        for candidate in _java_test_files(files):
            text = _read(workspace / candidate, root)
            if text is None:
                continue
            if any(name in text for name in wanted):
                # Maven selects by class name, so that is what a selection is.
                selected.append(Path(candidate).stem)
            if len(selected) >= MAX_IMPACTED_TESTS:
                break
        return sorted(set(selected))


_READERS: dict[str, _Reader] = {"python": _PythonReader(), "java": _JavaReader()}


# ── source reading ───────────────────────────────────────────────────────────


def _python_spans(tree: ast.Module) -> list[tuple[int, int, str, str]]:
    """``(start, end, qualified name, kind)`` for every definition in the file."""

    spans: list[tuple[int, int, str, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                walk(child, prefix)
                continue
            name = f"{prefix}.{child.name}" if prefix else child.name
            kind = "class" if isinstance(child, ast.ClassDef) else "function"
            if kind == "function" and prefix:
                kind = "method"
            end = child.end_lineno or child.lineno
            # From the first decorator: a line added to a decorator belongs to
            # the thing it decorates, not to the module around it.
            start = min(
                [child.lineno, *[item.lineno for item in child.decorator_list]]
            )
            spans.append((start, end, name, kind))
            walk(child, name)

    walk(tree, "")
    return spans


def _java_spans(source: str) -> list[tuple[int, str, str]]:
    """``(line, qualified name, kind)`` for each type and method declaration.

    Declaration lines only, and the enclosing type is whichever was declared
    last before them. That is wrong for a nested class declared after a method
    it does not contain; it is right for the shape every reviewed Java file
    actually has, and the cost of being wrong is a symbol name in the evidence,
    not a verdict.
    """

    spans: list[tuple[int, str, str]] = []
    type_name = ""
    for index, line in enumerate(source.splitlines(), start=1):
        match = _JAVA_TYPE_RE.match(line)
        if match:
            type_name = match.group("name")
            spans.append((index, type_name, "class"))
            continue
        method = _JAVA_METHOD_RE.match(line)
        if method and method.group("name") not in _JAVA_KEYWORDS:
            name = method.group("name")
            spans.append((index, f"{type_name}.{name}" if type_name else name, "method"))
    return spans


def _repository_files(workspace: Path) -> list[str]:
    """What this repository holds, as posix paths relative to ``workspace``.

    git first, and this is the whole point of the function: a working tree
    holds more than the repository. A ``.venv`` carries its dependencies' own
    test suites, a vendored checkout carries somebody else's, and a git
    worktree under the repository carries a *second copy of these very tests* —
    which pytest then refuses to collect at all ("import file mismatch"),
    turning a narrowed suite into no suite. Asking git which files it tracks
    answers that in one call, and answers it the way the developer would.

    When git has no answer — no git, not a repository, a merge request
    materialised without its ``.git`` — the tree is walked instead, pruning the
    tooling directories. That is weaker (an untracked test is a real test, and
    a vendored one is not), which is why it is the fallback and not the rule.
    """

    tracked = intake.tracked_paths(workspace)
    if tracked is not None:
        return sorted(path for path in tracked if not is_runtime_path(Path(path)))
    found: list[str] = []
    for parent, directories, names in os.walk(workspace):
        # Pruned rather than filtered afterwards: descending into a ``.venv``
        # to throw the result away is the expensive half.
        directories[:] = [name for name in directories if name not in RUNTIME_PATH_PARTS]
        base = Path(parent).relative_to(workspace)
        found += [(base / name).as_posix() for name in names]
    return sorted(found)


def _python_test_files(files: Sequence[str]) -> list[str]:
    """Files a Python project would keep tests in, deduplicated and bounded."""

    seen: dict[str, None] = {}
    for path in files:
        if not path.endswith(".py"):
            continue
        head, _, name = path.rpartition("/")
        in_test_dir = head.split("/", maxsplit=1)[0] in _PYTHON_TEST_DIRS
        if in_test_dir or name.startswith("test_") or name.endswith("_test.py"):
            seen.setdefault(path, None)
    return list(seen)[:MAX_SCANNED_TESTS]


def _java_test_files(files: Sequence[str]) -> list[str]:
    return [
        path
        for path in files
        if path.endswith(_JAVA_TEST_SUFFIXES) and _in_java_test_root(path)
    ][:MAX_SCANNED_TESTS]


def _in_java_test_root(path: str) -> bool:
    """Under ``src/test/java``, at the repository root or one Maven module down."""

    parts = path.split("/")
    return any(
        parts[start : start + 3] == _JAVA_TEST_PARTS and len(parts) > start + 3
        for start in (0, 1)
    )


def _read(path: Path, root: Path) -> str | None:
    """Read a file inside ``root``; ``None`` when it cannot be read at all."""

    try:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            return None
        if resolved.stat().st_size > MAX_ANALYSED_BYTES:
            return None
        return resolved.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _module_name(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def _dedupe(symbols: list[ChangedSymbol]) -> list[ChangedSymbol]:
    """One entry per symbol, keeping the first line that reached it."""

    seen: dict[tuple[str, str], ChangedSymbol] = {}
    for symbol in symbols:
        seen.setdefault((symbol.file, symbol.name), symbol)
    return list(seen.values())


def changed_lines_for(change: ChangeRequest) -> Mapping[str, set[int]]:
    """The added lines a coverage reader is asked to judge, safe paths only."""

    return {
        path: lines
        for path, lines in changed_line_locations(change.diff).items()
        if lines and is_safe_relative_path(path)
    }


# ── changed-line coverage ────────────────────────────────────────────────────

#: Uncovered lines carried in the result. The developer needs the first
#: screenful; the coverage report itself holds the rest.
MAX_UNCOVERED_LINES = 200


@dataclass(frozen=True)
class ChangedLineCoverage:
    """How much of what this change added the suite actually executed."""

    total: int = 0
    covered: int = 0
    uncovered: tuple[tuple[str, int], ...] = ()
    #: Source files of this language that the coverage data never named. Their
    #: lines are in neither count: nobody measured them, and a denominator that
    #: quietly drops them would report a bar as met that was never checked.
    unmapped: tuple[str, ...] = ()
    #: Files the coverage configuration never asked anyone to measure -- a test
    #: file, a script outside the measured packages. Also in neither count, but
    #: for the opposite reason: there is no missing measurement here, only a
    #: file that was never in scope. Kept apart from ``unmapped`` because one
    #: is a gap in the evidence and the other is the configuration working.
    out_of_scope: tuple[str, ...] = ()


def changed_line_coverage(
    per_file: Mapping[str, tuple[set[int], set[int]]],
    wanted: Mapping[str, set[int]],
    suffixes: tuple[str, ...],
    scope: tuple[str, ...] | None = None,
) -> ChangedLineCoverage:
    """Judge each added line against one file's ``(executed, missing)`` sets.

    A line in neither set is not executable — a comment, a blank line, a
    closing bracket — and is counted in neither total nor covered: asking a
    developer to "cover" a docstring would make the bar meaningless.

    ``scope`` is the path prefixes the coverage tool was configured to measure.
    A changed file outside them is reported as out of scope rather than as
    missing data — the difference dogfooding ran into: ``source =
    ["honest_ci"]`` means test files are *never* in a coverage report, so
    reading their absence as "nobody measured this" blocked every change that
    touched a test. ``None`` is "the configuration does not say", and then
    every file of this language is judged, which is the fail-closed direction.

    Shared by every toolchain because the arithmetic is the same everywhere;
    only where the two sets come from differs (coverage.py's JSON, JaCoCo's
    XML), and that stays in each adapter.
    """

    total = 0
    covered = 0
    uncovered: list[tuple[str, int]] = []
    unmapped: list[str] = []
    out_of_scope: list[str] = []
    for path in sorted(wanted):
        if suffixes and not path.endswith(suffixes):
            continue
        if scope is not None and not _within(path, scope):
            out_of_scope.append(path)
            continue
        data = per_file.get(path)
        if data is None:
            unmapped.append(path)
            continue
        executed, missing = data
        for line in sorted(wanted[path]):
            if line in executed:
                total += 1
                covered += 1
            elif line in missing:
                total += 1
                if len(uncovered) < MAX_UNCOVERED_LINES:
                    uncovered.append((path, line))
    return ChangedLineCoverage(
        total=total,
        covered=covered,
        uncovered=tuple(uncovered),
        unmapped=tuple(unmapped),
        out_of_scope=tuple(out_of_scope),
    )


def _within(path: str, scope: tuple[str, ...]) -> bool:
    """Whether ``path`` sits under one of the measured prefixes."""

    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in scope)


__all__ = [
    "ChangedLineCoverage",
    "ChangedSymbol",
    "ImpactAnalysis",
    "analyse",
    "changed_line_coverage",
    "changed_lines_for",
]
