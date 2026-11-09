"""Untrusted Maven XML: located diagnostics and reusable JaCoCo line data."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

from .contracts import ToolDiagnostic
from .diffs import is_safe_relative_path
from .redaction import redact_text
from .testreports import MAX_REPORT_BYTES, ReportError

# JaCoCo emits this external declaration, which ElementTree does not fetch.
# Internal subsets and entity declarations are never needed by these tools.
_JACOCO_DTD = re.compile(
    r'<!DOCTYPE\s+report\s+PUBLIC\s+"-//JACOCO//DTD Report 1\.[01]//EN"\s+"report\.dtd"\s*>'
)


def read_xml(path: Path, expected: str) -> ET.Element:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_REPORT_BYTES + 1)
        if len(raw) > MAX_REPORT_BYTES:
            raise ReportError("the XML is over the size limit")
        text = raw.decode("utf-8-sig")
        if expected == "report":
            text = _JACOCO_DTD.sub("", text, count=1)
        if "<!DOCTYPE" in text or "<!ENTITY" in text:
            raise ReportError("the XML carries a DTD or an entity, which is not supported")
        root = ET.fromstring(text)
    except (OSError, UnicodeError, ET.ParseError) as exc:
        raise ReportError("the XML cannot be read or is not well-formed") from exc
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    if root.tag != expected:
        raise ReportError(f"the XML root element is <{root.tag}>, not <{expected}>")
    return root


def _number(element: ET.Element, name: str) -> int:
    try:
        value = int(element.attrib[name])
        if value < 0:
            raise ValueError
        return value
    except (KeyError, ValueError) as exc:
        raise ReportError(f"the XML attribute {name} is not a non-negative integer") from exc


def _module(path: Path) -> Path:
    return next(parent.parent for parent in path.parents if parent.name == "target")


def source_path(value: str, workspace: Path, module: Path, *, package_path: bool = False) -> str:
    value = value.replace("\\", "/")
    if value.startswith("/workspace/"):
        candidate = workspace / value.removeprefix("/workspace/")
    elif Path(value).is_absolute():
        candidate = Path(value)
    elif is_safe_relative_path(value):
        if package_path:
            candidates = [module / "src" / kind / "java" / value for kind in ("main", "test")]
            candidate = next((item for item in candidates if item.is_file()), candidates[0])
        else:
            candidate = module / value
    else:
        raise ReportError("a diagnostic names an unsafe source file path")
    try:
        relative = candidate.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as exc:
        raise ReportError("a diagnostic names a source file outside the workspace") from exc
    if not is_safe_relative_path(relative):
        raise ReportError("a diagnostic names an unsafe source file path")
    return relative


def read_diagnostics(path: Path, workspace: Path, tool: str) -> list[ToolDiagnostic]:
    root = read_xml(
        path, {"checkstyle": "checkstyle", "pmd": "pmd", "spotbugs": "BugCollection"}[tool]
    )
    module = _module(path)
    diagnostics: list[ToolDiagnostic] = []
    if tool == "spotbugs":
        for bug in root.findall("BugInstance"):
            sources = bug.findall(".//SourceLine")
            source = next((item for item in sources if item.get("primary") == "true"), None)
            if source is None:
                source = next((item for item in sources if item.get("sourcepath")), None)
            diagnostics.append(
                ToolDiagnostic(
                    tool=tool,
                    file=source_path(
                        source.get("sourcepath", ""), workspace, module, package_path=True
                    )
                    if source is not None and source.get("sourcepath")
                    else None,
                    line=(_number(source, "start") or None)
                    if source is not None and source.get("start") is not None
                    else None,
                    code=bug.get("type"),
                    message=redact_text(
                        bug.findtext("LongMessage")
                        or bug.findtext("ShortMessage")
                        or bug.get("type")
                        or ""
                    ),
                )
            )
    else:
        for file in root.findall("file"):
            for item in file.findall("error" if tool == "checkstyle" else "violation"):
                diagnostics.append(
                    ToolDiagnostic(
                        tool=tool,
                        file=source_path(file.get("name", ""), workspace, module),
                        line=_number(item, "line" if tool == "checkstyle" else "beginline") or None,
                        code=item.get("source" if tool == "checkstyle" else "rule"),
                        message=redact_text(
                            item.get("message", "")
                            if tool == "checkstyle"
                            else " ".join("".join(item.itertext()).split())
                        ),
                    )
                )
        if root.findall("error") or root.findall("processingerror") or root.findall("configerror"):
            raise ReportError(
                f"the {tool} report records errors in the tool itself, so the check cannot "
                "count as complete"
            )
    return diagnostics


def read_jacoco_totals(path: Path) -> tuple[int, int]:
    root = read_xml(path, "report")
    counter = root.find("counter[@type='LINE']")
    if counter is None:
        raise ReportError("the JaCoCo report has no top-level LINE counter")
    return _number(counter, "covered"), _number(counter, "missed")


def read_jacoco_line_data(
    paths: Sequence[Path], workspace: Path
) -> dict[str, tuple[set[int], set[int]]]:
    """Workspace-relative source → ``(executed lines, missing lines)``.

    JaCoCo names sources by package path, which only becomes a file once it is
    resolved against the module that produced the report -- two modules can
    hold the same ``com/acme/Order.java``. A package path that resolves to no
    file in the workspace is left out entirely rather than guessed at: the
    caller then reports the changed file as one with no coverage data, which is
    the truth, instead of judging it against another module's numbers.
    """

    per_file: dict[str, tuple[set[int], set[int]]] = {}
    for path in paths:
        module = _module(path)
        for name, lines in read_jacoco_lines(path).items():
            resolved = _source_file(name, workspace, module)
            if resolved is None:
                continue
            executed, missing = per_file.setdefault(resolved, (set(), set()))
            for number, (missed_instructions, covered_instructions) in lines.items():
                # JaCoCo counts instructions; a line with any covered one ran.
                if covered_instructions > 0:
                    executed.add(number)
                elif missed_instructions > 0:
                    missing.add(number)
    return per_file


def _source_file(name: str, workspace: Path, module: Path) -> str | None:
    for kind in ("main", "test"):
        candidate = module / "src" / kind / "java" / name
        if not candidate.is_file():
            continue
        try:
            relative = candidate.resolve().relative_to(workspace.resolve()).as_posix()
        except ValueError:
            return None
        return relative if is_safe_relative_path(relative) else None
    return None


def read_jacoco_lines(path: Path) -> dict[str, dict[int, tuple[int, int]]]:
    """Package-relative source → line → (missed instructions, covered instructions).

    Keep package paths here: callers doing change-impact analysis must resolve
    them against each module, rather than guess when two modules share a name.
    """

    root = read_xml(path, "report")
    result: dict[str, dict[int, tuple[int, int]]] = {}
    for package in root.findall("package"):
        for source in package.findall("sourcefile"):
            name = "/".join(
                part for part in (package.get("name", ""), source.get("name", "")) if part
            )
            if not is_safe_relative_path(name):
                raise ReportError("a JaCoCo source file path is unsafe")
            result[name] = {
                _number(line, "nr"): (_number(line, "mi"), _number(line, "ci"))
                for line in source.findall("line")
            }
    return result
