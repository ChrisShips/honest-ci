"""Which toolchain judges a change: the scanner registry and the language router.

A team is rarely one language. The same deployment receives merge
requests from Java services and from Python tooling, and the gate semantics --
fail closed, judge only added lines, block on a suite that never ran -- are the
same for both. What differs is the toolchain: which scanners exist, how the
suite is run, where coverage comes from. That difference lives here, in three
parts:

* **The registry** (``SCANNER_ADAPTERS``) names every toolchain this build
  ships, keyed by language. It is also the whole list a ``tool`` rule may bind
  to: ``supported_tools()`` is the union of what the registered adapters
  declare, and nothing a configuration file says can add to it.
* **Detection** reads the build markers a repository already has -- ``pom.xml``
  says Maven, ``pyproject.toml`` says Python -- so a deployment serving several
  repositories needs no per-project switch. ``HONEST_CI_LANGUAGE`` forces one
  toolchain when the markers are ambiguous or absent.
* **The router** is itself a ``ScannerAdapter``, so the gates keep talking to
  one port. Tests and coverage go to the workspace's toolchain; each static
  tool goes to a toolchain that owns it, so a Java rule and a Python rule in
  one rule library each reach a scanner that can run them.

What the router never does is guess quietly. A workspace nobody can classify
falls back to the first registered toolchain and says so through
``toolchain()``, and a tool no toolchain knows is reported unavailable -- which
fail-closed turns into a block -- rather than silently skipped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..contracts import ChangeRequest, CoverageResult, TestResults, ToolReport
from ..settings import Settings
from .base import ScannerAdapter
from .java_scanner import JavaScannerAdapter
from .python_scanner import PythonScannerAdapter

#: The ``HONEST_CI_LANGUAGE`` value that means "read the workspace's markers".
LANGUAGE_AUTO = "auto"

#: Language → the toolchain that serves it. Order matters twice: it is the
#: order auto-detection tries markers in, and the first entry is the fallback
#: for a workspace that carries no marker at all.
SCANNER_ADAPTERS: dict[str, type[ScannerAdapter]] = {
    "python": PythonScannerAdapter,
    "java": JavaScannerAdapter,
}


class ScannerConfigError(ValueError):
    """The deployment names a toolchain this build does not have."""


@dataclass(frozen=True)
class ToolchainChoice:
    """Which toolchain judges a workspace, and how that was decided."""

    language: str
    adapter: ScannerAdapter
    #: ``configured`` (``HONEST_CI_LANGUAGE``), ``detected`` (a marker was found)
    #: or ``default`` (no marker: the first registered toolchain, and the
    #: evidence says so).
    basis: str
    reason: str

    @property
    def description(self) -> str:
        return f"{self.language} ({self.reason})"


def supported_tools(
    adapters: Mapping[str, type[ScannerAdapter]] | None = None,
) -> tuple[str, ...]:
    """Every static tool a rule may bind to, in registry order, without repeats."""

    registry = SCANNER_ADAPTERS if adapters is None else adapters
    tools: list[str] = []
    for adapter_cls in registry.values():
        for tool in adapter_cls.supported_tools:
            if tool not in tools:
                tools.append(tool)
    return tuple(tools)


def tool_languages(
    tool: str,
    adapters: Mapping[str, type[ScannerAdapter]] | None = None,
) -> tuple[str, ...]:
    """The toolchains that can run ``tool``, in registry order."""

    registry = SCANNER_ADAPTERS if adapters is None else adapters
    return tuple(
        language for language, adapter_cls in registry.items() if tool in adapter_cls.supported_tools
    )


def detect_languages(
    workspace: Path,
    adapters: Mapping[str, type[ScannerAdapter]] | None = None,
) -> list[tuple[str, str]]:
    """``(language, marker)`` for every registered toolchain whose marker is present.

    Registry order, first marker per toolchain. More than one entry means the
    repository carries several build systems and the router will take the
    first -- which is what ``HONEST_CI_LANGUAGE`` exists to override.
    """

    registry = SCANNER_ADAPTERS if adapters is None else adapters
    return _detect(workspace, {name: cls.markers for name, cls in registry.items()})


def build_scanner_adapter(settings: Settings, artifacts_path: Path) -> ScannerAdapter:
    """The router over every registered toolchain, checked against the settings.

    Raises ``ScannerConfigError`` for a ``HONEST_CI_LANGUAGE`` nobody implemented,
    at build time rather than on the first merge request: a deployment that
    cannot run its configured toolchain should refuse to start, not block every
    change with "tool not installed".
    """

    return LanguageRouter(settings, artifacts_path)


class LanguageRouter(ScannerAdapter):
    """One adapter per language, chosen per workspace, behind the one port."""

    language = LANGUAGE_AUTO

    def __init__(
        self,
        settings: Settings,
        artifacts_path: Path,
        adapters: Mapping[str, ScannerAdapter] | None = None,
    ) -> None:
        self.settings = settings
        if adapters is None:
            adapters = {
                name: adapter_cls(settings, artifacts_path)  # type: ignore[call-arg]
                for name, adapter_cls in SCANNER_ADAPTERS.items()
            }
        if not adapters:
            raise ScannerConfigError("No scanner toolchain is registered in this build.")
        self.adapters: dict[str, ScannerAdapter] = dict(adapters)
        self.supported_tools = tuple(
            dict.fromkeys(
                tool for adapter in self.adapters.values() for tool in adapter.supported_tools
            )
        )
        self.markers = tuple(
            dict.fromkeys(marker for adapter in self.adapters.values() for marker in adapter.markers)
        )
        forced = settings.language.strip().lower()
        if forced != LANGUAGE_AUTO and forced not in self.adapters:
            raise ScannerConfigError(
                f"HONEST_CI_LANGUAGE={settings.language!r} names a toolchain this build does not "
                f"have. Registered: {', '.join(self.adapters)}; or {LANGUAGE_AUTO} to read the "
                "workspace's build markers."
            )
        self._forced = None if forced == LANGUAGE_AUTO else forced

    # ── choosing ─────────────────────────────────────────────────────────

    def choose(self, workspace: Path) -> ToolchainChoice:
        """The toolchain for ``workspace``, with the reason the evidence records."""

        if self._forced is not None:
            return ToolchainChoice(
                self._forced,
                self.adapters[self._forced],
                "configured",
                f"HONEST_CI_LANGUAGE={self._forced}",
            )
        matches = _detect(
            workspace, {name: adapter.markers for name, adapter in self.adapters.items()}
        )
        if matches:
            language, marker = matches[0]
            reason = f"detected from {marker}"
            if len(matches) > 1:
                others = ", ".join(f"{name} ({found})" for name, found in matches[1:])
                reason += (
                    f"; the workspace also has {others}, and {language} comes first in "
                    "registry order; set HONEST_CI_LANGUAGE to use another"
                )
            return ToolchainChoice(language, self.adapters[language], "detected", reason)
        language = next(iter(self.adapters))
        markers = ", ".join(self.markers) or "none"
        return ToolchainChoice(
            language,
            self.adapters[language],
            "default",
            f"default: no language marker at the workspace root ({markers}); "
            "set HONEST_CI_LANGUAGE to choose one",
        )

    def toolchain(self, workspace: Path) -> str:
        return self.choose(workspace).description

    def language_at(self, workspace: Path) -> str:
        return self.choose(workspace).language

    def image(self, workspace: Path) -> str | None:
        return self.choose(workspace).adapter.image(workspace)

    def unselectable(self, workspace: Path) -> str | None:
        return self.choose(workspace).adapter.unselectable(workspace)

    # ── the port ─────────────────────────────────────────────────────────

    def run_static_checks(
        self,
        workspace: Path,
        change: ChangeRequest,
        tools: Sequence[str] | None = None,
        scopes: Mapping[str, Sequence[str]] | None = None,
    ) -> list[ToolReport]:
        """Send each requested tool to a toolchain that owns it; one report per tool.

        The workspace's own toolchain is preferred for a tool several of them
        declare (semgrep reads more than one language), so the report comes
        from the image and interpreter that fit the repository. ``scopes``
        travels with each tool to its owner unchanged: what a rule asked about
        does not depend on which toolchain answers.
        """

        requested = list(self.supported_tools) if tools is None else list(dict.fromkeys(tools))
        if not requested:
            return []
        primary = self.choose(workspace).language
        by_owner: dict[str, list[str]] = {}
        reports: dict[str, ToolReport] = {}
        for tool in requested:
            owner = self._owner(tool, primary)
            if owner is None:
                reports[tool] = _unknown_tool(tool, self.adapters)
            else:
                by_owner.setdefault(owner, []).append(tool)
        for language, names in by_owner.items():
            adapter = self.adapters[language]
            for report in adapter.run_static_checks(
                workspace, change, tools=names, scopes=scopes
            ):
                # An adapter answers once per tool it was asked about; an
                # answer for a tool nobody asked about is not evidence here.
                if report.name in names:
                    reports.setdefault(report.name, report)
        return [reports.get(tool) or _silent_tool(tool, self._owner(tool, primary)) for tool in requested]

    def run_tests(self, workspace: Path, selection: list[str] | None = None) -> TestResults:
        return self.choose(workspace).adapter.run_tests(workspace, selection)

    def measure_coverage(
        self,
        workspace: Path,
        lines: Mapping[str, set[int]] | None = None,
    ) -> CoverageResult:
        return self.choose(workspace).adapter.measure_coverage(workspace, lines)

    # ── internals ────────────────────────────────────────────────────────

    def _owner(self, tool: str, primary: str) -> str | None:
        if tool in self.adapters[primary].supported_tools:
            return primary
        for language, adapter in self.adapters.items():
            if tool in adapter.supported_tools:
                return language
        return None


def _detect(workspace: Path, markers: Mapping[str, Sequence[str]]) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for language, names in markers.items():
        for marker in names:
            if (workspace / marker).is_file():
                matches.append((language, marker))
                break
    return matches


def _unknown_tool(tool: str, adapters: Mapping[str, ScannerAdapter]) -> ToolReport:
    known = "; ".join(
        f"{language}: {', '.join(adapter.supported_tools) or '-'}"
        for language, adapter in adapters.items()
    )
    reason = f"No registered toolchain runs {tool!r}. Registered: {known}."
    return ToolReport(
        name=tool,
        command=f"{tool} (not executed)",
        available=False,
        passed=False,
        stderr=reason,
        findings=[reason],
    )


def _silent_tool(tool: str, owner: str | None) -> ToolReport:
    reason = f"The {owner or 'responsible'} toolchain returned no report for {tool!r}."
    return ToolReport(
        name=tool,
        command=f"{tool} (no report)",
        available=False,
        passed=False,
        stderr=reason,
        findings=[reason],
    )


__all__ = [
    "LANGUAGE_AUTO",
    "SCANNER_ADAPTERS",
    "LanguageRouter",
    "ScannerConfigError",
    "ToolchainChoice",
    "build_scanner_adapter",
    "detect_languages",
    "supported_tools",
    "tool_languages",
]
