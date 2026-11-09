"""The gate template.

A gate does two things and nothing else: look at the change, and return a
verdict with the evidence behind it. Routing, retries, invalidation and
checkpoints belong to the runner — a gate that decides where the run goes next
would put control flow in four places instead of one.

The template also enforces the rule that separates a verification layer from a
demo: **a tool that could not run never counts as a pass.** With
``fail_closed`` on (the default), an unavailable scanner turns a PASS into a
BLOCK with an explicit finding, instead of being quietly noted and waved
through.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ..contracts import Finding, GateName, GateReport, ToolReport, Verdict, WorkspaceKind
from ..diffs import markdown_code_block
from ..observability import span

if TYPE_CHECKING:
    from ..runner import RuntimeServices
    from ..state import RunState

#: How much of one tool's reason the fail-closed finding carries. The reasons
#: the adapters write lead with the cause and fit well inside it; what runs
#: past it is a tool's own log, which the evidence pack keeps whole.
_REASON_LIMIT = 1200


def unavailable_reasons(tools: Sequence[ToolReport]) -> list[tuple[list[str], str]]:
    """Why each tool that never ran did not, grouped by what it said.

    The adapter writes the reason into the tool's report, and a reader who is
    told only "these checks did not run" has to open the JSON for the one
    sentence that says what to do. Grouped because tools stopped by one cause
    -- a Maven that could not start -- each say the same thing, and a person
    needs it once. A tool that said nothing groups under ``""``.
    """

    groups: dict[str, list[str]] = {}
    for tool in tools:
        if tool.available:
            continue
        reason = next(
            (text.strip() for text in (*tool.findings, tool.stderr, tool.stdout) if text.strip()),
            "",
        )
        groups.setdefault(reason, []).append(tool.name)
    return [(names, reason) for reason, names in groups.items()]


class Gate(ABC):
    #: Gate identity; also the key used for reports, attempts and timing.
    name: GateName

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services

    @abstractmethod
    def evaluate(self, state: RunState) -> GateReport:
        """Judge the change and return a report. Never mutate ``state``."""

        raise NotImplementedError

    def _workspace(self, state: RunState) -> Path:
        """Where this gate's tools run.

        ``workspace_path`` is set by whoever materialised the change (a forge
        adapter, or the local-diff entry point). Falling back to the deployment
        root is what makes a local run work against the tree it is already in.
        """

        if state.workspace_path:
            return Path(state.workspace_path)
        return self.services.workspace_root

    def _workspace_kind(self, state: RunState) -> WorkspaceKind:
        """Which tree the gate verified, without naming the host it sits on.

        The absolute path is a fact about the operator's machine, and an
        evidence pack handed to an auditor is the wrong place for one: the
        auditor cannot check it, and it dates the pack to one checkout on one
        disk. The single bit a reader actually needs from it is whether the
        change was verified in a tree materialised for this run, or in the
        deployment's own working copy -- the second means the verdict covers
        whatever state that copy happened to be in. The path itself stays in
        the ledger, where ``honest-ci inspect <run>`` still answers "which
        directory" for the operator who owns the machine.
        """

        # Resolved on both sides: a local run records the workspace through
        # ``Path.resolve()``, so on a platform where the root is reached via a
        # symlink (macOS ``/var`` -> ``/private/var``) the two spellings of one
        # directory would otherwise read as two different trees.
        if self._workspace(state).resolve() == self.services.workspace_root.resolve():
            return "deployment-root"
        return "checkout"

    def run(self, state: RunState) -> GateReport:
        attempt = state.attempts.get(self.name, 0) + 1
        start = time.perf_counter()
        with span(f"gate.{self.name}", run_id=state.run_id, attempt=attempt):
            report = self.evaluate(state)
            report = self._enforce_fail_closed(report)
            if report.trace_id is None:
                report.trace_id = state.trace_id or state.run_id
            report.duration_s = round(time.perf_counter() - start, 3)
            return report

    def _enforce_fail_closed(self, report: GateReport) -> GateReport:
        """Turn "the check never ran" into a block, not into a pass."""

        if not self.services.settings.fail_closed:
            return report
        if report.verdict not in {Verdict.PASS, Verdict.SKIPPED}:
            return report
        missing = [tool.name for tool in report.tool_reports if not tool.available]
        if not missing:
            return report
        detail = (
            "These checks did not run, so this change has no evidence behind it: "
            + ", ".join(sorted(missing))
            + ". Fix the toolchain, or turn off HONEST_CI_FAIL_CLOSED deliberately."
        )
        for names, reason in unavailable_reasons(report.tool_reports):
            if not reason:
                continue
            clipped = reason if len(reason) <= _REASON_LIMIT else reason[:_REASON_LIMIT] + " …"
            # As code, because the reason may be the tool's own output, and
            # that echoes whatever the change under verification gave it.
            detail += (
                f"\n\nWhy {', '.join(sorted(names))} did not run:\n" + markdown_code_block(clipped)
            )
        finding = Finding(
            id=f"{self.name}.unavailable",
            gate=self.name,
            severity="high",
            title="Required check tools unavailable",
            detail=detail,
            decided_by="deterministic",
        )
        return report.model_copy(
            update={
                "verdict": Verdict.BLOCKED,
                "findings": [*report.findings, finding],
                "summary": report.summary or "Required checks were unavailable.",
            }
        )
