"""The trust boundary between deterministic gates and model judgment.

Nothing a model returns is believed on its own. Two guards apply:

* **Citation.** A model may only talk about something the deterministic side
  already put on the table. For an explanation that means the gate's own
  evidence items, which carry stable ids; for a judgment-class rule it means
  the rule ids in the team's own rule library. A response that cites
  anything else is discarded whole. Review comments the platform carried are
  sent as *context* and are deliberately not citable: they are somebody's
  prose, not evidence this side produced, and a model allowed to ground a
  finding in one could pass an invention off as a colleague's objection.
* **Location.** A review finding must point at a line the change actually
  added. A comment on code nobody touched is not a review finding, it is noise.

Both guards are whole-response: one ungrounded finding discards the report
rather than being dropped quietly, because a model that invented one citation
has not earned the benefit of the doubt on the others, and a partially kept
report is one nobody can describe in an evidence pack.

Model failures here are non-fatal by design: the deterministic verdict and its
findings stand, and the run continues without the advisory text. What a caller
must not do is read "nothing came back" as "nothing was found" — that is why
the judgment entry point returns ``None`` for "not judged" and an empty
``RuleJudgment`` only for "judged, and clean".
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from .contracts import (
    BlockBrief,
    ChangeRequest,
    EvidencePack,
    Finding,
    GateName,
    GateReport,
    ModelFinding,
    ModelReviewReport,
    PlatformReview,
    ProposedTests,
    ToolReport,
)
from .diffs import (
    added_lines,
    changed_line_locations,
    is_safe_relative_path,
    markdown_path,
    markdown_prose,
)
from .impact import ChangedSymbol
from .redaction import redact_obj, redact_text
from .rules import Rule

if TYPE_CHECKING:
    from .adapters.base import LLMAdapter

_MAX_EVIDENCE_ITEMS = 20
_MAX_DETAIL_CHARS = 1200
#: Added lines handed to a model as the inventory of what it may cite. Bounded
#: for the same reason the diff is: one 4000-line refactor must not decide how
#: much every other review costs.
_MAX_JUDGMENT_LOCATIONS = 400
_MAX_LOCATION_CHARS = 240
#: Platform review comments handed to a model as context. Bounded for the same
#: reason the diff and the locations are, and smaller because a comment thread
#: is prose: it is background, not the thing being judged.
_MAX_JUDGMENT_REVIEWS = 40
_MAX_REVIEW_CHARS = 600
#: Where each toolchain's generated cases may be written, and nowhere else. A
#: proposal naming any other path is discarded whole: a model that can choose
#: where a file lands can overwrite the suite that is judging it.
GENERATED_TEST_DIRS = {
    "python": "tests/honest_ci_generated",
    "java": "src/test/java/honest_ci/generated",
}
#: Generated cases per change, and bytes per case. A regression case for one
#: changed symbol is a few dozen lines; anything larger is not what was asked
#: for, and running it would spend the suite's budget on something nobody
#: reviewed.
MAX_GENERATED_TESTS = 5
MAX_GENERATED_TEST_CHARS = 20_000
#: Changed symbols offered as generation targets. Same bound as the adapter's,
#: kept here too because this side is the one that has to hold when an adapter
#: is replaced.
_MAX_TESTGEN_SYMBOLS = 40
_LOGGER = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)

#: Transport-level failures worth one more attempt, as opposed to a real block.
_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "rate limit",
    "429",
    "connection",
    "temporar",
    "service unavailable",
    "502",
    "503",
    "504",
)


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    detail: str

    def as_payload(self) -> dict[str, str]:
        return {"id": self.id, "detail": self.detail}


@dataclass(frozen=True)
class RuleJudgment:
    """What a model made of the judgment rules, after the trust boundary.

    ``findings`` is what a human is asked to adjudicate. ``below_threshold`` is
    the rest: grounded doubts the model was not confident enough about, which
    belong in "What this run could not verify" rather than among the findings
    a person has to rule on. ``questions`` is what the model could not
    resolve at all, and it is the most useful thing it produces — "I am not
    sure" written down is a review agenda, a wrong confident answer is not.
    """

    summary: str = ""
    findings: tuple[Finding, ...] = ()
    below_threshold: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()


# ── evidence extraction ──────────────────────────────────────────────────────


def gate_evidence_items(report: GateReport) -> list[EvidenceItem]:
    """Bounded, citable evidence for everything a gate decided."""

    items: list[EvidenceItem] = []
    for index, finding in enumerate(report.findings, start=1):
        identifier = finding.id or f"{report.gate}.finding.{index}"
        location = ""
        if finding.file:
            # A test-integrity finding on a removed line counts that line on
            # the old side of the diff; unmarked, the model would read it as a
            # line of the new file. A whole-file finding has no line to give.
            line = f":{finding.line_start}" if finding.line_start else ""
            side = " (base)" if finding.side == "base" else ""
            location = f" ({finding.file}{line}{side})"
        rule = f"[{finding.rule_id}] " if finding.rule_id else ""
        items.append(
            EvidenceItem(
                id=_safe_id(identifier),
                detail=_safe_detail(f"{rule}{finding.title}{location} — {finding.detail}".strip()),
            )
        )
    for index, tool in enumerate(report.tool_reports, start=1):
        if tool.passed:
            continue
        items.append(_tool_evidence(report.gate, index, tool))
    if report.test_results is not None:
        results = report.test_results
        if results.failed or results.errors:
            for index, failure in enumerate(results.failures[:10], start=1):
                items.append(
                    EvidenceItem(
                        id=f"{report.gate}.test.{index}",
                        detail=_safe_detail(f"Failing test: {failure}"),
                    )
                )
    return items[:_MAX_EVIDENCE_ITEMS] or [
        EvidenceItem(
            id=f"{report.gate}.gate",
            detail="The gate blocked this change without a detailed finding.",
        )
    ]


def deterministic_feedback(report: GateReport) -> str:
    """Format the authoritative gate verdict without involving a model."""

    return _format_deterministic_feedback(report.gate, gate_evidence_items(report))


def guarded_block_brief(
    report: GateReport,
    attempt: int,
    llm: LLMAdapter,
) -> tuple[str, BlockBrief | None]:
    """Explain a block, keeping the gate authoritative.

    Returns the deterministic feedback (always) and an advisory brief (only when
    the model cited this gate's own evidence).
    """

    items = gate_evidence_items(report)
    feedback = _format_deterministic_feedback(report.gate, items)
    source_fingerprint = _source_fingerprint(report.gate, items)
    brief = _guarded_model_output(
        label=f"{report.gate} block brief",
        generate=lambda: llm.explain_block(
            gate=report.gate,
            attempt=attempt,
            source_fingerprint=source_fingerprint,
            summary=report.summary,
            deterministic_feedback=feedback,
            evidence_items=[item.as_payload() for item in items],
        ),
        validate=lambda candidate: validate_block_brief(
            candidate,
            report.gate,
            attempt,
            source_fingerprint,
            items,
        ),
    )
    if brief is None:
        return feedback, None
    return _format_advisory_feedback(feedback, brief), brief


def guarded_review_report(
    evidence: EvidencePack,
    llm: LLMAdapter,
) -> ModelReviewReport | None:
    """Produce a grounded reading aid for a human reviewer, or nothing at all."""

    return _guarded_model_output(
        label="review report",
        generate=lambda: llm.review_change(evidence),
        validate=lambda candidate: validate_review_report(candidate, evidence),
    )


def guarded_rule_judgment(
    change: ChangeRequest,
    rules: Sequence[Rule],
    fingerprint: str,
    llm: LLMAdapter,
    *,
    default_min_confidence: float,
    gate: GateName = "review",
) -> RuleJudgment | None:
    """Put the judgment-class rules to a model, with code still in charge.

    ``None`` means those rules were **not judged**: the call failed, or what
    came back cited a rule this library does not contain or a line this change
    did not add. The caller has to treat that as unreviewed, which is why it is
    distinct from a ``RuleJudgment`` holding no findings — that one means the
    model did read the change against the rules and had nothing to raise.

    Three things the model is not allowed to decide, and where each is decided
    instead: the severity of a finding comes from the rule file, the confidence
    a finding must clear comes from the rule file or this deployment, and what
    a kept finding does to the verdict is the gate's call.
    """

    if not rules:
        return RuleJudgment()
    by_id = {rule.id: rule for rule in rules}
    report = _guarded_model_output(
        label="rule judgment",
        generate=lambda: llm.judge_rules(
            change=change,
            fingerprint=fingerprint,
            rules=[_rule_payload(rule) for rule in rules],
            locations=_location_payload(change, rules),
            reviews=_review_payload(change),
        ),
        validate=lambda candidate: validate_rule_judgment(candidate, change, fingerprint, by_id),
    )
    if report is None:
        return None

    findings: list[Finding] = []
    below: list[str] = []
    for index, item in enumerate(report.findings, start=1):
        rule = by_id[str(item.rule_id)]
        threshold = (
            rule.min_confidence if rule.min_confidence is not None else default_min_confidence
        )
        if item.confidence < threshold:
            # Recorded, but not as a finding. A finding is something a person
            # has to rule on, and guesses the model itself is not sure of would
            # bury the ones worth ruling on.
            below.append(
                # A file the change added lines to, under the name it chose;
                # this line is Markdown in the merge request note.
                f"[{rule.id}] {rule.title} · {markdown_path(item.file)}:{item.line_start}"
                f" (confidence {item.confidence:.2f} < {threshold:.2f})"
            )
            continue
        findings.append(_judgment_finding(rule, item, gate, index))
    return RuleJudgment(
        summary=report.summary.strip(),
        findings=tuple(findings),
        below_threshold=tuple(below),
        questions=tuple(item for item in report.review_questions if item.strip()),
    )


def guarded_test_proposal(
    change: ChangeRequest,
    symbols: Sequence[ChangedSymbol],
    fingerprint: str,
    llm: LLMAdapter,
    *,
    language: str,
    workspace: Path,
) -> ProposedTests | None:
    """Ask a model for regression cases, and let none of them become a file lightly.

    ``None`` means nothing usable came back: the call failed, or what came back
    did not survive the checks below. An empty ``ProposedTests`` is different
    and is returned as itself -- the model read the change and proposed
    nothing, which is an answer.

    Everything a proposal contains is untrusted input that is about to be
    *written into the tree the suite runs in*, which makes this the strictest
    guard in the file. Five checks, and any failure discards the whole
    proposal rather than the offending case:

    * the path sits inside this toolchain's generated-test directory,
    * it does not name a file that already exists -- a generated case must
      never be able to replace the suite that is judging the change,
    * the case is not larger than a regression case has any reason to be,
    * every ``targets`` entry is a symbol this change actually touched,
    * the file parses, so a syntax error is caught here rather than arriving
      as a mysterious test-suite error later.

    Whole-proposal because a model that put a file outside the sandbox
    directory, or aimed a case at a symbol nobody touched, has not earned the
    benefit of the doubt on the rest -- the same rule the citation guard
    applies to findings.
    """

    directory = GENERATED_TEST_DIRS.get(language)
    if directory is None or not symbols:
        return None
    wanted = {symbol.name for symbol in symbols} | {symbol.simple_name for symbol in symbols}
    return _guarded_model_output(
        label="test proposal",
        generate=lambda: llm.propose_tests(
            change=change,
            symbols=[_symbol_payload(symbol) for symbol in symbols[:_MAX_TESTGEN_SYMBOLS]],
            fingerprint=fingerprint,
            language=language,
            test_directory=directory,
        ),
        validate=lambda candidate: validate_test_proposal(
            candidate,
            fingerprint,
            wanted,
            language=language,
            workspace=workspace,
        ),
    )


# ── validation ───────────────────────────────────────────────────────────────


def validate_block_brief(
    brief: BlockBrief,
    gate: GateName,
    attempt: int,
    source_fingerprint: str,
    items: list[EvidenceItem],
) -> bool:
    allowed_ids = {item.id for item in items}
    return bool(
        brief.gate == gate
        and brief.attempt == attempt
        and brief.source_fingerprint == source_fingerprint
        and brief.summary.strip()
        and any(action.strip() for action in brief.recommended_actions)
        and any(evidence_id.strip() for evidence_id in brief.cited_evidence_ids)
        and set(brief.cited_evidence_ids).issubset(allowed_ids)
    )


def validate_review_report(report: ModelReviewReport, evidence: EvidencePack) -> bool:
    """Accept only reports bound to this evidence and grounded in changed lines."""

    if report.evidence_fingerprint != evidence.fingerprint:
        return False
    if not report.summary.strip() or not report.change_summary.strip():
        return False
    return _lands_on_added_lines(report.findings, evidence.change)


def validate_rule_judgment(
    report: ModelReviewReport,
    change: ChangeRequest,
    fingerprint: str,
    rules: dict[str, Rule],
) -> bool:
    """Accept only judgments bound to this change and to rules that exist.

    The rule id is the citation here: a judgment-class finding that names a
    rule the team never wrote is exactly the failure the citation guard
    exists for, and it is worse than a hallucinated evidence id — it would
    reach a developer as "your J-07 does not allow this" about a rule nobody
    can look up. ``change_summary`` is not required: this call asks the model to judge
    named rules, not to describe the change.
    """

    if report.evidence_fingerprint != fingerprint:
        return False
    if not report.summary.strip():
        return False
    if any(finding.rule_id not in rules for finding in report.findings):
        return False
    return _lands_on_added_lines(report.findings, change)


def validate_test_proposal(
    proposal: ProposedTests,
    fingerprint: str,
    targets: set[str],
    *,
    language: str,
    workspace: Path,
) -> bool:
    """Accept only cases that may safely become files in this workspace."""

    if proposal.evidence_fingerprint != fingerprint:
        return False
    if len(proposal.tests) > MAX_GENERATED_TESTS:
        return False
    directory = GENERATED_TEST_DIRS.get(language)
    if directory is None:
        return False
    seen: set[str] = set()
    for test in proposal.tests:
        path = test.path.replace("\\", "/").strip()
        if path in seen or not _is_generated_test_path(path, directory, language):
            return False
        seen.add(path)
        if (workspace / path).exists():
            # Not "overwrite it": a generated case that can replace an existing
            # file can replace the test that would have caught the change.
            return False
        if not test.content.strip() or len(test.content) > MAX_GENERATED_TEST_CHARS:
            return False
        if not test.targets or not set(test.targets).issubset(targets):
            return False
        if not _parses(test.content, path, language):
            return False
    return True


def _is_generated_test_path(path: str, directory: str, language: str) -> bool:
    if not is_safe_relative_path(path) or not path.startswith(f"{directory}/"):
        return False
    name = path.rsplit("/", maxsplit=1)[-1]
    if language == "python":
        return name.startswith("test_") and name.endswith(".py")
    return name.endswith(("Test.java", "Tests.java", "IT.java"))


def _parses(content: str, path: str, language: str) -> bool:
    """Whether the file is syntactically what it claims to be.

    Python gets the interpreter's own parser. Java gets brackets and the class
    name, because a Java parser is a dependency this product does not want in a
    private deployment (AGENTS.md) -- and the two failures that actually reach
    a sandbox as an unexplainable error are an unbalanced file and a class the
    compiler will not accept under that file name.
    """

    if language == "python":
        try:
            ast.parse(content)
        except (SyntaxError, ValueError):
            return False
        return True
    expected = path.rsplit("/", maxsplit=1)[-1].removesuffix(".java")
    if not re.search(rf"\b(?:class|interface|enum|record)\s+{re.escape(expected)}\b", content):
        return False
    depth = 0
    for character in content:
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def redact_model(model: ModelT) -> ModelT:
    """Apply persisted-data redaction and revalidate the same model type."""

    payload = redact_obj(model.model_dump(mode="json"))
    return type(model).model_validate(payload)


def is_transient_failure(message: str) -> bool:
    """Classify transport/service failures before retrying a gate."""

    lowered = message.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


# ── internals ────────────────────────────────────────────────────────────────


def _lands_on_added_lines(findings: Sequence[ModelFinding], change: ChangeRequest) -> bool:
    """Whether every proposed finding points at a line this change added."""

    changed_files = {item.path for item in change.files_changed} or set(
        changed_line_locations(change.diff)
    )
    changed_lines = changed_line_locations(change.diff)
    for finding in findings:
        if finding.file not in changed_files or finding.line_end < finding.line_start:
            return False
        available_lines = changed_lines.get(finding.file, set())
        if not available_lines:
            return False
        if any(
            line not in available_lines
            for line in range(finding.line_start, finding.line_end + 1)
        ):
            return False
    return True


def _rule_payload(rule: Rule) -> dict[str, str]:
    """One judgment rule as the model sees it: an id, a name, and a criterion."""

    return {
        "rule_id": rule.id,
        "title": rule.title,
        "criterion": _safe_detail(rule.detail),
        "paths": ", ".join(rule.paths),
    }


def _symbol_payload(symbol: ChangedSymbol) -> dict[str, str]:
    """One changed symbol, as the model is allowed to see it."""

    return {
        "file": symbol.file,
        "symbol": symbol.name,
        "kind": symbol.kind,
        "line": str(symbol.line),
    }


def _location_payload(change: ChangeRequest, rules: Sequence[Rule]) -> list[dict[str, str]]:
    """The added lines in these rules' scope — the only ones a finding may cite.

    Sent even though the diff is sent too: naming the citable lines outright is
    what keeps a whole response from being discarded over a line number the
    model counted wrong, and a discarded response costs a real review.
    """

    locations: list[dict[str, str]] = []
    for path, lines in sorted(added_lines(change.diff).items()):
        if not any(rule.applies_to(path) for rule in rules):
            continue
        for line in lines:
            if len(locations) >= _MAX_JUDGMENT_LOCATIONS:
                return locations
            locations.append(
                {
                    "file": path,
                    "line": str(line.line),
                    "text": redact_text(line.text)[:_MAX_LOCATION_CHARS],
                }
            )
    return locations


def _review_payload(change: ChangeRequest) -> list[dict[str, str]]:
    """The comments already on this change, as context for the model.

    Context and nothing else. ``validate_rule_judgment`` still accepts only a
    rule id this library contains and a line this change added, so a comment
    can never become a citation: if it could, an invented claim would only have
    to be laundered through somebody else's sentence to reach a developer as
    "your J-07 does not allow this".

    Located comments come first and in file order, because those are the ones a
    model can line up against the diff it is reading; the rest follow by author.
    """

    located = sorted(
        (review for review in change.platform_reviews if review.path),
        key=lambda review: (review.path or "", review.line or 0),
    )
    loose = sorted(
        (review for review in change.platform_reviews if not review.path),
        key=lambda review: review.author,
    )
    return [
        {
            "author": review.author + (" (bot)" if review.is_bot else ""),
            "location": _review_location(review),
            "state": _review_state(review),
            "body": redact_text(review.body)[:_MAX_REVIEW_CHARS],
        }
        for review in (*located, *loose)[:_MAX_JUDGMENT_REVIEWS]
    ]


def _review_location(review: PlatformReview) -> str:
    if not review.path:
        return ""
    return f"{review.path}:{review.line}" if review.line else review.path


def _review_state(review: PlatformReview) -> str:
    if review.resolved is None:
        return "comment"
    return "resolved" if review.resolved else "unresolved"


def _judgment_finding(rule: Rule, item: ModelFinding, gate: GateName, index: int) -> Finding:
    """One accepted proposal, recorded under the rule that authorised it.

    Severity comes from the rule file rather than from ``item.severity``: a
    model that could raise its own finding's severity would be choosing how
    loudly to interrupt a team, which is the rule author's call.

    The model's reasoning and its question are Markdown wherever a finding is
    shown -- the evidence pack, the merge request note, an inline comment on
    the line -- and the model wrote them after reading the diff, so both are
    made inert here, once, before they are mixed with the rule's own text.
    """

    detail = rule.detail
    if item.detail.strip():
        detail = f"{detail}\nThe model's reasoning: {markdown_prose(item.detail.strip())}"
    question = item.suggested_question
    return Finding(
        id=f"{gate}.{rule.id}.model.{index}",
        gate=gate,
        rule_id=rule.id,
        severity=rule.severity,
        title=rule.title,
        detail=detail,
        file=item.file,
        line_start=item.line_start,
        line_end=item.line_end,
        cited_evidence_ids=[rule.id],
        decided_by="model",
        confidence=item.confidence,
        suggested_fix=rule.fix or (markdown_prose(question) if question else question),
    )


def _guarded_model_output(
    *,
    label: str,
    generate: Callable[[], ModelT],
    validate: Callable[[ModelT], bool],
) -> ModelT | None:
    """Keep generation, redaction and grounding in one fail-soft boundary."""

    try:
        candidate = redact_model(generate())
        return candidate if validate(candidate) else None
    except Exception as exc:  # noqa: BLE001 - advisory output must remain non-fatal.
        # The exception message is withheld because provider failures can carry
        # unredacted payload fragments. The exception class still makes a
        # programming regression visible instead of silently swallowing it.
        _LOGGER.warning("Discarded %s after %s.", label, type(exc).__name__)
        return None


def _tool_evidence(gate: str, index: int, tool: ToolReport) -> EvidenceItem:
    excerpts = list(tool.findings[:3])
    output = tool.stderr or tool.stdout
    if output:
        excerpts.append("\n".join(output.splitlines()[-12:]))
    detail = f"Tool {tool.name} {'failed' if tool.available else 'never ran'}"
    if tool.exit_code is not None:
        detail += f" (exit={tool.exit_code})"
    if excerpts:
        detail += ": " + " | ".join(excerpts)
    return EvidenceItem(f"{gate}.tool.{_safe_id(tool.name)}.{index}", _safe_detail(detail))


def _format_deterministic_feedback(gate: str, items: list[EvidenceItem]) -> str:
    lines = [
        f"GATE BLOCKED: {gate}",
        "The deterministic checks remain authoritative. Do not bypass rules, scanners, tests or review.",
        "EVIDENCE:",
    ]
    lines.extend(f"- [{item.id}] {item.detail}" for item in items)
    lines.append("Address the cited issue(s), then push again to re-run the same gate.")
    return "\n".join(lines)


def _format_advisory_feedback(feedback: str, brief: BlockBrief) -> str:
    lines = [
        feedback,
        "",
        "MODEL BRIEF (advisory; it does not change the gate verdict):",
        brief.summary.strip(),
        "Recommended actions:",
    ]
    lines.extend(f"- {action}" for action in brief.recommended_actions[:8])
    lines.append("Cited evidence: " + ", ".join(brief.cited_evidence_ids))
    return "\n".join(lines)


def _source_fingerprint(gate: str, items: list[EvidenceItem]) -> str:
    material = {"gate": gate, "evidence": [item.as_payload() for item in items]}
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    return normalized or "item"


def _safe_detail(value: str) -> str:
    redacted = redact_text(value)
    if len(redacted) <= _MAX_DETAIL_CHARS:
        return redacted
    return redacted[:_MAX_DETAIL_CHARS] + "…"
