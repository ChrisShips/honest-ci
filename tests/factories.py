"""Shared fixtures: a change to verify, and gates that behave on command."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from honest_ci.adapters.base import ForgeAdapter, ForgeError, LLMAdapter, ScannerAdapter
from honest_ci.adapters.python_scanner import PythonScannerAdapter
from honest_ci.contracts import (
    ApprovalReading,
    BlockBrief,
    ChangedFile,
    ChangeRequest,
    CoverageResult,
    EvidencePack,
    Finding,
    GateName,
    GateReport,
    LLMCallRecord,
    ModelFinding,
    ModelReviewReport,
    PlatformApprovals,
    PlatformReview,
    ProposedTest,
    ProposedTests,
    TestResults,
    ToolReport,
    Verdict,
)
from honest_ci.gates.base import Gate
from honest_ci.grounding import guarded_rule_judgment
from honest_ci.license_check import Ownership
from honest_ci.report import MARKER_PREFIX
from honest_ci.rules import Rule
from honest_ci.settings import Settings
from honest_ci.state import RunState, RunStatus
from honest_ci.tools import ToolRunner

#: CJK ideographs, their punctuation and the full-width forms: a Chinese word
#: left in what honest-ci prints or writes brings some of them along.
CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")


def cjk_lines(text: str) -> list[str]:
    """The lines of ``text`` that hold a CJK character, so a failure shows them."""

    return [line for line in text.splitlines() if CJK.search(line)]


DIFF = """diff --git a/src/app/refund.py b/src/app/refund.py
--- a/src/app/refund.py
+++ b/src/app/refund.py
@@ -10,2 +10,5 @@ def split_refund(order, amount):
     total = 0
+    share = float(amount) / len(order.items)
+    for item in order.items:
+        item.refund = share
     return total
"""

#: A change that edits the code and, beside it, deletes one test, skips another
#: and rewrites the expected value of a third to what the code now returns.
WEAKENED_TESTS_DIFF = """diff --git a/src/app/pricing.py b/src/app/pricing.py
--- a/src/app/pricing.py
+++ b/src/app/pricing.py
@@ -1,2 +1,2 @@
 def total(amount):
-    return amount * 2
+    return amount * 3
diff --git a/tests/test_pricing.py b/tests/test_pricing.py
--- a/tests/test_pricing.py
+++ b/tests/test_pricing.py
@@ -1,8 +1,7 @@
 import pytest
 from app.pricing import total
-def test_total_doubles():
-    assert total(2) == 4
+@pytest.mark.skip(reason="flaky")
 def test_total_of_zero():
     assert total(0) == 0
 def test_total_of_three():
-    assert total(3) == 6
+    assert total(3) == 9
"""


#: One character of each kind a file name may not hold, keyed by a test id:
#: the ASCII controls a terminal or a Markdown renderer acts on, a C1 control,
#: a bidi override and a line separator.
REFUSED_NAME_CHARACTERS = {
    "tab": "\t",
    "newline": "\n",
    "escape": "\x1b",
    "delete": "\x7f",
    "c1-csi": "\x9b",
    "right-to-left-override": "‮",
    "line-separator": " ",
}


def make_change(**overrides: object) -> ChangeRequest:
    payload: dict[str, object] = {
        "forge": "github",
        "repo": "team/orders",
        "number": 482,
        "title": "feat: 订单退款按明细拆分",
        "head_sha": "a" * 40,
        "base_sha": "b" * 40,
        "source_branch": "feat/refund-split",
        "target_branch": "main",
        "diff": DIFF,
        "files_changed": [
            ChangedFile(path="src/app/refund.py", change_type="modified", added_lines=3)
        ],
        "ai_generated_ratio": 0.71,
    }
    payload.update(overrides)
    return ChangeRequest.model_validate(payload)


def make_weakened_change(**overrides: object) -> ChangeRequest:
    """``WEAKENED_TESTS_DIFF`` as a change, with the file list a forge would send."""

    payload: dict[str, object] = {
        "diff": WEAKENED_TESTS_DIFF,
        "files_changed": [
            ChangedFile(path="src/app/pricing.py", added_lines=1, removed_lines=1),
            ChangedFile(path="tests/test_pricing.py", added_lines=2, removed_lines=3),
        ],
    }
    payload.update(overrides)
    return make_change(**payload)


#: Markdown that renders as bold text vouching for the change, an image that is
#: a tracking pixel, and a link. Protocol-relative, so a file name can hold it:
#: a colon is refused in a path.
_PAYLOAD = "**Approved by security** ![x](//example.invalid/p.png) [y](//example.invalid)"

#: The payload twice with a backtick between, so it renders wherever it lands:
#: written as it is, the first copy does; in a one-backtick span, the backtick
#: closes the span and the second copy does.
HOSTILE_TEXT = f"{_PAYLOAD} ` {_PAYLOAD}"

#: The same as a file name. Nothing in it is refused as a path -- no colon,
#: nothing unprintable -- so it reaches every place a file name can.
HOSTILE_NAME = f"src/{HOSTILE_TEXT}.py"

#: Lines a change adds that would render inside the bot's own comment if they
#: were quoted as they are. Beside the bare payload and the backtick between
#: two, the payload behind a lone carriage return, which CommonMark ends a line
#: at, and ahead of a ``token=`` that the renderers' last redaction pass would
#: read the closing fence into, with a value and without.
HOSTILE_LINES = (
    f'x = "{_PAYLOAD}"',
    f'x = "{HOSTILE_TEXT}"',
    f'x = "\r{_PAYLOAD}"',
    f'x = "{_PAYLOAD}"  # token=abc',
    f'x = "{_PAYLOAD}"  # token=',
)

HOSTILE_DIFF = (
    "--- a/src/app/refund.py\n+++ b/src/app/refund.py\n"
    f"@@ -1,0 +1,{len(HOSTILE_LINES)} @@\n" + "".join(f"+{line}\n" for line in HOSTILE_LINES)
)


#: Output a test run or a scanner could print, many lines of it, each written to
#: end a code block or render past its end: the fences that close one, a blank
#: line that ends a quote holding one, raw HTML (an image by another route), a
#: line ending in ``token=``, whose value the renderers' last redaction pass
#: once took from the next line, and an escape sequence and a lone carriage
#: return, which CommonMark ends a line at.
HOSTILE_OUTPUT = "\n".join(
    (
        "E   AssertionError",
        "```",
        _PAYLOAD,
        "~~~~~~~~",
        "",
        '<img alt="Approved by security" src="//example.invalid/p.png">',
        "token=",
        _PAYLOAD,
        f"\x1b[2J\r{_PAYLOAD}",
    )
)

#: Prose a model steered by the diff it read could write, in every form a
#: platform fetches, links or notifies from: the payload's image and link, a
#: reference definition, raw HTML, a ``<…>`` autolink, a bare URL, a ``www.``
#: address, an e-mail address, mentions and an entity that decodes to one, a
#: ``mermaid`` block, and a code block left open to run over whatever follows.
#: It opens with a code block, which a report writing it after a label on the
#: same line would no longer read as one. The ``token=`` ends a line before a
#: fence, which a table cell joins onto the same line; there the final
#: redaction pass would take the fence as its value and open the payload after
#: it.
HOSTILE_PROSE = "\n".join(
    (
        "```",
        _PAYLOAD,
        "```",
        HOSTILE_TEXT,
        "[ref]: //example.invalid",
        "",
        "[ref] <img src=//example.invalid/p.png> <https://example.invalid>",
        "See https://example.invalid/login, www.example.invalid or security@example.invalid.",
        "Ping @reviewer, @org/team and &#64;all.",
        "~~~mermaid",
        "graph TD; A-->B; click A href \"https://example.invalid\"",
        "~~~",
        "token=",
        "```",
        _PAYLOAD,
        "```",
        "```",
        _PAYLOAD,
    )
)


def markdown_tokens(markdown: str) -> list[Any]:
    """Every token a CommonMark renderer, with GFM tables, reads out of ``markdown``.

    Blocks and the inline tokens inside them: a code block, raw HTML and a
    paragraph are blocks, and a code span, a link and an image are inline.
    """

    markdown_it = pytest.importorskip("markdown_it")
    parser = markdown_it.MarkdownIt("commonmark").enable("table")
    blocks = parser.parse(markdown)
    return [*blocks, *(child for token in blocks for child in token.children or [])]


def markdown_outside_code(markdown: str) -> list[Any]:
    """The inline tokens a CommonMark renderer, with GFM tables, reads out of ``markdown``."""

    return [token for token in markdown_tokens(markdown) if not token.block]


#: Tokens whose content is shown as it is written.
_CODE_TOKENS = frozenset({"code_inline", "code_block", "fence"})


def rendered_as_markdown(markdown: str, text: str) -> list[Any]:
    """Tokens that carry ``text`` anywhere but in code, and every image or link.

    Empty is the property: whoever wrote ``text`` got none of it rendered. Raw
    HTML counts as carrying it too, since it is passed through to be rendered.
    """

    return [
        token
        for token in markdown_tokens(markdown)
        if token.type in {"image", "link_open"}
        # An inline token's content is its source; its children say what of it rendered.
        or (token.type not in _CODE_TOKENS | {"inline"} and text in token.content)
    ]


#: What GitHub and GitLab link or notify from in text: a URL by its scheme or
#: ``www.``, an e-mail address, and a mention -- an ``@`` that does not follow a
#: word character, as ``team/orders#482@aaaa`` in a report's own heading does.
_REACHING_RE = re.compile(
    r"(?i)://|www\.|mailto:|xmpp:"
    r"|(?<![a-z0-9_])@[a-z0-9_]|[a-z0-9._%+\-]@[a-z0-9\-]+\.[a-z0-9]"
)


def reaching_markdown(markdown: str) -> list[Any]:
    """Tokens of ``markdown`` a platform would fetch, link or notify from.

    Images, links and raw HTML, and text holding a URL, an address or a
    mention: the platforms find those in the rendered HTML, so text is not
    inert just because no Markdown link was parsed. This product's own markers
    are HTML comments by design. Empty is the property for model prose, which
    may format but must not reach out.
    """

    return [
        token
        for token in markdown_tokens(markdown)
        if token.type in {"image", "link_open"}
        or (
            token.type in {"html_block", "html_inline"}
            and not token.content.startswith(MARKER_PREFIX)
        )
        or (token.type not in _CODE_TOKENS | {"inline"} and _REACHING_RE.search(token.content))
    ]


#: A heading, with the section number the evidence pack puts in front of its titles.
_HEADING_RE = re.compile(r"^(#{1,6}) (?:\d+\. )?(.*)$")


def headings(markdown: str) -> list[str]:
    """The title of every heading in ``markdown``, without the pack's section numbers."""

    return [match.group(2) for line in markdown.splitlines() if (match := _HEADING_RE.match(line))]


def section(markdown: str, title: str) -> str:
    """The body under the first heading whose title starts with ``title``.

    It runs to the next heading at the same level or above, so a pack section
    carries its subsections. A test names a section by its title rather than
    its number, which moves whenever a section is added ahead of it.
    """

    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match is None or not match.group(2).startswith(title):
            continue
        level = len(match.group(1))
        body: list[str] = []
        for rest in lines[index + 1 :]:
            below = _HEADING_RE.match(rest)
            if below is not None and len(below.group(1)) <= level:
                break
            body.append(rest)
        return "\n".join(body).strip("\n")
    raise AssertionError(f"no heading starts with {title!r}")


def table_row(markdown: str, label: str) -> list[str]:
    """The other cells of the first table row whose first cell is ``label``.

    Split on the cell separator itself: a pipe inside a cell is escaped, so it
    never has a space on both sides.
    """

    for line in markdown.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
        if cells[0] == label:
            return cells[1:]
    raise AssertionError(f"no table row is labelled {label!r}")


def unverified(markdown: str) -> list[str]:
    """The items under "What this run could not verify", in the pack or in the comment."""

    body = section(markdown, "What this run could not verify")
    return [line.removeprefix("- ") for line in body.splitlines() if line.startswith("- ")]


def field(output: str, label: str) -> str:
    """The value on the terminal line ``label`` heads, as ``Label : value``.

    Found by its label rather than by the whole line, so a test names the fact
    it reads instead of how the words around it happen to be put.
    """

    for line in output.splitlines():
        name, colon, value = line.partition(":")
        if colon and name.strip() == label:
            return value.strip()
    raise AssertionError(f"no {label!r} line in:\n{output}")


def make_settings() -> Settings:
    """Settings that never shell out to docker and never call a provider."""

    return Settings(sandbox_mode="process", fail_closed=True)


def fill_repository(root: Path, coordinates: Sequence[str]) -> Path:
    """A local Maven repository holding ``coordinates`` (``dependency:get`` order), laid out as Maven does."""

    for coordinate in coordinates:
        group, artifact, version = coordinate.split(":")[:3]
        directory = root / group.replace(".", "/") / artifact / version
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{artifact}-{version}.jar").write_bytes(b"")
    return root


def recorded_commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every argv the adapter hands a tool, in order, without running any tool."""

    commands: list[list[str]] = []

    def record(
        self: ToolRunner,
        name: str,
        command: list[str],
        report_name: str | None = None,
        allow_network: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> ToolReport:
        commands.append(command)
        return ToolReport(name=name, command=" ".join(command))

    monkeypatch.setattr(ToolRunner, "module_available", lambda *args, **kwargs: True)
    monkeypatch.setattr(ToolRunner, "run", record)
    return commands


def scanner_arguments(
    change: ChangeRequest, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Every argument the Python scanners are handed for ``change``, none of them run."""

    commands = recorded_commands(monkeypatch)
    PythonScannerAdapter(make_settings(), workspace.parent / "artifacts").run_static_checks(
        workspace, change
    )
    return [argument for command in commands for argument in command]


class ScriptedGate(Gate):
    """A gate that returns a prepared verdict, or raises a prepared error."""

    def __init__(
        self,
        services: object,
        name: GateName,
        verdicts: list[Verdict] | None = None,
        error: Exception | None = None,
        error_times: int = 0,
        tool_reports: list[ToolReport] | None = None,
        findings: list[Finding] | None = None,
    ) -> None:
        super().__init__(services)  # type: ignore[arg-type]
        self.name = name
        self.verdicts = verdicts or [Verdict.PASS]
        self.error = error
        self.error_times = error_times
        self.tool_reports = tool_reports or []
        self.findings = findings or []
        self.calls = 0

    def evaluate(self, state: RunState) -> GateReport:
        self.calls += 1
        if self.error is not None and self.calls <= self.error_times:
            raise self.error
        index = min(self.calls - 1, len(self.verdicts) - 1)
        verdict = self.verdicts[index]
        return GateReport(
            gate=self.name,
            verdict=verdict,
            summary=f"{self.name} → {verdict.value}",
            findings=self.findings,
            tool_reports=self.tool_reports,
        )


class ScriptedLLM(LLMAdapter):
    """A model that answers out of the call it was handed.

    Nothing here weakens grounding: the brief cites the evidence ids the gate
    really produced unless a test asks for ``cite`` it was never given, and the
    reading aid is bound to the evidence's own fingerprint unless a test moves
    it. That is how "the model was asked" and "the model was believed" stay
    separate facts in the assertions.
    """

    def __init__(
        self,
        *,
        cite: list[str] | None = None,
        fingerprint: str | None = None,
        error: Exception | None = None,
        judgments: list[ModelFinding] | None = None,
        questions: list[str] | None = None,
        tests: list[ProposedTest] | None = None,
    ) -> None:
        self.call_records: list[LLMCallRecord] = []
        self.cite = cite
        self.fingerprint = fingerprint
        self.error = error
        self.judgments = judgments
        self.questions = questions
        self.tests = tests
        self.explained: list[tuple[str, int]] = []
        self.reviewed: list[str] = []
        #: Rule ids and citable lines it was handed, so a test can assert on
        #: what the gate chose to ask about rather than only on the answer.
        self.judged: list[tuple[str, ...]] = []
        self.locations: list[dict[str, str]] = []
        #: The platform's own review comments it was handed as context, so a
        #: test can assert they arrived without being citable.
        self.reviews: list[dict[str, str]] = []
        #: The symbols and the directory a test-generation call was handed, so
        #: a test can assert the gate asked about what the change touched.
        self.testgen_symbols: list[tuple[str, ...]] = []
        self.testgen_directory: str | None = None

    def explain_block(
        self,
        *,
        gate: GateName,
        attempt: int,
        source_fingerprint: str,
        summary: str,
        deterministic_feedback: str,
        evidence_items: list[dict[str, str]],
    ) -> BlockBrief:
        self.explained.append((gate, attempt))
        self._record("BlockBrief")
        if self.error is not None:
            raise self.error
        return BlockBrief(
            gate=gate,
            attempt=attempt,
            source_fingerprint=source_fingerprint,
            summary="模型解释：金额分摊用了 float。",
            root_causes=["未使用 Decimal"],
            recommended_actions=["改用 decimal.Decimal 并补一条尾差断言"],
            cited_evidence_ids=self.cite or [evidence_items[0]["id"]],
            confidence=0.6,
        )

    def judge_rules(
        self,
        *,
        change: ChangeRequest,
        fingerprint: str,
        rules: list[dict[str, str]],
        locations: list[dict[str, str]],
        reviews: Sequence[dict[str, str]] = (),
    ) -> ModelReviewReport:
        self.judged.append(tuple(rule["rule_id"] for rule in rules))
        self.locations = list(locations)
        self.reviews = list(reviews)
        self._record("ModelReviewReport")
        if self.error is not None:
            raise self.error
        return ModelReviewReport(
            evidence_fingerprint=self.fingerprint or fingerprint,
            summary="按判断类规则读过这次变更。",
            change_summary="",
            findings=list(self.judgments or []),
            review_questions=list(self.questions or []),
        )

    def propose_tests(
        self,
        *,
        change: ChangeRequest,
        symbols: Sequence[dict[str, str]],
        fingerprint: str,
        language: str,
        test_directory: str,
    ) -> ProposedTests:
        self.testgen_symbols.append(tuple(str(item["symbol"]) for item in symbols))
        self.testgen_directory = test_directory
        self._record("ProposedTests")
        if self.error is not None:
            raise self.error
        return ProposedTests(
            evidence_fingerprint=self.fingerprint or fingerprint,
            summary="为这次改动的符号补了回归用例。",
            tests=list(self.tests or []),
        )

    def review_change(self, evidence: EvidencePack) -> ModelReviewReport:
        self.reviewed.append(evidence.run_id)
        self._record("ModelReviewReport")
        if self.error is not None:
            raise self.error
        return ModelReviewReport(
            evidence_fingerprint=self.fingerprint or evidence.fingerprint,
            summary="重点看退款分摊的精度与边界。",
            change_summary="按明细拆分退款金额",
            review_questions=["尾差落在谁头上？"],
            confidence=0.5,
        )

    def _record(self, schema: str) -> None:
        self.call_records.append(
            LLMCallRecord(
                provider="scripted",
                model="scripted-1",
                schema_name=schema,
                schema_valid=self.error is None,
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
            )
        )


def hostile_judgment_finding() -> Finding:
    """A judgment-rule finding, as grounding accepts it, argued in ``HOSTILE_PROSE``.

    Both the model's reasoning and its question for the reviewer are hostile,
    and the rule has no fix of its own, so the question is what a reader is
    told to do.
    """

    proposal = ModelFinding(
        rule_id="J-01",
        title="Bypasses the state machine",
        detail=HOSTILE_PROSE,
        file="src/app/refund.py",
        line_start=13,
        line_end=13,
        confidence=0.9,
        suggested_question=HOSTILE_PROSE,
    )
    rule = Rule(
        id="J-01",
        title="Order status changes go through the state machine",
        kind="judgment",
        severity="high",
    )
    judgment = guarded_rule_judgment(
        make_change(),
        [rule],
        "fp-1",
        ScriptedLLM(judgments=[proposal]),
        default_min_confidence=0.5,
    )
    assert judgment is not None
    (finding,) = judgment.findings
    return finding


class StubScanner(ScannerAdapter):
    """A scanner that hands back prepared reports without a toolchain.

    ``tests`` and ``coverage`` are what the regression gate will be handed;
    the defaults are a suite that never started and a coverage tool that is
    not there, so a test that forgets to say otherwise sees the gate fail
    closed rather than pass on nothing.

    ``generated`` answers the second ``run_tests`` call, which is the one the
    gate makes for model-written cases -- the first is always the repository's
    own suite.
    """

    #: Where each toolchain's generated cases land, watched so a test can say
    #: what was on disk at the moment the suite ran.
    generated_dirs = ("tests/honest_ci_generated", "src/test/java/honest_ci/generated")

    def __init__(
        self,
        reports: list[ToolReport] | None = None,
        *,
        tests: TestResults | None = None,
        coverage: CoverageResult | None = None,
        generated: TestResults | None = None,
    ) -> None:
        self.reports = reports or []
        self.requested: list[str] | None = None
        self.tests = tests
        self.coverage = coverage
        self.generated = generated
        self.tests_run: list[Path] = []
        self.coverage_measured: list[Path] = []
        #: What the gate asked for, so a test can assert on the selection and
        #: the changed lines rather than only on the verdict they produced.
        self.selections: list[list[str] | None] = []
        self.coverage_lines: list[Mapping[str, set[int]] | None] = []
        #: Generated test files present at each ``run_tests`` call, so a test
        #: can assert they were there when the suite ran and gone afterwards.
        self.generated_files_seen: list[list[str]] = []

    def run_static_checks(
        self,
        workspace: Path,
        change: ChangeRequest,
        tools: Sequence[str] | None = None,
        scopes: Mapping[str, Sequence[str]] | None = None,
    ) -> list[ToolReport]:
        self.requested = None if tools is None else list(tools)
        #: What each tool was actually asked about, so a test can assert on the
        #: scope a rule's ``paths`` produced rather than only on the verdict.
        self.scopes = None if scopes is None else {k: list(v) for k, v in scopes.items()}
        if tools is None:
            return list(self.reports)
        return [report for report in self.reports if report.name in tools]

    def run_tests(self, workspace: Path, selection: list[str] | None = None) -> TestResults:
        self.tests_run.append(workspace)
        self.selections.append(None if selection is None else list(selection))
        self.generated_files_seen.append(self.generated_files(workspace))
        results = self.tests if len(self.tests_run) == 1 else (self.generated or self.tests)
        if results is None:
            return TestResults(
                available=False, errors=1, failures=["pytest is not installed (stub)."]
            )
        return results

    def generated_files(self, workspace: Path) -> list[str]:
        return sorted(
            path.relative_to(workspace).as_posix()
            for directory in self.generated_dirs
            for path in (workspace / directory).rglob("*")
            if path.is_file()
        )

    def measure_coverage(
        self,
        workspace: Path,
        lines: Mapping[str, set[int]] | None = None,
    ) -> CoverageResult:
        self.coverage_measured.append(workspace)
        self.coverage_lines.append(lines)
        if self.coverage is None:
            return CoverageResult(available=False, stderr="coverage is not installed (stub).")
        return self.coverage


class StubForge(ForgeAdapter):
    """A platform that hands back a prepared change and records what was written.

    Everything it records is what the caller decided to publish, so a test can
    assert on the two things that matter at this boundary: that a verdict
    reached the platform at all, and that it reached it unchanged.
    """

    def __init__(
        self,
        change: ChangeRequest | None = None,
        *,
        fails_on: str | None = None,
        reviews: Sequence[PlatformReview] = (),
        approvals: PlatformApprovals | None = None,
        ownership: Ownership | None = None,
    ) -> None:
        self.change = change if change is not None else make_change()
        self.fails_on = fails_on
        self.reviews = list(reviews)
        #: What ``fetch_approvals`` hands over; ``None`` is a platform whose
        #: approvals are not read, like the port's default.
        self.approvals = approvals
        #: What ``ownership`` says of every change; ``None``, the port's
        #: default, is a platform that doesn't say.
        self._ownership = ownership
        self.approvals_fetched: list[str] = []
        #: The approval reading each ``publish_report`` was given.
        self.readings: list[ApprovalReading | None] = []
        #: The license note each ``publish_report`` was given.
        self.license_notes: list[str | None] = []
        #: The bodies ``publish_note`` wrote, in order.
        self.notes: list[str] = []
        #: The summaries ``publish_error`` was given. Recorded besides the
        #: blocked status the port's default sends for them.
        self.errors: list[str] = []
        self.fetched: list[tuple[str, int]] = []
        self.reviews_fetched: list[str] = []
        self.reports: list[tuple[str, RunStatus]] = []
        self.statuses: list[tuple[Verdict, str]] = []
        self.progress: list[str] = []
        #: Where each status linked, in the order sent: ("progress" or "status", url).
        self.links: list[tuple[str, str | None]] = []
        self.inline: list[str] = []
        self.updated: list[str] = []
        self.events: list[str] = []
        self._comment_keys: set[tuple[str, int | None]] = set()

    def fetch_change(self, repo: str, number: int) -> ChangeRequest:
        self._maybe_fail("fetch_change")
        self.fetched.append((repo, number))
        return self.change.model_copy(update={"repo": repo, "number": number})

    def fetch_reviews(self, change: ChangeRequest) -> list[PlatformReview]:
        # Deliberately not in ``events``: that list is the write-back sequence,
        # and reading the existing discussion is what a dry run does too.
        self._maybe_fail("fetch_reviews")
        self.reviews_fetched.append(change.ref)
        return list(self.reviews)

    def fetch_approvals(self, change: ChangeRequest) -> PlatformApprovals | None:
        # Read-only like ``fetch_reviews``, so not in ``events`` either.
        self._maybe_fail("fetch_approvals")
        self.approvals_fetched.append(change.ref)
        return self.approvals

    def materialize(self, change: ChangeRequest, workspace: Path) -> Path:
        self._maybe_fail("materialize")
        checkout = workspace / f"{change.head_sha[:12]}"
        checkout.mkdir(parents=True, exist_ok=True)
        return checkout

    def publish_report(
        self,
        change: ChangeRequest,
        evidence: EvidencePack,
        status: RunStatus,
        approvals: ApprovalReading | None = None,
        evidence_at: str | None = None,
        license_note: str | None = None,
    ) -> None:
        self._maybe_fail("publish_report")
        key = (change.repo, change.number)
        if key in self._comment_keys:
            self.updated.append(evidence.run_id)
        self._comment_keys.add(key)
        self.events.append("report")
        self.reports.append((evidence.run_id, status))
        self.readings.append(approvals)
        self.license_notes.append(license_note)

    def publish_note(self, change: ChangeRequest, body: str) -> None:
        self._maybe_fail("publish_note")
        self._comment_keys.add((change.repo, change.number))
        self.events.append("note")
        self.notes.append(body)

    def ownership(self, change: ChangeRequest) -> Ownership | None:
        return self._ownership

    def publish_progress(
        self,
        change: ChangeRequest,
        summary: str,
        url: str | None = None,
    ) -> None:
        self._maybe_fail("publish_progress")
        self.events.append("progress")
        self.progress.append(summary)
        self.links.append(("progress", url))

    def publish_inline(
        self,
        change: ChangeRequest,
        evidence: EvidencePack,
        findings: Sequence[Finding],
    ) -> list[str]:
        self._maybe_fail("publish_inline")
        self.events.append("inline")
        posted = [item.id for item in findings if item.id not in self.inline]
        self.inline.extend(posted)
        return posted

    def publish_status(
        self,
        change: ChangeRequest,
        verdict: Verdict,
        summary: str,
        url: str | None = None,
    ) -> None:
        self._maybe_fail("publish_status")
        self.events.append("status")
        self.statuses.append((verdict, summary))
        self.links.append(("status", url))

    def publish_error(
        self,
        change: ChangeRequest,
        summary: str,
        url: str | None = None,
    ) -> None:
        self.errors.append(summary)
        super().publish_error(change, summary, url)

    def _maybe_fail(self, method: str) -> None:
        if self.fails_on == method:
            raise ForgeError(f"the platform refused {method}")
