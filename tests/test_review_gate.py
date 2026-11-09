from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from honest_ci.adapters.base import AdapterSet
from honest_ci.adapters.deepseek import DeepSeekAdapter
from honest_ci.contracts import (
    ChangedFile,
    EvidencePack,
    GateReport,
    ModelFinding,
    PlatformReview,
    ToolDiagnostic,
    ToolReport,
    Verdict,
)
from honest_ci.gates import GATE_REGISTRY
from honest_ci.gates.base import Gate
from honest_ci.report import render_evidence, render_forge_comment
from honest_ci.runner import GateRunner
from honest_ci.settings import MODEL_NOT_ENABLED, Settings
from honest_ci.state import RunState
from honest_ci.storage import SQLiteLedger

from .factories import (
    DIFF,
    HOSTILE_OUTPUT,
    HOSTILE_TEXT,
    WEAKENED_TESTS_DIFF,
    ScriptedLLM,
    StubScanner,
    cjk_lines,
    make_change,
    make_settings,
    make_weakened_change,
    markdown_tokens,
    rendered_as_markdown,
)

FLOAT_RULE = """
[[rule]]
id = "R-01"
title = "金额计算不得使用 float"
severity = "high"
kind = "regex"
pattern = 'float\\('
paths = ["**/*.py"]
"""

RUFF_RULE = """
[[rule]]
id = "T-01"
title = "静态检查必须通过"
severity = "high"
kind = "tool"
tool = "ruff"
"""

JUDGMENT_RULE = """
[[rule]]
id = "J-01"
title = "订单状态变更必须走状态机"
detail = "直接给状态字段赋值会绕开状态机，退款后的对账因此漏过两次。"
severity = "high"
kind = "judgment"
paths = ["src/**/*.py"]
min_confidence = 0.7
"""


def build(
    tmp_path: Path,
    rules: str | None = None,
    scanner: StubScanner | None = None,
    llm: ScriptedLLM | None = None,
    settings: Settings | None = None,
    policy: str | None = None,
) -> Gate:
    if rules is not None:
        (tmp_path / "honest-ci-rules.toml").write_text(rules, encoding="utf-8")
    if policy is not None:
        (tmp_path / "honest-ci-policy.toml").write_text(policy, encoding="utf-8")
    runner = GateRunner(
        workspace_root=tmp_path,
        settings=settings or make_settings(),
        adapters=AdapterSet(
            ledger=SQLiteLedger(tmp_path / "ledger.sqlite3"),
            scanner=scanner,
            llm=llm,
        ),
    )
    return runner.gates["review"]


def state() -> RunState:
    return RunState(run_id="r1", change=make_change())


def ruff_report(diagnostics: list[ToolDiagnostic], passed: bool = False) -> ToolReport:
    return ToolReport(
        name="ruff",
        command="ruff check",
        available=True,
        passed=passed,
        exit_code=0 if passed else 1,
        diagnostics=diagnostics,
    )


def test_the_review_gate_is_deployed() -> None:
    assert "review" in GATE_REGISTRY


def test_a_gate_with_no_rule_library_reviewed_nothing_and_says_so(tmp_path: Path) -> None:
    report = build(tmp_path).run(state())

    # An empty rule library is not a clean bill of health: nothing was checked.
    assert report.verdict is Verdict.NEEDS_HUMAN
    assert report.summary.startswith("No team rule library is configured")
    assert report.metrics["rules_total"] == 0


def test_a_blocking_rule_names_the_rule_and_the_line_it_fired_on(tmp_path: Path) -> None:
    report = build(tmp_path, FLOAT_RULE).run(state())

    assert report.verdict is Verdict.BLOCKED
    finding = report.findings[0]
    assert finding.rule_id == "R-01"
    assert (finding.file, finding.line_start) == ("src/app/refund.py", 11)
    assert report.verifier_report["rules_evaluated"][0]["id"] == "R-01"
    assert report.metrics["findings_blocking"] == 1


def test_a_rule_below_the_blocking_bar_is_recorded_and_let_through(tmp_path: Path) -> None:
    report = build(tmp_path, FLOAT_RULE.replace('severity = "high"', 'severity = "low"')).run(
        state()
    )

    assert report.verdict is Verdict.PASS
    assert len(report.findings) == 1
    assert report.metrics["findings_blocking"] == 0


def test_the_rule_file_decides_which_scanners_run(tmp_path: Path) -> None:
    scanner = StubScanner([ruff_report([], passed=True)])
    build(tmp_path, RUFF_RULE, scanner).run(state())

    # Not "every scanner we know about": the team's rule file is what makes
    # a scanner mandatory, and that choice has to be auditable.
    assert scanner.requested == ["ruff"]


def test_a_tool_rule_only_hands_the_scanner_the_paths_it_covers(tmp_path: Path) -> None:
    # The fourth bug dogfooding found: a rule's ``paths`` was a comment. mypy
    # got the change's test files in a repository whose own mypy configuration
    # only ever looks at ``src``, and produced 23 diagnostics nobody asked for.
    scanner = StubScanner([ruff_report([], passed=True)])
    change = make_change(
        files_changed=[
            ChangedFile(path="src/app/refund.py", change_type="modified", added_lines=3),
            ChangedFile(path="tests/test_refund.py", change_type="modified", added_lines=2),
        ]
    )
    scoped = RUFF_RULE + '\npaths = ["src/**"]\n'

    build(tmp_path, scoped, scanner).run(RunState(run_id="r1", change=change))

    assert scanner.scopes == {"ruff": ["src/app/refund.py"]}


def test_a_rule_that_narrows_nothing_still_asks_about_every_changed_file(
    tmp_path: Path,
) -> None:
    scanner = StubScanner([ruff_report([], passed=True)])
    change = make_change(
        files_changed=[
            ChangedFile(path="src/app/refund.py", change_type="modified", added_lines=3),
            ChangedFile(path="tests/test_refund.py", change_type="modified", added_lines=2),
        ]
    )

    build(tmp_path, RUFF_RULE, scanner).run(RunState(run_id="r1", change=change))

    assert scanner.scopes == {"ruff": ["src/app/refund.py", "tests/test_refund.py"]}


def test_two_rules_on_one_scanner_ask_about_the_union_of_their_paths(
    tmp_path: Path,
) -> None:
    scanner = StubScanner([ruff_report([], passed=True)])
    change = make_change(
        files_changed=[
            ChangedFile(path="src/app/refund.py", change_type="modified", added_lines=3),
            ChangedFile(path="tests/test_refund.py", change_type="modified", added_lines=2),
            ChangedFile(path="docs/guide.md", change_type="modified", added_lines=1),
        ]
    )
    rules = (
        RUFF_RULE
        + '\npaths = ["src/**"]\n'
        + RUFF_RULE.replace('id = "T-01"', 'id = "T-02"')
        + '\npaths = ["tests/**"]\n'
    )

    build(tmp_path, rules, scanner).run(RunState(run_id="r1", change=change))

    # Each rule is a separate reason to run the scanner; running it on less
    # than one of them asked for would leave that rule unreviewed.
    assert scanner.scopes == {"ruff": ["src/app/refund.py", "tests/test_refund.py"]}


def test_a_scanner_finding_on_an_added_line_blocks_under_its_rule(tmp_path: Path) -> None:
    scanner = StubScanner(
        [
            ruff_report(
                [
                    ToolDiagnostic(
                        tool="ruff",
                        file="src/app/refund.py",
                        line=11,
                        code="F841",
                        message="Local variable `share` is assigned but never used",
                    )
                ]
            )
        ]
    )
    report = build(tmp_path, RUFF_RULE, scanner).run(state())

    assert report.verdict is Verdict.BLOCKED
    finding = report.findings[0]
    assert finding.rule_id == "T-01"
    assert finding.line_start == 11
    assert "F841" in finding.detail


def test_a_scanner_finding_on_untouched_code_is_evidence_not_a_block(tmp_path: Path) -> None:
    scanner = StubScanner(
        [
            ruff_report(
                [
                    ToolDiagnostic(
                        tool="ruff", file="src/app/refund.py", line=400, message="pre-existing"
                    )
                ]
            )
        ]
    )
    report = build(tmp_path, RUFF_RULE, scanner).run(state())

    # The change did not introduce it. Blocking on it is how a review gate turns
    # into a noise generator nobody trusts.
    assert report.verdict is Verdict.PASS
    assert report.findings == []
    assert report.metrics["diagnostics_total"] == 1
    assert report.metrics["diagnostics_on_added_lines"] == 0
    assert "so they are recorded and do not block" in report.summary


def test_without_a_diff_no_diagnostic_is_dropped(tmp_path: Path) -> None:
    scanner = StubScanner(
        [
            ruff_report(
                [
                    ToolDiagnostic(
                        tool="ruff", file="src/app/refund.py", line=400, message="unattributable"
                    )
                ]
            )
        ]
    )
    gate = build(tmp_path, RUFF_RULE, scanner)
    report = gate.run(RunState(run_id="r1", change=make_change(diff="")))

    # Not knowing which lines are new is a reason to be stricter, not looser.
    assert report.verdict is Verdict.BLOCKED
    assert report.verifier_report["line_attribution"] == "unavailable"


def test_a_scanner_that_never_ran_blocks_instead_of_passing(tmp_path: Path) -> None:
    scanner = StubScanner(
        [
            ToolReport(
                name="ruff",
                command="ruff (not executed)",
                available=False,
                passed=False,
                stderr="ruff is not installed in the execution environment.",
            )
        ]
    )
    report = build(tmp_path, RUFF_RULE, scanner).run(state())

    assert report.verdict is Verdict.BLOCKED
    assert any(finding.id == "review.unavailable" for finding in report.findings)
    # Why, where a person reads first: the summary is what the terminal, the
    # evidence pack's gate table and the pull request comment show.
    assert (
        "scanners that did not start: ruff — `ruff is not installed in the execution environment.`"
        in report.summary
    )


def test_a_rule_whose_scanner_is_missing_entirely_blocks(tmp_path: Path) -> None:
    # No ScannerAdapter at all: the rule cannot be evaluated, so it must not
    # silently stop firing.
    report = build(tmp_path, RUFF_RULE, scanner=None).run(state())

    assert report.verdict is Verdict.BLOCKED
    assert report.metrics["tools_unavailable"] == 1


def test_a_scanner_that_failed_without_parseable_output_is_missing_evidence(
    tmp_path: Path,
) -> None:
    scanner = StubScanner(
        [
            ToolReport(
                name="ruff",
                command="ruff check",
                available=True,
                passed=False,
                exit_code=2,
                stderr="error: invalid rule selector `XYZ`",
            )
        ]
    )
    report = build(tmp_path, RUFF_RULE, scanner).run(state())

    # It ran and failed, and nothing it printed can be attributed. That is a
    # broken check, not a clean file.
    assert report.verdict is Verdict.BLOCKED
    assert "invalid rule selector" in report.findings[0].detail


def test_the_gate_reports_the_rule_version_that_judged_the_change(tmp_path: Path) -> None:
    report = build(tmp_path, 'version = "v9"\n' + FLOAT_RULE).run(state())

    assert report.rule_version == "v9"
    # Named relative to the deployment root: the pack is handed over, and the
    # absolute path would say more about the machine than about the rules.
    assert report.verifier_report["rule_source"] == "honest-ci-rules.toml"


def test_the_pack_names_the_kind_of_tree_that_was_scanned_not_its_path(
    tmp_path: Path,
) -> None:
    gate = build(tmp_path, FLOAT_RULE)
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    in_place = gate.run(state())
    materialised = gate.run(
        RunState(run_id="r2", change=make_change(), workspace_path=str(checkout))
    )

    # An auditor cannot check a path on the operator's disk, so the pack keeps
    # the one thing the path was carrying: whether the change was scanned in a
    # tree materialised for this run, or in the deployment's own working copy.
    assert in_place.verifier_report["workspace_kind"] == "deployment-root"
    assert materialised.verifier_report["workspace_kind"] == "checkout"
    assert "workspace" not in in_place.verifier_report


def test_two_rules_on_one_scanner_do_not_double_count_its_diagnostics(tmp_path: Path) -> None:
    rules = RUFF_RULE + """
[[rule]]
id = "T-02"
title = "静态检查（观察级）"
severity = "low"
kind = "tool"
tool = "ruff"
"""
    scanner = StubScanner(
        [
            ruff_report(
                [ToolDiagnostic(tool="ruff", file="src/app/refund.py", line=11, message="x")]
            )
        ]
    )
    report = build(tmp_path, rules, scanner).run(state())

    # One diagnostic that violates two rules is two findings and one diagnostic.
    assert len(report.findings) == 2
    assert report.metrics["diagnostics_total"] == 1
    assert report.metrics["diagnostics_on_added_lines"] == 1


# ── judgment-class rules (v1) ────────────────────────────────────────────────


def proposal(**overrides: Any) -> ModelFinding:
    payload: dict[str, Any] = {
        "rule_id": "J-01",
        "title": "退款直接改了明细状态",
        "detail": "item.refund 被直接赋值。",
        "file": "src/app/refund.py",
        "line_start": 13,
        "line_end": 13,
        "confidence": 0.9,
    }
    payload.update(overrides)
    return ModelFinding.model_validate(payload)


def judgment_of(report: GateReport) -> dict[str, Any]:
    payload: dict[str, Any] = report.verifier_report["judgment"]
    return payload


def test_a_judgment_rule_with_no_model_is_unread_and_goes_to_a_human(tmp_path: Path) -> None:
    report = build(tmp_path, JUDGMENT_RULE).run(state())

    # Not a pass, and not a block either: nobody looked, and the model was
    # never entitled to a verdict, so the change goes in front of a person.
    assert report.verdict is Verdict.NEEDS_HUMAN
    assert judgment_of(report)["judged"] is False
    assert judgment_of(report)["not_judged_reason"] == MODEL_NOT_ENABLED
    assert report.metrics["judgment_rules_unread"] == 1


def test_a_named_provider_without_its_key_says_which_key_is_missing(tmp_path: Path) -> None:
    settings = replace(make_settings(), model_provider="deepseek")
    report = build(tmp_path, JUDGMENT_RULE, settings=settings).run(state())

    # A different fix from "nobody enabled a model", so a different sentence.
    assert report.verdict is Verdict.NEEDS_HUMAN
    reason = str(judgment_of(report)["not_judged_reason"])
    assert "no model credential is configured (DEEPSEEK_API_KEY)" in reason


def test_turning_model_judgment_off_is_not_the_same_as_skipping_the_rule(
    tmp_path: Path,
) -> None:
    llm = ScriptedLLM()
    settings = replace(make_settings(), model_judgment=False)
    report = build(tmp_path, JUDGMENT_RULE, llm=llm, settings=settings).run(state())

    assert report.verdict is Verdict.NEEDS_HUMAN
    assert llm.call_records == []
    assert "HONEST_CI_MODEL_JUDGMENT=0" in str(judgment_of(report)["not_judged_reason"])


def test_turning_the_model_off_leaves_every_deterministic_finding_word_for_word(
    tmp_path: Path,
) -> None:
    """独立性：关掉模型判定，确定性结论一字不变。

    「独立」能变成回归门禁的只有这一条含义：使用方可以关掉模型、可以换一家模型，
    本部署判出来的拦截项一个不多一个不少。它保护的是这条承诺，不是某个字段。

    判的是逐字相同（id、title、detail、行号、严重级别全在内），不是「数量一样」：
    一个把规则 id 编号顺序挪了位的改动，照样会让审计方手里两份证据对不上。
    """

    deterministic_and_judgment = FLOAT_RULE + JUDGMENT_RULE

    with_model = build(
        tmp_path,
        deterministic_and_judgment,
        llm=ScriptedLLM(judgments=[proposal()]),
    ).run(state())
    without_model = build(
        tmp_path,
        deterministic_and_judgment,
        llm=ScriptedLLM(judgments=[proposal()]),
        settings=replace(make_settings(), model_judgment=False),
    ).run(state())

    def deterministic(report: GateReport) -> list[dict[str, Any]]:
        return [
            finding.model_dump()
            for finding in report.findings
            if finding.decided_by == "deterministic"
        ]

    # Non-empty, or the equality below would hold on a gate that found nothing.
    assert deterministic(with_model)
    assert deterministic(with_model) == deterministic(without_model)
    assert [finding.id for finding in with_model.findings if finding.blocking] == [
        finding.id for finding in without_model.findings if finding.blocking
    ]
    # And the model really did speak in the first run -- otherwise this test
    # would pass just as well on a build where judgment never runs at all.
    assert any(finding.decided_by == "model" for finding in with_model.findings)
    assert all(finding.decided_by == "deterministic" for finding in without_model.findings)


def test_a_model_finding_routes_to_a_human_instead_of_blocking(tmp_path: Path) -> None:
    llm = ScriptedLLM(judgments=[proposal()])
    report = build(tmp_path, JUDGMENT_RULE, llm=llm).run(state())

    assert report.verdict is Verdict.NEEDS_HUMAN
    finding = report.findings[0]
    assert (finding.rule_id, finding.decided_by) == ("J-01", "model")
    # The rule is severity high and the finding carries that, but nothing a
    # model decided counts as blocking -- that is the invariant, not a detail.
    assert finding.severity == "high"
    assert report.metrics["findings_blocking"] == 0
    assert report.metrics["judgment_findings"] == 1


def test_a_deterministic_block_outranks_a_model_finding(tmp_path: Path) -> None:
    llm = ScriptedLLM(judgments=[proposal()])
    report = build(tmp_path, FLOAT_RULE + JUDGMENT_RULE, llm=llm).run(state())

    assert report.verdict is Verdict.BLOCKED
    assert {finding.rule_id for finding in report.findings} == {"R-01", "J-01"}


def test_a_clean_judgment_lets_the_change_through(tmp_path: Path) -> None:
    llm = ScriptedLLM(judgments=[], questions=["退款失败时状态回滚到哪一步？"])
    report = build(tmp_path, JUDGMENT_RULE, llm=llm).run(state())

    # The model was asked and raised nothing. Its open questions are recorded
    # for a reviewer but do not by themselves interrupt anyone.
    assert report.verdict is Verdict.PASS
    assert judgment_of(report)["questions"] == ["退款失败时状态回滚到哪一步？"]


def test_doubt_below_the_threshold_is_recorded_without_interrupting_anyone(
    tmp_path: Path,
) -> None:
    llm = ScriptedLLM(judgments=[proposal(confidence=0.4)])
    report = build(tmp_path, JUDGMENT_RULE, llm=llm).run(state())

    assert report.verdict is Verdict.PASS
    assert report.findings == []
    assert "(confidence 0.40 < 0.70)" in str(judgment_of(report)["below_threshold"][0])


def test_output_the_trust_boundary_threw_away_leaves_the_rule_unread(tmp_path: Path) -> None:
    llm = ScriptedLLM(judgments=[proposal()], fingerprint="fp-from-another-commit")
    report = build(tmp_path, JUDGMENT_RULE, llm=llm).run(state())

    assert report.verdict is Verdict.NEEDS_HUMAN
    assert report.findings == []
    assert judgment_of(report)["judged"] is False
    assert "discarded whole" in str(judgment_of(report)["not_judged_reason"])


def test_only_judgment_rules_are_put_to_the_model(tmp_path: Path) -> None:
    llm = ScriptedLLM(judgments=[])
    build(tmp_path, FLOAT_RULE + RUFF_RULE + JUDGMENT_RULE, StubScanner(), llm).run(state())

    # One call, carrying one rule: the deterministic rules are not the model's
    # business, and paying a provider to re-judge a regex would be absurd.
    assert llm.judged == [("J-01",)]


def test_a_library_without_judgment_rules_never_calls_the_model(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    report = build(tmp_path, FLOAT_RULE, llm=llm).run(state())

    assert llm.call_records == []
    assert "judgment" not in report.verifier_report


def test_the_model_only_sees_lines_it_is_allowed_to_cite(tmp_path: Path) -> None:
    llm = ScriptedLLM(judgments=[])
    build(tmp_path, JUDGMENT_RULE, llm=llm).run(state())

    assert [item["line"] for item in llm.locations] == ["11", "12", "13"]
    assert {item["file"] for item in llm.locations} == {"src/app/refund.py"}


# ── review comments the platform already carried ─────────────────────────────

OPEN_THREAD = PlatformReview(
    author="wangwei",
    body="这里的分摊是不是应该走 Decimal？",
    path="src/app/refund.py",
    line=11,
    resolved=False,
)
RESOLVED_THREAD = PlatformReview(
    author="liyan", body="已确认，没问题。", path="src/app/refund.py", line=12, resolved=True
)
PLAIN_COMMENT = PlatformReview(author="duo", body="顺带一提：这条 MR 很大。", is_bot=True)

IGNORE_POLICY = '[policy]\nversion = "v1"\nunresolved_reviews = "ignore"\n'


def reviewed(*reviews: PlatformReview, **overrides: object) -> RunState:
    return RunState(run_id="r1", change=make_change(platform_reviews=list(reviews), **overrides))


def platform_of(report: GateReport) -> dict[str, Any]:
    section: dict[str, Any] = report.verifier_report["platform_reviews"]
    return section


def test_an_unresolved_thread_hands_the_change_to_a_person(tmp_path: Path) -> None:
    report = build(tmp_path, FLOAT_RULE.replace('severity = "high"', 'severity = "low"')).run(
        reviewed(OPEN_THREAD, RESOLVED_THREAD)
    )

    # Somebody asked a question on this change and nobody answered it. An
    # automated pass there would be a machine overruling a person on the record.
    assert report.verdict is Verdict.NEEDS_HUMAN
    finding = next(item for item in report.findings if item.id.startswith("review.platform-"))
    assert finding.decided_by == "deterministic"
    assert finding.title == "Unresolved review threads on the change: 1"
    assert "wangwei" in finding.detail and "liyan" not in finding.detail
    assert report.metrics["platform_reviews_unresolved"] == 1


def test_a_thread_on_a_name_holding_a_backtick_names_the_file_whole(tmp_path: Path) -> None:
    thread = OPEN_THREAD.model_copy(update={"path": "src/app/a`b.py"})
    report = build(tmp_path, FLOAT_RULE.replace('severity = "high"', 'severity = "low"')).run(
        reviewed(thread)
    )

    # The path is whatever the commenter's client sent, and the detail is
    # Markdown wherever the finding is shown: a single-backtick span would
    # close at the name's own backtick.
    finding = next(item for item in report.findings if item.id.startswith("review.platform-"))
    assert "- wangwei · ``src/app/a`b.py``:11 · " in finding.detail


def test_what_a_scanner_or_a_commenter_quoted_never_renders_as_markdown(
    tmp_path: Path,
) -> None:
    diagnostic = ToolDiagnostic(
        tool="ruff",
        file="src/app/refund.py",
        line=11,
        code="F841",
        message=f"Local variable {HOSTILE_TEXT} is assigned but never used",
    )
    scanner = StubScanner([ruff_report([diagnostic])])
    thread = OPEN_THREAD.model_copy(update={"body": HOSTILE_TEXT})

    report = build(tmp_path, RUFF_RULE, scanner).run(reviewed(thread))

    # A scanner quotes the code it flags, and a comment is anyone's text; the
    # detail is Markdown on the line comment and in the pack.
    assert len(report.findings) == 2
    for finding in report.findings:
        assert "Approved by security" in finding.detail
        assert rendered_as_markdown(finding.detail, "Approved by security") == []


def test_what_a_broken_scanner_printed_is_quoted_as_a_block_it_cannot_close(
    tmp_path: Path,
) -> None:
    broken = ruff_report([]).model_copy(update={"stderr": HOSTILE_OUTPUT})

    report = build(tmp_path, RUFF_RULE, StubScanner([broken])).run(reviewed())
    evidence = EvidencePack(
        run_id="r1",
        change=make_change(),
        gate_reports=[report],
        verdict=report.verdict,
        fingerprint="f" * 64,
    )

    # A scanner that chokes echoes the source it choked on: the change's text.
    (finding,) = report.findings
    assert finding.id == "review.T-01.1"
    text = render_evidence(evidence)
    assert rendered_as_markdown(text, "Approved by security") == []
    [block] = [
        token.content
        for token in markdown_tokens(text)
        if token.type == "fence" and "Approved by security" in token.content
    ]
    assert block.count("Approved by security") == 4
    assert "\ntoken=\n" in block
    assert "\n\n" in block


def test_an_unresolved_thread_asks_for_a_person_rather_than_blocking(tmp_path: Path) -> None:
    report = build(tmp_path, FLOAT_RULE.replace('severity = "high"', 'severity = "low"')).run(
        reviewed(OPEN_THREAD)
    )

    # A block is terminal and only a new commit clears it; what an unanswered
    # question needs is the person who asked it.
    finding = next(item for item in report.findings if item.id.startswith("review.platform-"))
    assert finding.severity == "medium" and not finding.blocking
    assert report.verdict is Verdict.NEEDS_HUMAN


def test_a_policy_that_ignores_open_threads_keeps_them_as_evidence_only(tmp_path: Path) -> None:
    report = build(
        tmp_path,
        FLOAT_RULE.replace('severity = "high"', 'severity = "low"'),
        policy=IGNORE_POLICY,
    ).run(reviewed(OPEN_THREAD))

    assert report.verdict is Verdict.PASS
    assert not [item for item in report.findings if item.id.startswith("review.platform-")]
    assert platform_of(report) == {
        "policy": "ignore",
        "total": 1,
        "unresolved": 1,
        "bots": 0,
        "not_read_reason": None,
    }
    assert '1 unresolved, which the policy (unresolved_reviews = "ignore") only records' in (
        report.summary
    )


def test_resolved_and_plain_comments_are_evidence_and_not_a_hand_off(tmp_path: Path) -> None:
    report = build(tmp_path, FLOAT_RULE.replace('severity = "high"', 'severity = "low"')).run(
        reviewed(RESOLVED_THREAD, PLAIN_COMMENT)
    )

    # A comment the platform never tracked as resolvable is not an open
    # question, and an answered one is not either.
    assert report.verdict is Verdict.PASS
    assert platform_of(report)["unresolved"] == 0 and platform_of(report)["bots"] == 1
    assert "2 review comments already on the change" in report.summary


def test_a_change_nobody_commented_on_grows_no_review_section(tmp_path: Path) -> None:
    report = build(tmp_path, FLOAT_RULE.replace('severity = "high"', 'severity = "low"')).run(
        state()
    )

    assert report.verdict is Verdict.PASS
    assert "platform_reviews" not in report.verifier_report
    assert report.metrics["platform_reviews"] == 0


def test_reviews_nobody_could_read_leave_the_gate_running_and_say_so(tmp_path: Path) -> None:
    report = build(tmp_path, FLOAT_RULE.replace('severity = "high"', 'severity = "low"')).run(
        RunState(run_id="r1", change=make_change(platform_reviews_error="GitLab 返回 403"))
    )

    # Reading the discussion is fail-soft, but an empty list must not be how a
    # failed read looks: "not read" is not "nobody raised a problem".
    assert report.verdict is Verdict.PASS
    assert platform_of(report)["not_read_reason"] == "GitLab 返回 403"
    assert "the review comments already on the change could not be read" in report.summary


def test_the_model_is_told_what_people_already_said(tmp_path: Path) -> None:
    llm = ScriptedLLM(judgments=[])
    build(tmp_path, JUDGMENT_RULE, llm=llm).run(reviewed(PLAIN_COMMENT, OPEN_THREAD))

    # Located comments first, so the model can line them up against the diff.
    assert [item["location"] for item in llm.reviews] == ["src/app/refund.py:11", ""]
    assert llm.reviews[0]["state"] == "unresolved" and llm.reviews[1]["state"] == "comment"
    # A bot's comment is marked as one; a person's is not.
    assert "Decimal" in llm.reviews[0]["body"]
    assert llm.reviews[1]["author"].startswith(PLAIN_COMMENT.author)
    assert "bot" in llm.reviews[1]["author"] and "bot" not in llm.reviews[0]["author"]


def test_a_comment_can_never_become_a_citation(tmp_path: Path) -> None:
    # The model answers by naming a rule that only exists in a comment thread.
    llm = ScriptedLLM(
        judgments=[
            ModelFinding(
                rule_id="J-99",
                title="评审意见里提过的问题",
                file="src/app/refund.py",
                line_start=11,
                line_end=11,
                confidence=0.95,
            )
        ]
    )
    report = build(tmp_path, JUDGMENT_RULE, llm=llm).run(reviewed(OPEN_THREAD))

    # Whole response discarded: somebody else's sentence is not evidence this
    # side produced, so it cannot launder an invented rule id into a finding.
    assert not [item for item in report.findings if item.decided_by == "model"]
    assert judgment_of(report)["judged"] is False


def test_a_model_that_is_never_called_still_sees_no_reviews(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    build(tmp_path, FLOAT_RULE, llm=llm).run(reviewed(OPEN_THREAD))

    assert llm.reviews == []


def test_what_honest_ci_writes_around_the_teams_words_in_a_prompt_is_english(
    tmp_path: Path,
) -> None:
    # The rules and the comments are the team's own words, in whatever language
    # the team writes. The instruction, the separators and the bot marker are
    # honest-ci's, so on an English change the whole prompt reads in English.
    prompts: list[str] = []

    def transport(prompt: str, settings: Settings) -> tuple[str, dict[str, Any]]:
        prompts.append(prompt)
        return "not json", {}

    settings = Settings(
        sandbox_mode="process",
        model_provider="deepseek",
        deepseek_api_key="sk-test-key-value-1234",
        deepseek_max_retries=0,
    )
    (tmp_path / "honest-ci-rules.toml").write_text(
        '[[rule]]\nid = "J-02"\ntitle = "Order status changes go through the state machine"\n'
        'detail = "Assigning the status field directly skips the state machine."\n'
        'severity = "high"\nkind = "judgment"\npaths = ["src/**/*.py", "lib/**/*.py"]\n',
        encoding="utf-8",
    )
    runner = GateRunner(
        workspace_root=tmp_path,
        settings=settings,
        adapters=AdapterSet(
            ledger=SQLiteLedger(tmp_path / "ledger.sqlite3"),
            llm=DeepSeekAdapter(settings, transport),
        ),
    )
    thread = OPEN_THREAD.model_copy(update={"body": "Should this split use Decimal?"})
    bot = PLAIN_COMMENT.model_copy(update={"body": "Heads up: this change is large."})

    runner.gates["review"].run(
        reviewed(thread, bot, title="feat: refund each line item", description="")
    )

    assert prompts, "the judgment rule was never put to the model"
    assert "src/**/*.py" in prompts[0] and "lib/**/*.py" in prompts[0]
    assert cjk_lines(prompts[0]) == []


# ── test integrity ───────────────────────────────────────────────────────────

INTEGRITY_RULE = """
[[rule]]
id = "I-01"
title = "Tests do not disappear"
kind = "integrity"
check = "deleted_test"
"""


def weakened(**overrides: Any) -> RunState:
    return RunState(run_id="r1", change=make_weakened_change(**overrides))


def test_a_deleted_test_sends_the_change_to_a_person(tmp_path: Path) -> None:
    report = build(tmp_path, INTEGRITY_RULE).run(weakened())

    assert report.verdict is Verdict.NEEDS_HUMAN
    (finding,) = report.findings
    assert (finding.rule_id, finding.side, finding.blocking) == ("I-01", "base", False)
    assert "test integrity: 1 hit (1 for a human to review)" in report.summary
    assert report.metrics["integrity_rules"] == 1
    assert report.metrics["integrity_findings"] == 1
    evaluated = report.verifier_report["rules_evaluated"][0]
    assert (evaluated["check"], evaluated["action"]) == ("deleted_test", "needs_human")


def test_a_blocking_integrity_rule_blocks_like_any_other_high_finding(tmp_path: Path) -> None:
    report = build(tmp_path, INTEGRITY_RULE + 'action = "block"\n').run(weakened())

    assert report.verdict is Verdict.BLOCKED
    assert report.metrics["findings_blocking"] == 1
    assert "1 blocking" in report.summary


def test_a_recorded_hit_is_written_down_and_lets_the_change_through(tmp_path: Path) -> None:
    report = build(tmp_path, INTEGRITY_RULE + 'action = "record"\n').run(weakened())

    assert report.verdict is Verdict.PASS
    assert [finding.rule_id for finding in report.findings] == ["I-01"]
    assert "1 recorded only" in report.summary


def test_a_change_that_leaves_its_tests_alone_passes_the_integrity_rules(tmp_path: Path) -> None:
    report = build(tmp_path, INTEGRITY_RULE).run(state())

    assert report.verdict is Verdict.PASS
    assert report.findings == []
    assert "test integrity" not in report.summary


def test_a_block_elsewhere_outranks_an_integrity_hand_off(tmp_path: Path) -> None:
    report = build(tmp_path, FLOAT_RULE + INTEGRITY_RULE).run(
        weakened(diff=DIFF + WEAKENED_TESTS_DIFF)
    )

    assert report.verdict is Verdict.BLOCKED
    assert {finding.rule_id for finding in report.findings} == {"R-01", "I-01"}


def test_the_comment_on_the_change_says_which_test_went_away(tmp_path: Path) -> None:
    # Through the gate rather than a hand-built report: the comment finds the
    # rule's action in what the gate records, and that record is what this
    # checks.
    report = build(tmp_path, INTEGRITY_RULE).run(weakened())
    evidence = EvidencePack(
        run_id="r1",
        change=make_weakened_change(),
        gate_reports=[report],
        verdict=report.verdict,
        fingerprint="f" * 64,
    )

    text = render_forge_comment(evidence, status="awaiting_human")

    assert "### Test integrity (1)" in text
    assert "**[I-01] Tests do not disappear** · needs a human · `tests/test_pricing.py`:3 (base)" in (
        text
    )
    assert "`test_total_doubles` is removed" in text
    assert "Removed: `def test_total_doubles():`" in text
    # Listed on its own above, so not counted again with the findings that do not block.
    assert "non-blocking finding" not in text
