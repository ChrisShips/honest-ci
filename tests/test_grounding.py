from __future__ import annotations

from pathlib import Path
from typing import Any

from honest_ci.contracts import (
    BlockBrief,
    ChangedFile,
    EvidencePack,
    Finding,
    GateReport,
    ModelFinding,
    ModelReviewReport,
    ProposedTest,
    ProposedTests,
    ToolReport,
    Verdict,
)
from honest_ci.grounding import (
    MAX_GENERATED_TEST_CHARS,
    MAX_GENERATED_TESTS,
    RuleJudgment,
    changed_line_locations,
    deterministic_feedback,
    gate_evidence_items,
    guarded_rule_judgment,
    guarded_test_proposal,
    is_transient_failure,
    validate_block_brief,
    validate_review_report,
)
from honest_ci.impact import ChangedSymbol
from honest_ci.rules import Rule

from .factories import (
    HOSTILE_NAME,
    HOSTILE_PROSE,
    ScriptedLLM,
    make_change,
    reaching_markdown,
    rendered_as_markdown,
)

FINGERPRINT = "fp-1"


def evidence() -> EvidencePack:
    return EvidencePack(run_id="r1", change=make_change(), fingerprint=FINGERPRINT)


def blocked_report() -> GateReport:
    return GateReport(
        gate="review",
        verdict=Verdict.BLOCKED,
        summary="金额计算使用 float",
        findings=[
            Finding(
                id="review.C-12",
                gate="review",
                rule_id="C-12",
                severity="high",
                title="金额计算使用 float，规范要求 decimal",
                file="src/app/refund.py",
                line_start=11,
                line_end=11,
            )
        ],
        tool_reports=[
            ToolReport(name="ruff", command="ruff check", available=False, passed=False)
        ],
    )


def test_changed_line_locations_tracks_added_lines() -> None:
    locations = changed_line_locations(make_change().diff)
    assert locations["src/app/refund.py"] == {11, 12, 13}


def test_evidence_items_cover_findings_and_tools_that_never_ran() -> None:
    items = gate_evidence_items(blocked_report())
    ids = [item.id for item in items]
    assert "review.C-12" in ids
    assert any(item_id.startswith("review.tool.ruff") for item_id in ids)
    assert any("never ran" in item.detail for item in items)


def test_an_evidence_item_on_a_removed_line_or_a_whole_file_says_where_it_is() -> None:
    removed = Finding(
        id="review.I-01.1",
        gate="review",
        rule_id="I-01",
        severity="medium",
        title="Tests do not disappear",
        file="tests/test_pricing.py",
        line_start=3,
        line_end=3,
        side="base",
    )
    whole_file = removed.model_copy(
        update={
            "id": "review.I-01.2",
            "file": "tests/test_old.py",
            "line_start": None,
            "line_end": None,
        }
    )
    report = GateReport(gate="review", verdict=Verdict.NEEDS_HUMAN, findings=[removed, whole_file])

    details = [item.detail for item in gate_evidence_items(report)]

    assert "(tests/test_pricing.py:3 (base))" in details[0]
    assert "(tests/test_old.py (base))" in details[1]


def test_deterministic_feedback_states_the_gate_is_authoritative() -> None:
    feedback = deterministic_feedback(blocked_report())
    assert "GATE BLOCKED: review" in feedback
    assert "Do not bypass" in feedback
    assert "[review.C-12]" in feedback


def test_brief_is_rejected_when_it_cites_evidence_the_gate_never_produced() -> None:
    items = gate_evidence_items(blocked_report())
    good = BlockBrief(
        gate="review",
        attempt=1,
        source_fingerprint="src-1",
        summary="float 用于金额",
        recommended_actions=["改用 Decimal"],
        cited_evidence_ids=[items[0].id],
    )
    assert validate_block_brief(good, "review", 1, "src-1", items)

    invented = good.model_copy(update={"cited_evidence_ids": ["review.made-up"]})
    assert not validate_block_brief(invented, "review", 1, "src-1", items)

    wrong_source = good.model_copy(update={"source_fingerprint": "other"})
    assert not validate_block_brief(wrong_source, "review", 1, "src-1", items)

    no_action = good.model_copy(update={"recommended_actions": []})
    assert not validate_block_brief(no_action, "review", 1, "src-1", items)


def model_report(**overrides: object) -> ModelReviewReport:
    payload: dict[str, object] = {
        "evidence_fingerprint": FINGERPRINT,
        "summary": "退款分摊逻辑有精度风险",
        "change_summary": "新增按明细分摊",
        "findings": [
            ModelFinding(
                rule_id="C-12",
                severity="high",
                title="float 用于金额分摊",
                file="src/app/refund.py",
                line_start=11,
                line_end=11,
                confidence=0.8,
            )
        ],
    }
    payload.update(overrides)
    return ModelReviewReport.model_validate(payload)


def test_review_report_must_be_bound_to_this_evidence() -> None:
    assert validate_review_report(model_report(), evidence())
    stale = model_report(evidence_fingerprint="fp-other")
    assert not validate_review_report(stale, evidence())


def test_review_finding_must_point_at_a_line_the_change_added() -> None:
    untouched_line = model_report(
        findings=[
            ModelFinding(
                title="unrelated",
                file="src/app/refund.py",
                line_start=99,
                line_end=99,
            )
        ]
    )
    assert not validate_review_report(untouched_line, evidence())

    untouched_file = model_report(
        findings=[ModelFinding(title="elsewhere", file="src/app/other.py", line_start=1, line_end=1)]
    )
    assert not validate_review_report(untouched_file, evidence())


# ── judgment-class rules ─────────────────────────────────────────────────────


def judgment_rule(**overrides: Any) -> Rule:
    payload: dict[str, Any] = {
        "id": "J-01",
        "title": "订单状态变更必须走状态机",
        "kind": "judgment",
        "severity": "high",
        "detail": "直接给 order.status 赋值绕开了状态机，退款以后的对账和通知都会漏掉。",
        "min_confidence": 0.7,
    }
    payload.update(overrides)
    return Rule(**payload)


def judge(
    rules: list[Rule] | None = None,
    **llm_kwargs: Any,
) -> tuple[ScriptedLLM, RuleJudgment | None]:
    """Run one judgment through the boundary, keeping the model and its answer."""

    llm = ScriptedLLM(**llm_kwargs)
    judgment = guarded_rule_judgment(
        make_change(),
        rules if rules is not None else [judgment_rule()],
        FINGERPRINT,
        llm,
        default_min_confidence=0.5,
    )
    return llm, judgment


def proposal(**overrides: Any) -> ModelFinding:
    payload: dict[str, Any] = {
        "rule_id": "J-01",
        "severity": "info",
        "title": "绕开了状态机",
        "detail": "直接赋值 item.refund。",
        "file": "src/app/refund.py",
        "line_start": 13,
        "line_end": 13,
        "confidence": 0.9,
    }
    payload.update(overrides)
    return ModelFinding.model_validate(payload)


def test_an_accepted_judgment_is_model_decided_and_does_not_block() -> None:
    _, judgment = judge(judgments=[proposal()])

    assert judgment is not None
    finding = judgment.findings[0]
    assert finding.decided_by == "model"
    assert finding.rule_id == "J-01"
    assert (finding.file, finding.line_start) == ("src/app/refund.py", 13)
    # The rule says high, and high is what a reviewer is told to weigh it at --
    # but a model finding never blocks, so the gate is free to route instead.
    assert finding.severity == "high"
    assert not finding.blocking


def test_a_model_cannot_raise_the_severity_its_rule_declared() -> None:
    _, judgment = judge(
        [judgment_rule(severity="low")],
        judgments=[proposal(severity="critical")],
    )

    assert judgment is not None
    assert judgment.findings[0].severity == "low"


def test_a_judgment_bound_to_another_change_is_discarded() -> None:
    _, judgment = judge(judgments=[proposal()], fingerprint="fp-other")

    # None, not an empty result: "nobody judged these rules" and "they were
    # judged and came back clean" must not read the same to the caller.
    assert judgment is None


def test_a_judgment_citing_a_rule_the_library_does_not_have_is_discarded() -> None:
    _, judgment = judge(judgments=[proposal(rule_id="J-99")])

    assert judgment is None


def test_a_judgment_finding_must_point_at_a_line_the_change_added() -> None:
    _, untouched_line = judge(judgments=[proposal(line_start=99, line_end=99)])
    _, untouched_file = judge(judgments=[proposal(file="src/app/other.py")])

    assert untouched_line is None
    assert untouched_file is None


def test_one_ungrounded_finding_discards_the_whole_judgment() -> None:
    _, judgment = judge(judgments=[proposal(), proposal(line_start=99, line_end=99)])

    assert judgment is None


def test_doubt_below_the_rules_threshold_is_recorded_but_is_not_a_finding() -> None:
    _, judgment = judge(judgments=[proposal(confidence=0.4)])

    assert judgment is not None
    assert judgment.findings == ()
    assert "(confidence 0.40 < 0.70)" in judgment.below_threshold[0]


def test_a_doubt_below_the_bar_names_its_file_as_code_it_cannot_close() -> None:
    change = make_change(
        diff=f"--- a/{HOSTILE_NAME}\n+++ b/{HOSTILE_NAME}\n@@ -12,0 +13 @@\n+x = 1\n",
        files_changed=[ChangedFile(path=HOSTILE_NAME, added_lines=1)],
    )
    judgment = guarded_rule_judgment(
        change,
        [judgment_rule()],
        FINGERPRINT,
        ScriptedLLM(judgments=[proposal(file=HOSTILE_NAME, confidence=0.4)]),
        default_min_confidence=0.5,
    )

    # The line lands in the merge request note's limits, under the name the
    # change gave the file.
    assert judgment is not None
    (doubt,) = judgment.below_threshold
    assert rendered_as_markdown(doubt, "Approved by security") == []


def test_the_models_reasoning_and_question_reach_the_finding_inert() -> None:
    _, judgment = judge(
        judgments=[proposal(detail=HOSTILE_PROSE, suggested_question=HOSTILE_PROSE)]
    )

    assert judgment is not None
    (finding,) = judgment.findings
    # The rule's own text is the deployment's, and it is kept as it was written.
    assert finding.detail.startswith(f"{judgment_rule().detail}\nThe model's reasoning: ")
    assert reaching_markdown(finding.detail) == []
    assert finding.suggested_fix is not None
    assert reaching_markdown(finding.suggested_fix) == []


def test_a_rules_own_threshold_overrides_the_deployment_default() -> None:
    _, judgment = judge([judgment_rule(min_confidence=None)], judgments=[proposal(confidence=0.6)])

    # 0.6 clears the deployment default of 0.5 once the rule stops asking for 0.7.
    assert judgment is not None
    assert len(judgment.findings) == 1


def test_a_failed_model_call_leaves_the_rules_unjudged_without_raising() -> None:
    llm, judgment = judge(error=RuntimeError("provider exploded"))

    assert judgment is None
    # The call is still on the record: a run that spent tokens on output nobody
    # kept has to account for them.
    assert llm.call_records[0].schema_valid is False


def test_the_model_is_only_shown_lines_inside_the_rules_scope() -> None:
    llm, judgment = judge([judgment_rule(paths=("src/app/other.py",))])

    assert judgment is not None
    assert llm.judged == [("J-01",)]
    assert llm.locations == []


def test_no_judgment_rules_means_no_model_call_at_all() -> None:
    llm, judgment = judge([])

    assert judgment is not None
    assert judgment.findings == ()
    assert llm.call_records == []


def test_transient_failures_are_recognised() -> None:
    assert is_transient_failure("HTTP 503 service unavailable")
    assert is_transient_failure("Connection reset by peer")
    assert not is_transient_failure("Model output failed BlockBrief validation")


# ── generated regression cases ───────────────────────────────────────────────

PYTHON_CASE = "def test_split_refund_divides_evenly():\n    assert True\n"


def symbol(**overrides: Any) -> ChangedSymbol:
    payload: dict[str, Any] = {
        "file": "src/app/refund.py",
        "name": "split_refund",
        "kind": "function",
        "line": 10,
    }
    payload.update(overrides)
    return ChangedSymbol(**payload)


def propose(
    tmp_path: Path,
    tests: list[ProposedTest] | None = None,
    *,
    language: str = "python",
    **llm_kwargs: Any,
) -> tuple[ScriptedLLM, ProposedTests | None]:
    llm = ScriptedLLM(
        tests=[ProposedTest(path="tests/honest_ci_generated/test_refund.py", content=PYTHON_CASE, targets=["split_refund"])]
        if tests is None
        else tests,
        **llm_kwargs,
    )
    proposal = guarded_test_proposal(
        make_change(),
        [symbol()],
        FINGERPRINT,
        llm,
        language=language,
        workspace=tmp_path,
    )
    return llm, proposal


def test_a_grounded_proposal_is_kept_with_what_the_model_was_asked(tmp_path: Path) -> None:
    llm, proposal = propose(tmp_path)

    assert proposal is not None
    assert [test.path for test in proposal.tests] == ["tests/honest_ci_generated/test_refund.py"]
    assert llm.testgen_symbols == [("split_refund",)]
    assert llm.testgen_directory == "tests/honest_ci_generated"


def test_a_case_written_outside_the_generated_directory_discards_everything(
    tmp_path: Path,
) -> None:
    _, proposal = propose(
        tmp_path,
        [
            ProposedTest(path="tests/honest_ci_generated/test_ok.py", content=PYTHON_CASE, targets=["split_refund"]),
            ProposedTest(path="src/app/refund.py", content=PYTHON_CASE, targets=["split_refund"]),
        ],
    )

    # Whole proposal, not just the offending case: a model that chose where a
    # file lands has not earned the benefit of the doubt on the other one.
    assert proposal is None


def test_a_path_that_climbs_out_of_the_workspace_is_refused(tmp_path: Path) -> None:
    _, proposal = propose(
        tmp_path,
        [
            ProposedTest(
                path="tests/honest_ci_generated/../../../etc/test_x.py",
                content=PYTHON_CASE,
                targets=["split_refund"],
            )
        ],
    )

    assert proposal is None


def test_a_case_that_would_overwrite_an_existing_file_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "tests" / "honest_ci_generated" / "test_refund.py"
    target.parent.mkdir(parents=True)
    target.write_text("def test_the_one_that_was_already_here():\n    assert False\n", "utf-8")

    _, proposal = propose(tmp_path)

    # The file that is already there might be the very test that would have
    # caught this change. Generation never gets to replace it.
    assert proposal is None


def test_a_case_aimed_at_a_symbol_the_change_never_touched_is_refused(
    tmp_path: Path,
) -> None:
    _, proposal = propose(
        tmp_path,
        [
            ProposedTest(
                path="tests/honest_ci_generated/test_refund.py",
                content=PYTHON_CASE,
                targets=["charge_card"],
            )
        ],
    )

    assert proposal is None


def test_a_case_that_does_not_parse_is_refused(tmp_path: Path) -> None:
    _, proposal = propose(
        tmp_path,
        [
            ProposedTest(
                path="tests/honest_ci_generated/test_refund.py",
                content="def test_broken(:\n",
                targets=["split_refund"],
            )
        ],
    )

    assert proposal is None


def test_a_case_larger_than_a_regression_case_has_any_reason_to_be_is_refused(
    tmp_path: Path,
) -> None:
    body = "def test_big():\n" + "    assert True\n" * 3000

    _, proposal = propose(
        tmp_path,
        [
            ProposedTest(
                path="tests/honest_ci_generated/test_refund.py",
                content=body,
                targets=["split_refund"],
            )
        ],
    )

    assert len(body) > MAX_GENERATED_TEST_CHARS
    assert proposal is None


def test_more_cases_than_the_cap_are_refused(tmp_path: Path) -> None:
    _, proposal = propose(
        tmp_path,
        [
            ProposedTest(
                path=f"tests/honest_ci_generated/test_{index}.py",
                content=PYTHON_CASE,
                targets=["split_refund"],
            )
            for index in range(MAX_GENERATED_TESTS + 1)
        ],
    )

    assert proposal is None


def test_a_proposal_bound_to_another_change_is_discarded(tmp_path: Path) -> None:
    _, proposal = propose(tmp_path, fingerprint="fp-other")

    assert proposal is None


def test_a_failed_model_call_produces_no_files_and_does_not_raise(tmp_path: Path) -> None:
    _, proposal = propose(tmp_path, error=RuntimeError("provider down"))

    assert proposal is None


def test_a_java_case_must_declare_the_class_its_file_is_named_after(
    tmp_path: Path,
) -> None:
    good = ProposedTest(
        path="src/test/java/honest_ci/generated/RefundTest.java",
        content="package honest_ci.generated;\nclass RefundTest { void t() {} }\n",
        targets=["split_refund"],
    )
    mismatched = good.model_copy(
        update={"content": "package honest_ci.generated;\nclass Other { void t() {} }\n"}
    )
    unbalanced = good.model_copy(
        update={"content": "package honest_ci.generated;\nclass RefundTest { void t() {\n"}
    )

    assert propose(tmp_path, [good], language="java")[1] is not None
    assert propose(tmp_path, [mismatched], language="java")[1] is None
    assert propose(tmp_path, [unbalanced], language="java")[1] is None


def test_a_language_with_nowhere_to_put_generated_cases_is_never_asked(
    tmp_path: Path,
) -> None:
    llm, proposal = propose(tmp_path, language="rust")

    assert proposal is None
    assert llm.testgen_symbols == []


def test_a_change_with_no_symbols_is_never_asked(tmp_path: Path) -> None:
    llm = ScriptedLLM()

    proposal = guarded_test_proposal(
        make_change(), [], FINGERPRINT, llm, language="python", workspace=tmp_path
    )

    assert proposal is None
    assert llm.testgen_symbols == []
