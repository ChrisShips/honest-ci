from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import pytest

from honest_ci.adapters.base import AdapterSet
from honest_ci.contracts import GATE_SEQUENCE, EvidencePack, Finding, Verdict
from honest_ci.runner import GateRunner
from honest_ci.settings import Settings
from honest_ci.state import RunState
from honest_ci.storage import SQLiteLedger

from .factories import ScriptedGate, ScriptedLLM, headings, make_change, section, table_row

T = TypeVar("T")


def present(value: T | None) -> T:
    """``value``, asserted to be there before anything reads through it.

    ``RunState``'s optional fields and the ledger's lookups are ``None`` for
    reasons a test should state rather than dereference past: "the run shipped
    no evidence pack" and "the pack says the wrong thing" are different
    failures, and only the assertion here tells them apart.
    """

    assert value is not None
    return value


def build(
    tmp_path: Path,
    llm: ScriptedLLM | None = None,
    model_explain: bool = True,
    disabled_gates: tuple[str, ...] = (),
    model_provider: str | None = None,
    **scripted: dict[str, object],
) -> GateRunner:
    """A runner whose gates are scripted; retries never sleep in tests."""

    settings = Settings(
        sandbox_mode="process",
        retry_backoff_seconds=(0.0,),
        max_attempts=3,
        model_explain=model_explain,
        disabled_gates=disabled_gates,
        model_provider=model_provider,
    )
    runner = GateRunner(
        workspace_root=tmp_path,
        settings=settings,
        adapters=AdapterSet(ledger=SQLiteLedger(tmp_path / settings.ledger_path), llm=llm),
    )
    runner.gates = {
        name: ScriptedGate(runner.services, name=name, **kwargs)  # type: ignore[arg-type]
        for name, kwargs in scripted.items()
    }
    return runner


def all_passing(tmp_path: Path, model_provider: str | None = None) -> GateRunner:
    gates: dict[str, dict[str, object]] = {name: {} for name in GATE_SEQUENCE}
    # mypy cannot see that no gate is named after one of build's own keyword
    # parameters, so it checks every gate's script against each of them.
    return build(tmp_path, model_provider=model_provider, **gates)  # type: ignore[arg-type]


def test_a_clean_change_runs_every_gate_and_ships_evidence(tmp_path: Path) -> None:
    runner = all_passing(tmp_path)
    state = runner.verify_change(make_change())

    assert state.status == "mergeable"
    assert [report.gate for report in present(state.evidence).gate_reports] == list(GATE_SEQUENCE)
    assert present(state.evidence).verdict is Verdict.PASS
    assert state.metrics.gates_run == len(GATE_SEQUENCE)
    assert state.metrics.ai_generated_ratio == 0.71

    evidence_file = runner.artifact_store.run_dir(state.run_id) / "evidence.json"
    assert json.loads(evidence_file.read_text(encoding="utf-8"))["run_id"] == state.run_id

    record = runner.services.ledger.load_run_record(state.run_id)
    assert record is not None
    assert record.outcome == "mergeable"
    assert record.change_ref.startswith("team/orders#482@")


def test_a_block_stops_the_sequence_and_explains_itself(tmp_path: Path) -> None:
    runner = build(
        tmp_path,
        spec={},
        review={
            "verdicts": [Verdict.BLOCKED],
            "findings": [
                Finding(
                    id="review.C-12",
                    gate="review",
                    rule_id="C-12",
                    severity="high",
                    title="金额计算使用 float",
                )
            ],
        },
        regression={},
        metrics={},
    )
    state = runner.verify_change(make_change())

    assert state.status == "blocked"
    assert "GATE BLOCKED: review" in (state.last_feedback or "")
    assert "[review.C-12]" in (state.last_feedback or "")
    # Downstream gates never ran, and the evidence pack says so rather than
    # implying they passed.
    assert "regression" not in state.gate_reports
    assert present(state.evidence).verdict is Verdict.BLOCKED
    assert state.metrics.blocking_findings == 1
    assert present(runner.services.ledger.load_run_record(state.run_id)).outcome == "blocked"


BLOCKING_FINDING = Finding(
    id="review.C-12",
    gate="review",
    rule_id="C-12",
    severity="high",
    title="金额计算使用 float",
    file="src/app/refund.py",
    line_start=11,
)


def blocking_review(tmp_path: Path, llm: ScriptedLLM | None = None, **extra: object) -> GateRunner:
    return build(
        tmp_path,
        llm=llm,
        **extra,  # type: ignore[arg-type]
        spec={},
        review={"verdicts": [Verdict.BLOCKED], "findings": [BLOCKING_FINDING]},
        regression={},
        metrics={},
    )


def test_a_block_is_explained_by_the_model_without_changing_the_verdict(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    state = blocking_review(tmp_path, llm).verify_change(make_change())

    assert state.status == "blocked"  # the model does not soften a verdict
    assert llm.explained == [("review", 1)]
    assert [brief.gate for brief in state.briefs] == ["review"]

    feedback = state.last_feedback or ""
    # Deterministic evidence first, model text after it and labelled advisory.
    assert feedback.index("GATE BLOCKED: review") < feedback.index("MODEL BRIEF")
    assert "[review.C-12]" in feedback

    assert present(state.evidence).briefs[0].summary.startswith("模型解释")
    assert present(state.evidence).llm_calls[0].schema_name == "BlockBrief"
    assert state.metrics.token_usage["total_tokens"] == 120


def test_the_evidence_file_on_disk_parses_back_into_an_evidence_pack(tmp_path: Path) -> None:
    runner = blocking_review(tmp_path, ScriptedLLM())
    state = runner.verify_change(make_change())

    # ``evidence.json`` is the machine-readable half of what a reviewer is
    # handed, so redaction has to leave it loadable by the contract that
    # produced it. Token counters are where that used to break: they carry a
    # credential word in their name but hold integers.
    written = json.loads(
        (runner.artifact_store.run_dir(state.run_id) / "evidence.json").read_text(encoding="utf-8")
    )
    parsed = EvidencePack.model_validate(written)

    assert parsed.run_id == state.run_id
    assert parsed.metrics.token_usage == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert parsed.llm_calls[0].total_tokens == 120


def test_a_brief_citing_evidence_the_gate_never_produced_is_dropped(tmp_path: Path) -> None:
    llm = ScriptedLLM(cite=["review.invented"])
    state = blocking_review(tmp_path, llm).verify_change(make_change())

    assert llm.explained  # it was asked
    assert state.briefs == []  # and not believed
    assert "MODEL BRIEF" not in (state.last_feedback or "")
    assert "GATE BLOCKED: review" in (state.last_feedback or "")


def test_a_model_failure_leaves_the_block_and_its_evidence_intact(tmp_path: Path) -> None:
    llm = ScriptedLLM(error=ConnectionError("provider unreachable"))
    state = blocking_review(tmp_path, llm).verify_change(make_change())

    # Fail-soft: an advisory paragraph is never worth failing a run over, and a
    # blocked change stays blocked with the evidence code produced.
    assert state.status == "blocked"
    assert state.briefs == []
    assert "[review.C-12]" in (state.last_feedback or "")
    # The attempt is still accounted for: tokens spent on discarded text are
    # tokens spent.
    assert present(state.evidence).llm_calls[0].schema_valid is False


def test_explanations_can_be_turned_off_without_touching_the_verdicts(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    state = blocking_review(tmp_path, llm, model_explain=False).verify_change(make_change())

    assert llm.explained == []  # nothing left the deployment
    assert state.status == "blocked"
    assert "[review.C-12]" in (state.last_feedback or "")


def test_a_human_handoff_carries_a_reading_aid_bound_to_its_evidence(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    runner = build(tmp_path, llm=llm, spec={"verdicts": [Verdict.NEEDS_HUMAN]}, review={})
    state = runner.verify_change(make_change())

    ticket = state.approval_ticket
    assert ticket is not None
    assert ticket.assistant_report is not None
    assert ticket.assistant_report.evidence_fingerprint == present(state.evidence).fingerprint
    # The call the reading aid cost is inside the evidence it was made for.
    assert [call.schema_name for call in present(state.evidence).llm_calls] == ["ModelReviewReport"]


def test_a_reading_aid_bound_to_other_evidence_is_dropped(tmp_path: Path) -> None:
    llm = ScriptedLLM(fingerprint="fp-somewhere-else")
    runner = build(tmp_path, llm=llm, spec={"verdicts": [Verdict.NEEDS_HUMAN]}, review={})
    state = runner.verify_change(make_change())

    assert llm.reviewed  # asked
    assert state.approval_ticket is not None
    assert present(state.approval_ticket).assistant_report is None  # not believed
    assert present(state.approval_ticket).fingerprint == present(state.evidence).fingerprint


def test_a_rerun_replaces_the_explanation_it_re_ran(tmp_path: Path) -> None:
    runner = blocking_review(tmp_path, ScriptedLLM())
    state = runner.verify_change(make_change())
    assert len(state.briefs) == 1

    reran = runner.rerun_gate(state.run_id, "review", reason="规则改了")

    # Two explanations of the same gate would leave a reader guessing which one
    # describes the verdict they are looking at.
    assert [brief.gate for brief in reran.briefs] == ["review"]
    assert present(reran.evidence).briefs == reran.briefs


def test_the_evidence_pack_ships_with_a_readable_rendering(tmp_path: Path) -> None:
    runner = blocking_review(tmp_path, ScriptedLLM())
    state = runner.verify_change(make_change())

    summary = runner.summary_path(state)
    assert summary is not None and summary.is_file()
    text = summary.read_text(encoding="utf-8")
    assert headings(text)[0].startswith("Evidence pack")
    assert "C-12" in section(text, "Blocking findings")
    assert "Why gate review blocked (attempt 1)" in headings(text)

    # It is an artifact like any other: in the manifest, with its own digest.
    names = {record.name for record in present(state.artifact_manifest).artifacts}
    assert {"evidence.json", "evidence.md"} <= names


def test_needs_human_waits_for_a_decision_bound_to_the_evidence(tmp_path: Path) -> None:
    runner = build(
        tmp_path,
        spec={"verdicts": [Verdict.NEEDS_HUMAN]},
        review={},
        regression={},
        metrics={},
    )
    state = runner.verify_change(make_change())

    assert state.status == "awaiting_human"
    assert state.approval_ticket is not None
    assert present(state.approval_ticket).fingerprint == present(state.evidence).fingerprint

    # A crashed-process resume must not be able to skip the approval.
    with pytest.raises(ValueError, match="approval"):
        runner.resume_run(state.run_id)

    approved = runner.resume_approval(
        state.run_id,
        approved=True,
        reason="产品确认口径",
        reviewer="chris",
        review_minutes=6.0,
    )
    assert approved.status == "mergeable"
    assert present(approved.approval_decision).approved
    assert approved.metrics.human_review_minutes == 6.0

    stored = present(runner.services.ledger.load_approval(state.run_id))
    assert stored["approved"] == 1
    assert stored["reviewer"] == "chris"


def test_a_stale_approval_is_refused(tmp_path: Path) -> None:
    runner = build(tmp_path, spec={"verdicts": [Verdict.NEEDS_HUMAN]})
    state = runner.verify_change(make_change())

    # Someone pushed a new commit: the evidence in the ledger no longer matches
    # what the reviewer read.
    moved = present(state.evidence).model_copy(update={"fingerprint": "fp-new"})
    runner.services.ledger.save_evidence(state.run_id, moved, "fp-new")

    with pytest.raises(ValueError, match="Stale approval"):
        runner.resume_approval(state.run_id, approved=True)


REVIEW_URL = "https://github.com/octo-org/orders/pull/7#pullrequestreview-1"


def waiting_on_a_person(tmp_path: Path) -> tuple[GateRunner, RunState]:
    runner = build(
        tmp_path,
        spec={"verdicts": [Verdict.NEEDS_HUMAN]},
        review={},
        regression={},
        metrics={},
    )
    state = runner.verify_change(make_change())
    assert state.status == "awaiting_human"
    return runner, state


def test_a_decision_is_written_beside_the_evidence_it_was_given_on(tmp_path: Path) -> None:
    runner, state = waiting_on_a_person(tmp_path)
    evidence = present(state.evidence)
    run_dir = runner.artifact_store.run_dir(state.run_id)
    given = datetime(2026, 9, 1, 10, 5, tzinfo=UTC)

    approved = runner.resume_approval(
        state.run_id,
        approved=True,
        reason="Checked the rounding with finance.",
        reviewer="octo-lead",
        fingerprint=evidence.fingerprint,
        url=REVIEW_URL,
        decided_at=given,
    )

    # A run in CI loses its ledger with the job; the artifacts are what is left.
    recorded = json.loads((run_dir / "approval.json").read_text(encoding="utf-8"))
    assert recorded["approved"] is True
    assert recorded["fingerprint"] == evidence.fingerprint
    assert (recorded["reviewer"], recorded["url"]) == ("octo-lead", REVIEW_URL)
    assert datetime.fromisoformat(recorded["decided_at"]) == given
    decision = present(approved.approval_decision)
    assert (decision.url, decision.decided_at) == (REVIEW_URL, given)


def test_the_reading_of_an_approved_run_shows_the_decision_and_the_manifest_vouches_for_it(
    tmp_path: Path,
) -> None:
    runner, state = waiting_on_a_person(tmp_path)
    run_dir = runner.artifact_store.run_dir(state.run_id)
    gates_wrote = (run_dir / "evidence.json").read_bytes()

    approved = runner.resume_approval(
        state.run_id, approved=True, reason="Checked with finance.", reviewer="octo-lead"
    )

    text = present(runner.summary_path(approved)).read_text(encoding="utf-8")
    assert "- Decision: **approved by a person**" in section(text, "Human decision")
    assert table_row(text, "Outcome")[0].startswith("**approved**")
    # What the gates found is theirs: a signature given afterwards is not part of it.
    assert (run_dir / "evidence.json").read_bytes() == gates_wrote

    manifest = present(approved.artifact_manifest)
    assert present(approved.evidence).artifact_manifest == manifest
    assert present(runner.services.ledger.load_evidence(state.run_id)).artifact_manifest == manifest
    for record in manifest.artifacts:
        content = (runner.artifact_store.root / record.path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == record.sha256, record.name
    assert {"evidence.json", "evidence.md", "approval.json"} <= {
        record.name for record in manifest.artifacts
    }
    # Like the copy written with the run, the reading lists what it was made
    # from and not itself.
    listed = section(text, "Artifacts")
    assert "`approval.json`" in listed and "`evidence.json`" in listed
    assert "| `evidence.md` |" not in listed


def test_a_rejection_is_written_beside_the_evidence_too(tmp_path: Path) -> None:
    runner, state = waiting_on_a_person(tmp_path)

    rejected = runner.resume_approval(state.run_id, approved=False, reason="Needs a rounding test.")

    assert rejected.status == "blocked"
    run_dir = runner.artifact_store.run_dir(state.run_id)
    assert json.loads((run_dir / "approval.json").read_text(encoding="utf-8"))["approved"] is False
    text = present(runner.summary_path(rejected)).read_text(encoding="utf-8")
    assert "- Decision: **rejected by a person**" in section(text, "Human decision")
    assert table_row(text, "Outcome")[0].startswith("**blocked**")


def test_a_decision_given_on_other_evidence_is_refused_and_leaves_the_run_waiting(
    tmp_path: Path,
) -> None:
    runner, state = waiting_on_a_person(tmp_path)

    with pytest.raises(ValueError, match="Stale approval"):
        runner.resume_approval(
            state.run_id, approved=True, reason="Looks right.", fingerprint="fp-other"
        )

    assert present(runner.services.ledger.load_state(state.run_id)).status == "awaiting_human"
    assert not (runner.artifact_store.run_dir(state.run_id) / "approval.json").exists()


def test_a_transient_failure_is_retried_within_budget(tmp_path: Path) -> None:
    runner = build(
        tmp_path,
        spec={"error": ConnectionError("upstream reset"), "error_times": 1},
        review={},
        regression={},
        metrics={},
    )
    state = runner.verify_change(make_change())

    assert state.status == "mergeable"
    assert state.attempts["spec"] == 2
    exceptions = [item for item in state.execution_history if item.outcome == "exception"]
    assert exceptions[0].route == "retry_current"
    assert exceptions[0].failure_class == "transient_exception"


def test_a_fatal_failure_ends_the_run_with_a_reason(tmp_path: Path) -> None:
    runner = build(
        tmp_path,
        spec={"error": ValueError("rule file is malformed"), "error_times": 9},
        review={},
    )
    state = runner.verify_change(make_change())

    assert state.status == "failed"
    record = present(runner.services.ledger.load_run_record(state.run_id))
    assert record.outcome == "failed"
    assert "rule file is malformed" in record.failure_reasons[0]


def test_an_undeployed_gate_is_skipped_never_passed(tmp_path: Path) -> None:
    runner = build(tmp_path, spec={}, review={})
    state = runner.verify_change(make_change())

    regression = state.gate_reports["regression"]
    assert regression.verdict is Verdict.SKIPPED
    assert "has no regression gate" in regression.summary
    assert "not a pass" in regression.summary
    assert state.metrics.gates_run == 2  # skipped gates are not counted as run

    # The policy still requires regression and nothing ran it, so two passing
    # gates do not add up to a verified change.
    assert state.status == "awaiting_human"
    assert "regression" in present(state.approval_ticket).evidence_summary


def test_a_disabled_gate_is_skipped_never_passed_and_never_run(tmp_path: Path) -> None:
    runner = build(
        tmp_path,
        disabled_gates=("regression",),
        **{name: {} for name in GATE_SEQUENCE},  # type: ignore[arg-type]
    )
    state = runner.verify_change(make_change())

    regression = state.gate_reports["regression"]
    assert regression.verdict is Verdict.SKIPPED
    assert "switched the regression gate off" in regression.summary
    assert "not a pass" in regression.summary
    # Switched off means not asked, not asked-and-ignored.
    assert runner.gates["regression"].calls == 0  # type: ignore[attr-defined]
    assert state.metrics.gates_run == len(GATE_SEQUENCE) - 1
    # An operator switch does not rewrite the policy: regression is still
    # required, still has no conclusion, and the change still waits for a person.
    assert state.status == "awaiting_human"


def test_a_partial_deployment_is_mergeable_only_when_the_policy_says_so(tmp_path: Path) -> None:
    # Deploying one gate makes the installation accountable for one gate. That
    # narrowing is a line the team writes in their own policy file, where it
    # can be audited -- never something the code decides for them.
    (tmp_path / "honest-ci-policy.toml").write_text(
        """
        [policy]
        version = "v1"
        required_checks = ["review"]
        """,
        encoding="utf-8",
    )
    runner = build(tmp_path, review={})
    state = runner.verify_change(make_change())

    assert state.status == "mergeable"
    assert state.gate_reports["spec"].verdict is Verdict.SKIPPED


def test_a_required_check_that_no_deployment_can_run_still_ends_at_a_human(
    tmp_path: Path,
) -> None:
    (tmp_path / "honest-ci-policy.toml").write_text(
        """
        [policy]
        version = "v1"
        required_checks = ["review", "deploy"]
        """,
        encoding="utf-8",
    )
    runner = build(tmp_path, **{name: {} for name in GATE_SEQUENCE})  # type: ignore[arg-type]
    state = runner.verify_change(make_change())

    # A policy naming a gate this product does not have is a misconfiguration,
    # and an unsatisfiable requirement must fail towards a human.
    assert state.status == "awaiting_human"
    assert "deploy" in present(state.approval_ticket).evidence_summary


def test_rerunning_a_gate_invalidates_what_was_derived_from_it(tmp_path: Path) -> None:
    runner = build(
        tmp_path,
        spec={},
        review={"verdicts": [Verdict.BLOCKED, Verdict.PASS]},
        regression={},
        metrics={},
    )
    first = runner.verify_change(make_change())
    assert first.status == "blocked"

    second = runner.rerun_gate(first.run_id, "review", reason="规则误报，已修正")

    assert second.status == "mergeable"
    assert second.gate_reports["review"].verdict is Verdict.PASS
    assert present(second.evidence).verdict is Verdict.PASS
    assert second.attempts["spec"] == 1  # upstream gate was not re-run


def test_status_and_inspect_read_the_ledger(tmp_path: Path) -> None:
    runner = all_passing(tmp_path)
    state = runner.verify_change(make_change())

    loaded = runner.status(state.run_id)
    assert isinstance(loaded, RunState)
    assert loaded.status == "mergeable"

    payload = runner.inspect(state.run_id)
    assert payload["run_record"]["outcome"] == "mergeable"
    assert payload["artifact_manifest"]["run_id"] == state.run_id
    assert runner.status("missing") is None


def test_the_run_record_points_at_the_evidence_pack_itself(tmp_path: Path) -> None:
    runner = all_passing(tmp_path)
    state = runner.verify_change(make_change())

    # A reader of the ledger follows this field to the artifact on disk, so it has to be
    # the evidence pack and not the ledger that indexes it.
    record = runner.services.ledger.load_run_record(state.run_id)
    assert record is not None
    assert record.evidence_path is not None
    evidence = Path(record.evidence_path)
    assert evidence == runner.evidence_path(state)
    assert json.loads(evidence.read_text(encoding="utf-8"))["run_id"] == state.run_id


def test_a_run_with_no_evidence_yet_has_no_evidence_path(tmp_path: Path) -> None:
    runner = all_passing(tmp_path)

    assert runner.evidence_path(runner.build_state(make_change())) is None


def test_unknown_gate_names_are_rejected(tmp_path: Path) -> None:
    runner = all_passing(tmp_path)
    state = runner.verify_change(make_change())
    with pytest.raises(ValueError, match="Unknown gate"):
        runner.rerun_gate(state.run_id, "deploy")


def test_a_change_outside_automated_judgment_ends_at_a_human(tmp_path: Path) -> None:
    from honest_ci.contracts import ChangedFile

    runner = build(tmp_path, **{name: {} for name in GATE_SEQUENCE})  # type: ignore[arg-type]
    change = make_change(files_changed=[ChangedFile(path="src/payment/refund.py", added_lines=4)])

    state = runner.verify_change(change)

    # Every gate passed, but the change touches a sensitive path: "nothing
    # blocked it" must not become "merge it".
    assert all(report.verdict is Verdict.PASS for report in state.gate_reports.values())
    assert state.status == "awaiting_human"
    # The reviewer is told why it reached them: the guardrail, and the path that tripped it.
    summary = present(state.approval_ticket).evidence_summary
    assert all(item in summary for item in present(state.policy_decision).violations)
    assert "src/payment/refund.py" in summary


def test_a_deployment_with_no_gates_cannot_declare_a_change_mergeable(tmp_path: Path) -> None:
    runner = build(tmp_path)  # no gate implementations at all
    state = runner.verify_change(make_change())

    assert state.metrics.gates_run == 0
    assert state.status == "awaiting_human"
    # The reviewer is told that every gate was skipped, so nothing automated stands behind it.
    summary = present(state.approval_ticket).evidence_summary
    assert all(f"{gate}: skipped" in summary for gate in state.gate_reports)
    assert "no gate" in summary


def test_a_resume_can_re_point_the_run_at_a_tree_that_came_back(tmp_path: Path) -> None:
    """A checkout is a cache; a restart elsewhere does not have the old one.

    Without re-pointing, the resumed gates would scan whatever happened to be
    at the deployment root and the evidence pack would name a tree nobody
    verified against.
    """

    runner = build(tmp_path, review={"verdicts": [Verdict.PASS]})
    state = runner.build_state(make_change()).model_copy(
        update={"workspace_path": str(tmp_path / "gone"), "current_gate": "review"}
    )
    runner.services.ledger.save_state(state)

    resumed = runner.resume_run(state.run_id, workspace_path=str(tmp_path / "fresh"))

    assert resumed.workspace_path == str(tmp_path / "fresh")


def test_a_resume_that_is_not_told_otherwise_keeps_the_recorded_tree(tmp_path: Path) -> None:
    runner = build(tmp_path, review={"verdicts": [Verdict.PASS]})
    state = runner.build_state(make_change()).model_copy(
        update={"workspace_path": str(tmp_path / "checkout"), "current_gate": "review"}
    )
    runner.services.ledger.save_state(state)

    resumed = runner.resume_run(state.run_id)

    assert resumed.workspace_path == str(tmp_path / "checkout")


# ── annotations: a person's reading of a finding, never a verdict ────────────


def test_annotating_a_false_positive_leaves_the_verdict_alone(tmp_path: Path) -> None:
    runner = blocking_review(tmp_path)
    run_id = runner.verify_change(make_change()).run_id

    state = runner.annotate_finding(
        run_id, "review.C-12", false_positive=True, reason="日志里的 float", annotator="chris"
    )

    assert state.status == "blocked"  # the block stands
    assert state.gate_reports["review"].verdict is Verdict.BLOCKED
    assert present(state.evidence).findings[0].false_positive is True
    assert state.metrics.false_positives == 1
    assert present(state.evidence).metrics.false_positives == 1

    ledger = runner.services.ledger
    note = ledger.list_annotations(run_id)[0]
    assert note.fingerprint == present(state.evidence).fingerprint
    assert (note.false_positive, note.reason, note.annotator) == (True, "日志里的 float", "chris")
    assert present(ledger.load_evidence(run_id)).findings[0].false_positive is True
    assert present(ledger.load_run_record(run_id)).metrics.false_positives == 1
    # And the checkpoint agrees, so ``honest-ci evidence`` renders the flag.
    assert present(present(runner.status(run_id)).evidence).findings[0].false_positive is True


def test_confirming_a_finding_counts_as_annotated_but_not_as_a_false_positive(
    tmp_path: Path,
) -> None:
    runner = blocking_review(tmp_path)
    run_id = runner.verify_change(make_change()).run_id

    state = runner.annotate_finding(run_id, "review.C-12", false_positive=False)

    assert present(state.evidence).findings[0].false_positive is False
    # Zero, not None: somebody looked, and found nothing wrong with the finding.
    assert state.metrics.false_positives == 0


def test_annotating_needs_a_finding_the_run_produced(tmp_path: Path) -> None:
    runner = blocking_review(tmp_path)
    run_id = runner.verify_change(make_change()).run_id

    with pytest.raises(ValueError, match="no finding 'review.X-1'"):
        runner.annotate_finding(run_id, "review.X-1", false_positive=True)


def test_annotating_a_run_without_a_verdict_is_refused(tmp_path: Path) -> None:
    runner = blocking_review(tmp_path)
    state = runner.build_state(make_change())
    runner.services.ledger.save_state(state)

    with pytest.raises(ValueError, match="no evidence pack"):
        runner.annotate_finding(state.run_id, "review.C-12", false_positive=True)
    with pytest.raises(ValueError, match="not reached a verdict"):
        runner.record_review_minutes(state.run_id, 3)


def test_review_minutes_can_be_recorded_on_a_run_nobody_had_to_approve(tmp_path: Path) -> None:
    """Confirming a block costs minutes too, and ``approve`` cannot reach a blocked run."""

    runner = blocking_review(tmp_path)
    run_id = runner.verify_change(make_change()).run_id

    state = runner.record_review_minutes(run_id, 7.5)

    assert state.status == "blocked"
    assert state.metrics.human_review_minutes == 7.5
    assert present(
        runner.services.ledger.load_run_record(run_id)
    ).metrics.human_review_minutes == 7.5


JUDGMENT_RULE = """
version = "v4"

[[rule]]
id = "J-01"
title = "订单状态变更必须走状态机"
detail = "直接给状态字段赋值会绕开状态机。"
severity = "high"
kind = "judgment"
paths = ["src/**/*.py"]
"""


def test_a_judgment_rule_call_is_accounted_for_in_the_runs_evidence(tmp_path: Path) -> None:
    """The gate spends the tokens; the run has to be the thing that accounts for them."""

    (tmp_path / "honest-ci-rules.toml").write_text(JUDGMENT_RULE, encoding="utf-8")
    llm = ScriptedLLM(judgments=[])
    settings = Settings(sandbox_mode="process", retry_backoff_seconds=(0.0,))
    runner = GateRunner(
        workspace_root=tmp_path,
        settings=settings,
        adapters=AdapterSet(ledger=SQLiteLedger(tmp_path / settings.ledger_path), llm=llm),
    )
    state = runner.verify_change(make_change())

    assert llm.judged == [("J-01",)]
    # The rule file owns its version, and that version reaches the fingerprint
    # an approval will be bound to.
    assert state.evidence is not None
    assert present(state.evidence).rule_version == "v4"
    schemas = [call.schema_name for call in present(state.evidence).llm_calls]
    assert "ModelReviewReport" in schemas
    calls = json.loads((runner.artifact_store.run_dir(state.run_id) / "llm-calls.json").read_text())
    assert len(calls) == len(present(state.evidence).llm_calls)


def test_turning_explanations_off_does_not_turn_judgment_off(tmp_path: Path) -> None:
    """The two switches are separate because they send different things."""

    (tmp_path / "honest-ci-rules.toml").write_text(JUDGMENT_RULE, encoding="utf-8")
    llm = ScriptedLLM(judgments=[])
    settings = Settings(
        sandbox_mode="process", retry_backoff_seconds=(0.0,), model_explain=False
    )
    runner = GateRunner(
        workspace_root=tmp_path,
        settings=settings,
        adapters=AdapterSet(ledger=SQLiteLedger(tmp_path / settings.ledger_path), llm=llm),
    )
    runner.verify_change(make_change())

    assert llm.judged == [("J-01",)]
    assert llm.explained == []


# ── the configuration snapshot ───────────────────────────────────────────────


def test_the_pack_records_the_rule_and_policy_files_that_judged_the_change(
    tmp_path: Path,
) -> None:
    rules = tmp_path / "honest-ci-rules.toml"
    rules.write_text(JUDGMENT_RULE, encoding="utf-8")
    policy = tmp_path / "honest-ci-policy.toml"
    policy.write_text('version = "v9"\nrequired_checks = ["spec", "review"]\n', encoding="utf-8")

    state = all_passing(tmp_path).verify_change(make_change())

    config = present(state.evidence).configuration
    assert config is not None
    assert config.rules_path == "honest-ci-rules.toml"
    assert config.rules_sha256 == hashlib.sha256(rules.read_bytes()).hexdigest()
    assert config.rule_version == "v4"
    assert config.policy_path == "honest-ci-policy.toml"
    assert config.policy_sha256 == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert config.policy_version == "v9"
    assert config.required_checks == ["spec", "review"]


def test_editing_a_rule_changes_the_snapshot_digest(tmp_path: Path) -> None:
    """Two runs judged by different rule files must not look alike in evidence."""

    rules = tmp_path / "honest-ci-rules.toml"
    rules.write_text(JUDGMENT_RULE, encoding="utf-8")
    before = present(all_passing(tmp_path).verify_change(make_change()).evidence).configuration

    rules.write_text(JUDGMENT_RULE.replace("high", "medium"), encoding="utf-8")
    after = present(all_passing(tmp_path).verify_change(make_change()).evidence).configuration

    assert before is not None and after is not None
    assert before.rules_sha256 != after.rules_sha256
    assert before.digest != after.digest


def test_no_rule_file_is_recorded_as_absent_rather_than_as_an_empty_digest(
    tmp_path: Path,
) -> None:
    state = all_passing(tmp_path).verify_change(make_change())

    config = present(state.evidence).configuration
    assert config is not None
    assert config.rules_path is None
    assert config.rules_sha256 is None
    assert config.policy_path is None


def test_the_snapshot_records_which_gates_this_deployment_runs(tmp_path: Path) -> None:
    runner = build(tmp_path, disabled_gates=("metrics",), spec={}, review={})
    state = runner.verify_change(make_change())

    config = present(state.evidence).configuration
    assert config is not None
    assert config.deployed_gates == ["spec", "review"]
    assert config.disabled_gates == ["metrics"]
    assert config.fail_closed is True


def test_the_snapshot_names_no_model_when_none_was_enabled(tmp_path: Path) -> None:
    """"没问过" and "问了某个模型" have to be different lines in the evidence."""

    state = all_passing(tmp_path).verify_change(make_change())

    config = present(state.evidence).configuration
    assert config is not None
    assert config.model is None
    assert config.model_provider == ""


def test_the_snapshot_names_the_provider_whose_key_was_missing(tmp_path: Path) -> None:
    state = all_passing(tmp_path, model_provider="qwen").verify_change(make_change())

    config = present(state.evidence).configuration
    assert config is not None
    assert config.model is None
    assert config.model_provider == "qwen"


def test_a_key_in_the_environment_does_not_switch_a_model_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key exported for another tool is not consent to send this code anywhere."""

    monkeypatch.delenv("HONEST_CI_MODEL_PROVIDER", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abcdefghijklmnopqrstuvwx")

    runner = GateRunner(workspace_root=tmp_path, settings=Settings.from_env())

    assert runner.adapters.llm is None


def test_the_snapshot_survives_a_json_round_trip(tmp_path: Path) -> None:
    """The pack is read back from a ledger and from a file, months later."""

    state = all_passing(tmp_path).verify_change(make_change())

    restored = EvidencePack.model_validate_json(present(state.evidence).model_dump_json())

    assert present(restored.configuration).digest == present(
        present(state.evidence).configuration
    ).digest
    written = json.loads(
        (
            all_passing(tmp_path).artifact_store.run_dir(state.run_id) / "evidence.json"
        ).read_text(encoding="utf-8")
    )
    assert written["configuration"]["rule_version"] == present(state.evidence).rule_version


def test_an_old_pack_without_a_snapshot_still_loads(tmp_path: Path) -> None:
    """Packs written before this feature existed are still evidence."""

    state = all_passing(tmp_path).verify_change(make_change())
    payload = present(state.evidence).model_dump(mode="json")
    del payload["configuration"]

    assert EvidencePack.model_validate(payload).configuration is None


def test_the_snapshot_stays_out_of_the_approval_fingerprint(tmp_path: Path) -> None:
    """Tuning a threshold must not silently invalidate an existing signature."""

    strict = Settings(
        sandbox_mode="process", retry_backoff_seconds=(0.0,), coverage_threshold=0.9
    )
    relaxed = replace(strict, coverage_threshold=0.1)
    fingerprints = []
    digests = []
    for index, settings in enumerate((strict, relaxed)):
        root = tmp_path / f"run-{index}"
        root.mkdir()
        runner = GateRunner(
            workspace_root=root,
            settings=settings,
            adapters=AdapterSet(ledger=SQLiteLedger(root / settings.ledger_path)),
        )
        runner.gates = {
            name: ScriptedGate(runner.services, name=name) for name in GATE_SEQUENCE
        }
        evidence = runner.verify_change(make_change()).evidence
        fingerprints.append(present(evidence).fingerprint)
        digests.append(present(present(evidence).configuration).digest)

    assert fingerprints[0] == fingerprints[1]
    assert digests[0] != digests[1]
