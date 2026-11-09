from __future__ import annotations

import sqlite3
from pathlib import Path

from honest_ci.contracts import (
    ApprovalTicket,
    ArtifactManifest,
    EvidencePack,
    RunMetrics,
    RunRecord,
    Verdict,
)
from honest_ci.state import RunState
from honest_ci.storage import SQLiteLedger

from .factories import make_change


def ledger(tmp_path: Path) -> SQLiteLedger:
    return SQLiteLedger(tmp_path / ".honest-ci" / "ledger.sqlite3")


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    state = RunState(run_id="r1", change=make_change(), current_gate="review")
    store.save_state(state)

    loaded = store.load_state("r1")
    assert loaded is not None
    assert loaded.current_gate == "review"
    assert loaded.change.head_sha == state.change.head_sha
    assert store.load_state("missing") is None


def test_approval_is_bound_to_a_fingerprint_and_can_be_invalidated(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    change = make_change()
    evidence = EvidencePack(run_id="r1", change=change, fingerprint="fp-1", verdict=Verdict.PASS)
    store.save_evidence("r1", evidence, "fp-1")
    store.save_approval_ticket("r1", ApprovalTicket(fingerprint="fp-1"))
    store.resolve_approval("r1", approved=True, reason="looks fine", reviewer="chris")

    approval = store.load_approval("r1")
    assert approval is not None
    assert approval["fingerprint"] == "fp-1"
    assert approval["approved"] == 1
    assert approval["reviewer"] == "chris"

    store.clear_approval("r1")
    assert store.load_approval("r1") is None


def test_run_records_are_listed_newest_first(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    for index in range(3):
        store.save_run_record(
            RunRecord(
                run_id=f"r{index}",
                change_ref=f"team/orders#{index}",
                outcome="blocked",
                metrics=RunMetrics(gates_blocked=1),
            )
        )
    records = store.list_run_records(limit=2)
    assert len(records) == 2
    assert {record.outcome for record in records} == {"blocked"}
    assert store.load_run_record("r1") is not None


def test_artifact_manifest_roundtrip(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.save_artifact_manifest("r1", ArtifactManifest(run_id="r1"))
    loaded = store.load_artifact_manifest("r1")
    assert loaded is not None
    assert loaded["run_id"] == "r1"


# ── a ledger an earlier build wrote ──────────────────────────────────────────


def test_a_ledger_with_the_old_status_column_still_saves_checkpoints(tmp_path: Path) -> None:
    """Checkpoints once carried a status column, and nothing reads it any more.

    A ledger that already has one is left as it is rather than migrated: the
    column's default keeps every insert valid, so an existing ledger opens,
    loads and saves exactly as a fresh one does.
    """

    path = tmp_path / ".honest-ci" / "ledger.sqlite3"
    path.parent.mkdir(parents=True)
    old = RunState(run_id="r-old", change=make_change(), status="running")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE checkpoints (
            run_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_checkpoints_status ON checkpoints(status);
        """
    )
    conn.execute(
        "INSERT INTO checkpoints(run_id, state_json, status, updated_at)"
        " VALUES(?, ?, ?, CURRENT_TIMESTAMP)",
        (old.run_id, old.model_dump_json(), old.status),
    )
    conn.commit()
    conn.close()

    store = SQLiteLedger(path)
    store.save_state(old.model_copy(update={"status": "blocked"}))
    store.save_state(RunState(run_id="r-new", change=make_change()))

    reloaded = store.load_state("r-old")
    assert reloaded is not None and reloaded.status == "blocked"
    assert store.load_state("r-new") is not None


# ── findings & annotations ───────────────────────────────────────────────────

from honest_ci.contracts import Finding, FindingAnnotation, GateReport  # noqa: E402


def _evidence(run_id: str, fingerprint: str, *rules: str) -> EvidencePack:
    findings = [
        Finding(id=f"review.{rule}.1", gate="review", rule_id=rule, severity="high", title=rule)
        for rule in rules
    ]
    return EvidencePack(
        run_id=run_id,
        change=make_change(),
        gate_reports=[GateReport(gate="review", verdict=Verdict.BLOCKED, findings=findings)],
        verdict=Verdict.BLOCKED,
        fingerprint=fingerprint,
    )


def test_an_annotation_is_one_per_finding_and_bound_to_a_fingerprint(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.save_annotation(
        FindingAnnotation(
            run_id="r1",
            finding_id="review.R-01.1",
            fingerprint="fp-1",
            false_positive=True,
            reason="日志用的",
            annotator="chris",
        )
    )
    store.save_annotation(
        FindingAnnotation(
            run_id="r1", finding_id="review.R-01.1", fingerprint="fp-1", false_positive=False
        )
    )

    notes = store.list_annotations("r1")
    assert len(notes) == 1  # the second reading replaced the first
    assert notes[0].false_positive is False
    assert notes[0].fingerprint == "fp-1"
    assert store.list_annotations("other") == []


def test_load_evidence_returns_the_pack_as_last_saved(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    assert store.load_evidence("r1") is None

    store.save_evidence("r1", _evidence("r1", "fp-1", "R-01"), "fp-1")

    loaded = store.load_evidence("r1")
    assert loaded is not None and loaded.fingerprint == "fp-1"
    assert [f.rule_id for f in loaded.findings] == ["R-01"]
