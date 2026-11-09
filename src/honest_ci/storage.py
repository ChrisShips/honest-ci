"""The ledger: durable checkpoints, evidence, approvals and run accounting.

SQLite is deliberate. A private deployment should not need a database server,
and the whole ledger is one file an operator can copy, inspect and keep. That
single-file property is also why the journal mode is left alone: WAL would be
faster when several runs write at once, but it would turn the ledger into three
files that have to be checkpointed before any of them can be copied. The
contention it would have relieved is handled with a busy timeout instead —
every write here is small and short.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .contracts import (
    ApprovalTicket,
    ArtifactManifest,
    EvidencePack,
    FindingAnnotation,
    RunRecord,
)
from .state import RunState

#: How long a writer waits for another writer's lock before giving up.
BUSY_TIMEOUT_SECONDS = 30.0


class SQLiteLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        # Two runs started together on one ledger both checkpoint at every gate;
        # the default five seconds is enough for that, but a run that loses a
        # checkpoint is a run nobody can resume, so the wait is generous.
        conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        return conn

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        conn = self._connect()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        conn = self._connect()
        try:
            row: sqlite3.Row | None = conn.execute(sql, params).fetchone()
            return row
        finally:
            conn.close()

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        conn = self._connect()
        try:
            return list(conn.execute(sql, params).fetchall())
        finally:
            conn.close()

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_records (
                    run_id TEXT PRIMARY KEY,
                    change_ref TEXT NOT NULL DEFAULT '',
                    outcome TEXT NOT NULL DEFAULT '',
                    record_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    run_id TEXT PRIMARY KEY,
                    evidence_json TEXT NOT NULL,
                    ticket_json TEXT,
                    fingerprint TEXT NOT NULL,
                    approved INTEGER,
                    reason TEXT,
                    reviewer TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                -- What a person said about a finding, bound to the fingerprint
                -- the finding was produced under (see FindingAnnotation).
                CREATE TABLE IF NOT EXISTS annotations (
                    run_id TEXT NOT NULL,
                    finding_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    false_positive INTEGER NOT NULL,
                    reason TEXT,
                    annotator TEXT,
                    annotated_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, finding_id)
                );

                CREATE INDEX IF NOT EXISTS idx_run_records_change
                    ON run_records(change_ref);
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ── checkpoints ──────────────────────────────────────────────────────

    def save_state(self, state: RunState) -> None:
        self._execute(
            """
            INSERT INTO checkpoints(run_id, state_json, updated_at)
            VALUES(?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id) DO UPDATE SET
                state_json=excluded.state_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (state.run_id, state.model_dump_json()),
        )

    def load_state(self, run_id: str) -> RunState | None:
        row = self._fetchone("SELECT state_json FROM checkpoints WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        return RunState.model_validate_json(row["state_json"])

    # ── evidence & approvals ─────────────────────────────────────────────

    def save_evidence(self, run_id: str, evidence: EvidencePack, fingerprint: str) -> None:
        self._execute(
            """
            INSERT INTO approvals(run_id, evidence_json, fingerprint, approved, reason, updated_at)
            VALUES(?, ?, ?, NULL, NULL, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id) DO UPDATE SET
                evidence_json=excluded.evidence_json,
                fingerprint=excluded.fingerprint,
                updated_at=CURRENT_TIMESTAMP
            """,
            (run_id, evidence.model_dump_json(), fingerprint),
        )

    def save_approval_ticket(self, run_id: str, ticket: ApprovalTicket) -> None:
        self._execute(
            "UPDATE approvals SET ticket_json = ?, updated_at = CURRENT_TIMESTAMP WHERE run_id = ?",
            (ticket.model_dump_json(), run_id),
        )

    def resolve_approval(
        self,
        run_id: str,
        approved: bool,
        reason: str | None = None,
        reviewer: str | None = None,
    ) -> None:
        self._execute(
            """
            UPDATE approvals
            SET approved = ?, reason = ?, reviewer = ?, updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
            """,
            (1 if approved else 0, reason, reviewer, run_id),
        )

    def clear_approval(self, run_id: str) -> None:
        """Invalidate a prior approval once an upstream artifact changes."""

        self._execute("DELETE FROM approvals WHERE run_id = ?", (run_id,))

    def load_approval(self, run_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM approvals WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        return dict(row)

    def load_evidence(self, run_id: str) -> EvidencePack | None:
        """The evidence pack as last saved, without the rest of the run state."""

        row = self._fetchone("SELECT evidence_json FROM approvals WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        return EvidencePack.model_validate_json(row["evidence_json"])

    # ── annotations ──────────────────────────────────────────────────────

    def save_annotation(self, annotation: FindingAnnotation) -> None:
        """Record what a person said about one finding.

        One annotation per finding: a second reading replaces the first. The
        ledger is the current human judgment, not a discussion thread.
        """

        self._execute(
            """
            INSERT INTO annotations(
                run_id, finding_id, fingerprint, false_positive, reason, annotator,
                annotated_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id, finding_id) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                false_positive=excluded.false_positive,
                reason=excluded.reason,
                annotator=excluded.annotator,
                annotated_at=excluded.annotated_at,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                annotation.run_id,
                annotation.finding_id,
                annotation.fingerprint,
                1 if annotation.false_positive else 0,
                annotation.reason,
                annotation.annotator,
                annotation.annotated_at.isoformat(),
            ),
        )

    def list_annotations(self, run_id: str | None = None) -> list[FindingAnnotation]:
        sql = (
            "SELECT run_id, finding_id, fingerprint, false_positive, reason, annotator, "
            "annotated_at FROM annotations"
        )
        params: tuple[Any, ...] = ()
        if run_id is not None:
            sql += " WHERE run_id = ?"
            params = (run_id,)
        sql += " ORDER BY updated_at, run_id, finding_id"
        return [
            FindingAnnotation(
                run_id=str(row["run_id"]),
                finding_id=str(row["finding_id"]),
                fingerprint=str(row["fingerprint"]),
                false_positive=bool(row["false_positive"]),
                reason=row["reason"],
                annotator=row["annotator"],
                annotated_at=row["annotated_at"],
            )
            for row in self._fetchall(sql, params)
        ]

    # ── run accounting ───────────────────────────────────────────────────

    def save_run_record(self, record: RunRecord) -> None:
        self._execute(
            """
            INSERT INTO run_records(run_id, change_ref, outcome, record_json, updated_at)
            VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id) DO UPDATE SET
                change_ref=excluded.change_ref,
                outcome=excluded.outcome,
                record_json=excluded.record_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (record.run_id, record.change_ref, record.outcome, record.model_dump_json()),
        )

    def load_run_record(self, run_id: str) -> RunRecord | None:
        row = self._fetchone("SELECT record_json FROM run_records WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        return RunRecord.model_validate_json(row["record_json"])

    def list_run_records(self, limit: int = 50, offset: int = 0) -> list[RunRecord]:
        """Newest-first feed, for ``honest-ci runs``."""

        rows = self._fetchall(
            "SELECT record_json FROM run_records ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (max(1, limit), max(0, offset)),
        )
        return [RunRecord.model_validate_json(row["record_json"]) for row in rows]

    # ── artifacts ────────────────────────────────────────────────────────

    def save_artifact_manifest(self, run_id: str, manifest: ArtifactManifest) -> None:
        self._execute(
            """
            INSERT INTO artifacts(run_id, manifest_json, updated_at)
            VALUES(?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id) DO UPDATE SET
                manifest_json=excluded.manifest_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (run_id, manifest.model_dump_json()),
        )

    def load_artifact_manifest(self, run_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT manifest_json FROM artifacts WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        loaded: dict[str, Any] = json.loads(row["manifest_json"])
        return loaded
