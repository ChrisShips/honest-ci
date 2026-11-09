"""The evidence contract itself: what blocks, what is measured, what version it is.

`contracts.py` is mostly declarations, and the gates' own tests cover them by
using them. What is here instead are the four properties that have to hold
*across* versions of this software — an old pack still loads, a new pack says
which format it is, only this deployment's own code may block, and a number
nobody measured is not reported as zero. Each of them is a promise to somebody
outside this repository, so each gets an assertion rather than a convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from honest_ci.contracts import (
    EVIDENCE_SCHEMA_VERSION,
    GATE_SEQUENCE,
    EvidencePack,
    Finding,
    GateReport,
    RunMetrics,
    Verdict,
    evidence_schema,
)

from .factories import make_change

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_SCHEMA = REPO_ROOT / "docs" / "evidence.schema.json"


def pack(**overrides: Any) -> EvidencePack:
    payload: dict[str, Any] = {
        "run_id": "r1",
        "change": make_change(),
        "fingerprint": "f" * 64,
    }
    payload.update(overrides)
    return EvidencePack.model_validate(payload)


# ── the pipeline, and the names packs may still carry ────────────────────────


def test_measurement_is_not_a_gate_in_the_pipeline() -> None:
    # 度量不判定任何东西，所以它不在流水线里 —— 每次运行的度量写进证据包。
    assert "metrics" not in GATE_SEQUENCE
    assert GATE_SEQUENCE == ("spec", "review", "regression")


def test_a_pack_written_when_metrics_was_still_a_gate_still_loads() -> None:
    """Old evidence keeps reading, which is what "只增不破坏" means in practice.

    Dropping ``metrics`` from ``GateName`` as well as from the pipeline would
    have turned every pack, ledger row and checkpoint written before today into
    a ``ValidationError`` — the removal an auditor would experience as their
    evidence becoming unreadable.
    """

    old = pack(
        gate_reports=[
            GateReport(gate="review", verdict=Verdict.PASS),
            GateReport(gate="metrics", verdict=Verdict.SKIPPED, summary="未部署"),
        ],
    )
    reloaded = EvidencePack.model_validate(json.loads(old.model_dump_json()))

    assert [report.gate for report in reloaded.gate_reports] == ["review", "metrics"]


# ── the format's own version ─────────────────────────────────────────────────


def test_every_pack_says_which_format_version_it_is() -> None:
    assert pack().schema_version == EVIDENCE_SCHEMA_VERSION


def test_a_pack_written_before_the_field_existed_reads_as_v1() -> None:
    # The "no field means v1" rule lives in the default, so no verifier has to
    # carry that branch forever.
    payload = json.loads(pack().model_dump_json())
    del payload["schema_version"]

    assert EvidencePack.model_validate(payload).schema_version == "v1"


def test_a_finding_written_before_it_had_a_side_is_on_the_head() -> None:
    # Every finding before the integrity checks read an added line, so a pack
    # from then has to go on meaning exactly that.
    finding = Finding(id="review.R-01.1", gate="review", title="x", file="a.py", line_start=3)
    review = GateReport(gate="review", verdict=Verdict.PASS, findings=[finding])
    payload = json.loads(pack(gate_reports=[review]).model_dump_json())
    del payload["gate_reports"][0]["findings"][0]["side"]

    (loaded,) = EvidencePack.model_validate(payload).findings

    assert loaded.side == "head"


def test_the_committed_schema_is_the_schema_this_build_writes() -> None:
    """The add-only rule, as a check that fails CI rather than a discipline.

    If this fails: a field was added — regenerate with ``honest-ci schema >
    docs/evidence.schema.json`` and confirm nothing was removed or narrowed. A
    field's *meaning* changed — bump ``EVIDENCE_SCHEMA_VERSION`` first, because
    packs already handed out were written under the old meaning.
    """

    committed = json.loads(COMMITTED_SCHEMA.read_text(encoding="utf-8"))

    assert committed == evidence_schema(), (
        "docs/evidence.schema.json 与本次构建写出的证据格式不一致。"
        "加字段：重新生成（honest-ci schema > docs/evidence.schema.json）。"
        "改字段语义：先升 EVIDENCE_SCHEMA_VERSION。"
    )


def test_the_published_schema_carries_its_own_version() -> None:
    schema = evidence_schema()

    assert schema["x-schema-version"] == EVIDENCE_SCHEMA_VERSION
    assert "schema_version" in schema["properties"]


# ── who may block ────────────────────────────────────────────────────────────


def finding(**overrides: Any) -> Finding:
    payload: dict[str, Any] = {
        "id": "review.R-01.1",
        "gate": "review",
        "severity": "high",
        "title": "金额计算用了 float",
    }
    payload.update(overrides)
    return Finding.model_validate(payload)


def test_only_this_deployments_own_code_blocks() -> None:
    assert finding(decided_by="deterministic").blocking


@pytest.mark.parametrize("source", ["model", "human"])
def test_nothing_else_blocks_however_severe_its_rule(source: str) -> None:
    """The allow-list, stated as the property it exists for.

    A block is terminal — ``resume_approval`` reaches runs waiting on a human,
    not blocked ones — so anything this deployment did not decide itself routes
    the change to a person instead of ending it.
    """

    assert not finding(severity="critical", decided_by=source).blocking


def test_a_source_this_build_does_not_know_yet_does_not_block() -> None:
    """外部结论永不拦截，一处收口 —— and the收口 holds before the source exists.

    ``external:<source>`` is the next one (平台 AI 评审 / CI / SAST). It is
    validated here, not constructed past validation, because the contract
    deliberately accepts a value this build was never written against: what
    such a value must do is be advisory, not be rejected.
    """

    external = finding(severity="critical", decided_by="external:gitlab-duo")

    assert external.decided_by == "external:gitlab-duo"
    assert not external.blocking


def test_the_published_contract_does_not_close_the_set_of_decision_sources() -> None:
    """The open type, as the property it exists for rather than a declaration.

    A closed ``enum`` here would put every verifier written against the
    published schema on a collision course with the first pack carrying
    ``external:<source>``: obeying the schema, it would have to call that pack
    malformed. Widening cannot widen who blocks — that is ``BLOCKING_SOURCES``,
    and ``test_nothing_else_blocks_however_severe_its_rule`` holds it closed.
    """

    decided_by = evidence_schema()["$defs"]["Finding"]["properties"]["decided_by"]

    assert decided_by["type"] == "string"
    assert "enum" not in decided_by


# ── what was measured, and what was not ──────────────────────────────────────


def test_a_defect_count_nobody_measured_is_not_reported_as_zero() -> None:
    """``escaped_defects`` has no writer yet, so a pack must not claim a count.

    ``0`` here would be this product telling an auditor that it looked for
    escaped defects and found none. ``None`` is what the rest of this model
    already says for "not measured" (``human_review_minutes``,
    ``false_positives``, ``ai_generated_ratio``).
    """

    metrics = RunMetrics()

    assert metrics.escaped_defects is None
    assert json.loads(metrics.model_dump_json())["escaped_defects"] is None
