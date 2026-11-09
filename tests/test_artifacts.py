from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from honest_ci.artifacts import ArtifactStore
from honest_ci.contracts import GateReport, Verdict


def test_json_artifacts_are_redacted_and_fingerprinted(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    report = GateReport(
        gate="review",
        verdict=Verdict.BLOCKED,
        summary="token=glpat-abcdefghijklmnopqrst leaked into the summary",
    )
    manifest = store.write_json("run-1", "gate-review.json", report)

    record = manifest.artifacts[0]
    written = (store.root / record.path).read_bytes()
    assert "glpat-" not in written.decode("utf-8")
    assert record.sha256 == hashlib.sha256(written).hexdigest()
    assert record.size_bytes == len(written)


def test_rewriting_an_artifact_replaces_its_record(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    manifest = store.write_text("run-1", "notes.txt", "first")
    manifest = store.write_text("run-1", "notes.txt", "second", manifest=manifest)

    assert len(manifest.artifacts) == 1
    assert (store.root / manifest.artifacts[0].path).read_text(encoding="utf-8") == "second"

    on_disk = json.loads((store.run_dir("run-1") / "manifest.json").read_text(encoding="utf-8"))
    assert len(on_disk["artifacts"]) == 1


def test_artifact_names_cannot_escape_the_run_directory(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    manifest = store.write_text("run-1", "../../escape.txt", "contained")

    record = manifest.artifacts[0]
    assert record.name == "escape.txt"
    assert (store.run_dir("run-1") / "escape.txt").is_file()
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize(
    "value",
    [
        'password = "hunter2"',
        "token:",
        "db_password:",
        "api_key=",
        "secret: 'y'",
        'token="',
        "token=\\",
        "a line ending in token=\nand the next one",
    ],
)
def test_json_artifacts_read_back_whatever_their_strings_hold(tmp_path: Path, value: str) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    store.write_json("run-1", "e.json", {"text": value, "list": [value, "after"], "z": 1})

    written = json.loads((store.run_dir("run-1") / "e.json").read_text(encoding="utf-8"))
    assert set(written) == {"text", "list", "z"}
    assert written["list"][1] == "after"
    assert written["z"] == 1


def test_a_gate_report_quoting_a_credential_line_reads_back_redacted(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    report = GateReport(gate="review", verdict=Verdict.BLOCKED, summary='password = "hunter2"')
    store.write_json("run-1", "gate-review.json", report)

    text = (store.run_dir("run-1") / "gate-review.json").read_text(encoding="utf-8")
    assert GateReport.model_validate_json(text).summary == 'password = "[REDACTED]"'


def test_a_diff_artifact_keeps_every_line_and_its_marker(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    diff = "@@ -1,2 +1,2 @@\n-  password:\n+  password: hunter2\n"
    store.write_text("run-1", "change.diff", diff)

    written = (store.run_dir("run-1") / "change.diff").read_text(encoding="utf-8")
    assert "hunter2" not in written
    assert [line[0] for line in written.splitlines()] == ["@", "-", "+"]
