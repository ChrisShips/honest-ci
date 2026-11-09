"""Evidence on disk: redacted content plus a SHA-256 manifest.

Everything a gate produces is written through here, so an evidence pack handed
to a reviewer or an auditor is auditable (every artifact has a fingerprint) and
free of credentials.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .contracts import ArtifactManifest, ArtifactRecord
from .redaction import redact_obj, redact_text
from .utils import write_text


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(
        self,
        run_id: str,
        name: str,
        payload: Any,
        manifest: ArtifactManifest | None = None,
    ) -> ArtifactManifest:
        if isinstance(payload, BaseModel):
            payload = payload.model_dump(mode="json")
        # Redacted as data and not again as text: every string in it, keys
        # included, has met the patterns, and a pass over the serialized text
        # would read JSON's own escapes, quotes and commas as part of a value.
        # ``{"text": "token:", "z": 1}`` lost the comma after ``token:``, and an
        # evidence pack that does not parse is no evidence at all.
        payload = redact_obj(payload)
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        return self._write(run_id, name, text, "json", manifest)

    def write_text(
        self,
        run_id: str,
        name: str,
        text: str,
        kind: str = "text",
        manifest: ArtifactManifest | None = None,
    ) -> ArtifactManifest:
        return self._write(run_id, name, redact_text(text), kind, manifest)

    def _write(
        self,
        run_id: str,
        name: str,
        text: str,
        kind: str,
        manifest: ArtifactManifest | None,
    ) -> ArtifactManifest:
        """Write ``text``, which the caller has already redacted, and record it."""

        safe_name = _safe_name(name)
        path = self.run_dir(run_id) / safe_name
        write_text(path, text)
        record = _record(self.root, path, kind)
        current = manifest or ArtifactManifest(run_id=run_id)
        deduped = [item for item in current.artifacts if item.name != record.name]
        updated = current.model_copy(update={"artifacts": [*deduped, record]})
        self.save_manifest(updated)
        return updated

    def save_manifest(self, manifest: ArtifactManifest) -> None:
        path = self.run_dir(manifest.run_id) / "manifest.json"
        payload = manifest.model_dump(mode="json")
        write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _record(root: Path, path: Path, kind: str) -> ArtifactRecord:
    content = path.read_bytes()
    return ArtifactRecord(
        name=path.name,
        path=str(path.relative_to(root)).replace("\\", "/"),
        sha256=hashlib.sha256(content).hexdigest(),
        kind=kind,
        size_bytes=len(content),
    )


def _safe_name(name: str) -> str:
    cleaned = name.replace("\\", "/").split("/")[-1]
    return "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in cleaned)
