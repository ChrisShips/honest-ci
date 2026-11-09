"""GitHub's answers on disk, one file each, so a run can stop anywhere.

A run of a thousand pull requests per group takes hours and is expected to be
interrupted. Everything it asks GitHub is kept, keyed by the question, so
running the same command again replays what is here and asks only for the
rest -- and a run scaled from 50 to 1000 per group asks nothing twice. The
cache is also the snapshot the numbers come from: a pull request merged after
it was first read stays unmerged here until the cache is deleted.

Entries are written here rather than through ``ArtifactStore``, and they are
not redacted. They are GitHub's own answers about public pull requests -- the
token is only ever in a request header, and nothing GitHub sends back contains
it -- kept verbatim so that a resumed run sees exactly what a fresh one saw.
The product's redaction is tuned for evidence, where masking too much is the
safe mistake; over this data it rewrites the repository names, file paths and
diffs the benchmark reads. ``src/github_webhook_handlers/app.py`` becomes
``src/[REDACTED]/app.py`` -- ``github_`` and sixteen more characters is a
token's shape -- and a test that changes its expected ``password="old"`` to
``password="new"`` looks unchanged, since both lines then read
``password="[REDACTED]"``.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Cached:
    value: Any


class Cache:
    """One gzipped JSON file per answer, written atomically.

    Atomic because a run is expected to be killed: an interrupted write leaves
    no entry, never half of one that a resumed run would trust.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, kind: str, key: str) -> Cached | None:
        try:
            with gzip.open(self._path(kind, key), "rt", encoding="utf-8") as handle:
                return Cached(json.load(handle)["value"])
        except FileNotFoundError:
            return None
        except (OSError, EOFError, ValueError, KeyError, TypeError):
            # Writes are atomic, so this is disk trouble rather than an
            # interrupted run; asking GitHub again is cheaper than stopping.
            return None

    def put(self, kind: str, key: str, value: Any) -> None:
        path = self._path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"value": value}, ensure_ascii=False).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with (
                os.fdopen(descriptor, "wb") as raw,
                gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle,
            ):
                handle.write(payload)
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise

    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / key[:2] / f"{key}.json.gz"


def cache_key(*parts: Any) -> str:
    """The identity of a question: what it asks, never who asks it."""

    text = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
