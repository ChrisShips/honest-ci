"""Model-written regression cases: proposed, run in the sandbox, then removed.

The regression gate's v0 question was "does the repository's suite still
pass". Its last v1 question is the one a reviewer asks when the suite is thin:
**is there a single test aimed at this change**. A model writes the cases,
deterministic code decides everything else.

Four properties hold this together, and each of them is a line in this file:

* **A generated case can never block.** It is a `decided_by="model"` finding,
  and `Finding.blocking` is false for those — at most the change goes to a
  human. A model-written test that fails means one of two things (the test is
  wrong, or the code is), and deciding which is exactly the judgment a person
  is for.
* **Nothing it writes survives the run.** The files are written into the tree
  the suite runs in, run once by path, and deleted in a ``finally``. A
  verification pass that leaves files behind has changed the thing it was
  verifying — and for ``honest-ci verify`` that tree is the developer's own
  working copy.
* **Nothing it writes is trusted.** Paths, contents and targets all pass
  ``grounding.guarded_test_proposal`` first; this module writes only what came
  back from there, and re-checks the one thing that can change in between —
  whether the file already exists.
* **Not generating is recorded, not assumed away.** Every way this can produce
  nothing returns a reason, and the reason reaches "What this run could not
  verify". "No generated case failed" must never be readable as "the change
  was covered".

The results never touch ``TestResults.passed``/``failed``: they have their own
``generated*`` fields, so no arithmetic can let a model-written case decide a
verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .adapters.base import LLMAdapter, ScannerAdapter
from .artifacts import ArtifactStore
from .contracts import ChangeRequest, ProposedTest, TestResults
from .grounding import GENERATED_TEST_DIRS, guarded_test_proposal
from .impact import ChangedSymbol
from .settings import MODEL_NOT_ENABLED, Settings

#: Failing generated cases quoted in the evidence. The same bound the suite's
#: own failures get, for the same reason.
MAX_GENERATED_FAILURES = 20


@dataclass(frozen=True)
class GeneratedTests:
    """What came of asking for regression cases — including nothing, and why.

    ``skipped_reason`` and ``results`` are mutually exclusive: either cases ran
    or they did not, and a reader must never have to infer which from an empty
    count. ``artifacts`` holds where the generated text was kept for a human to
    pick up, because the files themselves are gone by the time anyone reads
    this.
    """

    results: TestResults | None = None
    proposed: int = 0
    paths: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    skipped_reason: str = ""

    @property
    def ran(self) -> bool:
        return self.results is not None

    @property
    def failed(self) -> int:
        results = self.results
        if results is None:
            return 0
        return results.failed + results.errors

    def as_dict(self) -> dict[str, object]:
        """The shape the evidence pack carries, JSON-serialisable as it is."""

        return {
            "proposed": self.proposed,
            "ran": self.ran,
            "passed": self.results.passed if self.results else 0,
            "failed": self.failed,
            "paths": list(self.paths),
            "targets": list(self.targets),
            "artifacts": list(self.artifacts),
            "not_generated_reason": self.skipped_reason or None,
        }


def propose_and_run(
    *,
    change: ChangeRequest,
    symbols: tuple[ChangedSymbol, ...],
    fingerprint: str,
    language: str,
    workspace: Path,
    scanner: ScannerAdapter,
    llm: LLMAdapter | None,
    settings: Settings,
    artifacts: ArtifactStore,
) -> GeneratedTests:
    """Generate regression cases, run them once, and leave the tree as it was.

    Every early return carries the sentence the evidence will print. They are
    deliberately the same shapes the judgment-class rules use — switched off,
    no model enabled, no credential, call failed, discarded at the trust
    boundary — because a reader who has learned to read one line of "What
    this run could not verify" should not have to learn a second vocabulary
    for this one.
    """

    if not settings.model_testgen:
        return GeneratedTests(
            skipped_reason="regression test generation is switched off in this deployment "
            "(HONEST_CI_MODEL_TESTGEN=0)"
        )
    if llm is None:
        if settings.model_provider is None:
            return GeneratedTests(skipped_reason=MODEL_NOT_ENABLED)
        return GeneratedTests(
            skipped_reason=f"no model credential is configured ({settings.credential_hint})"
        )
    if language not in GENERATED_TEST_DIRS:
        return GeneratedTests(
            skipped_reason=f"the {language} toolchain has no place for generated tests yet"
        )
    if not symbols:
        return GeneratedTests(
            skipped_reason="impact analysis found no changed symbol to write tests for"
        )
    unselectable = scanner.unselectable(workspace)
    if unselectable is not None:
        # Run anyway, the "generated" counts would be the whole suite's. Asked
        # before the model is, because nothing would run what it wrote.
        return GeneratedTests(
            skipped_reason=f"the generated tests could not be run on their own: {unselectable}"
        )

    proposal = guarded_test_proposal(
        change, symbols, fingerprint, llm, language=language, workspace=workspace
    )
    if proposal is None:
        return GeneratedTests(
            skipped_reason="the model call failed, or the tests it returned failed the trust "
            "boundary and were discarded whole"
        )
    if not proposal.tests:
        return GeneratedTests(skipped_reason="the model proposed no regression test for this change")

    # Kept in pairs: a case that was not written has no path, and its
    # targets are not something this run exercised.
    written: list[tuple[ProposedTest, Path]] = []
    created: list[Path] = []
    try:
        for test in proposal.tests:
            target = workspace / test.path
            if target.exists():
                # Checked at the boundary too; checked again here because this
                # is the moment a file would be overwritten, and nothing about
                # a generated case is worth that risk.
                continue
            created += _missing_parents(target.parent, workspace)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(test.content, encoding="utf-8")
            written.append((test, target))
        if not written:
            return GeneratedTests(
                skipped_reason="every generated test path already exists in the workspace, "
                "so no file was written"
            )
        paths = tuple(path.relative_to(workspace).as_posix() for _, path in written)
        results = scanner.run_tests(workspace, list(_selection(paths, language)))
        return GeneratedTests(
            results=results,
            proposed=len(proposal.tests),
            paths=paths,
            targets=tuple(dict.fromkeys(n for test, _ in written for n in test.targets)),
            artifacts=_keep(artifacts, [test for test, _ in written], paths),
        )
    finally:
        _remove([path for _, path in written], created)


def _selection(paths: tuple[str, ...], language: str) -> tuple[str, ...]:
    """The generated files, named the way this toolchain selects tests."""

    if language != "java":
        return paths
    # Maven selects by class name, and the guard already checked that each
    # file's class is named after the file.
    return tuple(path.rsplit("/", maxsplit=1)[-1].removesuffix(".java") for path in paths)


def _keep(
    artifacts: ArtifactStore,
    tests: list[ProposedTest],
    paths: tuple[str, ...],
) -> tuple[str, ...]:
    """Save the generated text where a developer can pick it up afterwards.

    Under an id of its own rather than the run's: ``write_text`` rewrites the
    manifest it is handed, and the run's manifest belongs to the runner. The
    same reason ``ToolRunner`` keeps its logs apart, and the paths travel in
    the evidence either way.
    """

    store_id = f"generated-tests-{uuid4().hex}"
    manifest = None
    kept: list[str] = []
    for test, path in zip(tests, paths, strict=True):
        manifest = artifacts.write_text(store_id, path, test.content, "text", manifest)
        kept.append(manifest.artifacts[-1].path)
    return tuple(kept)


def _missing_parents(directory: Path, workspace: Path) -> list[Path]:
    """The directories this write is about to create, deepest first."""

    missing: list[Path] = []
    current = directory
    while current != workspace and not current.exists():
        missing.append(current)
        current = current.parent
    return missing


def _remove(written: list[Path], created: list[Path]) -> None:
    """Put the tree back: the files, then only the directories this run made.

    Only the ones it made: an empty ``tests/`` that was already there is the
    repository's, and a verification pass does not get to tidy it away.
    """

    for path in written:
        path.unlink(missing_ok=True)
    for directory in sorted(set(created), key=lambda item: len(item.parts), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


__all__ = ["MAX_GENERATED_FAILURES", "GeneratedTests", "propose_and_run"]
