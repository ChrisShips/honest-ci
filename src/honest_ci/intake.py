"""Intake: turning a local change into a ``ChangeRequest``.

The gates never see a repository or a patch file — they see a change. This is
where one is built without any platform credential, which is what makes a
first try possible: a team's own repo, its own rule file, on a laptop, before
any platform integration exists.

Four properties are why this is a module and not a few lines of CLI glue:

* **A local change is identified by its content, not by a commit.** A working
  tree holds edits no commit names, so calling ``HEAD`` the head sha would let
  an approval survive an edit — the exact thing the fingerprint exists to
  prevent. The identity of a local change is the hash of the diff that was
  verified; ``head_sha`` is passed explicitly only when a real commit is what
  is under review.
* **What git cannot diff is still part of the change.** An untracked file is
  invisible to ``git diff``, and "0 findings" on a change whose new file was
  never read is precisely the failure this product exists to prevent. Untracked
  files are inlined as new-file hunks instead — without ``git add -N``, because
  a verification layer does not write to the index of the tree it is judging.
* **Anything that cannot be read is named, not dropped.** A binary, oversized
  or undecodable file still enters ``files_changed``, so the structural
  guardrails see it and the evidence pack says it was not reviewed.
* **An unreadable change raises.** "git is not installed" rendered as an empty
  diff would be a clean verdict on a change nobody looked at.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .contracts import ChangedFile, ChangeRequest
from .diffs import diff_label, is_safe_relative_path, split_lines
from .utils import is_runtime_path

#: A local run is interactive. A git call that hangs should surface as an error
#: rather than stall the operator standing in front of the terminal.
GIT_TIMEOUT_SECONDS = 60

#: Untracked files larger than this are recorded by name but not inlined: a
#: generated asset is not review material, and it would bury the real diff in
#: the evidence pack.
MAX_UNTRACKED_BYTES = 512_000

#: ``--no-renames`` makes a rename a delete plus an add, which is what keeps the
#: three views of the change (diff text, line counts, file list) consistent and
#: the new file's content reviewable line by line. ``--no-ext-diff`` keeps a
#: repo-configured external differ from replacing the unified diff every other
#: layer parses.
_DIFF_FLAGS = ("--no-ext-diff", "--no-renames", "--no-color")

#: git's status letters, narrowed to the contract's vocabulary. With
#: --no-renames only A / D / M / T can appear.
_ChangeType = Literal["added", "modified", "deleted", "renamed"]
_CHANGE_TYPES: dict[str, _ChangeType] = {
    "A": "added",
    "D": "deleted",
    "M": "modified",
    "T": "modified",
}


class IntakeError(RuntimeError):
    """The change could not be read. Never downgraded into an empty diff."""


@dataclass(frozen=True)
class LocalChangeSource:
    """A diff plus where it came from, so the evidence can name its origin."""

    diff: str
    origin: str
    files: tuple[ChangedFile, ...] = ()
    #: Untracked files inlined into the diff, and ones only recorded by name.
    untracked_included: tuple[str, ...] = ()
    untracked_recorded: tuple[str, ...] = ()
    head_commit: str | None = None
    base_commit: str | None = None
    branch: str | None = None


# ── sources ──────────────────────────────────────────────────────────────────


def from_patch(text: str, origin: str) -> LocalChangeSource:
    """Take a unified diff as given.

    No file list is derived here on purpose: ``diffs.py`` is the one parser
    everything else reads a diff through, and a second one in this module is how
    the policy layer and the rule engine would start disagreeing about which
    files a change touched.
    """

    return LocalChangeSource(diff=text, origin=origin)


def read_patch_file(path: Path) -> str:
    try:
        # Bytes first: text mode would turn a "\r" inside a line into a line
        # break, which is not where the diff breaks it.
        return path.read_bytes().decode("utf-8")
    except OSError as exc:
        raise IntakeError(f"Could not read the patch file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise IntakeError(f"The patch file is not UTF-8 text: {path}") from exc


def from_git(repo: Path, base: str | None = None) -> LocalChangeSource:
    """Build a change from a git working tree.

    With ``base``, the comparison is against the merge base with that ref — the
    same set of changes the merge request would carry. Without it, the
    comparison is against ``HEAD``: what is uncommitted right now, which is what
    a first try actually wants to review.

    Either way the *working tree* is one side of the comparison, never a commit
    pair, because the tree is what the scanners will run over.
    """

    head = _revision(repo, "HEAD")
    branch = _revision(repo, "HEAD", abbrev=True)
    base_commit: str | None = None
    if base is not None:
        base_commit = _git(repo, "merge-base", base, "HEAD").strip()
        revisions = [base_commit]
        origin = f"git diff (working tree ↔ {base_commit[:12]}, the merge base with {base})"
    elif head is not None:
        revisions = ["HEAD"]
        origin = "git diff (working tree ↔ HEAD)"
    else:
        # A repository with no commits: everything is untracked, and asking for
        # HEAD would fail. The untracked pass below carries the whole change.
        revisions = []
        origin = "git diff (working tree ↔ empty repository)"

    diff = _git(repo, "diff", *_DIFF_FLAGS, *revisions)
    tracked = _tracked_files(repo, revisions)
    untracked_diff, included, recorded = _inline_untracked(repo, _untracked_paths(repo))

    return LocalChangeSource(
        diff=diff + untracked_diff,
        origin=origin,
        files=(*tracked, *included, *recorded),
        untracked_included=tuple(item.path for item in included),
        untracked_recorded=tuple(item.path for item in recorded),
        head_commit=head,
        base_commit=base_commit,
        branch=branch,
    )


# ── the change ───────────────────────────────────────────────────────────────


def content_identity(diff: str) -> str:
    """Identity of a change that has no commit of its own.

    The fingerprint an approval is bound to is built from this, so it has to
    move whenever the reviewed content moves: same diff, same identity; one
    edited line, a new identity and a dead approval.
    """

    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def build_change(
    source: LocalChangeSource,
    *,
    repo: str,
    head_sha: str | None = None,
    title: str = "",
    author: str | None = None,
    target_branch: str | None = None,
    ai_generated_ratio: float | None = None,
) -> ChangeRequest:
    """Assemble the ``ChangeRequest`` the gates will judge."""

    if not source.diff.strip() and not source.files:
        raise IntakeError(
            "There is nothing to verify: the working tree matches the base, and the patch "
            "is empty. (Already committed? Try --base <target branch>.)"
        )
    return ChangeRequest(
        forge="local",
        repo=repo,
        title=title or f"Local change · {source.origin}",
        description=_description(source),
        author=author,
        source_branch=source.branch,
        target_branch=target_branch,
        head_sha=head_sha or content_identity(source.diff),
        base_sha=source.base_commit,
        diff=source.diff,
        files_changed=list(source.files),
        ai_generated_ratio=ai_generated_ratio,
    )


def _description(source: LocalChangeSource) -> str:
    """What the evidence pack needs in order to reproduce this run."""

    lines = [f"Source: {source.origin}"]
    if source.head_commit:
        lines.append(f"Working tree HEAD: {source.head_commit}")
    if source.untracked_included:
        lines.append("Untracked files included: " + ", ".join(source.untracked_included))
    if source.untracked_recorded:
        # Named rather than dropped: the structural guardrails still see the
        # path, and a reader can tell "not reviewed" from "reviewed and clean".
        lines.append(
            "Untracked files that cannot be reviewed as text (binary, too large or "
            "unreadable), recorded but not reviewed: " + ", ".join(source.untracked_recorded)
        )
    return "\n".join(lines)


# ── git ──────────────────────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> str:
    command = ["git", "--no-pager", "-C", str(repo), *args]
    try:
        # Bytes, not text=True: text mode rewrites every "\r" as "\n", which
        # would split a changed line that holds one into a line with a "+"
        # and a line without. Python runs what follows a lone "\r"; the
        # rules would never see it.
        proc = subprocess.run(
            command,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
            env=_git_env(),
        )
    except FileNotFoundError as exc:
        raise IntakeError(
            "git not found: a local run needs git, or pass a patch file with --diff."
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise IntakeError(f"git {args[0]} could not run: {exc}") from exc
    # A path or a line that is not UTF-8 must not take the whole run down; it
    # will fail the path check or show as a replacement character in the
    # evidence.
    stdout = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        detail = (proc.stderr.decode("utf-8", errors="replace") or stdout).strip().splitlines()
        reason = detail[-1] if detail else f"exit code {proc.returncode}"
        raise IntakeError(f"git {args[0]} failed: {reason}")
    return stdout


def _git_env() -> dict[str, str]:
    """Environment for read-only git.

    ``GIT_OPTIONAL_LOCKS=0`` is the one that matters: without it git refreshes
    (and writes) the index of the tree under verification, and a check that
    modifies what it is checking is not a check.
    """

    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_PAGER"] = "cat"
    env["LC_ALL"] = "C"
    return env


def _revision(repo: Path, ref: str, abbrev: bool = False) -> str | None:
    """Resolve a ref, or ``None`` when there is nothing to resolve.

    A repository with no commits is a legitimate state for a fresh clone, so
    it is not an error here — unlike "this is not a repository", which is.
    """

    args = ["rev-parse", *(["--abbrev-ref"] if abbrev else []), ref]
    try:
        return _git(repo, *args).strip() or None
    except IntakeError:
        _git(repo, "rev-parse", "--git-dir")  # re-raises if it is not a repo
        return None


def _tracked_files(repo: Path, revisions: list[str]) -> tuple[ChangedFile, ...]:
    """File list from git itself, not from re-parsing the diff.

    Binary files and deletions do not carry ``+++`` headers, so a diff-derived
    list would quietly lose them — and losing a deleted ``auth/`` file is how a
    sensitive-path guardrail stops firing.
    """

    counts = _numstat(repo, revisions)
    statuses = _name_status(repo, revisions)
    files: list[ChangedFile] = []
    for path in sorted(set(counts) | set(statuses)):
        added, removed = counts.get(path, (0, 0))
        files.append(
            ChangedFile(
                path=path,
                change_type=_CHANGE_TYPES.get(statuses.get(path, "M"), "modified"),
                added_lines=added,
                removed_lines=removed,
            )
        )
    return tuple(files)


def _numstat(repo: Path, revisions: list[str]) -> dict[str, tuple[int, int]]:
    raw = _git(repo, "diff", *_DIFF_FLAGS, "--numstat", "-z", *revisions)
    counts: dict[str, tuple[int, int]] = {}
    for record in raw.split("\0"):
        if not record.strip():
            continue
        added, _, rest = record.partition("\t")
        removed, _, path = rest.partition("\t")
        if path:
            counts[path] = (_line_count(added), _line_count(removed))
    return counts


def _line_count(value: str) -> int:
    # "-" is git's marker for a binary file: it has no line counts, and
    # pretending it has zero changed lines is the honest reading.
    return int(value) if value.isdigit() else 0


def _name_status(repo: Path, revisions: list[str]) -> dict[str, str]:
    raw = _git(repo, "diff", *_DIFF_FLAGS, "--name-status", "-z", *revisions)
    fields = [field_value for field_value in raw.split("\0") if field_value]
    # With --no-renames every record is exactly (status, path); nothing emits
    # the three-field rename form that would shift this pairing.
    return {fields[index + 1]: fields[index] for index in range(0, len(fields) - 1, 2)}


# ── untracked files ──────────────────────────────────────────────────────────


def tracked_paths(repo: Path) -> tuple[str, ...] | None:
    """Every file git tracks in ``repo``, or ``None`` when git cannot say.

    Public because "which files belong to this repository" is a question
    outside intake too: anything that walks a working tree to find candidates
    (``impact.py`` looking for test files) otherwise walks whatever else is
    lying in the directory — a ``.venv``, a vendored checkout, a second copy of
    the repository inside a worktree.

    ``None`` rather than an empty tuple when git has no answer — no git, not a
    repository, a materialised merge request copied without its ``.git`` — so
    the caller can fall back to walking instead of reading "this repository
    tracks no files".
    """

    try:
        raw = _git(repo, "ls-files", "-z")
    except IntakeError:
        return None
    return tuple(path for path in raw.split("\0") if path)


def _untracked_paths(repo: Path) -> list[str]:
    raw = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    return [path for path in raw.split("\0") if path and not is_runtime_path(Path(path))]


def _inline_untracked(
    repo: Path, paths: list[str]
) -> tuple[str, list[ChangedFile], list[ChangedFile]]:
    """Turn untracked files into new-file hunks, or record why they could not be.

    ``git add -N`` would make them diffable in one line, and it is exactly what
    this must not do: writing to the index of the tree under verification means
    the run changed the thing it was judging.
    """

    root = repo.resolve()
    chunks: list[str] = []
    included: list[ChangedFile] = []
    recorded: list[ChangedFile] = []

    for path in paths:
        text = _readable_text(root, path)
        if text is None:
            recorded.append(ChangedFile(path=path, change_type="added"))
            continue
        # Split where git would, so the lines of this hunk are the lines the
        # tree check reads back out of the file.
        lines = split_lines(text)
        chunks.append(_new_file_hunk(path, text, lines))
        included.append(ChangedFile(path=path, change_type="added", added_lines=len(lines)))
    return "".join(chunks), included, recorded


def _readable_text(root: Path, path: str) -> str | None:
    """The file's content, or ``None`` when it cannot be reviewed as text."""

    if not is_safe_relative_path(path):
        return None
    candidate = root / path
    if candidate.is_symlink():
        # A symlink out of the tree would drag an unrelated file into the
        # evidence under a path that belongs to this repo.
        return None
    resolved = candidate.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(root):
        return None
    try:
        data = candidate.read_bytes()
    except OSError:
        return None
    if len(data) > MAX_UNTRACKED_BYTES:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _new_file_hunk(path: str, text: str, lines: list[str]) -> str:
    header = (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ {diff_label('b/' + path)}\n"
    )
    if not lines:
        # An empty file has no hunk, which is also what git emits. The ``+++``
        # header alone is enough for the diff parser to report the file.
        return header
    body = "".join(f"+{line}\n" for line in lines)
    tail = "" if text.endswith("\n") else "\\ No newline at end of file\n"
    return f"{header}@@ -0,0 +1,{len(lines)} @@\n{body}{tail}"


__all__ = [
    "GIT_TIMEOUT_SECONDS",
    "MAX_UNTRACKED_BYTES",
    "IntakeError",
    "LocalChangeSource",
    "build_change",
    "content_identity",
    "from_git",
    "from_patch",
    "read_patch_file",
    "tracked_paths",
]
