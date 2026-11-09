from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from honest_ci import intake
from honest_ci.diffs import AddedLine, added_lines
from honest_ci.gates.regression import check_tree
from honest_ci.integrity import detect
from honest_ci.policy import PolicyConfig, PolicyEngine

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="本地 diff 入口需要 git")

PATCH = """diff --git a/src/app/refund.py b/src/app/refund.py
--- a/src/app/refund.py
+++ b/src/app/refund.py
@@ -10,2 +10,3 @@ def split_refund(order, amount):
     total = 0
+    share = float(amount) / len(order.items)
     return total
"""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "dev@example.com")
    _git(tmp_path, "config", "user.name", "Dev")
    (tmp_path / "app.py").write_text("def total():\n    return 0\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "--quiet", "-m", "init")
    return tmp_path


# ── identity ─────────────────────────────────────────────────────────────────


def test_the_same_diff_always_has_the_same_identity() -> None:
    first = intake.build_change(intake.from_patch(PATCH, "补丁"), repo="team/orders")
    second = intake.build_change(intake.from_patch(PATCH, "补丁"), repo="team/orders")

    assert first.head_sha == second.head_sha


def test_an_edited_diff_gets_a_new_identity_so_an_approval_cannot_survive_it() -> None:
    original = intake.build_change(intake.from_patch(PATCH, "补丁"), repo="team/orders")
    edited = intake.build_change(
        intake.from_patch(PATCH.replace("float(amount)", "Decimal(amount)"), "补丁"),
        repo="team/orders",
    )

    assert original.head_sha != edited.head_sha


def test_a_working_tree_change_is_not_identified_by_the_head_commit(repo: Path) -> None:
    # Two different working trees share one HEAD. Reusing it as the head sha
    # would let an approval granted on one of them carry over to the other.
    (repo / "app.py").write_text("def total():\n    return 1\n", encoding="utf-8")
    head = _git(repo, "rev-parse", "HEAD").strip()

    change = intake.build_change(intake.from_git(repo), repo="local")

    assert change.head_sha != head
    assert change.description.count(head) == 1  # recorded, just not as the identity


def test_an_explicit_head_sha_wins(repo: Path) -> None:
    (repo / "app.py").write_text("def total():\n    return 1\n", encoding="utf-8")

    change = intake.build_change(intake.from_git(repo), repo="local", head_sha="c" * 40)

    assert change.head_sha == "c" * 40


# ── what git cannot diff ─────────────────────────────────────────────────────


def test_an_untracked_file_is_reviewed_rather_than_invisible(repo: Path) -> None:
    (repo / "fee.py").write_text("def fee(amount):\n    return amount * 0.01\n", encoding="utf-8")

    source = intake.from_git(repo)

    assert source.untracked_included == ("fee.py",)
    assert [text for _, text in added_lines(source.diff)["fee.py"]] == [
        "def fee(amount):",
        "    return amount * 0.01",
    ]
    assert any(item.path == "fee.py" and item.added_lines == 2 for item in source.files)


@pytest.mark.parametrize("content", [b"x = 1", b"x = 1\r"], ids=["bare", "carriage-return"])
def test_an_untracked_file_without_a_trailing_newline_still_parses(
    repo: Path, content: bytes
) -> None:
    # A lone "\r" does not end a line for git, so a file that ends in one has
    # no final newline either, and the hunk written for it says so.
    (repo / "fee.py").write_bytes(content)

    source = intake.from_git(repo)

    assert [text for _, text in added_lines(source.diff)["fee.py"]] == ["x = 1"]
    assert source.diff.endswith("\\ No newline at end of file\n")


def test_an_untracked_binary_is_recorded_by_name_not_dropped(repo: Path) -> None:
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

    source = intake.from_git(repo)

    # It cannot be reviewed as text, but the guardrails still have to see the
    # path, and the evidence has to say it was not reviewed.
    assert source.untracked_recorded == ("logo.png",)
    assert "logo.png" not in added_lines(source.diff)
    assert any(item.path == "logo.png" for item in source.files)


def test_an_oversized_untracked_file_is_recorded_not_inlined(repo: Path) -> None:
    (repo / "bundle.js").write_text("x" * (intake.MAX_UNTRACKED_BYTES + 1), encoding="utf-8")

    source = intake.from_git(repo)

    assert source.untracked_recorded == ("bundle.js",)


def test_an_untracked_symlink_is_not_followed_out_of_the_tree(
    repo: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "secret.env"
    outside.write_text("TOKEN=abc\n", encoding="utf-8")
    (repo / "link.env").symlink_to(outside)

    source = intake.from_git(repo)

    assert "TOKEN=abc" not in source.diff
    assert source.untracked_recorded == ("link.env",)


def test_reading_the_change_does_not_write_to_the_index(repo: Path) -> None:
    # A check that modifies the tree it is checking is not a check. ``git add
    # -N`` would make untracked files diffable in one line, and this is why it
    # is not used.
    (repo / "fee.py").write_text("x = 1\n", encoding="utf-8")
    index = repo / ".git" / "index"
    before = index.read_bytes()

    intake.from_git(repo)

    assert index.read_bytes() == before


# ── the file list ────────────────────────────────────────────────────────────


def test_a_deleted_file_stays_in_the_file_list(repo: Path) -> None:
    # A deleted file has no ``+++`` header, so a diff-derived list loses it —
    # and with it every guardrail that fires on the path.
    (repo / "app.py").unlink()

    source = intake.from_git(repo)

    assert [(item.path, item.change_type) for item in source.files] == [("app.py", "deleted")]


def test_a_moved_file_reaches_the_guardrails_under_both_of_its_names(repo: Path) -> None:
    workflows = repo / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "deploy.yml").write_text("on: push\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "deploy workflow")
    (repo / "docs").mkdir()
    _git(repo, "mv", ".github/workflows/deploy.yml", "docs/deploy.yml")

    change = intake.build_change(intake.from_git(repo), repo="local")
    decision = PolicyEngine(PolicyConfig(version="v1")).evaluate_change(change)

    # Without rename detection git reads a move as what it is on disk: one
    # name gone, one name new. This is the reading the forge adapters, which
    # report the same move as one renamed file, are held to.
    assert [(item.path, item.change_type) for item in change.files_changed] == [
        (".github/workflows/deploy.yml", "deleted"),
        ("docs/deploy.yml", "added"),
    ]
    assert decision.violations == [
        "sensitive paths require human review: `.github/workflows/deploy.yml`"
    ]
    assert decision.changed_files == [".github/workflows/deploy.yml", "docs/deploy.yml"]


def test_a_test_file_moved_out_of_collection_is_seen_and_counts_no_lines(repo: Path) -> None:
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_refund.py").write_text(
        "def test_refunds_in_full():\n    assert True\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "refund test")
    (tests / "disabled").mkdir()
    _git(repo, "mv", "tests/test_refund.py", "tests/disabled/refund.py")

    change = intake.build_change(intake.from_git(repo), repo="local")
    hits = detect(change.diff, change.renames)

    # The test is still defined, in the file it moved to, so a match by name
    # alone finds it added back; pytest no longer collects that file.
    assert [(hit.check, hit.file, hit.line) for hit in hits] == [
        ("deleted_test", "tests/test_refund.py", 1)
    ]
    assert "`tests/disabled/refund.py`" in hits[0].detail
    # Moved line for line: no lines, as a forge counts the same rename.
    assert change.changed_lines == 0


def test_a_pasted_patch_that_only_renames_files_reaches_the_guardrails(repo: Path) -> None:
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".github" / "workflows" / "deploy.yml").write_text("on: push\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_refund.py").write_text(
        "def test_refunds_in_full():\n    assert True\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "workflow and test")
    (repo / "docs").mkdir()
    (repo / "tests" / "disabled").mkdir()
    _git(repo, "mv", ".github/workflows/deploy.yml", "docs/deploy.yml")
    _git(repo, "mv", "tests/test_refund.py", "tests/disabled/refund.py")
    (repo / "app.py").write_text("def total():\n    return 1\n", encoding="utf-8")
    # What a developer pastes: git diff detects renames by default, and shows
    # a move without edits as headers alone -- no "---"/"+++", no hunk.
    patch = _git(repo, "diff", "-M", "HEAD")
    assert "rename to docs/deploy.yml" in patch and "+++ b/docs/deploy.yml" not in patch

    change = intake.build_change(intake.from_patch(patch, "补丁"), repo="local")
    decision = PolicyEngine(PolicyConfig(version="v1")).evaluate_change(change)
    hits = detect(change.diff, change.renames)

    assert change.touched_paths == [
        ".github/workflows/deploy.yml",
        "app.py",
        "docs/deploy.yml",
        "tests/disabled/refund.py",
        "tests/test_refund.py",
    ]
    assert decision.violations == [
        "sensitive paths require human review: `.github/workflows/deploy.yml`"
    ]
    assert [(hit.check, hit.file, hit.line) for hit in hits] == [
        ("deleted_test", "tests/test_refund.py", None)
    ]
    assert change.changed_lines == 2


def test_line_counts_come_from_git(repo: Path) -> None:
    (repo / "app.py").write_text("def total():\n    return 0\n    # note\n", encoding="utf-8")

    source = intake.from_git(repo)

    changed = next(item for item in source.files if item.path == "app.py")
    assert (changed.added_lines, changed.removed_lines) == (1, 0)
    assert changed.change_type == "modified"


def test_comparing_against_a_base_covers_committed_and_uncommitted_work(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "app.py").write_text("def total():\n    return 1\n", encoding="utf-8")
    _git(repo, "commit", "--quiet", "-am", "committed work")
    (repo / "app.py").write_text("def total():\n    return 2\n", encoding="utf-8")

    source = intake.from_git(repo, base=base)

    assert source.base_commit == base
    assert "return 2" in source.diff
    assert "return 0" in source.diff  # the base's content is the other side


def test_a_non_ascii_named_file_is_keyed_by_its_real_path(repo: Path) -> None:
    # git quotes such a name in the diff's headers and not in the -z file
    # list. Read as written, the diff would name a file no rule or glob
    # matches, and the tree check would call the change absent from the very
    # tree it was read from.
    _git(repo, "config", "core.quotePath", "true")  # the default, whatever a global config says
    env_file = repo / "config" / "密钥.env"
    env_file.parent.mkdir()
    env_file.write_text("LOG_LEVEL=debug\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "config")
    env_file.write_text("LOG_LEVEL=info\n", encoding="utf-8")

    source = intake.from_git(repo)

    assert added_lines(source.diff) == {"config/密钥.env": [AddedLine(1, "LOG_LEVEL=info")]}
    assert [item.path for item in source.files] == ["config/密钥.env"]
    assert check_tree(repo, intake.build_change(source, repo="local")).status == "present"


#: Names git leaves unquoted, one character away from ``app.py``. A space is
#: never quoted; a no-break space is not once core.quotePath is off.
WHITESPACE_TWINS = [("true", "app.py "), ("false", "app.py "), ("false", "app.py\u00a0")]
WHITESPACE_TWIN_IDS = ["space", "space-unquoted", "no-break-space-unquoted"]


@pytest.mark.parametrize(("quote_path", "name"), WHITESPACE_TWINS, ids=WHITESPACE_TWIN_IDS)
def test_a_name_ending_in_whitespace_keeps_its_own_lines(
    repo: Path, quote_path: str, name: str
) -> None:
    # Trimmed back to app.py, the twin's lines would be judged as app.py's:
    # by every rule, and by the tree check, which would then find the change
    # missing from the very tree it was read from.
    _git(repo, "config", "core.quotePath", quote_path)
    twin = repo / name
    twin.write_text("def total():\n    return 0\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "twin")
    twin.write_text("def total():\n    return 0\nimport os\n", encoding="utf-8")

    source = intake.from_git(repo)

    assert added_lines(source.diff) == {name: [AddedLine(3, "import os")]}
    assert [item.path for item in source.files] == [name]
    assert check_tree(repo, intake.build_change(source, repo="local")).status == "present"


@pytest.mark.parametrize(("quote_path", "name"), WHITESPACE_TWINS, ids=WHITESPACE_TWIN_IDS)
def test_an_untracked_name_ending_in_whitespace_keeps_its_own_lines(
    repo: Path, quote_path: str, name: str
) -> None:
    _git(repo, "config", "core.quotePath", quote_path)
    (repo / name).write_text("import os\n", encoding="utf-8")

    source = intake.from_git(repo)

    assert added_lines(source.diff) == {name: [AddedLine(1, "import os")]}
    assert source.untracked_included == (name,)
    assert check_tree(repo, intake.build_change(source, repo="local")).status == "present"


def test_an_untracked_name_with_a_space_gets_the_header_git_would_write(repo: Path) -> None:
    # Git ends a ``---``/``+++`` label that holds a space with a tab, so GNU
    # patch does not cut the name at the space. The hunk written for an
    # untracked file sits in the same diff as git's own, and reads the same.
    (repo / "my fee.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-N", "my fee.py")
    written_by_git = _git(repo, "diff", "--no-color")
    _git(repo, "reset", "--quiet")

    source = intake.from_git(repo)

    assert "--- /dev/null\n+++ b/my fee.py\t\n" in written_by_git
    assert "--- /dev/null\n+++ b/my fee.py\t\n" in source.diff


# ── where a line ends ────────────────────────────────────────────────────────


def test_a_changed_line_holding_a_carriage_return_is_read_whole(repo: Path) -> None:
    # Python runs what follows a lone "\r". Read in text mode, git's output
    # breaks the line there, and the rules see only the part before it.
    (repo / "app.py").write_bytes(b"def total():\n    return 0\nx = 1\ros.system(cmd)\n")

    source = intake.from_git(repo)

    assert added_lines(source.diff)["app.py"] == [AddedLine(3, "x = 1\ros.system(cmd)")]


def test_a_patch_file_keeps_its_line_breaks(tmp_path: Path) -> None:
    patch = tmp_path / "change.patch"
    hidden = PATCH.replace("len(order.items)", "len(order.items)\ros.system(cmd)")
    patch.write_bytes(hidden.encode())

    text = intake.read_patch_file(patch)

    assert [line for _, line in added_lines(text)["src/app/refund.py"]] == [
        "    share = float(amount) / len(order.items)\ros.system(cmd)"
    ]


def test_an_untracked_file_is_split_where_git_would_split_it(repo: Path) -> None:
    # A form feed ends a line for str.splitlines, not for git. The hunk written
    # for the file has to number it the way the tree check reads the file
    # back, or a local run would find its own change missing.
    (repo / "fee.py").write_text("x = 1\x0cy = 2\nz = 3\n", encoding="utf-8")

    source = intake.from_git(repo)

    assert added_lines(source.diff)["fee.py"] == [
        AddedLine(1, "x = 1\x0cy = 2"),
        AddedLine(2, "z = 3"),
    ]
    assert check_tree(repo, intake.build_change(source, repo="local")).status == "present"


# ── failures are failures, never empty diffs ─────────────────────────────────


def test_a_missing_patch_file_raises(tmp_path: Path) -> None:
    with pytest.raises(intake.IntakeError):
        intake.read_patch_file(tmp_path / "nope.patch")


def test_a_directory_that_is_not_a_repository_raises(tmp_path: Path) -> None:
    with pytest.raises(intake.IntakeError, match="git"):
        intake.from_git(tmp_path)


def test_an_unknown_base_ref_raises(repo: Path) -> None:
    with pytest.raises(intake.IntakeError):
        intake.from_git(repo, base="no-such-branch")


def test_a_change_with_nothing_in_it_raises(repo: Path) -> None:
    # The refusal names the way out: a change that is already committed is
    # found by comparing against the target branch.
    with pytest.raises(intake.IntakeError, match="--base"):
        intake.build_change(intake.from_git(repo), repo="local")


def test_a_repository_without_commits_is_read_from_its_untracked_files(tmp_path: Path) -> None:
    _git(tmp_path, "init", "--quiet")
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

    change = intake.build_change(intake.from_git(tmp_path), repo="local")

    assert "x = 1" in change.diff
    assert change.base_sha is None


# ── what the repository holds ────────────────────────────────────────────────


def test_tracked_paths_names_what_git_tracks_and_nothing_else(repo: Path) -> None:
    (repo / "scratch.py").write_text("x = 1\n", encoding="utf-8")

    tracked = intake.tracked_paths(repo)

    assert tracked == ("app.py",)


def test_tracked_paths_says_it_cannot_tell_rather_than_saying_nothing(tmp_path: Path) -> None:
    # A merge request copied without its ``.git``. "git has no answer" and
    # "this repository is empty" send a caller in opposite directions, so they
    # are never the same value.
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

    assert intake.tracked_paths(tmp_path) is None
