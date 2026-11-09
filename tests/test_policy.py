from __future__ import annotations

from pathlib import Path

import pytest

from honest_ci.contracts import ChangedFile
from honest_ci.policy import (
    PolicyConfig,
    PolicyEngine,
    count_changed_lines,
    extract_changed_files,
    normalize_diff_path,
)

from .factories import DIFF, make_change


def engine() -> PolicyEngine:
    return PolicyEngine(PolicyConfig(version="v1"))


def test_ordinary_change_stays_on_the_automated_path() -> None:
    decision = engine().evaluate_change(make_change())
    assert decision.allowed
    assert decision.risk_level == "low"
    assert decision.changed_files == ["src/app/refund.py"]


def test_sensitive_path_escalates_to_a_human() -> None:
    change = make_change(
        files_changed=[ChangedFile(path="src/payment/refund.py", added_lines=2)],
    )
    decision = engine().evaluate_change(change)
    assert not decision.allowed
    assert decision.risk_level == "high"
    assert any("sensitive paths" in violation for violation in decision.violations)


@pytest.mark.parametrize("path", ["honest-ci-rules.toml", "honest-ci-policy.toml"])
def test_a_change_to_the_gate_s_own_configuration_goes_to_a_person(path: str) -> None:
    # Deleting a test and switching off the rule that would report it look
    # fine together; a change that edits the checks is not judged by them.
    change = make_change(files_changed=[ChangedFile(path=path, added_lines=1, removed_lines=1)])

    decision = engine().evaluate_change(change)

    assert not decision.allowed
    assert any(path in violation for violation in decision.violations)


def test_blast_radius_beyond_the_limit_escalates() -> None:
    files = [ChangedFile(path=f"src/app/mod_{index}.py", added_lines=90) for index in range(30)]
    decision = PolicyEngine(PolicyConfig(version="v1", max_changed_files=10)).evaluate_change(
        make_change(files_changed=files)
    )
    assert not decision.allowed
    assert any("changed files" in violation for violation in decision.violations)


def test_unsafe_paths_are_reported() -> None:
    change = make_change(files_changed=[ChangedFile(path="../../etc/passwd")])
    decision = engine().evaluate_change(change)
    assert not decision.allowed
    assert any("unsafe path" in violation for violation in decision.violations)


def test_a_rename_counts_toward_the_file_limit_under_both_names() -> None:
    change = make_change(
        files_changed=[
            ChangedFile(
                path="src/app/refunds.py",
                change_type="renamed",
                previous_path="src/app/refund.py",
                added_lines=1,
            ),
            ChangedFile(path="src/app/fees.py", added_lines=1),
        ]
    )

    decision = PolicyEngine(PolicyConfig(version="v1", max_changed_files=2)).evaluate_change(
        change
    )

    # Local intake reads this change as three files, and one limit cannot hold
    # a change on a laptop that it passes on a platform. Nor is a move small:
    # everything that used the old name was moved along with it.
    assert decision.changed_files == ["src/app/fees.py", "src/app/refund.py", "src/app/refunds.py"]
    assert decision.violations == ["changed files 3 exceeds automated limit 2"]


#: A 40-line module moved to another package, as git shows it without rename
#: detection: the whole file gone under one name, the whole file new under the
#: other.
MODULE = [f"line_{index} = {index}" for index in range(40)]
LOCAL_MOVE = (
    "--- a/src/app/ledger.py\n+++ /dev/null\n@@ -1,40 +0,0 @@\n"
    + "".join(f"-{line}\n" for line in MODULE)
    + "--- /dev/null\n+++ b/src/billing/ledger.py\n@@ -0,0 +1,40 @@\n"
    + "".join(f"+{line}\n" for line in MODULE)
)
LOCAL_FILES = [
    ChangedFile(path="src/app/refund.py", added_lines=3),
    ChangedFile(path="src/app/ledger.py", change_type="deleted", removed_lines=40),
    ChangedFile(path="src/billing/ledger.py", change_type="added", added_lines=40),
]


@pytest.mark.parametrize(
    ("diff", "files"),
    [
        (DIFF + LOCAL_MOVE, LOCAL_FILES),
        (DIFF + LOCAL_MOVE, []),
        (
            DIFF,
            [
                ChangedFile(path="src/app/refund.py", added_lines=3),
                ChangedFile(
                    path="src/billing/ledger.py",
                    change_type="renamed",
                    previous_path="src/app/ledger.py",
                ),
            ],
        ),
    ],
    ids=["local", "patch", "forge"],
)
def test_a_file_moved_without_edits_counts_no_lines_wherever_the_change_came_from(
    diff: str, files: list[ChangedFile]
) -> None:
    change = make_change(diff=diff, files_changed=files)

    decision = PolicyEngine(PolicyConfig(version="v1", max_changed_lines=10)).evaluate_change(
        change
    )

    # A forge sends no lines for a pure rename, so it counts none; read as a
    # deletion and an addition it would count eighty, and the same change
    # would be held on a laptop and passed on the platform. The move is not
    # waved through for it: both of its names still count as files.
    assert decision.changed_lines == 3
    assert decision.allowed
    assert decision.changed_files == [
        "src/app/ledger.py",
        "src/app/refund.py",
        "src/billing/ledger.py",
    ]


def test_a_file_moved_with_edits_still_counts_every_line_locally() -> None:
    edited = LOCAL_MOVE.replace("+line_39 = 39\n", "+line_39 = 40\n")
    change = make_change(diff=DIFF + edited, files_changed=[])

    decision = PolicyEngine(PolicyConfig(version="v1", max_changed_lines=10)).evaluate_change(
        change
    )

    # The platform would count two lines here. Locally, pairing the deletion
    # with the addition would take a similarity guess, and a limit that
    # shrinks on a guess is one a change can be shaped to pass: the larger
    # count stands, and the change goes to a person.
    assert decision.changed_lines == 83
    assert decision.violations == ["changed lines 83 exceeds automated limit 10"]


@pytest.mark.parametrize(
    "patch",
    [
        "--- a/.github/workflows/deploy.yml\n+++ /dev/null\n@@ -1 +0,0 @@\n-on: push\n",
        "diff --git a/.github/workflows/deploy.yml b/docs/deploy.yml\n"
        "similarity index 50%\n"
        "rename from .github/workflows/deploy.yml\n"
        "rename to docs/deploy.yml\n"
        "--- a/.github/workflows/deploy.yml\n+++ b/docs/deploy.yml\n"
        "@@ -1 +1 @@\n-on: push\n+on: workflow_dispatch\n",
    ],
    ids=["deleted", "renamed"],
)
def test_a_patch_that_takes_a_file_out_of_a_sensitive_path_goes_to_a_person(patch: str) -> None:
    change = make_change(files_changed=[], diff=DIFF + patch)

    decision = engine().evaluate_change(change)

    # A patch handed over with --diff has no file list, so its headers are the
    # inventory -- both sides of them. Read from the new side alone, a
    # deletion's name is /dev/null and a move's is the one it moved to.
    assert decision.violations == [
        "sensitive paths require human review: `.github/workflows/deploy.yml`"
    ]
    assert ".github/workflows/deploy.yml" in decision.changed_files


def test_policy_file_overrides_defaults(tmp_path: Path) -> None:
    (tmp_path / "honest-ci-policy.toml").write_text(
        """
        [policy]
        version = "v7"
        max_changed_files = 3
        sensitive_paths = ["src/app/**"]
        """,
        encoding="utf-8",
    )
    config = PolicyConfig.load(tmp_path, Path("honest-ci-policy.toml"), "v1")
    assert config.version == "v7"
    assert config.max_changed_files == 3
    decision = PolicyEngine(config).evaluate_change(make_change())
    assert not decision.allowed


def test_diff_helpers() -> None:
    assert extract_changed_files(DIFF) == ["src/app/refund.py"]
    assert count_changed_lines(DIFF) == 3
    assert normalize_diff_path("b/src/app/refund.py") == "src/app/refund.py"
    assert normalize_diff_path("/dev/null") is None


def write_policy(tmp_path: Path, body: str) -> PolicyConfig:
    (tmp_path / "honest-ci-policy.toml").write_text(body, encoding="utf-8")
    return PolicyConfig.load(tmp_path, Path("honest-ci-policy.toml"), "v1")


def test_open_review_threads_hand_the_change_to_a_person_unless_told_otherwise(
    tmp_path: Path,
) -> None:
    assert PolicyConfig(version="v1").unresolved_reviews == "needs_human"
    assert write_policy(tmp_path, '[policy]\nversion = "v1"\n').unresolved_reviews == "needs_human"


def test_a_team_that_leaves_threads_open_can_say_so_in_one_auditable_line(
    tmp_path: Path,
) -> None:
    body = '[policy]\nversion = "v1"\nunresolved_reviews = "ignore"\n'
    assert write_policy(tmp_path, body).unresolved_reviews == "ignore"


def test_a_word_nobody_recognises_is_refused_rather_than_defaulted(tmp_path: Path) -> None:
    # Falling back to the safe verdict would be the right answer reached the
    # wrong way: the team wrote configuration that was never configuration.
    with pytest.raises(ValueError, match="unresolved_reviews"):
        write_policy(tmp_path, '[policy]\nunresolved_reviews = "warn"\n')
    with pytest.raises(ValueError, match="unresolved_reviews"):
        write_policy(tmp_path, "[policy]\nunresolved_reviews = true\n")
