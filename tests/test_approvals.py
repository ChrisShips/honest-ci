"""Which approvals given on the platform approve a run's evidence.

``read_approvals`` reads nothing but its arguments, so every rule is shown here
on the reviews themselves: the three conditions the approval has to meet, the
window that binds it to the rule and policy versions as well as the commit, and
what a later review from the same person does to an earlier one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from honest_ci.approvals import approval_window, latest_decisions, read_approvals
from honest_ci.contracts import (
    ApprovalReading,
    ApprovalWindow,
    PlatformApproval,
    PlatformApprovals,
)

HEAD = "a" * 40
FINGERPRINT = "f" * 64
#: When the conclusion for ``FINGERPRINT`` reached the change, as its window records it.
PUBLISHED = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
RECORDED = ApprovalWindow(fingerprint=FINGERPRINT, after=PUBLISHED)
#: The placeholder the gitleaks config already allows by exact value.
TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz012345"


def minutes(offset: int) -> datetime:
    """``offset`` minutes after the conclusion was published."""

    return PUBLISHED + timedelta(minutes=offset)


def review(**overrides: Any) -> PlatformApproval:
    """An approval that counts, unless ``overrides`` says otherwise."""

    payload: dict[str, Any] = {
        "reviewer": "octo-lead",
        "state": "approved",
        "commit_sha": HEAD,
        "reason": "Checked the rounding with finance; one refund per line item is intended.",
        "can_push": True,
        "submitted_at": minutes(5),
        "url": "https://github.com/octo-org/orders/pull/7#pullrequestreview-1",
    }
    payload.update(overrides)
    return PlatformApproval.model_validate(payload)


def read(
    *reviews: PlatformApproval,
    window: ApprovalWindow | None = RECORDED,
    fingerprint: str = FINGERPRINT,
) -> ApprovalReading:
    found = PlatformApprovals(reviews=list(reviews), window=window)
    return read_approvals(found, head_sha=HEAD, fingerprint=fingerprint)


# ── the three conditions ─────────────────────────────────────────────────────


def test_an_approval_on_this_commit_with_a_reason_from_someone_who_can_push_counts() -> None:
    approval = review()

    reading = read(approval)

    assert reading.approval == approval
    assert reading.uncounted == [] and reading.objections == []
    assert reading.window == RECORDED
    assert reading.error is None


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        ({"can_push": False}, "the reviewer cannot push to this repository"),
        ({"commit_sha": "b" * 40}, "it was given on another commit (bbbbbbbbbbbb)"),
        ({"commit_sha": ""}, "it was given on another commit (unknown)"),
        ({"reason": ""}, "it gives no reason"),
        ({"reason": "  \n\t "}, "it gives no reason"),
        ({"edited_by": "octo-agent"}, "someone other than the reviewer edited its text"),
        ({"submitted_at": minutes(-1)}, "it was given before this conclusion was published"),
        ({"submitted_at": None}, "it was given before this conclusion was published"),
        ({"is_bot": True}, "a bot gave it, and an approval has to come from a person"),
        ({"reviewer": ""}, "the account that gave it no longer exists"),
    ],
    ids=[
        "cannot-push",
        "other-commit",
        "commit-not-given",
        "no-reason",
        "blank-reason",
        "edited-by-someone-else",
        "before-the-conclusion",
        "undated",
        "bot",
        "deleted-account",
    ],
)
def test_an_approval_that_misses_a_condition_does_not_count_and_says_which(
    overrides: dict[str, Any], why: str
) -> None:
    approval = review(**overrides)

    reading = read(approval)

    assert reading.approval is None
    [uncounted] = reading.uncounted
    assert uncounted.review == approval
    assert uncounted.why.startswith(why)


def test_an_approval_without_a_reason_is_told_to_approve_again_not_to_edit() -> None:
    [uncounted] = read(review(reason="")).uncounted

    # Editing the review starts no run, so the check would not move until
    # something else started one; a new review starts one at once.
    assert "approve again" in uncounted.why


def test_an_approval_whose_reviewer_edited_its_text_still_counts() -> None:
    # Only somebody else's edit puts words in the reviewer's mouth; logins
    # match whatever case the platform writes them in.
    assert read(review(edited_by="Octo-Lead")).approval is not None


def test_the_head_commit_is_matched_whatever_case_the_platform_writes_it_in() -> None:
    assert read(review(commit_sha=HEAD.upper())).approval is not None


def test_an_approval_given_in_the_same_second_as_the_window_opened_does_not_count() -> None:
    # Timestamps come to the second: one given in that second cannot be told
    # apart from the review the window was opened after.
    [uncounted] = read(review(submitted_at=PUBLISHED)).uncounted

    assert uncounted.why.startswith("it was given before this conclusion was published")


def test_a_reason_that_quotes_a_credential_is_redacted_before_it_is_kept() -> None:
    approval = review(reason=f"Rotated {TOKEN} after this, the split is fine.")

    reading = read(approval)

    assert reading.approval is not None
    assert TOKEN not in reading.approval.reason
    assert "the split is fine" in reading.approval.reason


def test_a_timestamp_sent_without_a_zone_is_read_as_utc() -> None:
    naive = review(submitted_at=datetime(2026, 9, 1, 10, 5))

    assert naive.submitted_at == minutes(5)
    assert read(naive).approval == naive


# ── the window: binding to the rule and policy versions ──────────────────────


def test_a_window_recorded_for_this_fingerprint_is_kept_as_it_was() -> None:
    later = review(reviewer="octo-qa", state="changes_requested", submitted_at=minutes(30))

    # Reviews given since do not move it: they were given on this evidence.
    assert approval_window(PlatformApprovals(reviews=[later], window=RECORDED), FINGERPRINT) == (
        RECORDED
    )


def test_new_evidence_opens_its_window_after_every_review_already_given() -> None:
    # The same commit, judged again under another rule or policy version: the
    # approval was given on the conclusion the old versions reached.
    earlier = review(submitted_at=minutes(5))
    other = review(reviewer="octo-qa", state="dismissed", submitted_at=minutes(9))
    old = ApprovalWindow(fingerprint="e" * 64, after=PUBLISHED)

    reading = read(earlier, other, window=old)

    assert reading.window == ApprovalWindow(fingerprint=FINGERPRINT, after=minutes(9))
    assert reading.approval is None
    [uncounted] = reading.uncounted
    assert uncounted.review == earlier
    assert uncounted.why.startswith("it was given before this conclusion was published")


def test_an_approval_github_moved_onto_a_new_commit_does_not_count() -> None:
    # After a push that leaves the diff as it was, an empty commit say, GitHub
    # reports the approval on the new head. The new commit is new evidence all
    # the same, so its window opens after the approval.
    moved = review(commit_sha=HEAD, submitted_at=minutes(5))
    previous_commit = ApprovalWindow(fingerprint="e" * 64, after=PUBLISHED)

    reading = read(moved, window=previous_commit)

    assert reading.approval is None
    [uncounted] = reading.uncounted
    assert uncounted.why.startswith("it was given before this conclusion was published")


def test_after_new_evidence_an_approval_given_on_it_counts() -> None:
    old = ApprovalWindow(fingerprint="e" * 64, after=PUBLISHED)
    first = read(review(submitted_at=minutes(5)), window=old)
    assert first.window is not None

    # The next run reads the window the first one recorded in its comment.
    again = review(submitted_at=minutes(12), url=None)
    reading = read(review(submitted_at=minutes(5)), again, window=first.window)

    assert reading.approval == again


def test_the_first_conclusion_on_a_change_opens_its_window_after_the_reviews_already_there() -> (
    None
):
    # Nothing recorded yet: an approval already on the change was given before
    # anyone could read this conclusion.
    reading = read(review(submitted_at=minutes(5)), window=None)

    assert reading.window == ApprovalWindow(fingerprint=FINGERPRINT, after=minutes(5))
    assert reading.approval is None


def test_a_change_nobody_has_reviewed_opens_its_window_from_the_start() -> None:
    reading = read(window=None)

    assert reading.window == ApprovalWindow(fingerprint=FINGERPRINT, after=None)
    assert reading.approval is None and reading.uncounted == []

    # With nothing to have come before it, any approval on the commit is after it.
    approval = review(submitted_at=minutes(-600))
    assert read(approval, window=reading.window).approval == approval


def test_a_window_that_cannot_be_dated_leaves_undated_reviews_before_it() -> None:
    undated = review(submitted_at=None, reviewer="octo-qa")

    assert approval_window(PlatformApprovals(reviews=[undated]), FINGERPRINT).after is None
    assert read(undated, window=ApprovalWindow(fingerprint=FINGERPRINT)).approval is None


# ── later reviews from the same person, and from others ──────────────────────


def test_a_request_for_changes_after_an_approval_withdraws_it() -> None:
    objection = review(
        state="changes_requested", reason="Rounding test missing.", submitted_at=minutes(8)
    )

    reading = read(review(submitted_at=minutes(5)), objection)

    assert reading.approval is None
    # Not an approval that failed a condition: there is no approval any more.
    assert reading.uncounted == []
    assert reading.objections == [objection]


def test_a_dismissed_approval_no_longer_counts() -> None:
    reading = read(review(state="dismissed"))

    assert reading.approval is None
    assert reading.uncounted == [] and reading.objections == []


def test_an_approval_after_a_dismissed_one_counts() -> None:
    again = review(submitted_at=minutes(9))

    assert read(review(state="dismissed", submitted_at=minutes(5)), again).approval == again


def test_a_standing_request_for_changes_from_someone_who_can_push_holds_every_approval() -> None:
    objection = review(reviewer="octo-qa", state="changes_requested", submitted_at=minutes(2))

    reading = read(review(), objection)

    assert reading.approval is None
    [uncounted] = reading.uncounted
    assert uncounted.why.startswith("a request for changes stands")
    assert reading.objections == [objection]


@pytest.mark.parametrize(
    "overrides",
    [{"can_push": False}, {"is_bot": True}, {"reviewer": ""}],
    ids=["cannot-push", "bot", "deleted-account"],
)
def test_a_request_for_changes_that_no_approval_would_need_to_answer_holds_nothing(
    overrides: dict[str, Any],
) -> None:
    objection = review(
        **{
            "reviewer": "octo-qa",
            "state": "changes_requested",
            "submitted_at": minutes(2),
            **overrides,
        }
    )

    reading = read(review(), objection)

    assert reading.approval is not None
    assert reading.objections == []


def test_a_request_for_changes_its_author_later_approved_holds_nothing() -> None:
    withdrawn = review(reviewer="octo-qa", state="changes_requested", submitted_at=minutes(2))
    approval = review(reviewer="octo-qa", submitted_at=minutes(7))

    reading = read(withdrawn, approval)

    assert reading.objections == []
    assert reading.approval == approval


def test_the_earliest_approval_that_counts_is_the_one_recorded() -> None:
    later = review(reviewer="octo-lead", submitted_at=minutes(9))
    earlier = review(reviewer="octo-qa", submitted_at=minutes(6))

    assert read(later, earlier).approval == earlier


def test_one_person_is_one_reviewer_whatever_the_case_of_their_login() -> None:
    reading = read(
        review(reviewer="Octo-Lead", submitted_at=minutes(5)),
        review(reviewer="octo-lead", state="changes_requested", submitted_at=minutes(6)),
    )

    assert reading.approval is None
    assert [objection.reviewer for objection in reading.objections] == ["octo-lead"]


def test_reviews_from_deleted_accounts_stand_alone_and_supersede_nothing() -> None:
    approval = review(reviewer="", submitted_at=minutes(5))
    objection = review(reviewer="", state="changes_requested", submitted_at=minutes(6))

    assert latest_decisions([approval, objection]) == [approval, objection]


def test_two_reviews_in_the_same_second_keep_the_order_the_platform_gave() -> None:
    approval = review(submitted_at=minutes(5))
    objection = review(state="changes_requested", submitted_at=minutes(5))

    assert latest_decisions([approval, objection]) == [objection]
    assert latest_decisions([objection, approval]) == [approval]


def test_each_reviewers_latest_decision_comes_back_oldest_first() -> None:
    first = review(reviewer="octo-qa", submitted_at=minutes(3))
    replaced = review(reviewer="octo-lead", submitted_at=minutes(1))
    latest = review(reviewer="octo-lead", state="changes_requested", submitted_at=minutes(4))
    undated = review(reviewer="octo-ops", submitted_at=None)

    assert latest_decisions([latest, first, replaced, undated]) == [undated, first, latest]
