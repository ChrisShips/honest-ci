"""Approvals given on the platform, judged against the evidence they approve.

A run in GitHub Actions keeps no ledger: the runner goes away with the job, and
with it any record of who signed for what. So every run works the approval out
again from what the platform holds -- the decisive reviews on the pull request,
and the window this deployment recorded in its own conclusion comment -- and
the rules for what counts live here, in one function that reads nothing but
its arguments, rather than in the adapter that happened to fetch them.

Each reviewer's latest decisive review is the one read, as the platform reads
it: an approval followed by a request for changes, or one somebody dismissed,
is no longer an approval. An approval then counts only when

* a person gave it, not a bot, from an account that still exists;
* that person can push to the repository;
* it was given on the commit under verification;
* it was given after this evidence was published on the change, which binds it
  to the rule and policy versions as well as to the commit (``ApprovalWindow``)
  -- the commit a review names is not enough: GitHub moves an approval onto the
  new head when a push leaves the pull request's diff as it was;
* and its text gives a reason, because a sign-off without one is not evidence
  -- in the reviewer's own words: anyone who can push can edit a review's text,
  and a reason somebody else wrote there is not the reviewer's.

None counts while somebody who can push has a request for changes standing:
turning the check green past it would put an automated verdict over a person
on the record.

Nothing here touches a gate's verdict. An approval that counts becomes the
run's human decision through ``GateRunner.resume_approval``, the same door
``honest-ci approve`` uses, and a blocked run is never asked about approvals.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from .contracts import (
    ApprovalReading,
    ApprovalWindow,
    PlatformApproval,
    PlatformApprovals,
    UncountedApproval,
)

#: Where a review with no submission time sorts: before everything, so any
#: review the platform did date supersedes it.
_UNDATED = datetime.min.replace(tzinfo=UTC)


def read_approvals(
    found: PlatformApprovals, *, head_sha: str, fingerprint: str
) -> ApprovalReading:
    """What ``found`` amounts to for the evidence ``fingerprint`` of commit ``head_sha``."""

    window = approval_window(found, fingerprint)
    latest = latest_decisions(found.reviews)
    objections = [
        review
        for review in latest
        if review.state == "changes_requested"
        and review.reviewer
        and review.can_push
        and not review.is_bot
    ]
    counted: list[PlatformApproval] = []
    uncounted: list[UncountedApproval] = []
    for review in latest:
        if review.state != "approved":
            continue
        why = _why_not(review, head_sha=head_sha, window=window)
        if why is None and objections:
            why = "a request for changes stands, and no approval counts until it is withdrawn"
        if why is None:
            counted.append(review)
        else:
            uncounted.append(UncountedApproval(review=review, why=why))
    return ApprovalReading(
        window=window,
        approval=counted[0] if counted else None,
        uncounted=uncounted,
        objections=objections,
    )


def approval_window(found: PlatformApprovals, fingerprint: str) -> ApprovalWindow:
    """The window approvals of ``fingerprint`` are read in.

    The one the change carries, when it was recorded for this same fingerprint.
    Otherwise this evidence is new on the change -- a new commit, other rules,
    another policy, or the first run -- so every decisive review already there
    was given on something else, and the window opens after the latest of them.
    """

    recorded = found.window
    if recorded is not None and recorded.fingerprint == fingerprint:
        return recorded
    given = [review.submitted_at for review in found.reviews if review.submitted_at is not None]
    return ApprovalWindow(fingerprint=fingerprint, after=max(given, default=None))


def latest_decisions(reviews: Iterable[PlatformApproval]) -> list[PlatformApproval]:
    """Each reviewer's latest decisive review, oldest first.

    A review from an account that no longer exists belongs to nobody, so it
    stands alone rather than superseding another; it never counts either way.
    """

    latest: dict[str, PlatformApproval] = {}
    orphaned: list[PlatformApproval] = []
    # Sorted stably, so two reviews dated the same second keep the platform's order.
    for review in sorted(reviews, key=_submitted):
        if review.reviewer:
            # Logins are case-insensitive on every platform this reads.
            latest[review.reviewer.lower()] = review
        else:
            orphaned.append(review)
    return sorted([*latest.values(), *orphaned], key=_submitted)


def _why_not(
    review: PlatformApproval, *, head_sha: str, window: ApprovalWindow
) -> str | None:
    """Why this approval does not approve the evidence, or ``None`` when it does."""

    if review.is_bot:
        return "a bot gave it, and an approval has to come from a person"
    if not review.reviewer:
        return "the account that gave it no longer exists"
    if not review.can_push:
        return "the reviewer cannot push to this repository"
    if not review.commit_sha or review.commit_sha.lower() != head_sha.lower():
        shown = review.commit_sha[:12] or "unknown"
        return f"it was given on another commit ({shown}), and a new commit needs a new approval"
    if review.submitted_at is None or (
        window.after is not None and review.submitted_at <= window.after
    ):
        # Equal counts as before: timestamps come to the second, and a review
        # given in the same second as the latest one already there cannot be
        # told apart from it.
        return (
            "it was given before this conclusion was published, "
            "so it approved an earlier one; approve again"
        )
    if review.edited_by and review.edited_by.lower() != review.reviewer.lower():
        return "someone other than the reviewer edited its text; the reviewer has to approve again"
    if not review.reason.strip():
        # A new review rather than an edit of this one: an edit starts no run,
        # so the check would stay as it is until something else started one.
        return "it gives no reason; approve again and write the reason in the review's text"
    return None


def _submitted(review: PlatformApproval) -> datetime:
    return review.submitted_at or _UNDATED


__all__ = ["approval_window", "latest_decisions", "read_approvals"]
