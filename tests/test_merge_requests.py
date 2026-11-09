"""An approval given on the platform, carried into the run it approves.

``apply_platform_approvals`` is the one door a platform's reviews come in by.
The CLI tests walk through it end to end; these hold its edges: which runs
ask at all, what a refusal from the platform or from the runner leaves behind,
and how a signed run is told.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from honest_ci.adapters.base import ForgeError
from honest_ci.contracts import (
    ApprovalWindow,
    ChangeRequest,
    PlatformApproval,
    PlatformApprovals,
    Verdict,
)
from honest_ci.merge_requests import (
    apply_platform_approvals,
    outcome_label,
    publish_conclusion,
    status_summary,
)
from honest_ci.state import RunState

from .factories import StubForge, make_change
from .test_runner import all_passing, build, present, waiting_on_a_person

REVIEW_URL = "https://github.com/octo-org/orders/pull/7#pullrequestreview-1"
#: The placeholder the gitleaks config already allows by exact value.
TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz012345"


def approval(**overrides: Any) -> PlatformApproval:
    payload: dict[str, Any] = {
        "reviewer": "octo-lead",
        "state": "approved",
        "commit_sha": make_change().head_sha,
        "reason": "Checked the refund split with finance.",
        "can_push": True,
        "submitted_at": datetime(2026, 9, 1, 10, 5, tzinfo=UTC),
        "url": REVIEW_URL,
    }
    payload.update(overrides)
    return PlatformApproval.model_validate(payload)


def given_on(state: RunState, *reviews: PlatformApproval) -> PlatformApprovals:
    """``reviews``, in the window a conclusion on ``state``'s evidence recorded."""

    fingerprint = present(state.evidence).fingerprint
    return PlatformApprovals(reviews=list(reviews), window=ApprovalWindow(fingerprint=fingerprint))


def test_an_approval_that_counts_becomes_the_runs_human_decision(tmp_path: Path) -> None:
    runner, state = waiting_on_a_person(tmp_path)
    forge = StubForge(state.change, approvals=given_on(state, approval()))

    approved, reading = apply_platform_approvals(forge, runner, state)

    assert approved.status == "mergeable"
    decision = present(approved.approval_decision)
    assert decision.approved
    assert (decision.reviewer, decision.reason) == ("octo-lead", approval().reason)
    # Where and when it was given on the platform, not when this run read it.
    assert (decision.url, decision.decided_at) == (REVIEW_URL, approval().submitted_at)
    assert decision.fingerprint == present(state.evidence).fingerprint
    assert reading is not None and reading.approval == approval()
    assert present(runner.services.ledger.load_state(state.run_id)).status == "mergeable"


def test_the_conclusion_carries_what_the_approvals_came_to(tmp_path: Path) -> None:
    runner, state = waiting_on_a_person(tmp_path)
    forge = StubForge(state.change, approvals=given_on(state, approval()))
    approved, reading = apply_platform_approvals(forge, runner, state)

    published = publish_conclusion(forge, approved, approvals=reading)

    assert published.ok and published.sent
    assert forge.readings == [reading]
    assert forge.statuses == [(Verdict.PASS, published.summary)]
    assert "## honest-ci · approved" in published.comment


@pytest.mark.parametrize(
    ("verdict", "status"),
    [(Verdict.BLOCKED, "blocked"), (Verdict.PASS, "mergeable")],
)
def test_only_a_run_waiting_on_a_person_asks_the_platform_about_approvals(
    tmp_path: Path, verdict: Verdict, status: str
) -> None:
    runner = build(tmp_path, spec={"verdicts": [verdict]}, review={}, regression={}, metrics={})
    state = runner.verify_change(make_change())
    assert state.status == status
    forge = StubForge(state.change, approvals=given_on(state, approval()))

    after, reading = apply_platform_approvals(forge, runner, state)

    # A blocked change is fixed by a new commit, not a signature; a mergeable
    # one needs nobody's.
    assert (after, reading) == (state, None)
    assert forge.approvals_fetched == []


def test_a_platform_that_hands_over_no_approvals_leaves_the_run_as_it_was(
    tmp_path: Path,
) -> None:
    runner, state = waiting_on_a_person(tmp_path)
    forge = StubForge(state.change)

    after, reading = apply_platform_approvals(forge, runner, state)

    assert (after, reading) == (state, None)
    assert forge.approvals_fetched == [state.change.ref]


class _Refusing(StubForge):
    def fetch_approvals(self, change: ChangeRequest) -> PlatformApprovals | None:
        raise ForgeError(f"GitHub answered 401 to the token {TOKEN}")


def test_approvals_the_platform_will_not_hand_over_leave_the_run_waiting(
    tmp_path: Path,
) -> None:
    runner, state = waiting_on_a_person(tmp_path)

    after, reading = apply_platform_approvals(_Refusing(state.change), runner, state)

    assert after == state
    assert reading is not None and reading.error is not None
    assert "GitHub answered 401" in reading.error
    assert TOKEN not in reading.error
    # No window either: the next run then reads every review as given before it.
    assert reading.window is None and reading.approval is None


def test_an_approval_the_runner_refuses_leaves_the_run_waiting_and_says_why(
    tmp_path: Path,
) -> None:
    runner, state = waiting_on_a_person(tmp_path)
    # The ledger moved on to other evidence under the approval.
    moved = present(state.evidence).model_copy(update={"fingerprint": "fp-new"})
    runner.services.ledger.save_evidence(state.run_id, moved, "fp-new")
    forge = StubForge(state.change, approvals=given_on(state, approval()))

    after, reading = apply_platform_approvals(forge, runner, state)

    assert after == state
    assert reading is not None
    assert reading.approval is None
    assert "Stale approval" in (reading.error or "")
    assert present(runner.services.ledger.load_state(state.run_id)).status == "awaiting_human"


def test_a_signed_run_is_told_as_approved_and_never_as_merely_mergeable(tmp_path: Path) -> None:
    runner, state = waiting_on_a_person(tmp_path / "approved")
    approved = runner.resume_approval(state.run_id, approved=True, reason="Checked.")
    assert outcome_label(approved) == "approved"
    assert status_summary(approved).startswith("approved · ")

    runner, state = waiting_on_a_person(tmp_path / "rejected")
    rejected = runner.resume_approval(state.run_id, approved=False, reason="Needs a test.")
    assert outcome_label(rejected) == "blocked"

    clean = all_passing(tmp_path / "clean").verify_change(make_change())
    assert outcome_label(clean) == "mergeable"
