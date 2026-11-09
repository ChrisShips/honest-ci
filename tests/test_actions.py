"""What a run learns from the GitHub Actions runner, and what it refuses to guess."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from honest_ci.actions import FORK_READ_ONLY_EVENTS, ActionsContext

# The placeholder the gitleaks config already allows by exact value.
TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz012345"

BASE_REPOSITORY = {"id": 1001, "full_name": "octo-org/orders"}
FORK_REPOSITORY = {"id": 2002, "full_name": "octo-contributor/orders"}


def runner_environment(**overrides: str) -> dict[str, str]:
    """The variables a runner sets for a step of a pull_request workflow."""

    environment = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_WORKSPACE": "/home/runner/work/orders/orders",
        "GITHUB_REPOSITORY": "octo-org/orders",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "4242",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_EVENT_PATH": "/home/runner/work/_temp/_github_workflow/event.json",
        "GITHUB_STEP_SUMMARY": "/home/runner/work/_temp/_runner_file_commands/step_summary",
    }
    environment.update(overrides)
    return environment


def event_file(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "event.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def pull_request_event(
    head: dict[str, Any] | None, base: dict[str, Any] = BASE_REPOSITORY
) -> dict[str, Any]:
    return {"pull_request": {"number": 7, "head": {"repo": head}, "base": {"repo": base}}}


def context_for(tmp_path: Path, payload: Any, event_name: str = "pull_request") -> ActionsContext:
    return ActionsContext(event_name=event_name, event_path=event_file(tmp_path, payload))


# ── the job ──────────────────────────────────────────────────────────────────


def test_outside_actions_there_is_no_job_even_with_stray_variables() -> None:
    # A developer's shell that happens to export GITHUB_WORKSPACE must not make
    # a local run verify whatever directory it names.
    stray = {"GITHUB_WORKSPACE": "/tmp/elsewhere", "GITHUB_REPOSITORY": "octo-org/orders"}

    assert ActionsContext.from_env(stray) is None
    assert ActionsContext.from_env(runner_environment(GITHUB_ACTIONS="false")) is None


def test_inside_actions_the_runner_variables_describe_the_job() -> None:
    context = ActionsContext.from_env(runner_environment(GITHUB_SERVER_URL="https://github.com/"))

    assert context is not None
    assert context.workspace == Path("/home/runner/work/orders/orders")
    assert context.repository == "octo-org/orders"
    assert context.event_name == "pull_request"
    assert context.step_summary is not None
    assert context.run_url == "https://github.com/octo-org/orders/actions/runs/4242"


@pytest.mark.parametrize(
    "overrides",
    [
        {"GITHUB_RUN_ID": ""},
        {"GITHUB_RUN_ID": "4242/../../../settings"},
        {"GITHUB_SERVER_URL": "javascript:alert(1)"},
        {"GITHUB_REPOSITORY": ""},
    ],
    ids=["no-run-id", "run-id-with-a-path", "not-a-web-address", "no-repository"],
)
def test_a_run_link_is_left_out_rather_than_invented(overrides: dict[str, str]) -> None:
    context = ActionsContext.from_env(runner_environment(**overrides))

    # A status whose link points somewhere made up is worse than no link.
    assert context is not None and context.run_url is None


def test_the_evidence_is_where_the_action_uploads_it() -> None:
    context = ActionsContext.from_env(
        runner_environment(HONEST_CI_EVIDENCE_ARTIFACT="honest-ci-evidence-7-1")
    )

    assert context is not None
    assert context.evidence_at == (
        "the `honest-ci-evidence-7-1` artifact of "
        "[this workflow run](https://github.com/octo-org/orders/actions/runs/4242)"
    )


def test_without_a_run_link_the_artifact_is_still_named() -> None:
    context = ActionsContext.from_env(
        runner_environment(GITHUB_RUN_ID="", HONEST_CI_EVIDENCE_ARTIFACT="honest-ci-evidence-7-1")
    )

    assert context is not None
    assert context.evidence_at == "the `honest-ci-evidence-7-1` artifact of this workflow run"


@pytest.mark.parametrize(
    "name",
    ["", "evidence`](https://example.com)", "evidence/7", "x" * 129],
    ids=["unset", "markdown", "a-path", "too-long"],
)
def test_an_artifact_nobody_declared_is_not_promised(name: str) -> None:
    """A workflow running run-pr without the Action uploads nothing under a known name.

    And a name that could not be quoted safely inside the comment is not
    quoted: the comment says the evidence is kept only if it is uploaded.
    """

    context = ActionsContext.from_env(runner_environment(HONEST_CI_EVIDENCE_ARTIFACT=name))

    assert context is not None and context.evidence_artifact is None
    assert context.evidence_at == (
        "written on the runner of "
        "[this workflow run](https://github.com/octo-org/orders/actions/runs/4242), "
        "and kept only if the workflow uploads it"
    )


def test_the_workflow_owns_its_own_repository_in_any_case() -> None:
    context = ActionsContext(repository="Octo-Org/Orders")

    assert context.owns("octo-org/orders")
    assert not context.owns("octo-org/dashboards")
    assert not ActionsContext().owns("octo-org/orders")


# ── whether the token can write ──────────────────────────────────────────────


@pytest.mark.parametrize("event_name", sorted(FORK_READ_ONLY_EVENTS))
def test_a_pull_request_from_a_fork_gets_a_read_only_token(tmp_path: Path, event_name: str) -> None:
    reason = context_for(
        tmp_path, pull_request_event(FORK_REPOSITORY), event_name
    ).read_only_reason()

    assert reason is not None and "octo-contributor/orders" in reason


def test_a_pull_request_from_the_same_repository_can_be_written_to(tmp_path: Path) -> None:
    context = context_for(tmp_path, pull_request_event(dict(BASE_REPOSITORY)))

    assert context.read_only_reason() is None


def test_a_repository_renamed_since_is_still_the_same_repository(tmp_path: Path) -> None:
    # Compared by id: a rename changes the name, not the repository.
    renamed = {"id": 1001, "full_name": "octo-org/orders-legacy"}

    assert context_for(tmp_path, pull_request_event(renamed)).read_only_reason() is None


def test_a_different_repository_id_is_a_fork_whatever_the_name(tmp_path: Path) -> None:
    namesake = {"id": 2002, "full_name": "octo-org/orders"}

    assert context_for(tmp_path, pull_request_event(namesake)).read_only_reason() is not None


def test_without_ids_the_names_decide_in_any_case(tmp_path: Path) -> None:
    payload = pull_request_event(
        {"full_name": "Octo-Org/Orders"}, base={"full_name": "octo-org/orders"}
    )

    assert context_for(tmp_path, payload).read_only_reason() is None


def test_a_deleted_fork_is_still_a_fork(tmp_path: Path) -> None:
    # GitHub reports a head repository that no longer exists as null.
    reason = context_for(tmp_path, pull_request_event(None)).read_only_reason()

    assert reason is not None and "deleted fork" in reason


@pytest.mark.parametrize(
    "payload",
    [None, "not json", {"action": "opened"}],
    ids=["no-payload-file", "unreadable-payload", "no-pull-request-in-it"],
)
def test_a_payload_that_cannot_rule_a_fork_out_counts_as_one(tmp_path: Path, payload: Any) -> None:
    if payload is None:
        context = ActionsContext(event_name="pull_request", event_path=tmp_path / "missing.json")
    elif isinstance(payload, str):
        path = tmp_path / "event.json"
        path.write_text(payload, encoding="utf-8")
        context = ActionsContext(event_name="pull_request", event_path=path)
    else:
        context = context_for(tmp_path, payload)

    # Wrong one way, a same-repository pull request loses its comment and its
    # required status stays missing, which still blocks the merge. Wrong the
    # other way, every write fails on a 403 and the verdict is lost with it.
    reason = context.read_only_reason()
    assert reason is not None and "cannot be ruled out" in reason


@pytest.mark.parametrize("event_name", ["pull_request_target", "workflow_dispatch", "push", None])
def test_events_whose_token_github_does_not_restrict_are_left_alone(
    tmp_path: Path, event_name: str | None
) -> None:
    # pull_request_target runs with the base repository's token, which can
    # write. Whether a workflow should run a fork's code with that token is its
    # author's decision; skipping the write would only hide that it was made.
    context = ActionsContext(
        event_name=event_name, event_path=event_file(tmp_path, pull_request_event(FORK_REPOSITORY))
    )

    assert context.read_only_reason() is None


# ── the job summary ──────────────────────────────────────────────────────────


def test_the_job_summary_is_appended_to_and_redacted(tmp_path: Path) -> None:
    summary = tmp_path / "step_summary"
    summary.write_text("## from an earlier tool\n", encoding="utf-8")

    written = ActionsContext(step_summary=summary).write_step_summary(
        f"## honest-ci\n\nquoted by a scanner: {TOKEN}"
    )

    text = summary.read_text(encoding="utf-8")
    assert written
    # The file belongs to the step; another tool in it may have written first.
    assert text.startswith("## from an earlier tool\n")
    assert "## honest-ci" in text and text.endswith("\n")
    assert TOKEN not in text


def test_a_job_summary_with_nowhere_to_go_is_reported_rather_than_raised(tmp_path: Path) -> None:
    assert ActionsContext().write_step_summary("## honest-ci") is False
    # A directory where the file should be: the write fails, the run does not.
    assert ActionsContext(step_summary=tmp_path).write_step_summary("## honest-ci") is False
