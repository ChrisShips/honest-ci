"""Where the license key check asks for a key, and where it lets a run through.

A key is due only in CI, and there only on an organization's repository that
isn't public, or on a personal account's that isn't, for a pull request another
person opened. Everywhere else, and wherever the check can't tell, none is.
The three places it reads from are exercised as a run reaches them: an event
file as the runner writes it, the pull request the GitHub adapter fetched from
a GitHub on loopback, and a git checkout whose origin is looked up there. So is
the trial, from the workflow file's commits as that GitHub pages them. Last,
what the check makes of the key a run has, or of its trial: whether the gates
run, and what a run says in place of a verdict when they don't.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.parse
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from honest_ci.actions import ActionsContext
from honest_ci.adapters.github import MAX_STATUS_DESCRIPTION, GitHubForgeAdapter
from honest_ci.contracts import ChangeRequest
from honest_ci.license_check import (
    CI_VARIABLES,
    DEPENDABOT_ACTOR,
    KEY_VARIABLE,
    TRIAL_DAYS,
    Cleared,
    KeyDue,
    Ownership,
    Stopped,
    Trial,
    check_license,
    describe_key,
    github_repository,
    in_ci,
    key_due_on_pull_request,
    key_due_on_workspace,
    workflow_trial,
)
from honest_ci.report import VERDICT_MARKER
from honest_ci.settings import Settings

from .factories import StubForge, make_change
from .test_actions import event_file
from .test_github_adapter import (
    BASE_SHA,
    FILE_ENTRY,
    REPO,
    RUNNER_VARIABLES,
    TOKEN,
    FakeGitHub,
    Headed,
    Recorded,
    fake_github,
    git,
    make_adapter,
    serve_pull_request,
)
from .test_licensing import EXPIRES, TEST, b64url, issue, unb64url


@pytest.fixture(autouse=True)
def hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    # The runner sets these in the job that runs this suite in CI, where they
    # would make the adapter a step of a workflow of this repository.
    for name in RUNNER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    # A developer's own git config can rewrite a remote's URL (insteadOf).
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def github() -> Iterator[FakeGitHub]:
    with fake_github() as api:
        yield api


#: A job in a CI system other than GitHub Actions, as the check sees it.
IN_CI = {"CI": "true"}
#: A step of a GitHub Actions workflow, as the check sees it.
IN_ACTIONS = {"GITHUB_ACTIONS": "true", "CI": "true"}

ORG = {"login": "octo-org", "id": 9001, "type": "Organization"}
OWNER = {"login": "octo-dev", "id": 501, "type": "User"}
SOMEONE_ELSE = {"login": "octo-contributor", "id": 502, "type": "User"}
DEPENDABOT = {"login": "dependabot[bot]", "id": 49699333, "type": "Bot"}

ORGANIZATION_KEY = KeyDue(owner_id=9001, owner="octo-org", personal=False)
PERSONAL_KEY = KeyDue(owner_id=501, owner="octo-dev", personal=True)

#: What a platform says that is there to show it wasn't asked: were it asked,
#: every run would need an organization's key.
AN_ORGANIZATIONS_PRIVATE_REPOSITORY = Ownership(
    public=False, owner_id=9001, owner="octo-org", organization=True
)


def repository(owner: dict[str, Any], visibility: str = "private") -> dict[str, Any]:
    """A repository as GitHub describes it: alike in an event, a pull request and ``GET /repos``."""

    return {
        "id": 1001,
        "full_name": f"{owner['login']}/orders",
        "private": visibility != "public",
        "visibility": visibility,
        "owner": owner,
    }


def without(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def pull_request_event(repo: Any, author: Any) -> dict[str, Any]:
    """A ``pull_request`` event for a pull request ``author`` opened from a branch of ``repo``."""

    return {
        "action": "opened",
        "number": 7,
        "pull_request": {
            "number": 7,
            "user": author,
            "head": {"repo": repo},
            "base": {"repo": repo},
        },
        "repository": repo,
        "sender": author,
    }


def in_actions(tmp_path: Path, event: Any, event_name: str = "pull_request") -> ActionsContext:
    return ActionsContext(event_name=event_name, event_path=event_file(tmp_path, event))


class Platform(StubForge):
    """A platform that puts every pull request on ``says``, and counts how often it is asked."""

    def __init__(self, says: Ownership | None) -> None:
        super().__init__()
        self.says = says
        self.asked = 0

    def ownership(self, change: ChangeRequest) -> Ownership | None:
        self.asked += 1
        return self.says


def fetched(
    github: FakeGitHub, repo: Any, author: Any
) -> tuple[GitHubForgeAdapter, ChangeRequest]:
    """A pull request onto ``repo`` that ``author`` opened, as the adapter fetched it."""

    serve_pull_request(
        github, [FILE_ENTRY], base={"ref": "main", "sha": BASE_SHA, "repo": repo}, user=author
    )
    adapter = make_adapter(github)
    return adapter, adapter.fetch_change(REPO, 7)


def checkout(tmp_path: Path, origin: str | None) -> Path:
    """A git repository whose ``origin`` is ``origin``, or that has none."""

    workspace = tmp_path / "orders"
    workspace.mkdir()
    git(workspace, "init", "--quiet")
    if origin is not None:
        git(workspace, "remote", "add", "origin", origin)
    return workspace


def settings_for(github: FakeGitHub, token: str | None = TOKEN) -> Settings:
    """A deployment whose GitHub is ``github``, as ``make_adapter``'s is."""

    return Settings(github_base_url=github.base_url, github_token=token, forge_timeout_seconds=10)


def verify_outside_actions(
    workspace: Path, github: FakeGitHub, settings: Settings | None = None
) -> KeyDue | None:
    """What ``verify`` in another CI system learns, with ``github`` standing in for github.com."""

    return key_due_on_workspace(
        IN_CI, None, workspace, settings or settings_for(github), api=github.base_url
    )


def logged(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "honest_ci.license_check"
    ]


def said_why(caplog: pytest.LogCaptureFixture) -> str:
    """The one line the check logs when it can't tell; it says why."""

    [line] = logged(caplog)
    assert line.startswith(
        "The license check couldn't tell whether this run needs a license key, "
        "so it asks for none: "
    )
    return line


# ── in CI or not ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["true", "1", "True"])
def test_a_job_whose_ci_variable_is_set_is_in_ci(value: str) -> None:
    assert in_ci({"CI": value})


@pytest.mark.parametrize("value", ["false", "0", "False", "", " "])
def test_a_ci_variable_that_is_off_or_empty_says_nothing(value: str) -> None:
    assert not in_ci({"CI": value})


@pytest.mark.parametrize("name", CI_VARIABLES)
def test_a_ci_system_that_doesnt_set_ci_is_known_by_its_own_variable(name: str) -> None:
    assert in_ci({name: "set by the CI system"})
    assert not in_ci({name: ""})


def test_github_actions_is_ci_by_its_own_switch() -> None:
    assert in_ci({"GITHUB_ACTIONS": "true"})
    assert not in_ci({"GITHUB_ACTIONS": "false"})


def test_a_process_no_ci_system_started_is_not_in_ci() -> None:
    assert not in_ci({})
    # A developer's shell, with a stray runner variable in it.
    assert not in_ci({"HOME": "/Users/someone", "GITHUB_WORKSPACE": "/tmp/elsewhere"})


@pytest.mark.parametrize(
    ("repo", "author"),
    [
        pytest.param(repository(ORG), SOMEONE_ELSE, id="an organization's private repository"),
        pytest.param(
            repository(ORG, "internal"), SOMEONE_ELSE, id="an organization's internal repository"
        ),
        pytest.param(
            repository(OWNER),
            SOMEONE_ELSE,
            id="a personal private repository, someone else's pull request",
        ),
        pytest.param(
            repository(OWNER), OWNER, id="a personal private repository, its owner's pull request"
        ),
        pytest.param(
            repository(OWNER),
            DEPENDABOT,
            id="a personal private repository, Dependabot's pull request",
        ),
        pytest.param(repository(ORG, "public"), SOMEONE_ELSE, id="a public repository"),
    ],
)
def test_outside_ci_no_repository_is_asked_for_a_key(
    github: FakeGitHub, tmp_path: Path, repo: dict[str, Any], author: dict[str, Any]
) -> None:
    # On someone's own computer: no CI variable, and no runner.
    adapter, change = fetched(github, repo, author)
    github.on("GET", f"/repos/{repo['full_name']}", payload=repo)
    workspace = checkout(tmp_path, f"https://github.com/{repo['full_name']}.git")
    requests = len(github.recorded)

    assert key_due_on_pull_request({}, None, adapter, change) is None
    assert (
        key_due_on_workspace({}, None, workspace, settings_for(github), api=github.base_url)
        is None
    )
    # Not even looked up.
    assert len(github.recorded) == requests


# ── verify and resume in GitHub Actions: the event says ──────────────────────


@pytest.mark.parametrize(
    ("repo", "author", "due"),
    [
        pytest.param(
            repository(ORG), SOMEONE_ELSE, ORGANIZATION_KEY, id="an organization's private repository"
        ),
        pytest.param(
            repository(ORG, "internal"),
            SOMEONE_ELSE,
            ORGANIZATION_KEY,
            id="an organization's internal repository",
        ),
        pytest.param(
            repository(ORG),
            DEPENDABOT,
            ORGANIZATION_KEY,
            id="an organization's private repository, Dependabot's pull request",
        ),
        pytest.param(
            repository(ORG, "public"), SOMEONE_ELSE, None, id="an organization's public repository"
        ),
        pytest.param(
            repository(OWNER),
            OWNER,
            None,
            id="a personal private repository, its owner's pull request",
        ),
        pytest.param(
            repository(OWNER),
            SOMEONE_ELSE,
            PERSONAL_KEY,
            id="a personal private repository, someone else's pull request",
        ),
        pytest.param(
            repository(OWNER),
            DEPENDABOT,
            None,
            id="a personal private repository, Dependabot's pull request",
        ),
        pytest.param(repository(OWNER, "public"), SOMEONE_ELSE, None, id="a personal public one"),
    ],
)
def test_in_actions_on_a_pull_request_every_command_comes_to_the_same_answer(
    github: FakeGitHub,
    tmp_path: Path,
    repo: dict[str, Any],
    author: dict[str, Any],
    due: KeyDue | None,
) -> None:
    # verify and resume read the event; run-pr reads the pull request it
    # fetched, which on a pull_request run is the event's.
    actions = in_actions(tmp_path, pull_request_event(repo, author))
    adapter, change = fetched(github, repo, author)
    requests = len(github.recorded)

    assert key_due_on_workspace(IN_ACTIONS, actions, tmp_path, Settings()) == due
    assert key_due_on_pull_request(IN_ACTIONS, actions, adapter, change) == due
    # Neither asks GitHub anything more.
    assert len(github.recorded) == requests


@pytest.mark.parametrize(
    ("repo", "due"),
    [
        pytest.param(without(repository(ORG), "visibility"), ORGANIZATION_KEY, id="private"),
        pytest.param(without(repository(ORG), "private"), ORGANIZATION_KEY, id="visibility private"),
        pytest.param(
            without(repository(ORG, "internal"), "private"),
            ORGANIZATION_KEY,
            id="visibility internal",
        ),
        pytest.param(without(repository(ORG, "public"), "visibility"), None, id="not private"),
        pytest.param(without(repository(ORG, "public"), "private"), None, id="visibility public"),
    ],
)
def test_either_private_or_visibility_alone_says_whether_a_repository_is_public(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, repo: dict[str, Any], due: KeyDue | None
) -> None:
    actions = in_actions(tmp_path, {"repository": repo}, "push")

    assert key_due_on_workspace(IN_ACTIONS, actions, tmp_path, Settings()) == due
    # Told, not guessed: a check that couldn't tell would say so.
    assert logged(caplog) == []


def test_the_owner_is_known_by_login_when_the_event_gives_the_author_no_id(tmp_path: Path) -> None:
    author = {"login": "Octo-Dev", "type": "User"}
    actions = in_actions(tmp_path, pull_request_event(repository(OWNER), author))

    assert key_due_on_workspace(IN_ACTIONS, actions, tmp_path, Settings()) is None


@pytest.mark.parametrize("event_name", ["workflow_dispatch", "push", "schedule"])
def test_verify_in_a_workflow_no_pull_request_started_needs_no_key_on_a_personal_repository(
    tmp_path: Path, event_name: str
) -> None:
    # Nothing in the event says that anyone but the owner opens pull requests.
    actions = in_actions(tmp_path, {"repository": repository(OWNER)}, event_name)

    assert key_due_on_workspace(IN_ACTIONS, actions, tmp_path, Settings()) is None


@pytest.mark.parametrize("event_name", ["workflow_dispatch", "push", "schedule"])
def test_in_actions_an_organizations_private_repository_needs_a_key_without_a_pull_request_too(
    tmp_path: Path, event_name: str
) -> None:
    actions = in_actions(tmp_path, {"repository": repository(ORG)}, event_name)

    assert key_due_on_workspace(IN_ACTIONS, actions, tmp_path, Settings()) == ORGANIZATION_KEY


def test_a_pull_request_from_a_fork_runs_read_only_and_is_asked_for_no_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    event = pull_request_event(repository(ORG), SOMEONE_ELSE)
    event["pull_request"]["head"] = {"repo": {"id": 2002, "full_name": "octo-contributor/orders"}}
    actions = in_actions(tmp_path, event)
    platform = Platform(AN_ORGANIZATIONS_PRIVATE_REPOSITORY)

    assert actions.read_only_reason() is not None
    assert key_due_on_pull_request(IN_ACTIONS, actions, platform, make_change()) is None
    assert key_due_on_workspace(IN_ACTIONS, actions, tmp_path, Settings()) is None
    # Not a check that couldn't tell: a fork's run gets no secret to hold a key.
    assert logged(caplog) == []


@pytest.mark.parametrize("content", [None, "{not json", "[]"])
def test_an_event_that_cant_be_read_asks_for_no_key_and_the_log_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, content: str | None
) -> None:
    path = tmp_path / "event.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    actions = ActionsContext(event_name="pull_request", event_path=path)
    platform = Platform(AN_ORGANIZATIONS_PRIVATE_REPOSITORY)

    assert key_due_on_workspace(IN_ACTIONS, actions, tmp_path, Settings()) is None
    assert "the pull_request event couldn't be read" in said_why(caplog)
    # run-pr can't rule out a fork's pull request either, so it runs read-only
    # as a fork's does, and its own output says why.
    assert key_due_on_pull_request(IN_ACTIONS, actions, platform, make_change()) is None


@pytest.mark.parametrize(
    ("event", "reason"),
    [
        pytest.param(
            {**pull_request_event(repository(ORG), SOMEONE_ELSE), "repository": None},
            "the pull_request event doesn't describe its repository",
            id="no repository",
        ),
        pytest.param(
            pull_request_event(without(repository(ORG), "private", "visibility"), SOMEONE_ELSE),
            "the pull_request event doesn't say whether its repository is public",
            id="no visibility",
        ),
        pytest.param(
            pull_request_event(repository(without(ORG, "id")), SOMEONE_ELSE),
            "the pull_request event has no id for its repository's owner",
            id="an owner without an id",
        ),
        pytest.param(
            pull_request_event(repository({**ORG, "id": True}), SOMEONE_ELSE),
            "the pull_request event has no id for its repository's owner",
            id="an owner whose id is true",
        ),
        pytest.param(
            pull_request_event(repository({**ORG, "id": "9001"}), SOMEONE_ELSE),
            "the pull_request event has no id for its repository's owner",
            id="an owner whose id is a string",
        ),
        pytest.param(
            pull_request_event(repository({**ORG, "type": "Enterprise"}), SOMEONE_ELSE),
            'the pull_request event says its repository\'s owner is of type "Enterprise"',
            id="an owner of neither kind",
        ),
        pytest.param(
            pull_request_event(repository(OWNER), None),
            "the pull_request event doesn't describe the pull request's author",
            id="no author",
        ),
        pytest.param(
            pull_request_event(repository(OWNER), without(SOMEONE_ELSE, "login")),
            "the pull_request event has no login for the pull request's author",
            id="an author without a login",
        ),
        pytest.param(
            pull_request_event(repository(OWNER), without(SOMEONE_ELSE, "type")),
            "the pull_request event has no type for the pull request's author",
            id="an author without a type",
        ),
    ],
)
def test_an_event_that_leaves_out_what_a_key_depends_on_asks_for_none_and_the_log_says_what(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, event: dict[str, Any], reason: str
) -> None:
    actions = in_actions(tmp_path, event)

    assert key_due_on_workspace(IN_ACTIONS, actions, tmp_path, Settings()) is None
    assert reason in said_why(caplog)


def test_whatever_the_event_holds_is_redacted_before_the_log_repeats_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    actions = in_actions(tmp_path, pull_request_event(repository({**ORG, "type": TOKEN}), OWNER))

    assert key_due_on_workspace(IN_ACTIONS, actions, tmp_path, Settings()) is None
    line = said_why(caplog)
    assert "says its repository's owner is of type" in line
    assert TOKEN not in line


# ── run-pr: the pull request it fetched says, in Actions or not ──────────────


@pytest.mark.parametrize(
    ("repo", "author", "due"),
    [
        pytest.param(
            repository(ORG), SOMEONE_ELSE, ORGANIZATION_KEY, id="an organization's private repository"
        ),
        pytest.param(
            repository(OWNER),
            SOMEONE_ELSE,
            PERSONAL_KEY,
            id="a personal private repository, someone else's pull request",
        ),
        pytest.param(
            repository(OWNER),
            OWNER,
            None,
            id="a personal private repository, its owner's pull request",
        ),
        pytest.param(
            repository(OWNER),
            DEPENDABOT,
            None,
            id="a personal private repository, Dependabot's pull request",
        ),
        pytest.param(repository(ORG, "public"), SOMEONE_ELSE, None, id="a public repository"),
    ],
)
def test_run_pr_outside_actions_reads_the_pull_request_it_fetched_and_asks_github_nothing_more(
    github: FakeGitHub, repo: dict[str, Any], author: dict[str, Any], due: KeyDue | None
) -> None:
    adapter, change = fetched(github, repo, author)
    requests = len(github.recorded)

    assert key_due_on_pull_request(IN_CI, None, adapter, change) == due
    assert len(github.recorded) == requests


@pytest.mark.parametrize(
    ("event_name", "event"),
    [
        pytest.param(
            "workflow_dispatch",
            {"inputs": {"number": "7"}, "repository": repository(OWNER), "sender": OWNER},
            id="started by hand",
        ),
        pytest.param(
            "issue_comment",
            {
                "action": "created",
                "issue": {
                    "number": 7,
                    "pull_request": {"url": "https://api.github.com/repos/octo-dev/orders/pulls/7"},
                },
                "repository": repository(OWNER),
                "sender": OWNER,
            },
            id="started by a comment",
        ),
        pytest.param(
            "schedule",
            {"schedule": "0 * * * *", "repository": repository(OWNER)},
            id="on a schedule",
        ),
    ],
)
@pytest.mark.parametrize(
    ("author", "due"),
    [
        pytest.param(SOMEONE_ELSE, PERSONAL_KEY, id="someone else's pull request"),
        pytest.param(OWNER, None, id="its owner's pull request"),
    ],
)
def test_in_actions_run_pr_reads_the_pull_request_it_fetched_whatever_started_the_workflow(
    github: FakeGitHub,
    tmp_path: Path,
    event_name: str,
    event: dict[str, Any],
    author: dict[str, Any],
    due: KeyDue | None,
) -> None:
    # On a personal private repository: the event doesn't say who opened the
    # pull request, but the pull request does.
    actions = in_actions(tmp_path, event, event_name)
    adapter, change = fetched(github, repository(OWNER), author)
    requests = len(github.recorded)

    assert key_due_on_pull_request(IN_ACTIONS, actions, adapter, change) == due
    assert len(github.recorded) == requests


#: An organization's public repository, whose workflows run run-pr on pull
#: requests in the organization's other repositories.
CI_REPOSITORY = {**repository(ORG, "public"), "id": 1002, "full_name": "octo-org/ci"}


@pytest.mark.parametrize(
    ("event_name", "event"),
    [
        pytest.param(
            "schedule", {"schedule": "0 * * * *", "repository": CI_REPOSITORY}, id="on a schedule"
        ),
        pytest.param(
            "pull_request",
            pull_request_event(CI_REPOSITORY, SOMEONE_ELSE),
            id="on a pull request of its own",
        ),
    ],
)
def test_run_pr_in_another_repositorys_workflow_needs_the_key_of_the_pull_requests_repository(
    github: FakeGitHub, tmp_path: Path, event_name: str, event: dict[str, Any]
) -> None:
    actions = in_actions(tmp_path, event, event_name)
    adapter, change = fetched(github, repository(ORG), SOMEONE_ELSE)

    assert key_due_on_pull_request(IN_ACTIONS, actions, adapter, change) == ORGANIZATION_KEY


def test_what_the_check_reads_off_a_pull_request_stays_out_of_the_change(
    github: FakeGitHub,
) -> None:
    # The change goes into the evidence pack, and what the check finds must not.
    # created_at comes from the clock, not the pull request, and its digits can
    # spell either id.
    _, change = fetched(github, repository(ORG), SOMEONE_ELSE)
    kept = change.model_dump_json(exclude={"created_at"})

    for fact in ("Organization", '"private"', '"visibility"', "9001", "502"):
        assert fact not in kept


def test_a_pull_request_that_doesnt_describe_its_repository_is_asked_for_no_key(
    github: FakeGitHub, caplog: pytest.LogCaptureFixture
) -> None:
    adapter, change = fetched(github, None, SOMEONE_ELSE)

    assert key_due_on_pull_request(IN_CI, None, adapter, change) is None
    assert "the pull request doesn't describe its repository" in said_why(caplog)


def test_a_platform_that_doesnt_say_whose_repository_it_is_is_asked_for_no_key(
    github: FakeGitHub, caplog: pytest.LogCaptureFixture
) -> None:
    # The port's default, and the GitHub adapter on a pull request it never fetched.
    for platform in (StubForge(), make_adapter(github)):
        caplog.clear()

        assert key_due_on_pull_request(IN_CI, None, platform, make_change()) is None
        assert "the platform didn't say whose repository the pull request is on" in said_why(
            caplog
        )


# ── verify and resume outside Actions: GitHub says, of origin's repository ───


@pytest.mark.parametrize(
    "origin",
    [
        pytest.param("https://github.com/octo-org/orders.git", id="https"),
        pytest.param("git@github.com:octo-org/orders.git", id="ssh"),
    ],
)
@pytest.mark.parametrize(
    ("answers", "due"),
    [
        pytest.param({"/repos/octo-org/orders": repository(ORG, "public")}, None, id="200 public"),
        pytest.param({"/repos/octo-org/orders": repository(ORG)}, ORGANIZATION_KEY, id="200 private"),
        pytest.param(
            {"/repos/octo-org/orders": repository(ORG, "internal")},
            ORGANIZATION_KEY,
            id="200 internal",
        ),
        # What GitHub answers a run whose token can't see the repository, or
        # that has none; then the owner's account says it is an organization.
        pytest.param({"/users/octo-org": ORG}, ORGANIZATION_KEY, id="404"),
    ],
)
def test_verify_outside_actions_asks_github_about_the_repository_origin_names(
    github: FakeGitHub,
    tmp_path: Path,
    origin: str,
    answers: dict[str, Any],
    due: KeyDue | None,
) -> None:
    for path, payload in answers.items():
        github.on("GET", path, payload=payload)

    assert verify_outside_actions(checkout(tmp_path, origin), github) == due


@pytest.mark.parametrize(
    "answers",
    [
        pytest.param({"/repos/octo-dev/orders": repository(OWNER)}, id="200 private"),
        pytest.param({"/users/octo-dev": OWNER}, id="404"),
    ],
)
def test_verify_outside_actions_asks_no_personal_repository_for_a_key(
    github: FakeGitHub, tmp_path: Path, caplog: pytest.LogCaptureFixture, answers: dict[str, Any]
) -> None:
    # There is no pull request, so nothing says anyone but the owner opens them.
    for path, payload in answers.items():
        github.on("GET", path, payload=payload)
    workspace = checkout(tmp_path, "https://github.com/octo-dev/orders.git")

    assert verify_outside_actions(workspace, github) is None
    assert logged(caplog) == []


@pytest.mark.parametrize(
    ("answers", "reason"),
    [
        pytest.param(
            {"/repos/octo-org/orders": (500, {"message": "Server Error"})},
            "GitHub answered 500 to GET /repos/octo-org/orders",
            id="an error",
        ),
        pytest.param(
            {"/repos/octo-org/orders": (403, {"message": "API rate limit exceeded"})},
            "GitHub answered 403 to GET /repos/octo-org/orders",
            id="rate limited",
        ),
        pytest.param(
            {"/repos/octo-org/orders": (204, b"")},
            "GitHub answered 204 to GET /repos/octo-org/orders",
            id="no content",
        ),
        pytest.param(
            {"/repos/octo-org/orders": (200, b"<html>Sign in to your proxy</html>")},
            "GitHub's answer to GET /repos/octo-org/orders isn't JSON",
            id="a sign-in page",
        ),
        pytest.param(
            {"/repos/octo-org/orders": (200, without(repository(ORG), "private", "visibility"))},
            "GitHub's answer to GET /repos/octo-org/orders doesn't say whether its repository "
            "is public",
            id="no visibility",
        ),
        pytest.param({}, "GitHub answered 404 to GET /users/octo-org", id="404, and no such owner"),
        pytest.param(
            {"/users/octo-org": (200, without(ORG, "type"))},
            "GitHub's answer to GET /users/octo-org has no type for the account",
            id="404, and an owner without a type",
        ),
    ],
)
def test_when_github_cant_say_verify_is_asked_for_no_key_and_the_log_says_why(
    github: FakeGitHub,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    answers: dict[str, tuple[int, Any]],
    reason: str,
) -> None:
    for path, (status, payload) in answers.items():
        github.on("GET", path, status, payload)
    workspace = checkout(tmp_path, "https://github.com/octo-org/orders.git")

    assert verify_outside_actions(workspace, github) is None
    assert reason in said_why(caplog)


def test_when_github_cant_be_reached_verify_is_asked_for_no_key_and_the_log_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A port nothing listens on.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    api = f"http://127.0.0.1:{port}/api/v3"
    settings = Settings(github_base_url=api, github_token=TOKEN, forge_timeout_seconds=5)
    workspace = checkout(tmp_path, "https://github.com/octo-org/orders.git")

    assert key_due_on_workspace(IN_CI, None, workspace, settings, api=api) is None
    assert "GitHub couldn't be reached for GET /repos/octo-org/orders" in said_why(caplog)


@pytest.mark.parametrize(
    "origin",
    [
        pytest.param("https://gitlab.com/octo-org/orders.git", id="another platform"),
        pytest.param("git@gitlab.com:octo-org/orders.git", id="another platform over ssh"),
        pytest.param("https://github.example.com/octo-org/orders.git", id="GitHub Enterprise"),
        pytest.param(None, id="no origin"),
    ],
)
def test_verify_outside_actions_without_a_github_com_origin_is_asked_for_no_key(
    github: FakeGitHub, tmp_path: Path, caplog: pytest.LogCaptureFixture, origin: str | None
) -> None:
    assert verify_outside_actions(checkout(tmp_path, origin), github) is None
    # Nothing to look up, and nothing unclear about that.
    assert github.recorded == []
    assert logged(caplog) == []


def test_when_git_cant_be_run_verify_is_asked_for_no_key_and_the_log_says_so(
    github: FakeGitHub,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace = checkout(tmp_path, "https://github.com/octo-org/orders.git")
    monkeypatch.setenv("PATH", str(tmp_path / "no-git-here"))

    assert verify_outside_actions(workspace, github) is None
    assert "git couldn't be run to read origin" in said_why(caplog)
    assert github.recorded == []


def test_the_token_goes_to_the_repository_lookup_and_never_to_the_account_lookup(
    github: FakeGitHub, tmp_path: Path
) -> None:
    github.on("GET", "/users/octo-org", payload=ORG)

    verify_outside_actions(checkout(tmp_path, "https://github.com/octo-org/orders.git"), github)

    [repository_lookup] = github.calls("GET", "/repos/octo-org/orders")
    [account_lookup] = github.calls("GET", "/users/octo-org")
    assert repository_lookup.headers["authorization"] == f"Bearer {TOKEN}"
    assert "authorization" not in account_lookup.headers


def test_a_token_for_another_github_never_reaches_github_com(
    github: FakeGitHub, tmp_path: Path
) -> None:
    # A deployment set up for a GitHub Enterprise Server, verifying a checkout
    # of a github.com repository.
    settings = Settings(
        github_base_url="https://github.example.com/api/v3",
        github_token=TOKEN,
        forge_timeout_seconds=10,
    )
    github.on("GET", "/users/octo-org", payload=ORG)
    workspace = checkout(tmp_path, "https://github.com/octo-org/orders.git")

    assert verify_outside_actions(workspace, github, settings) == ORGANIZATION_KEY
    assert github.recorded
    assert all("authorization" not in call.headers for call in github.recorded)


def test_a_credential_in_origin_goes_nowhere(
    github: FakeGitHub, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Some CI systems check a repository out with a token in the remote's URL.
    origin = f"https://x-access-token:{TOKEN}@github.com/octo-org/orders.git"
    github.on("GET", "/repos/octo-org/orders", 500, {"message": "Server Error"})
    workspace = checkout(tmp_path, origin)

    assert verify_outside_actions(workspace, github, settings_for(github, token=None)) is None
    assert TOKEN not in said_why(caplog)
    assert github.recorded
    assert all(TOKEN not in str(call.headers) for call in github.recorded)


# ── which remotes name a github.com repository ───────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/octo-org/orders.git",
        "https://github.com/octo-org/orders",
        "https://github.com/octo-org/orders/",
        "https://ci-bot@github.com/octo-org/orders.git",
        "http://github.com/octo-org/orders.git",
        "git@github.com:octo-org/orders.git",
        "git@github.com:octo-org/orders",
        "GitHub.com:octo-org/orders.git",
        "ssh://git@github.com/octo-org/orders.git",
        "ssh://git@ssh.github.com:443/octo-org/orders.git",
        "git://github.com/octo-org/orders.git",
    ],
)
def test_each_way_a_remote_names_a_github_com_repository_is_read(url: str) -> None:
    assert github_repository(url) == ("octo-org", "orders")


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/octo-org/orders.git",
        "git@gitlab.com:octo-org/orders.git",
        "https://github.example.com/octo-org/orders.git",
        "https://github.com.example.net/octo-org/orders.git",
        "file://github.com/octo-org/orders.git",
        "https://github.com/octo-org",
        "https://github.com/octo-org/orders/pull/7",
        "https://github.com/octo-org/orders.git?ref=main",
        "https://github.com/../orders",
        "https://github.com:not-a-port/octo-org/orders.git",
        "/srv/git/orders.git",
        "../orders",
        "",
    ],
)
def test_a_remote_that_names_no_github_com_repository_is_not_read(url: str) -> None:
    assert github_repository(url) is None


# ── the trial: from the workflow file's first commit ─────────────────────────

WORKFLOW = ".github/workflows/honest-ci.yml"
#: The commit the run took its workflow file from: on a pull request, the
#: merge commit GitHub made for it.
WORKFLOW_SHA = "5f3c2a9e8d7b6a5f4e3d2c1b0a9f8e7d6c5b4a39"
COMMITS = "/repos/octo-org/orders/commits"


def trial_job(github: FakeGitHub, **variables: str) -> dict[str, str]:
    """A workflow's step on a pull request, as the runner describes it, on ``github``."""

    return {
        **IN_ACTIONS,
        "GITHUB_API_URL": github.base_url,
        "GITHUB_WORKFLOW_REF": f"{REPO}/{WORKFLOW}@refs/pull/7/merge",
        "GITHUB_WORKFLOW_SHA": WORKFLOW_SHA,
        **variables,
    }


def commit_made(committed: str, authored: str | None = None) -> dict[str, Any]:
    """A commit as the commits endpoint lists it: ``committed`` and ``authored`` are its dates."""

    return {
        "sha": committed.translate({ord(c): None for c in "-:TZ+"}).ljust(40, "0")[:40],
        "commit": {
            "author": {"name": "Octo Dev", "date": authored or committed},
            "committer": {"name": "Octo Dev", "date": committed},
            "message": "Run honest-ci on pull requests",
        },
    }


def serve_commits(github: FakeGitHub, commits: list[dict[str, Any]]) -> None:
    """The workflow file's ``commits``, newest first, paged as GitHub pages them.

    The links name the repository by its id, as GitHub's do, and every page
    links a next one, the last page included, as the commits endpoint's do.
    An empty list comes with no ``Link`` at all.
    """

    def route(call: Recorded) -> tuple[int, Any]:
        size = int(call.query["per_page"][0])
        page = int(call.query.get("page", ["1"])[0])
        items = commits[(page - 1) * size : page * size]
        if not commits:
            return 200, items
        last = -(-len(commits) // size)
        query = urllib.parse.urlencode(
            {key: values[0] for key, values in call.query.items() if key != "page"}
        )
        url = f"{github.base_url}/repositories/1001/commits?{query}&page="
        links = [f'<{url}{page + 1}>; rel="next"', f'<{url}{last}>; rel="last"']
        if page > 1:
            links += [f'<{url}1>; rel="first"', f'<{url}{page - 1}>; rel="prev"']
        return 200, Headed(items, {"Link": ", ".join(links)})

    github.on_call("GET", COMMITS, route)


def serve_history(github: FakeGitHub, *days: str) -> None:
    """The workflow file's commits, one at noon UTC on each of ``days``, newest first."""

    serve_commits(github, [commit_made(f"{day}T12:00:00Z") for day in days])


def no_trial_start(caplog: pytest.LogCaptureFixture) -> str:
    """The one line the check logs when it can't read the trial's first day; it says why."""

    [line] = logged(caplog)
    assert line.startswith(
        "The license check couldn't tell when this repository's 14-day trial started, "
        "so it lets the run go ahead without a license key: "
    )
    return line


def test_the_trial_starts_the_day_the_workflow_file_was_first_committed(
    github: FakeGitHub,
) -> None:
    serve_history(github, "2026-10-07", "2026-10-01", "2026-09-28")

    trial = workflow_trial(trial_job(github), settings_for(github))

    assert trial == Trial(workflow=WORKFLOW, started=date(2026, 9, 28))
    # From the commit the run took the file from, one commit a page; then
    # straight to the last page, at the endpoint asked, wherever a link points.
    first, last = github.calls("GET", COMMITS)
    assert first.query == {"path": [WORKFLOW], "sha": [WORKFLOW_SHA], "per_page": ["1"]}
    assert last.query == {**first.query, "page": ["3"]}


def test_a_workflow_file_committed_once_takes_one_request(github: FakeGitHub) -> None:
    serve_history(github, "2026-10-02")

    assert workflow_trial(trial_job(github), settings_for(github)) == Trial(
        workflow=WORKFLOW, started=date(2026, 10, 2)
    )
    assert len(github.calls("GET", COMMITS)) == 1


def test_a_page_without_links_is_the_only_page(github: FakeGitHub) -> None:
    github.on("GET", COMMITS, payload=[commit_made("2026-10-02T12:00:00Z")])

    assert workflow_trial(trial_job(github), settings_for(github)) == Trial(
        workflow=WORKFLOW, started=date(2026, 10, 2)
    )
    assert len(github.calls("GET", COMMITS)) == 1


def test_the_trial_counts_from_the_day_of_the_commit_not_of_its_change_in_utc(
    github: FakeGitHub,
) -> None:
    # Committed late in the evening west of UTC, which is the next day in
    # UTC, with a change first written months before.
    serve_commits(
        github,
        [commit_made("2026-09-28T22:30:00-05:00", authored="2026-06-01T09:00:00Z")],
    )

    trial = workflow_trial(trial_job(github), settings_for(github))

    assert trial is not None
    assert trial.started == date(2026, 9, 29)


@pytest.fixture
def west_of_utc() -> Iterator[None]:
    """A machine whose own zone is behind UTC, so that a local evening is UTC's next day."""

    before = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"
    time.tzset()
    try:
        yield
    finally:
        if before is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = before
        time.tzset()


@pytest.mark.usefixtures("west_of_utc")
def test_a_commit_date_without_a_zone_is_read_as_utc_not_as_the_machines_time(
    github: FakeGitHub,
) -> None:
    serve_commits(github, [commit_made("2026-10-02T23:30:00")])

    trial = workflow_trial(trial_job(github), settings_for(github))

    assert trial is not None
    assert trial.started == date(2026, 10, 2)


def test_the_trial_lasts_fourteen_days_its_first_day_included() -> None:
    trial = Trial(workflow=WORKFLOW, started=date(2026, 10, 1))

    assert TRIAL_DAYS == 14
    assert trial.works_until == date(2026, 10, 14)
    days = [date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 14), date(2026, 10, 15)]
    assert [trial.days_left(day) for day in days] == [14, 13, 1, 0]
    assert [trial.over(day) for day in days] == [False, False, False, True]
    assert trial.days_left(date(2027, 1, 1)) == 0


def test_a_commit_dated_ahead_of_the_clock_leaves_no_more_than_a_whole_trial() -> None:
    trial = Trial(workflow=WORKFLOW, started=date(2026, 10, 20))

    assert trial.days_left(date(2026, 10, 8)) == TRIAL_DAYS
    assert not trial.over(date(2026, 10, 8))


@pytest.mark.parametrize(
    ("workflow_ref", "workflow"),
    [
        pytest.param(
            f"{REPO}/.github/workflows/honest-ci.yml@refs/heads/fix@home",
            ".github/workflows/honest-ci.yml",
            id="an @ in the ref",
        ),
        pytest.param(
            f"{REPO}/.github/workflows/checks.yaml@refs/heads/main",
            ".github/workflows/checks.yaml",
            id=".yaml",
        ),
        pytest.param(
            f"{REPO}/.github/workflows/pull request checks.YML@refs/tags/v1.0",
            ".github/workflows/pull request checks.YML",
            id="a space, and the ending in capitals",
        ),
    ],
)
def test_the_workflow_file_is_read_off_the_ref_the_runner_gives(
    github: FakeGitHub, workflow_ref: str, workflow: str
) -> None:
    serve_history(github, "2026-10-02")

    trial = workflow_trial(
        trial_job(github, GITHUB_WORKFLOW_REF=workflow_ref), settings_for(github)
    )

    assert trial == Trial(workflow=workflow, started=date(2026, 10, 2))
    [call] = github.calls("GET", COMMITS)
    assert call.query["path"] == [workflow]


def test_the_trial_is_looked_up_with_the_runs_token(github: FakeGitHub) -> None:
    serve_history(github, "2026-10-07", "2026-10-01")

    workflow_trial(trial_job(github), settings_for(github))

    calls = github.calls("GET", COMMITS)
    assert len(calls) == 2
    assert all(call.headers["authorization"] == f"Bearer {TOKEN}" for call in calls)


def test_a_token_for_another_github_never_goes_to_the_one_the_run_is_on(
    github: FakeGitHub,
) -> None:
    # A deployment set up for github.com, running on a GitHub Enterprise Server.
    settings = Settings(
        github_base_url="https://api.github.com", github_token=TOKEN, forge_timeout_seconds=10
    )
    serve_history(github, "2026-10-07", "2026-10-01")

    assert workflow_trial(trial_job(github), settings) == Trial(
        workflow=WORKFLOW, started=date(2026, 10, 1)
    )
    assert github.recorded
    assert all("authorization" not in call.headers for call in github.recorded)


@pytest.mark.parametrize(
    ("variables", "reason"),
    [
        pytest.param(
            {"GITHUB_WORKFLOW_REF": ""},
            "GITHUB_WORKFLOW_REF doesn't name a workflow file in .github/workflows",
            id="no workflow ref",
        ),
        pytest.param(
            {"GITHUB_WORKFLOW_REF": f"{REPO}/dynamic/pages/pages-build-deployment@refs/heads/main"},
            "GITHUB_WORKFLOW_REF doesn't name a workflow file in .github/workflows",
            id="a workflow GitHub made",
        ),
        pytest.param(
            {"GITHUB_WORKFLOW_REF": f"{REPO}/.github/workflows/ci/honest-ci.yml@refs/heads/main"},
            "GITHUB_WORKFLOW_REF doesn't name a workflow file in .github/workflows",
            id="a file in a folder below",
        ),
        pytest.param(
            {"GITHUB_WORKFLOW_REF": f"{REPO}/.github/workflows/honest-ci.json@refs/heads/main"},
            "GITHUB_WORKFLOW_REF doesn't name a workflow file in .github/workflows",
            id="not YAML",
        ),
        pytest.param(
            {"GITHUB_WORKFLOW_REF": "../orders/.github/workflows/honest-ci.yml@refs/heads/main"},
            "GITHUB_WORKFLOW_REF doesn't name a workflow file in .github/workflows",
            id="an owner that is a path",
        ),
        pytest.param(
            {"GITHUB_WORKFLOW_SHA": ""},
            "GITHUB_WORKFLOW_SHA doesn't name the commit the workflow file is from",
            id="no workflow commit",
        ),
        pytest.param(
            {"GITHUB_WORKFLOW_SHA": "main"},
            "GITHUB_WORKFLOW_SHA doesn't name the commit the workflow file is from",
            id="a branch for a commit",
        ),
        pytest.param(
            {"GITHUB_API_URL": ""},
            "GITHUB_API_URL doesn't say where GitHub's API is",
            id="no API",
        ),
    ],
)
def test_a_run_that_cant_name_its_workflow_file_goes_ahead_and_asks_github_nothing(
    github: FakeGitHub,
    caplog: pytest.LogCaptureFixture,
    variables: dict[str, str],
    reason: str,
) -> None:
    serve_history(github, "2026-09-01")

    assert workflow_trial(trial_job(github, **variables), settings_for(github)) is None
    assert reason in no_trial_start(caplog)
    assert github.recorded == []


def _answering(status: int, payload: Any) -> Callable[[FakeGitHub], None]:
    return lambda github: github.on("GET", COMMITS, status, payload)


def _linked(link: str) -> Callable[[FakeGitHub], None]:
    """A first page with one commit, and ``link`` for its ``Link`` header."""

    page = Headed([commit_made("2026-10-07T12:00:00Z")], {"Link": link})
    return _answering(200, page)


def _shrunk(github: FakeGitHub) -> None:
    """A history that lost its first commits between the two requests, as after a force push."""

    first_page = Headed(
        [commit_made("2026-10-07T12:00:00Z")],
        {"Link": f'<{github.base_url}/repositories/1001/commits?page=3>; rel="last"'},
    )
    github.on_call("GET", COMMITS, lambda call: (200, [] if "page" in call.query else first_page))


@pytest.mark.parametrize(
    ("serve", "reason"),
    [
        pytest.param(
            _answering(404, {"message": "Not Found"}),
            "GitHub answered 404 to GET /repos/octo-org/orders/commits",
            id="a token that can't see the repository",
        ),
        pytest.param(
            _answering(500, {"message": "Server Error"}),
            "GitHub answered 500 to GET /repos/octo-org/orders/commits",
            id="an error",
        ),
        pytest.param(
            _answering(200, b"<html>Sign in to your proxy</html>"),
            "GitHub's answer to GET /repos/octo-org/orders/commits isn't JSON",
            id="a sign-in page",
        ),
        pytest.param(
            _answering(200, {"message": "Not a list"}),
            "GitHub's answer to GET /repos/octo-org/orders/commits isn't a list of commits",
            id="not a list",
        ),
        pytest.param(
            _answering(200, []),
            "GitHub's answer to GET /repos/octo-org/orders/commits lists no commit that touched "
            ".github/workflows/honest-ci.yml",
            id="no commits",
        ),
        pytest.param(
            _shrunk,
            "GitHub's answer to GET /repos/octo-org/orders/commits lists no commit that touched "
            ".github/workflows/honest-ci.yml",
            id="an empty last page",
        ),
        pytest.param(
            _linked('<https://api.github.com/repositories/1001/commits?page=2>; rel="next"'),
            "GitHub's answer to GET /repos/octo-org/orders/commits has more pages but doesn't say "
            "which is the last",
            id="a next page and no last",
        ),
        pytest.param(
            _linked(
                '<https://api.github.com/repositories/1001/commits?cursor=Y3Vyc29y>; rel="last"'
            ),
            "GitHub's answer to GET /repos/octo-org/orders/commits doesn't say which page is the "
            "last",
            id="a last page with no number",
        ),
        pytest.param(
            _linked('<https://api.github.com/repositories/1001/commits?page=0>; rel="last"'),
            "GitHub's answer to GET /repos/octo-org/orders/commits doesn't say which page is the "
            "last",
            id="page 0",
        ),
        pytest.param(
            _linked(
                f'<https://api.github.com/repositories/1001/commits?page={"9" * 5000}>; rel="last"'
            ),
            "GitHub's answer to GET /repos/octo-org/orders/commits doesn't say which page is the "
            "last",
            id="a page number longer than any history",
        ),
        pytest.param(
            _linked('<http://[::1/commits?page=2>; rel="last"'),
            "GitHub's answer to GET /repos/octo-org/orders/commits doesn't say which page is the "
            "last",
            id="a link that isn't a URL",
        ),
        pytest.param(
            _answering(200, [without(commit_made("2026-10-07T12:00:00Z"), "commit")]),
            "GitHub's answer to GET /repos/octo-org/orders/commits doesn't date the commit",
            id="no date",
        ),
        pytest.param(
            _answering(200, [commit_made(TOKEN)]),
            "GitHub's answer to GET /repos/octo-org/orders/commits dates the commit ",
            id="whatever the answer holds, redacted",
        ),
        pytest.param(
            _answering(200, [commit_made("yesterday")]),
            'GitHub\'s answer to GET /repos/octo-org/orders/commits dates the commit "yesterday", '
            "which isn't a date",
            id="not a date",
        ),
    ],
)
def test_when_github_cant_say_when_the_trial_started_the_run_goes_ahead_and_the_log_says_why(
    github: FakeGitHub,
    caplog: pytest.LogCaptureFixture,
    serve: Callable[[FakeGitHub], None],
    reason: str,
) -> None:
    serve(github)

    assert workflow_trial(trial_job(github), settings_for(github)) is None
    line = no_trial_start(caplog)
    assert reason in line
    assert TOKEN not in line


def test_when_github_cant_be_reached_the_run_goes_ahead_and_the_log_says_so(
    github: FakeGitHub, caplog: pytest.LogCaptureFixture
) -> None:
    # A port nothing listens on.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    job = trial_job(github, GITHUB_API_URL=f"http://127.0.0.1:{port}/api/v3")

    assert workflow_trial(job, settings_for(github)) is None
    assert "GitHub couldn't be reached for GET /repos/octo-org/orders/commits" in no_trial_start(
        caplog
    )


# ── whether the gates run ────────────────────────────────────────────────────

#: A day a key ``issue`` signs works on, months before its expiry date.
TODAY = date(2027, 6, 1)
#: The trial of the workflow file ``serve_history(github, TRIAL_START)`` makes.
TRIAL_START = "2026-10-01"


def org_key(**fields: Any) -> str:
    """A key the test key pair signed for octo-org, the owner ``ORGANIZATION_KEY`` names."""

    return issue(org_id=9001, **fields)


def extended(key: str) -> str:
    """``key`` with its expiry date moved on after it was signed, its signature as it was."""

    prefix, payload, signature = key.split(".")
    fields = {**json.loads(unb64url(payload)), "expires": "2099-12-31"}
    return f"{prefix}.{b64url(json.dumps(fields).encode())}.{signature}"


def check(
    due: KeyDue | None,
    environ: dict[str, str],
    today: date,
    settings: Settings | None = None,
) -> Cleared | Stopped:
    """What the check makes of a run, its keys verified against the test key pair's."""

    return check_license(due, environ, settings or Settings(), today=today, public_key=TEST)


def stops(
    due: KeyDue,
    environ: dict[str, str],
    today: date,
    settings: Settings | None = None,
) -> Stopped:
    """What the check says of a run it stops; fails the test when it lets the run go ahead."""

    checked = check(due, environ, today, settings)
    assert isinstance(checked, Stopped), checked
    return checked


def test_where_no_key_is_due_the_gates_run_and_no_key_is_read() -> None:
    environ = {**IN_ACTIONS, KEY_VARIABLE: "not a key"}

    assert check(None, environ, TODAY) == Cleared()


def test_a_key_for_the_repositorys_owner_lets_the_gates_run_and_asks_github_nothing(
    github: FakeGitHub,
) -> None:
    # In Actions, where a run without a key would look its trial up.
    job = trial_job(github, **{KEY_VARIABLE: org_key()})

    assert check(ORGANIZATION_KEY, job, TODAY, settings_for(github)) == Cleared()
    assert github.recorded == []


def test_a_key_pasted_with_a_line_break_after_it_works() -> None:
    environ = {**IN_CI, KEY_VARIABLE: f"{org_key()}\n"}

    assert check(ORGANIZATION_KEY, environ, TODAY) == Cleared()


def test_on_its_expiry_date_a_key_works_and_the_run_says_nothing_of_it() -> None:
    environ = {**IN_CI, KEY_VARIABLE: org_key()}

    assert check(ORGANIZATION_KEY, environ, EXPIRES) == Cleared()


@pytest.mark.parametrize(
    ("days_before", "left"), [(13, "14 days"), (0, "1 day")], ids=["first", "last"]
)
def test_a_trial_key_works_through_its_expiry_date_and_the_run_says_how_many_days_are_left(
    days_before: int, left: str
) -> None:
    """It stops the day after, with no grace days, so every run before then says so."""

    environ = {**IN_CI, KEY_VARIABLE: org_key(plan="trial", seats=0)}

    assert check(ORGANIZATION_KEY, environ, EXPIRES - timedelta(days=days_before)) == Cleared(
        note=(
            f"honest-ci is on a trial key here: {left} left, through 2027-11-13. After that it "
            "needs a subscription, which comes with a new key for `HONEST_CI_LICENSE_KEY`: "
            "https://honest-ci.dev/pricing."
        )
    )


@pytest.mark.parametrize(
    ("days_past", "left"), [(1, "30 days"), (30, "1 day")], ids=["first", "last"]
)
def test_a_key_past_its_expiry_date_works_through_its_grace_days_and_the_run_says_how_many_are_left(
    days_past: int, left: str
) -> None:
    environ = {**IN_CI, KEY_VARIABLE: org_key()}

    assert check(ORGANIZATION_KEY, environ, EXPIRES + timedelta(days=days_past)) == Cleared(
        note=(
            "The license key for octo-org expired on 2027-11-13 and keeps working through "
            f"2027-12-13: {left} left. A renewal comes with a new key for "
            "`HONEST_CI_LICENSE_KEY`; to renew, see https://honest-ci.dev/pricing."
        )
    )


@pytest.mark.parametrize(
    ("key", "today", "status"),
    [
        (
            org_key(),
            EXPIRES + timedelta(days=31),
            "it expired on 2027-11-13, more than 30 days ago",
        ),
        # A trial key has no grace days: the day after its expiry date, it stops.
        (
            org_key(plan="trial", seats=0),
            EXPIRES + timedelta(days=1),
            "it is a trial key that expired on 2027-11-13",
        ),
        (issue(org_id=4242, org="another-org"), TODAY, "it was issued for another-org (id 4242)"),
        # GitHub's longest login: the status still names the id whole.
        (issue(org_id=4242, org="o" * 39), TODAY, f"it was issued for {'o' * 39} (id 4242)"),
        (extended(org_key()), TODAY, "its signature doesn't match"),
        (org_key(v=2), TODAY, "it is a version 2 key"),
        (
            org_key().rsplit(".", 1)[0],
            TODAY,
            "it isn't an honest-ci license key, which is hci1.<payload>.<signature> on one line",
        ),
    ],
    ids=[
        "expired",
        "trial-expired",
        "another-owner",
        "longest-login",
        "changed",
        "later-format",
        "cut-short",
    ],
)
def test_a_key_that_doesnt_work_here_stops_the_run_which_says_why_and_never_the_key(
    key: str, today: date, status: str
) -> None:
    stopped = stops(ORGANIZATION_KEY, {**IN_CI, KEY_VARIABLE: key}, today)

    assert stopped.status == f"No gate ran. License key refused: {status}"
    assert len(stopped.status) <= MAX_STATUS_DESCRIPTION
    # The status has room for what is wrong; the explanation says all of it.
    assert f"was refused because {stopped.refusal}." in stopped.explanation[2]
    for text in (stopped.status, *stopped.explanation, stopped.comment):
        assert key not in text


def test_a_key_is_verified_against_the_public_key_honest_ci_ships() -> None:
    environ = {**IN_CI, KEY_VARIABLE: org_key()}

    checked = check_license(ORGANIZATION_KEY, environ, Settings(), today=TODAY)

    assert isinstance(checked, Stopped)
    assert checked.status == "No gate ran. License key refused: its signature doesn't match"


@pytest.mark.parametrize("value", ["", "  ", "\n"], ids=["empty", "spaces", "line-break"])
def test_an_empty_key_is_none_as_a_secret_nobody_set_passes_one(value: str) -> None:
    assert stops(ORGANIZATION_KEY, {**IN_CI, KEY_VARIABLE: value}, TODAY) == Stopped(
        ORGANIZATION_KEY, in_actions=False
    )


def test_outside_actions_a_run_without_a_key_stops_as_there_is_no_trial_to_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stopped = stops(ORGANIZATION_KEY, IN_CI, TODAY)

    assert stopped == Stopped(ORGANIZATION_KEY, in_actions=False)
    assert stopped.explanation == [
        "No gate ran. No license key, and this repository needs one in CI. "
        "This is not a verdict: honest-ci checked nothing in this change.",
        "",
        "This repository belongs to the organization octo-org and isn't public, so honest-ci "
        "needs a license key here when it runs in CI. Outside GitHub Actions there is no "
        "automatic trial: email support@honest-ci.dev for a 14-day trial key.",
        "",
        "- Get a key at https://honest-ci.dev/pricing. It comes by email, and a renewal comes "
        "with a new one.",
        "- Set it: in the CI job that runs honest-ci, set the environment variable "
        "`HONEST_CI_LICENSE_KEY` to the key, the way the job gets its other secrets.",
        "- Or stop running honest-ci here: remove the step that runs it, and its status from "
        "the branch's required status checks if it is one.",
    ]
    assert logged(caplog) == []


def test_in_actions_a_run_without_a_key_goes_ahead_on_its_trial_and_says_how_long_is_left(
    github: FakeGitHub,
) -> None:
    serve_history(github, "2026-10-07", TRIAL_START)

    assert check(
        ORGANIZATION_KEY, trial_job(github), date(2026, 10, 8), settings_for(github)
    ) == Cleared(
        note=(
            "honest-ci is on its 14-day trial here: 7 days left, through 2026-10-14, counted "
            "from 2026-10-01, when `.github/workflows/honest-ci.yml` was first committed. "
            "After that it needs a license key: https://honest-ci.dev/pricing."
        )
    )


def test_on_the_trials_last_day_the_run_goes_ahead_with_one_day_left(github: FakeGitHub) -> None:
    serve_history(github, TRIAL_START)

    checked = check(ORGANIZATION_KEY, trial_job(github), date(2026, 10, 14), settings_for(github))

    assert isinstance(checked, Cleared) and checked.note is not None
    assert "trial here: 1 day left, through 2026-10-14," in checked.note


def test_once_the_trial_is_over_the_run_stops_and_says_which_file_it_counted_from(
    github: FakeGitHub,
) -> None:
    serve_history(github, TRIAL_START)

    stopped = stops(ORGANIZATION_KEY, trial_job(github), date(2026, 10, 15), settings_for(github))

    assert stopped == Stopped(
        ORGANIZATION_KEY, in_actions=True, trial=Trial(WORKFLOW, date(2026, 10, 1))
    )
    # What the pull request shows, whole: it reads as no verdict, and the
    # next run, with a key or without, writes over it.
    assert stopped.comment == (
        f"{VERDICT_MARKER}\n"
        "## honest-ci · not run\n"
        "\n"
        "**No gate ran. No license key, and the 14-day trial ended on 2026-10-14.** "
        "This is not a verdict: honest-ci checked nothing in this change.\n"
        "\n"
        "This repository belongs to the organization octo-org and isn't public, so honest-ci "
        "needs a license key here when it runs in CI. Without one it ran on a 14-day trial, "
        "which counts from the day `.github/workflows/honest-ci.yml` was first committed: "
        "2026-10-01. If honest-ci was added to a workflow file that already existed, the trial "
        "started before honest-ci did: email support@honest-ci.dev for a trial key.\n"
        "\n"
        "- Get a key at https://honest-ci.dev/pricing. It comes by email, and a renewal comes "
        "with a new one.\n"
        # GitHub Free passes no organization secret to a private repository,
        # and the check can't see which plan the organization is on.
        "- Set it: save the key as an organization secret named `HONEST_CI_LICENSE_KEY` on "
        "GitHub Team or Enterprise, or as a secret of this repository on GitHub Free, which "
        "doesn't pass organization secrets to private repositories. The honest-ci step passes "
        "it on as `license-key: ${{ secrets.HONEST_CI_LICENSE_KEY }}` under its `with:`, as the "
        "workflow `honest-ci init` writes does.\n"
        "- Or stop running honest-ci here: remove its step from "
        "`.github/workflows/honest-ci.yml`, and its status from the branch's required status "
        "checks if it is one.\n"
    )


def test_a_run_that_cant_tell_when_its_trial_started_goes_ahead_and_the_log_says_so(
    github: FakeGitHub, caplog: pytest.LogCaptureFixture
) -> None:
    github.on("GET", COMMITS, status=500, payload={"message": "Server Error"})

    checked = check(ORGANIZATION_KEY, trial_job(github), date(2026, 10, 15), settings_for(github))

    assert checked == Cleared()
    assert "GitHub answered 500 to GET /repos/octo-org/orders/commits" in no_trial_start(caplog)


def test_a_key_given_during_the_trial_is_checked_all_the_same(github: FakeGitHub) -> None:
    serve_history(github, TRIAL_START)
    job = trial_job(github, **{KEY_VARIABLE: issue(org_id=4242, org="another-org")})

    stopped = stops(ORGANIZATION_KEY, job, date(2026, 10, 8), settings_for(github))

    # One that doesn't work is better found on the first day than on the fifteenth.
    assert stopped.trial is None
    assert stopped.explanation[2].endswith(
        "The key in `HONEST_CI_LICENSE_KEY` was refused because it was issued for another-org "
        "(id 4242), and this repository's owner has id 9001."
    )
    assert github.recorded == []


#: What a run says wherever a key is due because someone else opened the pull request.
FREE_FOR_A_MACHINE_ACCOUNT = (
    "If a machine account you use for an AI agent opened this pull request, email "
    "support@honest-ci.dev for a free key."
)


def test_on_a_personal_repository_the_trial_says_a_machine_accounts_key_is_free(
    github: FakeGitHub,
) -> None:
    serve_history(github, TRIAL_START)

    checked = check(PERSONAL_KEY, trial_job(github), date(2026, 10, 8), settings_for(github))

    assert isinstance(checked, Cleared) and checked.note is not None
    assert checked.note.endswith(f"https://honest-ci.dev/pricing. {FREE_FOR_A_MACHINE_ACCOUNT}")


def test_a_stop_on_a_personal_repository_says_why_and_where_its_secret_goes(
    github: FakeGitHub,
) -> None:
    serve_history(github, TRIAL_START)

    stopped = stops(PERSONAL_KEY, trial_job(github), date(2026, 10, 15), settings_for(github))

    _, _, why, _, _, set_it, _ = stopped.explanation
    assert why.startswith(
        "This repository belongs to octo-dev and isn't public, and someone other than octo-dev "
        "opened this pull request, so honest-ci needs a license key here when it runs in CI. "
        f"{FREE_FOR_A_MACHINE_ACCOUNT} Without one it ran on a 14-day trial"
    )
    # A personal account has no secrets of its own to share, only its repositories have.
    assert set_it == (
        "- Set it: save the key as a secret of this repository named `HONEST_CI_LICENSE_KEY` "
        "(a personal account has no account-wide secrets). The honest-ci step passes it on as "
        "`license-key: ${{ secrets.HONEST_CI_LICENSE_KEY }}` under its `with:`, as the workflow "
        "`honest-ci init` writes does."
    )


#: What a stop in a run Dependabot started adds to where the secret goes.
DEPENDABOT_SECRETS_ONLY = (
    " Dependabot started this run, and GitHub gives such runs Dependabot secrets only: save the "
    "key as a Dependabot secret too, at the same level."
)


@pytest.mark.parametrize("key", [None, "not a key"], ids=["trial-over", "refused"])
def test_a_stop_in_a_run_dependabot_started_says_the_key_goes_in_its_secrets_too(
    github: FakeGitHub, key: str | None
) -> None:
    """Such a run gets none of the Actions secrets a key is saved in for every other run."""

    serve_history(github, TRIAL_START)
    given = {} if key is None else {KEY_VARIABLE: key}
    job = trial_job(github, GITHUB_ACTOR=DEPENDABOT_ACTOR, **given)

    stopped = stops(ORGANIZATION_KEY, job, date(2026, 10, 15), settings_for(github))

    [set_it] = [line for line in stopped.explanation if line.startswith("- Set it:")]
    assert set_it.endswith(f"writes does.{DEPENDABOT_SECRETS_ONLY}")
    assert DEPENDABOT_SECRETS_ONLY.strip() in stopped.comment


@pytest.mark.parametrize(
    "environ",
    [
        {**IN_ACTIONS, "GITHUB_ACTOR": "octo-dev"},
        # Dependabot secrets are a thing of GitHub Actions only.
        {**IN_CI, "GITHUB_ACTOR": DEPENDABOT_ACTOR},
    ],
    ids=["someone-else", "outside-actions"],
)
def test_a_stop_in_a_run_dependabot_didnt_start_in_actions_says_nothing_of_its_secrets(
    environ: dict[str, str],
) -> None:
    stopped = stops(ORGANIZATION_KEY, {**environ, KEY_VARIABLE: "not a key"}, TODAY)

    assert not [line for line in stopped.explanation if "Dependabot" in line]


# ── what doctor says of a key ────────────────────────────────────────────────


def describe(key: str | None, today: date = TODAY) -> str:
    """What ``doctor`` says of ``key``, verified against the test key pair's."""

    environ = {} if key is None else {KEY_VARIABLE: key}
    return describe_key(environ, today=today, public_key=TEST)


@pytest.mark.parametrize("value", [None, "", " \n"], ids=["unset", "empty", "blank"])
def test_doctor_says_when_there_is_no_key(value: str | None) -> None:
    assert describe(value) == "none (HONEST_CI_LICENSE_KEY is not set)"


def test_doctor_says_whose_a_key_is_and_until_when_and_never_the_key() -> None:
    key = org_key()

    said = describe(key)

    assert said == (
        "signature verified · L-2026-0007 for octo-org (id 9001) · plan small, 3 seats · "
        "expires on 2027-11-13"
    )
    assert key.split(".")[2] not in said


@pytest.mark.parametrize(("seats", "said"), [(1, ", 1 seat"), (0, "")], ids=["one", "none"])
def test_doctor_counts_the_seats_a_key_names_when_it_names_any(seats: int, said: str) -> None:
    assert f" · plan trial{said} · " in describe(org_key(plan="trial", seats=seats))


def test_doctor_reads_a_key_for_another_owner_as_it_is() -> None:
    """It knows no repository to compare the key's owner with; a run does."""

    assert describe(issue(org_id=4242, org="another-org")).startswith(
        "signature verified · L-2026-0007 for another-org (id 4242) · "
    )


@pytest.mark.parametrize(
    ("days_past", "left"), [(1, "30 days"), (30, "1 day")], ids=["first", "last"]
)
def test_doctor_says_how_many_grace_days_a_key_past_its_expiry_date_has_left(
    days_past: int, left: str
) -> None:
    said = describe(org_key(), EXPIRES + timedelta(days=days_past))

    assert said.startswith("signature verified · ")
    assert said.endswith(f" · expired on 2027-11-13, works through 2027-12-13: {left} left")


def test_doctor_says_a_key_past_its_grace_days_is_refused() -> None:
    assert describe(org_key(), EXPIRES + timedelta(days=31)) == (
        "refused: it expired on 2027-11-13, more than 30 days ago · "
        "L-2026-0007 for octo-org (id 9001) · plan small, 3 seats"
    )


def test_doctor_says_a_trial_key_works_through_its_expiry_date_and_not_a_day_after() -> None:
    key = org_key(plan="trial", seats=0)

    assert describe(key, EXPIRES) == (
        "signature verified · L-2026-0007 for octo-org (id 9001) · plan trial · "
        "expires on 2027-11-13"
    )
    assert describe(key, EXPIRES + timedelta(days=1)) == (
        "refused: it is a trial key that expired on 2027-11-13 · "
        "L-2026-0007 for octo-org (id 9001) · plan trial"
    )


@pytest.mark.parametrize(
    ("key", "why"),
    [
        (extended(org_key()), "its signature doesn't match"),
        (org_key(v=2), "it is a version 2 key"),
        (org_key().rsplit(".", 1)[0], "it isn't an honest-ci license key"),
    ],
    ids=["changed", "later-format", "cut-short"],
)
def test_doctor_says_why_a_key_is_refused_and_never_repeats_it(key: str, why: str) -> None:
    said = describe(key)

    assert said.startswith(f"refused: {why}")
    assert key.split(".")[1] not in said


def test_doctor_verifies_a_key_against_the_public_key_honest_ci_ships() -> None:
    said = describe_key({KEY_VARIABLE: org_key()}, today=TODAY)

    assert said.startswith("refused: its signature doesn't match")
