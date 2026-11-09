"""The demo repository for GitHub, and what each of its pull requests should show.

``examples/python-agent-demo/github-demo.sh`` builds the repository that
GITHUB.md says to push: ``main`` with honest-ci installed by
``honest-ci init --github`` -- run for real, so building the repository checks
init as well -- and a branch for every pull request to open. GITHUB.md's table
says how each of those pull requests should end. Here every row runs through
``honest-ci verify`` the way the Action would run it: the head checked out, the
rules and the policy read at the base commit, the workflow's environment. A
row that is wrong, or a script that drifted from the checklist, fails here
rather than on GitHub.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from honest_ci.actions import workflow_template
from honest_ci.adapters.github import STATUS_STATE
from honest_ci.cli import EXIT_CODES, EXIT_USAGE, app
from honest_ci.merge_requests import STATUS_VERDICT
from honest_ci.scaffold import GITHUB_FILES

from .test_python_agent_demo import SCENARIOS
from .workflow_yaml import load

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "python-agent-demo"
SCRIPT = EXAMPLE / "github-demo.sh"
CHECKLIST = EXAMPLE / "GITHUB.md"

RULES = "honest-ci-rules.toml"
POLICY = "honest-ci-policy.toml"
WORKFLOW = ".github/workflows/honest-ci.yml"

needs_bash_and_git = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None,
    reason="the script is bash, and what it builds is a git repository",
)

#: git as a fixture needs it: nobody's own configuration, none of the ``GIT_*``
#: variables that a hook running the tests would pass down, and an author.
GIT_ENV = {
    **{name: value for name, value in os.environ.items() if not name.startswith("GIT_")},
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "demo",
    "GIT_AUTHOR_EMAIL": "demo@example.com",
    "GIT_COMMITTER_NAME": "demo",
    "GIT_COMMITTER_EMAIL": "demo@example.com",
}


@dataclass(frozen=True)
class PullRequest:
    head: str
    base: str
    #: The run's terminal state, which heads the comment.
    status: str
    #: What the ``honest-ci/verify`` commit status ends as.
    state: str
    exit_code: int
    #: The one rule whose hit decides the verdict; ``None`` when nothing may fire.
    rule: str | None


_ROW = re.compile(
    r"^\|\s*`(?P<head>[\w-]+)`\s*\|\s*`(?P<base>[\w-]+)`\s*\|\s*`honest-ci · (?P<status>\w+)`\s*\|"
    r"\s*`(?P<state>\w+)`\s*\|\s*(?P<code>\d+)\s*\|\s*(?:`(?P<rule>[\w.-]+)`|—)\s*\|$"
)


def _checklist() -> list[PullRequest]:
    return [
        PullRequest(
            match["head"],
            match["base"],
            match["status"],
            match["state"],
            int(match["code"]),
            match["rule"],
        )
        for match in map(_ROW.match, CHECKLIST.read_text(encoding="utf-8").splitlines())
        if match
    ]


PULL_REQUESTS = _checklist()


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        env=GIT_ENV,
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    return proc.stdout


def build(honest_ci: Path, target: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args, str(target)],
        env={**GIT_ENV, "HONEST_CI": str(honest_ci)},
        capture_output=True,
        encoding="utf-8",
        check=False,
        timeout=300,
    )


@pytest.fixture(scope="module")
def honest_ci(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The honest-ci under test, as one command the script can run."""

    command = tmp_path_factory.mktemp("bin") / "honest-ci"
    command.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(sys.executable)} -c 'from honest_ci.cli import app; app()' \"$@\"\n",
        encoding="utf-8",
    )
    command.chmod(0o755)
    return command


@pytest.fixture(scope="module")
def demo(honest_ci: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("github-demo") / "honest-ci-demo"
    built = build(honest_ci, target)
    assert built.returncode == 0, built.stdout + built.stderr
    return target


# ── the checklist ────────────────────────────────────────────────────────────


def test_each_row_shows_what_its_state_means_on_github() -> None:
    assert PULL_REQUESTS
    for pull in PULL_REQUESTS:
        assert EXIT_CODES[pull.status] == pull.exit_code, pull
        assert STATUS_STATE[STATUS_VERDICT[pull.status]] == pull.state, pull


def test_the_scenarios_end_as_the_demo_readme_says() -> None:
    """Two tables, one per place the scenarios are tried; they must not tell two stories."""

    rows = {pull.head: pull for pull in PULL_REQUESTS}
    # The README uses the gate's word; the comment heads with the run's.
    statuses = {"needs_human": "awaiting_human"}

    for scenario in SCENARIOS:
        pull = rows[scenario.name]
        assert pull.base == "main"
        assert (pull.status, pull.exit_code, pull.rule) == (
            statuses.get(scenario.verdict, scenario.verdict),
            scenario.exit_code,
            scenario.rule,
        )


#: A row of the approval steps: what to do, on which pull request, and how it ends.
_APPROVAL_ROW = re.compile(
    r"^\|\s*(?P<step>[A-Z][^|`]*?)\s*\|\s*`(?P<head>[\w-]+)`\s*\|"
    r"\s*`honest-ci · (?P<status>\w+)`\s*\|\s*`(?P<state>\w+)`\s*\|\s*(?P<code>\d+)\s*\|$"
)


def test_each_approval_step_ends_as_an_approval_can_end() -> None:
    steps = [
        match
        for match in map(_APPROVAL_ROW.match, CHECKLIST.read_text(encoding="utf-8").splitlines())
        if match
    ]
    opened = {pull.head: pull for pull in PULL_REQUESTS}

    # One that counts, and the three that do not: no reason, a later commit, a blocked change.
    assert [step["status"] for step in steps] == [
        "approved",
        "awaiting_human",
        "awaiting_human",
        "blocked",
    ]
    for step in steps:
        before = opened[step["head"]]
        # An approved run is mergeable to the merge rule; only its comment says who signed.
        after = "mergeable" if step["status"] == "approved" else step["status"]
        assert EXIT_CODES[after] == int(step["code"]), step.group(0)
        assert STATUS_STATE[STATUS_VERDICT[after]] == step["state"], step.group(0)
        if step["status"] == "approved":
            assert before.status == "awaiting_human", step.group(0)
        else:
            # An approval that does not count leaves the pull request where it was.
            assert step["status"] == before.status, step.group(0)


@needs_bash_and_git
def test_the_checklist_opens_a_pull_request_from_every_branch(demo: Path) -> None:
    branches = set(git(demo, "for-each-ref", "--format=%(refname:short)", "refs/heads").split())
    heads = [pull.head for pull in PULL_REQUESTS]

    assert len(heads) == len(set(heads)), heads
    assert set(heads) == branches - {"main", "before-honest-ci"}
    assert {pull.base for pull in PULL_REQUESTS} == {"main", "before-honest-ci"}


# ── the repository ───────────────────────────────────────────────────────────


@needs_bash_and_git
def test_main_is_the_shop_with_what_init_writes(demo: Path) -> None:
    for item in GITHUB_FILES:
        assert git(demo, "show", f"main:{item.path}") == item.content(), item.path
    assert git(demo, "diff", "--name-only", "before-honest-ci", "main").split() == sorted(
        item.path for item in GITHUB_FILES
    )
    # The install pull request's head is main itself, so that it installs
    # exactly what every other pull request is judged by.
    assert git(demo, "rev-parse", "install-honest-ci") == git(demo, "rev-parse", "main")


def _workflow_env() -> dict[str, str]:
    """What the workflow template sets on the Action's step."""

    steps = [step for job in load(workflow_template())["jobs"].values() for step in job["steps"]]
    action = [step for step in steps if step.get("uses", "").startswith("ChrisShips/honest-ci@")]
    assert len(action) == 1, steps
    env: dict[str, str] = action[0].get("env", {})
    return env


def _as_the_action_runs_it(
    monkeypatch: pytest.MonkeyPatch, checkout: Path, base: str, config: Path
) -> None:
    """The workflow's environment, with the rules and the policy as ``base`` has them.

    A file the base does not have is named anyway and left absent, as the
    Action leaves it, and the run reads that as "not configured".
    """

    for name in list(os.environ):
        if name.startswith(("HONEST_CI_", "GIT_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name, value in _workflow_env().items():
        monkeypatch.setenv(name, value)
    for variable, path in (("HONEST_CI_RULES_PATH", RULES), ("HONEST_CI_POLICY_PATH", POLICY)):
        at_base = subprocess.run(
            ["git", "-C", str(checkout), "show", f"{base}:{path}"],
            env=GIT_ENV,
            capture_output=True,
            check=False,
        )
        if at_base.returncode == 0:
            (config / path).write_bytes(at_base.stdout)
        monkeypatch.setenv(variable, str(config / path))


def _evidence(checkout: Path) -> dict[str, Any]:
    packs = list((checkout / ".honest-ci" / "artifacts").glob("*/evidence.json"))
    assert len(packs) == 1, packs
    pack = json.loads(packs[0].read_text(encoding="utf-8"))
    assert isinstance(pack, dict)
    return pack


def _verify(
    pull: PullRequest, demo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Result, dict[str, Any]]:
    """The pull request run as the Action would run it: its result and its evidence."""

    checkout = tmp_path / "checkout"
    git(tmp_path, "clone", "-q", "--branch", pull.head, str(demo), str(checkout))
    config = tmp_path / "config"
    config.mkdir()
    _as_the_action_runs_it(monkeypatch, checkout, f"origin/{pull.base}", config)
    monkeypatch.chdir(checkout)

    result = CliRunner().invoke(app, ["verify", "--base", f"origin/{pull.base}"])

    return result, _evidence(checkout)


@needs_bash_and_git
@pytest.mark.parametrize("pull", PULL_REQUESTS, ids=lambda pull: pull.head)
def test_each_pull_request_ends_where_the_checklist_says(
    pull: PullRequest, demo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, pack = _verify(pull, demo, tmp_path, monkeypatch)

    assert result.exit_code == pull.exit_code, result.stdout
    fired = {finding["rule_id"] for report in pack["gate_reports"] for finding in report["findings"]}
    # Exactly the checklist's rule: a pull request held for the right reason
    # and also tripping something else would be showing a different story.
    assert fired == ({pull.rule} if pull.rule else set())


@needs_bash_and_git
def test_the_install_pull_request_goes_to_a_person_for_the_reasons_given(
    demo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GITHUB.md explains why the first pull request is held, and says it is expected.

    Each reason on its own would hold it, so each is checked on its own.
    """

    install = next(pull for pull in PULL_REQUESTS if pull.base == "before-honest-ci")

    _, pack = _verify(install, demo, tmp_path, monkeypatch)

    review = next(report for report in pack["gate_reports"] if report["gate"] == "review")
    decision = pack["policy_decision"]
    sensitive = [text for text in decision["violations"] if text.startswith("sensitive paths")]
    # No rules at the base: nothing to review the change against.
    assert review["verdict"] == "needs_human"
    assert review["summary"].startswith("No team rule library is configured")
    # No policy at the base: the built-in one requires gates that do not run.
    assert {"spec", "regression"} <= set(decision["required_checks"])
    # And it changes the workflow and honest-ci's own two files.
    assert len(sensitive) == 1
    for path in (WORKFLOW, RULES, POLICY):
        assert f"`{path}`" in sensitive[0]


# ── the script ───────────────────────────────────────────────────────────────


@needs_bash_and_git
def test_the_action_can_be_called_at_another_ref(honest_ci: Path, tmp_path: Path) -> None:
    """For a commit that is not released yet: only the Action's reference changes."""

    built = build(honest_ci, tmp_path / "demo", "--action-ref", "feat/next")

    assert built.returncode == 0, built.stdout + built.stderr
    workflow = git(tmp_path / "demo", "show", f"main:{WORKFLOW}")
    changed = [
        (template, written)
        for template, written in zip(
            workflow_template().splitlines(), workflow.splitlines(), strict=True
        )
        if template != written
    ]
    assert len(changed) == 1, changed
    steps = [step for job in load(workflow)["jobs"].values() for step in job["steps"]]
    assert "ChrisShips/honest-ci@feat/next" in [step.get("uses") for step in steps]


@needs_bash_and_git
@pytest.mark.parametrize("ref", ["main\n  env: {}", "main # moved", "-main"])
def test_a_ref_that_could_carry_yaml_of_its_own_is_refused(
    honest_ci: Path, tmp_path: Path, ref: str
) -> None:
    built = build(honest_ci, tmp_path / "demo", "--action-ref", ref)

    assert built.returncode == EXIT_USAGE
    assert not (tmp_path / "demo").exists()


@needs_bash_and_git
def test_a_directory_with_something_in_it_is_left_alone(honest_ci: Path, tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("mine\n", encoding="utf-8")

    built = build(honest_ci, tmp_path)

    assert built.returncode == EXIT_USAGE
    assert [path.name for path in tmp_path.iterdir()] == ["notes.txt"]


@needs_bash_and_git
def test_the_script_says_when_it_has_no_honest_ci(tmp_path: Path) -> None:
    built = build(tmp_path / "missing" / "honest-ci", tmp_path / "demo")

    assert built.returncode == EXIT_USAGE
    assert "HONEST_CI" in built.stderr
    assert not (tmp_path / "demo").exists()


@needs_bash_and_git
def test_the_script_stops_and_says_why_when_init_fails(tmp_path: Path) -> None:
    refusing = tmp_path / "honest-ci"
    refusing.write_text("#!/bin/sh\necho 'init said no'\nexit 2\n", encoding="utf-8")
    refusing.chmod(0o755)

    built = build(refusing, tmp_path / "demo")

    assert built.returncode != 0
    assert "init said no" in built.stderr
