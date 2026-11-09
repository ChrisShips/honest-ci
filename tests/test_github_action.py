"""The GitHub Action and the workflow that calls it.

Two files decide what honest-ci may do inside somebody else's repository:
``action.yml`` and the workflow the package hands out to call it. What makes
them safe to install -- the one event that starts a run, what its token may do,
the commit every other action is pinned to -- is read off the parsed files on
every commit, because a workflow that lost any of those would still run and
still post a verdict.

The Action's shell steps run here for real, in a throwaway repository and with
a stand-in for honest-ci: they decide which rules judge a pull request and
which exit code a verdict becomes, and reading them proves neither.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from honest_ci import __version__
from honest_ci.actions import workflow_template
from honest_ci.cli import EXIT_CODES, EXIT_USAGE
from honest_ci.license_check import KEY_VARIABLE
from honest_ci.policy import PolicyConfig
from honest_ci.rules import RuleSet

from .workflow_yaml import UnsupportedYaml, load

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "honest_ci"
ACTION = REPO / "action.yml"
TEMPLATE = PACKAGE / "resources" / "github-workflow.yml"

#: What run-pr does with the token, one permission each: read the code and the
#: pull request, post the conclusion comment, set the commit status.
PERMISSIONS = {"contents": "read", "pull-requests": "write", "statuses": "write"}

#: ``owner/repo[/path]@<full commit SHA>``. A tag or a branch can be moved by
#: whoever controls the action's repository, and what it points to next runs
#: with this workflow's token; a commit cannot be moved.
PINNED = re.compile(r"[\w.-]+/[\w.-]+(?:/[\w./-]+)?@[0-9a-f]{40}")

#: The Action as the template calls it: the release this package is.
THIS_ACTION = f"ChrisShips/honest-ci@v{__version__}"

#: Where the Action installs honest-ci and runs it from.
VENV = "$RUNNER_TEMP/honest-ci/venv"

CHECK = "Check the trigger"
INSTALL = "Install honest-ci"
CONFIG = "Read the rules and the policy at the base commit"
VERIFY = "Verify the pull request"
UPLOAD = "Upload the evidence pack"

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="the Action's steps call git")


def action() -> dict[str, Any]:
    return load(ACTION.read_text(encoding="utf-8"))


def workflow() -> dict[str, Any]:
    return load(workflow_template())


def action_steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = action()["runs"]["steps"]
    return steps


def action_step(name: str) -> dict[str, Any]:
    matches = [step for step in action_steps() if step.get("name") == name]
    assert len(matches) == 1, f"action.yml has {len(matches)} steps named {name!r}"
    return matches[0]


def workflow_steps() -> list[dict[str, Any]]:
    return [step for job in workflow()["jobs"].values() for step in job["steps"]]


def uses(steps: list[dict[str, Any]]) -> list[str]:
    return [step["uses"] for step in steps if "uses" in step]


def strings(node: Any) -> Iterator[str]:
    """Every key and every value of a parsed document, comments aside."""

    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from strings(item)
    elif isinstance(node, str):
        yield node


# ── what starts a run, and what its token may do ─────────────────────────────


def test_the_workflow_runs_on_pull_requests_and_their_reviews_and_nothing_else() -> None:
    """``pull_request_target`` would hand the pull request's code a token that can write.

    honest-ci scans the tree it is given and, with the regression gate on, runs
    its tests. On ``pull_request_target`` that tree sits next to the base
    repository's secrets and a token that can push: the "pwn request".
    https://securitylab.github.com/resources/github-actions-preventing-pwn-requests/

    A review starts a run because an Approve review is how a person lets a
    change through; a dismissed one, so that an approval taken back stops
    counting.
    """

    document = workflow()

    assert document["on"] == {
        "pull_request": None,
        "pull_request_review": {"types": ["submitted", "dismissed"]},
    }
    # Nowhere outside a comment either: a condition or a step that names it is
    # somebody getting ready to support it.
    assert [text for text in strings(document) if "pull_request_target" in text] == []


def test_the_token_gets_three_permissions_and_no_more() -> None:
    """Exactly what run-pr uses, granted once, at the top of the workflow.

    A job's own ``permissions`` would replace these rather than add to them,
    so there is none: the three below are the whole grant.
    """

    document = workflow()

    assert document["permissions"] == PERMISSIONS
    assert [name for name, job in document["jobs"].items() if "permissions" in job] == []


def test_every_place_that_names_the_permissions_names_these() -> None:
    """What run-pr asks for when a write is refused, and what the Action says it needs."""

    cli = (PACKAGE / "cli.py").read_text(encoding="utf-8")
    documented = action()["inputs"]["github-token"]["description"]

    for name, level in PERMISSIONS.items():
        assert f"`{name}: {level}`" in cli
        assert f"{name}: {level}" in documented


def test_a_push_cancels_the_run_for_the_commit_it_replaced() -> None:
    """One group per pull request, and per workflow: concurrency groups are repository-wide."""

    concurrency = workflow()["concurrency"]

    assert "github.workflow" in concurrency["group"]
    assert "github.event.pull_request.number" in concurrency["group"]
    assert concurrency["cancel-in-progress"] == "true"


# ── what runs ────────────────────────────────────────────────────────────────


def test_every_action_from_elsewhere_is_pinned_to_a_commit() -> None:
    references = uses(workflow_steps()) + uses(action_steps())
    others = [reference for reference in references if reference != THIS_ACTION]

    # Named, so that a reader that stopped finding steps cannot pass vacuously.
    assert {reference.partition("@")[0] for reference in others} >= {
        "actions/checkout",
        "actions/setup-python",
        "actions/upload-artifact",
    }
    assert [reference for reference in others if not PINNED.fullmatch(reference)] == []


def test_the_workflow_calls_the_release_this_package_is() -> None:
    """The one reference that is a tag: a release's commit is not known until it exists.

    The template says how to replace the tag with that commit.
    """

    references = uses(workflow_steps())

    assert [ref for ref in references if ref.startswith("ChrisShips/honest-ci")] == [THIS_ACTION]


def test_the_checkout_is_the_head_commit_with_its_history_and_no_credential() -> None:
    """What the GitHub adapter insists on before it verifies a checkout.

    The head, because a pull_request run checks out a merge commit by default.
    The whole history, because the rules and the policy are read at the base
    commit. No credential left behind, because the pull request's own code
    runs over this tree.
    """

    steps = workflow_steps()
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
    honest_ci = next(step for step in steps if step.get("uses") == THIS_ACTION)

    assert checkout["with"] == {
        "ref": "${{ github.event.pull_request.head.sha }}",
        "fetch-depth": "0",
        "persist-credentials": "false",
    }
    assert steps.index(checkout) < steps.index(honest_ci)


def test_no_script_has_an_expression_spliced_into_it() -> None:
    """Values reach a script through ``env:``, where they stay data.

    An expression is substituted into a script's text before bash reads it, so
    a pull request title with a quote and a semicolon in it would run.
    """

    scripts = [step for step in action_steps() + workflow_steps() if "run" in step]

    assert scripts
    for step in scripts:
        assert "${{" not in step["run"], step.get("name")
        assert step["shell"] == "bash", step.get("name")


def test_honest_ci_is_installed_from_the_action_itself() -> None:
    """Not from a package index: what judges the pull request is the release the workflow pinned."""

    script = action_step(INSTALL)["run"]
    install = [line for line in script.splitlines() if " -m pip install " in line]

    assert f'"{VENV}"' in script
    assert len(install) == 1
    assert install[0].split()[-1] == '"$GITHUB_ACTION_PATH"'


def test_the_exit_codes_the_action_documents_are_the_ones_run_pr_uses() -> None:
    described = action()["outputs"]["exit-code"]["description"]

    assert {int(code) for code in re.findall(r"\b(\d) [a-z]", described)} == {
        *EXIT_CODES.values(),
        EXIT_USAGE,
    }


def test_the_evidence_is_uploaded_whatever_the_verdict() -> None:
    """A blocked change, or one held for a person, is the one whose evidence gets read."""

    upload = action_step(UPLOAD)

    assert "!cancelled()" in upload["if"]
    assert "steps.verify.outcome == 'failure'" in upload["if"]
    assert upload["with"]["path"] == "${{ steps.config.outputs.evidence }}"


def test_the_workflow_hands_the_action_the_license_key_from_a_secret() -> None:
    """Named as the check names it, so that the secret, the input and the variable are one name.

    Not required: on a public repository nobody sets the secret, and the
    workflow passes nothing.
    """

    [step] = [step for step in workflow_steps() if step.get("uses") == THIS_ACTION]

    assert step["with"] == {"license-key": f"${{{{ secrets.{KEY_VARIABLE} }}}}"}
    assert not action()["inputs"]["license-key"].get("required", False)
    assert "default" not in action()["inputs"]["license-key"]


def test_the_template_ships_in_the_package() -> None:
    """One copy, read the same way wherever the package is installed."""

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = pyproject["tool"]["setuptools"]["package-data"]["honest_ci"]

    assert TEMPLATE in {path for pattern in patterns for path in PACKAGE.glob(pattern)}
    assert workflow_template() == TEMPLATE.read_text(encoding="utf-8")


# ── the Action's steps, run ──────────────────────────────────────────────────

#: git as a fixture needs it: nobody's own configuration, and none of the
#: ``GIT_*`` variables that a hook running the tests would pass down.
GIT_ENV = {
    **{name: value for name, value in os.environ.items() if not name.startswith("GIT_")},
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}

BASE_RULES = 'version = "rules at the base commit"\n'
HEAD_RULES = 'version = "rules as the pull request loosened them"\n'
HEAD_POLICY = "[policy]\nsensitive_paths = []\n"

#: run-pr's stand-in: says what it was given, leaves evidence where it was told
#: to, and exits with the code the test asks for.
FAKE_HONEST_CI = """\
#!/usr/bin/env bash
set -eu
printf '%s\\n' "$@" > "$RUNNER_TEMP/argv"
for name in HONEST_CI_FORGE HONEST_CI_RULES_PATH HONEST_CI_POLICY_PATH \\
    HONEST_CI_ARTIFACTS_DIR HONEST_CI_EVIDENCE_ARTIFACT GITHUB_TOKEN; do
  printf '%s=%s\\n' "$name" "${!name-}"
done > "$RUNNER_TEMP/environment"
env > "$RUNNER_TEMP/whole-environment"
mkdir -p "$HONEST_CI_ARTIFACTS_DIR/run-1"
printf '{}\\n' > "$HONEST_CI_ARTIFACTS_DIR/run-1/evidence.json"
exit "$VERDICT_EXIT_CODE"
"""


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=test", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=GIT_ENV,
    )
    return proc.stdout.strip()


@dataclass(frozen=True)
class PullRequest:
    checkout: Path
    base: str


@dataclass(frozen=True)
class Ran:
    code: int
    log: str
    outputs: dict[str, str]


class Runner:
    """Enough of a GitHub-hosted runner to run the Action's shell steps.

    A step gets what a composite step gets: the runner's variables, the
    caller's ``env``, and its own ``env:`` with each expression replaced by a
    value this test supplies. An expression nobody supplied is a ``KeyError``
    rather than an empty string, so a step that starts reading something new
    cannot pass here by reading nothing.
    """

    def __init__(self, root: Path, checkout: Path, base: str) -> None:
        self.temp = root / "runner-temp"
        home = root / "home"
        self.temp.mkdir(parents=True)
        home.mkdir()
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": str(home),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "RUNNER_OS": "Linux",
            "RUNNER_TEMP": str(self.temp),
            "GITHUB_WORKSPACE": str(checkout),
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_REPOSITORY": "octo-org/orders",
        }
        self.context = {
            "github.event.pull_request.base.sha": base,
            "github.event.pull_request.number": "7",
            "github.run_attempt": "1",
            "inputs.github-token": "dummy",
            # What a workflow passes when the secret is unset, as it is
            # wherever no key is due.
            "inputs.license-key": "",
        }
        self.steps = 0

    def run(self, name: str, **env: str) -> Ran:
        step = action_step(name)
        self.steps += 1
        output = self.temp / f"output-{self.steps}"
        output.touch()
        script = self.temp / f"step-{self.steps}.sh"
        script.write_text(step["run"], encoding="utf-8")
        environ = {**self.env, **env, "GITHUB_OUTPUT": str(output)}
        environ.update({key: self.evaluate(value) for key, value in step.get("env", {}).items()})
        proc = subprocess.run(
            # How a composite step runs `shell: bash`.
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
            cwd=self.env["GITHUB_WORKSPACE"],
            env=environ,
            capture_output=True,
            text=True,
            check=False,
        )
        outputs: dict[str, str] = {}
        for line in output.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            assert separator, f"{name} wrote {line!r} to GITHUB_OUTPUT"
            outputs[key] = value
            self.context[f"steps.{step['id']}.outputs.{key}"] = value
        return Ran(proc.returncode, proc.stdout + proc.stderr, outputs)

    def evaluate(self, text: str) -> str:
        def value(match: re.Match[str]) -> str:
            expression = match[1].strip()
            if expression in self.context:
                return self.context[expression]
            if re.fullmatch(r"steps\.[\w-]+\.outputs\.[\w-]+", expression):
                # What the runner makes of an output the step never set.
                return ""
            raise KeyError(f"this test supplies no value for ${{{{ {expression} }}}}")

        return re.sub(r"\$\{\{(.*?)\}\}", value, text)

    def install_honest_ci_stand_in(self) -> None:
        executable = Path(VENV.replace("$RUNNER_TEMP", str(self.temp))) / "bin" / "honest-ci"
        executable.parent.mkdir(parents=True)
        executable.write_text(FAKE_HONEST_CI, encoding="utf-8")
        executable.chmod(0o755)

    def recorded(self, name: str) -> list[str]:
        return (self.temp / name).read_text(encoding="utf-8").splitlines()


@pytest.fixture
def pull(tmp_path: Path) -> PullRequest:
    """What actions/checkout leaves with ``ref: head.sha`` and ``fetch-depth: 0``.

    The base commit has rules and no policy. The pull request loosens the
    rules and brings a policy of its own.
    """

    checkout = tmp_path / "workspace"
    (checkout / "config").mkdir(parents=True)
    git(checkout, "init", "--quiet")
    (checkout / "honest-ci-rules.toml").write_text(BASE_RULES, encoding="utf-8")
    (checkout / "config" / "strict.toml").write_text(BASE_RULES, encoding="utf-8")
    (checkout / "linked.toml").symlink_to("honest-ci-rules.toml")
    git(checkout, "add", "-A")
    git(checkout, "commit", "--quiet", "-m", "base")
    base = git(checkout, "rev-parse", "HEAD")
    (checkout / "honest-ci-rules.toml").write_text(HEAD_RULES, encoding="utf-8")
    (checkout / "config" / "strict.toml").write_text(HEAD_RULES, encoding="utf-8")
    (checkout / "honest-ci-policy.toml").write_text(HEAD_POLICY, encoding="utf-8")
    git(checkout, "add", "-A")
    git(checkout, "commit", "--quiet", "-m", "loosen the rules")
    return PullRequest(checkout, base)


@pytest.fixture
def runner(tmp_path: Path, pull: PullRequest) -> Runner:
    return Runner(tmp_path / "runner", pull.checkout, pull.base)


@pytest.mark.parametrize(
    ("event", "runner_os", "code"),
    [
        ("pull_request", "Linux", 0),
        ("pull_request", "macOS", 0),
        ("pull_request_review", "Linux", 0),
        ("pull_request_review_comment", "Linux", EXIT_USAGE),
        ("pull_request_target", "Linux", EXIT_USAGE),
        ("push", "Linux", EXIT_USAGE),
        ("workflow_dispatch", "Linux", EXIT_USAGE),
        ("", "Linux", EXIT_USAGE),
        ("pull_request", "Windows", EXIT_USAGE),
    ],
)
def test_the_action_runs_for_a_pull_request_and_refuses_anything_else(
    tmp_path: Path, event: str, runner_os: str, code: int
) -> None:
    """Checked before anything is installed, since the event decides what the token may do."""

    ran = Runner(tmp_path, tmp_path, base="").run(
        CHECK, GITHUB_EVENT_NAME=event, RUNNER_OS=runner_os
    )

    assert ran.code == code
    assert ("::error title=honest-ci::" in ran.log) == (code != 0)


@needs_git
def test_the_rules_come_from_the_base_commit_not_the_pull_request(
    runner: Runner, pull: PullRequest
) -> None:
    """A pull request that loosens a rule is still judged by the rule it loosened."""

    ran = runner.run(CONFIG)

    assert ran.code == 0, ran.log
    root = Path(ran.outputs["root"])
    rules = Path(ran.outputs["rules"])
    assert rules == root / "honest-ci-rules.toml"
    assert rules.read_text(encoding="utf-8") == BASE_RULES
    assert (pull.checkout / "honest-ci-rules.toml").read_text(encoding="utf-8") == HEAD_RULES
    # Named in the evidence as the repository names it, not by a runner path.
    loaded = RuleSet.load(root, rules, "default")
    assert (loaded.version, loaded.source) == ("rules at the base commit", "honest-ci-rules.toml")


@needs_git
def test_a_file_the_base_commit_lacks_is_not_taken_from_the_pull_request(
    runner: Runner, pull: PullRequest
) -> None:
    """The pull request's own policy is a change like any other, not configuration."""

    ran = runner.run(CONFIG)

    assert ran.code == 0, ran.log
    root = Path(ran.outputs["root"])
    policy = Path(ran.outputs["policy"])
    assert (pull.checkout / "honest-ci-policy.toml").is_file()
    assert policy == root / "honest-ci-policy.toml"
    assert not policy.exists()
    assert "honest-ci-policy.toml does not exist at the base commit" in ran.log
    # Which honest-ci reads as "no policy file": the defaults judge, and the
    # evidence says that no file did.
    assert PolicyConfig.load(root, policy, "default").source is None


@needs_git
def test_a_configured_path_is_read_at_the_base_commit_too(runner: Runner) -> None:
    ran = runner.run(CONFIG, HONEST_CI_RULES_PATH="config/strict.toml")

    assert ran.code == 0, ran.log
    root = Path(ran.outputs["root"])
    rules = Path(ran.outputs["rules"])
    assert rules == root / "config" / "strict.toml"
    assert rules.read_text(encoding="utf-8") == BASE_RULES
    assert RuleSet.load(root, rules, "default").source == "config/strict.toml"


@needs_git
@pytest.mark.parametrize(
    "path",
    [
        "/etc/hosts",
        "../honest-ci-rules.toml",
        "config/../honest-ci-rules.toml",
        "./honest-ci-rules.toml",
        "config/",
        "rules file.toml",
    ],
)
def test_a_path_that_is_not_inside_the_repository_is_refused(runner: Runner, path: str) -> None:
    """The file is read at the base commit, so it has to be a name the commit can have."""

    ran = runner.run(CONFIG, HONEST_CI_RULES_PATH=path)

    assert ran.code == EXIT_USAGE
    assert "is not a path inside the repository" in ran.log
    assert "rules" not in ran.outputs


@needs_git
@pytest.mark.parametrize(("path", "mode"), [("config", "040000"), ("linked.toml", "120000")])
def test_a_path_that_is_not_a_regular_file_at_the_base_commit_is_refused(
    runner: Runner, path: str, mode: str
) -> None:
    """A link would be read as the name of its target, a directory not at all."""

    ran = runner.run(CONFIG, HONEST_CI_POLICY_PATH=path)

    assert ran.code == EXIT_USAGE
    assert f"is not a regular file (git mode {mode})" in ran.log


@needs_git
def test_a_checkout_without_the_base_commit_stops_the_run(tmp_path: Path, pull: PullRequest) -> None:
    """actions/checkout fetches one commit unless told otherwise, and it is the head."""

    shallow = tmp_path / "shallow"
    git(tmp_path, "clone", "--quiet", "--depth", "1", f"file://{pull.checkout}", str(shallow))

    ran = Runner(tmp_path / "runner", shallow, pull.base).run(CONFIG)

    assert ran.code == EXIT_CODES["failed"]
    assert "Check the repository out with 'fetch-depth: 0'" in ran.log
    assert ran.outputs == {}


@needs_git
@pytest.mark.parametrize("base", ["", "HEAD", "--help", "0" * 40])
def test_a_base_that_names_no_commit_here_stops_the_run(runner: Runner, base: str) -> None:
    runner.context["github.event.pull_request.base.sha"] = base

    ran = runner.run(CONFIG)

    assert ran.code == EXIT_CODES["failed"]
    assert "is not in the checkout" in ran.log
    assert ran.outputs == {}


@needs_git
def test_run_pr_is_handed_the_base_commit_s_configuration(runner: Runner) -> None:
    runner.install_honest_ci_stand_in()

    assert runner.run(CHECK).code == 0
    config = runner.run(CONFIG)
    verify = runner.run(VERIFY, VERDICT_EXIT_CODE="0")

    assert verify.code == 0, verify.log
    root = config.outputs["root"]
    assert runner.recorded("argv") == ["run-pr", "octo-org/orders", "7", "--workspace", root]
    assert runner.recorded("environment") == [
        "HONEST_CI_FORGE=github",
        f"HONEST_CI_RULES_PATH={root}/honest-ci-rules.toml",
        f"HONEST_CI_POLICY_PATH={root}/honest-ci-policy.toml",
        f"HONEST_CI_ARTIFACTS_DIR={root}/evidence",
        "HONEST_CI_EVIDENCE_ARTIFACT=honest-ci-evidence-7-1",
        "GITHUB_TOKEN=dummy",
    ]


@needs_git
def test_run_pr_is_told_the_name_the_evidence_is_uploaded_under(runner: Runner) -> None:
    """The comment quotes it, and a reader looks for exactly that name on the run's page."""

    runner.context["github.run_attempt"] = "2"
    runner.install_honest_ci_stand_in()
    runner.run(CHECK)
    runner.run(CONFIG)
    runner.run(VERIFY, VERDICT_EXIT_CODE=str(EXIT_CODES["awaiting_human"]))

    uploaded = runner.evaluate(action_step(UPLOAD)["with"]["name"])

    assert uploaded == "honest-ci-evidence-7-2"
    assert f"HONEST_CI_EVIDENCE_ARTIFACT={uploaded}" in runner.recorded("environment")


@needs_git
@pytest.mark.parametrize("code", sorted(set(EXIT_CODES.values())))
def test_the_verdict_is_the_action_s_exit_code(runner: Runner, code: int) -> None:
    """Passed through untouched, and recorded as the ``exit-code`` output.

    Any code but 0 fails the step, and with it the job.
    """

    runner.install_honest_ci_stand_in()
    runner.run(CHECK)
    runner.run(CONFIG)

    verify = runner.run(VERIFY, VERDICT_EXIT_CODE=str(code))

    assert verify.code == code
    assert verify.outputs == {"exit-code": str(code)}
    assert runner.evaluate(action()["outputs"]["exit-code"]["value"]) == str(code)


@needs_git
@pytest.mark.parametrize(
    ("given", "set_under_env", "read"),
    [
        ("key-from-the-input", None, "key-from-the-input"),
        # An input left out leaves alone the key the workflow set under env:.
        ("", "key-from-env", "key-from-env"),
        ("key-from-the-input", "key-from-env", "key-from-the-input"),
        ("", None, None),
    ],
)
def test_run_pr_reads_the_license_key_from_the_input_or_else_from_the_step_s_env(
    runner: Runner, given: str, set_under_env: str | None, read: str | None
) -> None:
    runner.context["inputs.license-key"] = given
    caller = {} if set_under_env is None else {KEY_VARIABLE: set_under_env}
    runner.install_honest_ci_stand_in()
    runner.run(CHECK)
    runner.run(CONFIG)

    verify = runner.run(VERIFY, VERDICT_EXIT_CODE="0", **caller)

    assert verify.code == 0, verify.log
    environment = runner.recorded("whole-environment")
    # Under the one name honest-ci reads, and under no other.
    assert [line for line in environment if line.startswith(f"{KEY_VARIABLE}=")] == (
        [] if read is None else [f"{KEY_VARIABLE}={read}"]
    )
    for passed_over in [key for key in (given, set_under_env) if key and key != read]:
        assert not [line for line in environment if passed_over in line]
    if read is not None:
        assert [line for line in environment if read in line] == [f"{KEY_VARIABLE}={read}"]
        assert read not in verify.log


@needs_git
def test_the_evidence_run_pr_writes_is_where_the_upload_looks(runner: Runner) -> None:
    runner.install_honest_ci_stand_in()
    runner.run(CHECK)
    runner.run(CONFIG)
    runner.run(VERIFY, VERDICT_EXIT_CODE=str(EXIT_CODES["blocked"]))

    uploaded = Path(runner.evaluate(action_step(UPLOAD)["with"]["path"]))

    assert (uploaded / "run-1" / "evidence.json").is_file()
    # upload-artifact leaves out anything hidden unless told otherwise.
    assert not [part for part in uploaded.relative_to(runner.temp).parts if part.startswith(".")]


# ── the YAML reader these tests stand on ─────────────────────────────────────


def test_the_reader_reads_the_subset_the_files_are_written_in() -> None:
    text = (
        "# a comment\n"
        "on:\n"
        "  pull_request:\n"
        "permissions:\n"
        "  contents: read  # why\n"
        "steps:\n"
        "  - uses: owner/repo@abc # v1\n"
        "    with:\n"
        "      depth: 0\n"
        '      version: "3.12"\n'
        "      quote: 'it''s'\n"
        "  - run: |\n"
        "      echo one\n"
        "\n"
        "        echo two\n"
        "    shell: bash\n"
        "description: >-\n"
        "  folded\n"
        "  text\n"
    )

    assert load(text) == {
        "on": {"pull_request": None},
        "permissions": {"contents": "read"},
        "steps": [
            {"uses": "owner/repo@abc", "with": {"depth": "0", "version": "3.12", "quote": "it's"}},
            {"run": "echo one\n\n  echo two\n", "shell": "bash"},
        ],
        "description": "folded text",
    }


@pytest.mark.parametrize(
    "text",
    [
        "on: [pull_request_target]\n",
        "on: {pull_request_target: {}}\n",
        "base: &base\n  a: b\nother: *base\n",
        "permissions: !!map {}\n",
        "on:\n\tpull_request:\n",
        "on:\n  pull_request:\non:\n  pull_request_target:\n",
        '"on": pull_request\n',
        "? on\n: pull_request\n",
        "---\non: pull_request\n",
        "jobs:\n- run\n",
        "on: - pull_request\n",
        "on:\n    pull_request:\n  push:\n",
        "run: |+\n  echo\n",
    ],
)
def test_the_reader_refuses_what_it_does_not_read(text: str) -> None:
    """Loudly: a file half read would pass these tests on the half."""

    with pytest.raises(UnsupportedYaml):
        load(text)
