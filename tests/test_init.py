"""``honest-ci init --github``: the configuration a first install starts from.

What it writes has one job: the pull request after the install gets a verdict
that means something, with nothing else set up. So the tests hold the files to
what the Action and the gates will make of them -- the paths the Action reads,
the gates the workflow runs, the rules that need nothing but the diff -- rather
than to their text. What the demo repository built from them shows on real
changes is in ``test_github_demo.py``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from honest_ci.actions import workflow_template
from honest_ci.cli import EXIT_USAGE, app
from honest_ci.gates import GATE_REGISTRY
from honest_ci.integrity import INTEGRITY_CHECKS
from honest_ci.policy import DEFAULT_SENSITIVE_PATHS, PolicyConfig
from honest_ci.rules import RuleSet
from honest_ci.scaffold import GITHUB_FILES
from honest_ci.settings import Settings

from .factories import CJK
from .workflow_yaml import load

REPO = Path(__file__).resolve().parents[1]
RESOURCES = REPO / "src" / "honest_ci" / "resources"

RULES = "honest-ci-rules.toml"
POLICY = "honest-ci-policy.toml"
WORKFLOW = ".github/workflows/honest-ci.yml"

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="init asks git for the root")


@pytest.fixture(autouse=True)
def _nobodys_git(monkeypatch: pytest.MonkeyPatch) -> None:
    # init asks git where the repository's root is. A GIT_DIR passed down by a
    # hook running the tests would answer for the wrong repository.
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def init(root: Path, *args: str) -> Result:
    return CliRunner().invoke(app, ["init", *args, "--workspace", str(root)])


def written(root: Path) -> dict[str, str]:
    return {
        path: (root / path).read_text(encoding="utf-8")
        for path in (RULES, POLICY, WORKFLOW)
        if (root / path).exists()
    }


def _git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)


# ── the command ──────────────────────────────────────────────────────────────


def test_init_writes_the_rules_the_policy_and_the_workflow(tmp_path: Path) -> None:
    result = init(tmp_path, "--github")

    assert result.exit_code == 0, result.stdout
    assert written(tmp_path) == {
        RULES: (RESOURCES / RULES).read_text(encoding="utf-8"),
        POLICY: (RESOURCES / POLICY).read_text(encoding="utf-8"),
        # The one template the package hands out, not a copy of it.
        WORKFLOW: workflow_template(),
    }
    for path in (RULES, POLICY, WORKFLOW):
        assert f"Created {path}" in result.stdout


def test_the_first_install_says_its_pull_request_goes_to_a_person(tmp_path: Path) -> None:
    """Expected, and said up front: otherwise the first thing a new user sees looks like a fault."""

    result = init(tmp_path, "--github")

    assert "first pull request goes to a person" in result.stdout
    assert Settings().forge_status_name in result.stdout


@needs_git
def test_init_writes_at_the_root_of_a_repository(tmp_path: Path) -> None:
    _git_init(tmp_path)

    result = init(tmp_path, "--github")

    assert result.exit_code == 0, result.stdout
    assert set(written(tmp_path)) == {RULES, POLICY, WORKFLOW}


def test_a_file_that_exists_is_kept_as_it_is(tmp_path: Path) -> None:
    (tmp_path / POLICY).write_text("[policy]\nrequired_checks = []\n", encoding="utf-8")

    result = init(tmp_path, "--github")

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / POLICY).read_text(encoding="utf-8") == "[policy]\nrequired_checks = []\n"
    assert f"Kept {POLICY}" in result.stdout
    assert f"Created {RULES}" in result.stdout
    assert f"Created {WORKFLOW}" in result.stdout


def test_running_init_again_changes_nothing(tmp_path: Path) -> None:
    init(tmp_path, "--github")
    (tmp_path / RULES).write_text('version = "ours"\n', encoding="utf-8")
    before = written(tmp_path)

    result = init(tmp_path, "--github")

    assert result.exit_code == 0, result.stdout
    assert written(tmp_path) == before
    assert [line for line in result.stdout.splitlines() if line] == [
        f"Kept {path}: it already exists, and init does not overwrite it."
        for path in (RULES, POLICY, WORKFLOW)
    ]


def test_a_symbolic_link_at_a_path_is_not_followed(tmp_path: Path) -> None:
    """A link is something that exists; writing through it would overwrite whatever it names."""

    elsewhere = tmp_path / "elsewhere.toml"
    elsewhere.write_text("somebody's file\n", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    (root / RULES).symlink_to(elsewhere)
    (root / POLICY).symlink_to(tmp_path / "missing.toml")

    result = init(root, "--github")

    assert result.exit_code == 0, result.stdout
    assert elsewhere.read_text(encoding="utf-8") == "somebody's file\n"
    assert not (tmp_path / "missing.toml").exists()
    assert f"Kept {RULES}" in result.stdout
    assert f"Kept {POLICY}" in result.stdout


def test_init_needs_to_be_told_the_platform(tmp_path: Path) -> None:
    result = init(tmp_path)

    assert result.exit_code == EXIT_USAGE
    assert "--github" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_init_needs_a_directory(tmp_path: Path) -> None:
    result = init(tmp_path / "missing", "--github")

    assert result.exit_code == EXIT_USAGE
    assert not (tmp_path / "missing").exists()


@needs_git
def test_init_refuses_a_directory_below_the_repository_root(tmp_path: Path) -> None:
    """Files there would look installed and do nothing: GitHub and the Action read the root."""

    _git_init(tmp_path)
    below = tmp_path / "services" / "shop"
    below.mkdir(parents=True)

    result = init(below, "--github")

    assert result.exit_code == EXIT_USAGE
    assert "Run init at its root" in result.stdout
    assert list(below.iterdir()) == []


def test_a_file_that_cannot_be_written_is_named(tmp_path: Path) -> None:
    (tmp_path / ".github").write_text("not a directory\n", encoding="utf-8")

    result = init(tmp_path, "--github")

    assert result.exit_code == 1
    assert f"Could not write {WORKFLOW}" in result.stdout


# ── what it writes ───────────────────────────────────────────────────────────


def test_the_files_go_where_the_action_and_the_settings_look() -> None:
    action = (REPO / "action.yml").read_text(encoding="utf-8")

    assert [item.path for item in GITHUB_FILES] == [RULES, POLICY, WORKFLOW]
    assert Settings().rules_path == Path(RULES)
    assert Settings().policy_path == Path(POLICY)
    assert f'copy rules "${{HONEST_CI_RULES_PATH:-{RULES}}}"' in action
    assert f'copy policy "${{HONEST_CI_POLICY_PATH:-{POLICY}}}"' in action


def test_the_package_ships_the_files_init_writes() -> None:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = pyproject["tool"]["setuptools"]["package-data"]["honest_ci"]
    package = REPO / "src" / "honest_ci"
    shipped = {path for pattern in patterns for path in package.glob(pattern)}

    assert {RESOURCES / RULES, RESOURCES / POLICY} <= shipped


def _rules(root: Path) -> RuleSet:
    return RuleSet.load(root, Path(RULES), "v1")


def test_every_rule_is_decided_from_the_diff_alone(tmp_path: Path) -> None:
    """No model, no scanner, no sandbox: the Action installs none of them.

    The test-integrity rules, one per check, so that a check added later is a
    rule a new install has from the start.
    """

    init(tmp_path, "--github")
    rules = _rules(tmp_path)

    assert {rule.kind for rule in rules.rules} == {"integrity"}
    assert sorted(rule.check or "" for rule in rules.rules) == sorted(INTEGRITY_CHECKS)


def _commented_out_rules(text: str) -> str:
    """The ``[[rule]]`` tables the file leaves commented out, uncommented."""

    kept: list[str] = []
    in_rule = False
    for line in text.splitlines():
        if line == "# [[rule]]":
            in_rule = True
            kept.append("[[rule]]")
        elif in_rule and re.match(r"# [a-z_]+ = ", line):
            kept.append(line.removeprefix("# "))
        else:
            in_rule = False
    return "\n".join(kept) + "\n"


def test_the_examples_for_a_teams_own_rules_load_once_uncommented(tmp_path: Path) -> None:
    """Somebody will uncomment them first thing; a broken one would stop every run."""

    init(tmp_path, "--github")
    (tmp_path / RULES).write_text(
        (tmp_path / RULES).read_text(encoding="utf-8")
        + _commented_out_rules((tmp_path / RULES).read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    kinds = {rule.id: rule.kind for rule in _rules(tmp_path).rules}

    assert {kinds[rule] for rule in ("R-01", "R-10")} == {"regex", "path"}


def _gates_the_workflow_runs(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    steps = [step for job in load(workflow_template())["jobs"].values() for step in job["steps"]]
    action = [step for step in steps if step.get("uses", "").startswith("ChrisShips/honest-ci@")]
    assert len(action) == 1, steps
    for name in list(os.environ):
        if name.startswith("HONEST_CI_"):
            monkeypatch.delenv(name)
    for name, value in action[0].get("env", {}).items():
        monkeypatch.setenv(name, value)
    return set(GATE_REGISTRY) - set(Settings.from_env().disabled_gates)


def test_the_policy_requires_only_the_gates_the_workflow_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A required gate that reaches no verdict holds every pull request for a person."""

    init(tmp_path, "--github")
    policy = PolicyConfig.load(tmp_path, Path(POLICY), "v1")

    assert "review" in policy.required_checks
    assert set(policy.required_checks) <= _gates_the_workflow_runs(monkeypatch)


def test_the_policy_names_honest_cis_own_files_as_sensitive(tmp_path: Path) -> None:
    """A policy that sets sensitive_paths replaces the built-in list rather than adding to it."""

    init(tmp_path, "--github")
    policy = PolicyConfig.load(tmp_path, Path(POLICY), "v1")

    assert {RULES, POLICY, ".github/**"} <= set(policy.sensitive_paths)
    assert policy.sensitive_paths == list(DEFAULT_SENSITIVE_PATHS)


def test_what_init_writes_and_says_is_english(tmp_path: Path) -> None:
    result = init(tmp_path, "--github")
    shown = {**written(tmp_path), "output": result.stdout}

    # What init writes goes to repositories whose readers may not read Chinese.
    assert {where: CJK.findall(text) for where, text in shown.items() if CJK.search(text)} == {}
