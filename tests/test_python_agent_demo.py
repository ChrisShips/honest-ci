"""The Python demo, run the way its README tells a reader to run it.

The demo is the first thing a stranger tries, and its README makes the claim:
four patches, the verdict each one gets, the exit code and the rule behind it.
That table is read here and every row is run, so the README cannot drift away
from what honest-ci does -- a detector change that moves a scenario to another
verdict, a patch that stopped applying, a row nobody wrote a patch for. The
expected results live in the table and nowhere else.

The setup is the README's: a git checkout of the demo, a patch applied to the
working tree, ``honest-ci verify`` with the regression gate switched off and
nothing else configured. No model and no Docker, because that is the promise
the demo makes, and because a scanner rule sneaking into its rule library would
make the verdicts depend on which machine ran them.

The same four also run the way the GitHub Action runs them, and what a reader
is shown -- the comment, the job summary, the commit status and the evidence
pack -- has to be English throughout.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from honest_ci import cli, intake
from honest_ci.cli import EXIT_CODES, app
from honest_ci.contracts import ApprovalReading, ChangeRequest, EvidencePack
from honest_ci.report import render_forge_comment
from honest_ci.state import RunStatus

from .factories import StubForge, cjk_lines

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "python-agent-demo"
README = EXAMPLE / "README.md"


@dataclass(frozen=True)
class Scenario:
    name: str
    verdict: str
    exit_code: int
    #: The one rule whose hit decides the verdict; ``None`` when nothing may fire.
    rule: str | None


_ROW = re.compile(
    r"^\|\s*`(?P<name>[\w-]+)`\s*\|[^|]*\|\s*`(?P<verdict>\w+)`\s*\|\s*(?P<code>\d+)\s*\|"
    r"\s*(?:`(?P<rule>[\w-]+)`|—)\s*\|$"
)


def _readme_table() -> list[Scenario]:
    return [
        Scenario(match["name"], match["verdict"], int(match["code"]), match["rule"])
        for match in map(_ROW.match, README.read_text(encoding="utf-8").splitlines())
        if match
    ]


SCENARIOS = _readme_table()
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="the demo is a git checkout")


def test_the_readme_has_one_row_for_each_scenario_patch() -> None:
    names = [scenario.name for scenario in SCENARIOS]
    patches = {path.stem for path in (EXAMPLE / "scenarios").glob("*.patch")}

    assert len(names) == len(set(names)), names
    assert set(names) == patches
    # The four the demo exists to show: one change that should merge, and one
    # for each way of turning a red suite green without fixing anything.
    assert {"clean", "delete-failing-test", "rewrite-expectation", "add-skip"} <= patches


def test_each_row_names_a_verdict_its_exit_code_stands_for() -> None:
    # The README uses the word a reader sees on the pull request; a run
    # records the same state as awaiting_human.
    statuses = {"needs_human": "awaiting_human"}

    for scenario in SCENARIOS:
        status = statuses.get(scenario.verdict, scenario.verdict)
        assert EXIT_CODES.get(status) == scenario.exit_code, scenario


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=demo", "-c", "user.email=demo@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _checkout(tmp_path: Path, scenario: str) -> Path:
    """The README's copy of the demo, committed, with one scenario applied on top."""

    repo = tmp_path / "honest-ci-demo"
    shutil.copytree(EXAMPLE, repo)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "baseline")
    _git(repo, "apply", str(Path("scenarios") / f"{scenario}.patch"))
    return repo


def _configured_by_the_readme(monkeypatch: pytest.MonkeyPatch) -> None:
    # Configured by the README and by nothing else: a developer's own rules
    # path, sandbox or model provider must not decide what a stranger sees.
    for name in list(os.environ):
        if name.startswith("HONEST_CI_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HONEST_CI_DISABLED_GATES", "regression")


def _evidence(repo: Path) -> dict[str, Any]:
    packs = list((repo / ".honest-ci" / "artifacts").glob("*/evidence.json"))
    assert len(packs) == 1, packs
    pack = json.loads(packs[0].read_text(encoding="utf-8"))
    assert isinstance(pack, dict)
    return pack


@needs_git
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_each_scenario_ends_where_the_readme_says(
    scenario: Scenario, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _checkout(tmp_path, scenario.name)
    _configured_by_the_readme(monkeypatch)
    monkeypatch.chdir(repo)

    result = CliRunner().invoke(app, ["verify"])

    assert result.exit_code == scenario.exit_code, result.stdout
    pack = _evidence(repo)
    fired = {finding["rule_id"] for report in pack["gate_reports"] for finding in report["findings"]}
    # Exactly the README's rule: a scenario held for the right reason and also
    # tripping something else would be telling the reader a different story.
    assert fired == ({scenario.rule} if scenario.rule else set())
    # Nothing ran a scanner, so nothing needed a sandbox.
    assert not any(report["tool_reports"] for report in pack["gate_reports"])


@needs_git
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_every_scenario_leaves_the_demo_suite_green(scenario: Scenario, tmp_path: Path) -> None:
    """The premise of the demo: CI passes all of them, the held ones included."""

    repo = _checkout(tmp_path, scenario.name)
    env = {name: value for name, value in os.environ.items() if name != "PYTEST_ADDOPTS"}

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr


# ── what a reader is shown is in English ─────────────────────────────────────

#: The repository the demo's pull request is opened on.
_REPO = "octo-org/honest-ci-demo"
#: The same repository as the pull_request event names it.
_REPOSITORY: dict[str, object] = {"id": 1001, "full_name": _REPO}


def _not_english(shown: dict[str, str]) -> dict[str, list[str]]:
    """The lines of each text that carry a CJK character; empty when none does."""

    found = {
        where: cjk_lines(text)
        for where, text in shown.items()
    }
    return {where: lines for where, lines in found.items() if lines}


def _rendered_evidence(artifacts: Path) -> str:
    packs = list(artifacts.glob("*/evidence.md"))
    assert len(packs) == 1, packs
    return packs[0].read_text(encoding="utf-8")


class _PullRequest(StubForge):
    """GitHub as the Action meets it: one pull request, its head checked out at ``checkout``.

    The comment it records is the one the GitHub adapter posts: the same
    rendering of the same evidence.
    """

    def __init__(self, change: ChangeRequest, checkout: Path) -> None:
        super().__init__(change)
        self.checkout = checkout
        self.comments: list[str] = []

    def materialize(self, change: ChangeRequest, workspace: Path) -> Path:
        return self.checkout

    def publish_report(
        self,
        change: ChangeRequest,
        evidence: EvidencePack,
        status: RunStatus,
        approvals: ApprovalReading | None = None,
        evidence_at: str | None = None,
        license_note: str | None = None,
    ) -> None:
        super().publish_report(change, evidence, status, approvals, evidence_at, license_note)
        self.comments.append(
            render_forge_comment(
                evidence,
                status=status,
                approvals=approvals,
                evidence_at=evidence_at,
                license_note=license_note,
            )
        )


def _pull_request(repo: Path, scenario: str) -> ChangeRequest:
    """The applied scenario as a pull request whose author wrote in English.

    The title and the branches are the author's words, not honest-ci's, so they
    are English here: a CJK character found below is then one honest-ci wrote.
    """

    change = intake.build_change(intake.from_git(repo), repo=_REPO)
    return change.model_copy(
        update={
            "forge": "github",
            "title": f"Demo scenario {scenario}",
            "description": "",
            "author": "octo-agent",
            "source_branch": f"agent/{scenario}",
            "target_branch": "main",
        }
    )


def _as_the_action_runs_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """The environment ``action.yml`` gives ``honest-ci run-pr`` on a pull_request run.

    Returns the run directory and the job summary file. The rules and the
    policy are copied into the run directory, as the Action copies them from
    the base commit.
    """

    root = tmp_path / "run"
    root.mkdir()
    for name in ("honest-ci-rules.toml", "honest-ci-policy.toml"):
        shutil.copyfile(EXAMPLE / name, root / name)
    runner_files = tmp_path / "runner"
    runner_files.mkdir()
    event = runner_files / "event.json"
    event.write_text(
        json.dumps(
            {
                "pull_request": {
                    "number": 7,
                    "head": {"repo": _REPOSITORY},
                    "base": {"repo": _REPOSITORY},
                }
            }
        ),
        encoding="utf-8",
    )
    summary = runner_files / "step_summary.md"
    # The suite may itself run inside a workflow, whose variables would leak in.
    for name in list(os.environ):
        if name.startswith("GITHUB_"):
            monkeypatch.delenv(name)
    for name, value in {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": _REPO,
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "4242",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_STEP_SUMMARY": str(summary),
        "HONEST_CI_FORGE": "github",
        "HONEST_CI_RULES_PATH": str(root / "honest-ci-rules.toml"),
        "HONEST_CI_POLICY_PATH": str(root / "honest-ci-policy.toml"),
        "HONEST_CI_ARTIFACTS_DIR": str(root / "evidence"),
    }.items():
        monkeypatch.setenv(name, value)
    return root, summary


@needs_git
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_what_the_pull_request_shows_is_in_english(
    scenario: Scenario,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The comment, the job summary, the commit status and the evidence pack.

    And the workflow log: what the step printed, and what it logged. Run the
    way the Action runs it, under the workflow template's configuration, which
    switches off the same gate the README does.
    """

    caplog.set_level(logging.DEBUG)
    repo = _checkout(tmp_path, scenario.name)
    _configured_by_the_readme(monkeypatch)
    root, summary = _as_the_action_runs_it(monkeypatch, tmp_path)
    forge = _PullRequest(_pull_request(repo, scenario.name), checkout=repo)
    monkeypatch.setattr(cli, "build_forge_adapter", lambda settings: forge)

    result = CliRunner().invoke(app, ["run-pr", _REPO, "7", "--workspace", str(root)])

    assert result.exit_code == scenario.exit_code, result.stdout
    assert forge.comments and forge.progress and forge.statuses
    shown = {
        "comment": "\n".join(forge.comments),
        "job summary": summary.read_text(encoding="utf-8"),
        "commit status": "\n".join([*forge.progress, *(text for _, text in forge.statuses)]),
        "evidence.md": _rendered_evidence(root / "evidence"),
        "terminal": result.output,
        "log": caplog.text,
    }
    assert _not_english(shown) == {}


@needs_git
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_what_a_readme_run_prints_and_writes_is_in_english(
    scenario: Scenario, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The commands the README has a reader type, and the evidence they leave behind."""

    repo = _checkout(tmp_path, scenario.name)
    _configured_by_the_readme(monkeypatch)
    monkeypatch.chdir(repo)
    cli_runner = CliRunner()

    verified = cli_runner.invoke(app, ["verify"])
    assert verified.exit_code == scenario.exit_code, verified.output
    [run_id] = [path.name for path in (repo / ".honest-ci" / "artifacts").iterdir()]
    doctor = cli_runner.invoke(app, ["doctor"])
    evidence = cli_runner.invoke(app, ["evidence", run_id])

    # Whether this machine has Docker decides doctor's exit code, not its language;
    # what matters here is that it ran to the end rather than into a traceback.
    assert doctor.exception is None or isinstance(doctor.exception, SystemExit), doctor.exception
    # Whatever the machine, the ledger verify has just written can take the next run.
    assert "✓ Ledger is writable" in doctor.output, doctor.output
    assert evidence.exit_code == 0, evidence.output
    shown = {
        "honest-ci verify": verified.output,
        "honest-ci doctor": doctor.output,
        "honest-ci evidence": evidence.output,
        "evidence.md": _rendered_evidence(repo / ".honest-ci" / "artifacts"),
    }
    assert _not_english(shown) == {}
