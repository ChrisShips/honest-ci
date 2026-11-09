from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner, Result

from honest_ci import cli, license_check
from honest_ci.cli import app
from honest_ci.contracts import (
    ChangedFile,
    ChangeRequest,
    PlatformApproval,
    PlatformApprovals,
    PlatformReview,
    RunMetrics,
    RunRecord,
    Verdict,
)
from honest_ci.license_check import KEY_VARIABLE
from honest_ci.maven import (
    MAVEN_CLASS_FILE_CEILINGS,
    MAVEN_JDK_CEILING,
    MAVEN_PLUGINS,
    MAVEN_STAMP,
    offline_coordinates,
)
from honest_ci.report import STATUS_HINT, VERDICT_MARKER
from honest_ci.settings import MODEL_NOT_ENABLED, Settings
from honest_ci.state import RunState
from honest_ci.storage import SQLiteLedger

from .factories import (
    StubForge,
    cjk_lines,
    field,
    fill_repository,
    headings,
    make_change,
    section,
    table_row,
    unverified,
)
from .test_license_check import (
    AN_ORGANIZATIONS_PRIVATE_REPOSITORY,
    ORG,
    SOMEONE_ELSE,
    pull_request_event,
    repository,
)
from .test_licensing import TEST, issue

runner = CliRunner()


def readiness(output: str) -> list[tuple[bool, str]]:
    """Doctor's readiness checks, in order: whether each is ready, and what it says."""

    checks: list[tuple[bool, str]] = []
    for line in output.splitlines():
        mark, _, text = line.strip().partition(" ")
        if mark in {"✓", "✗"}:
            checks.append((mark == "✓", text))
    return checks


def ready(output: str) -> list[str]:
    return [text for ok, text in readiness(output) if ok]


def not_ready(output: str) -> list[str]:
    return [text for ok, text in readiness(output) if not ok]


def gate_row(output: str, gate: str) -> tuple[str, str]:
    """What a run printed for ``gate``: its verdict, and the summary after it."""

    for line in output.splitlines():
        if line.startswith(f"  {gate} "):
            verdict, _, summary = line[len(gate) + 2 :].strip().partition(" ")
            return verdict, summary.strip()
    raise AssertionError(f"no row for {gate} in:\n{output}")


#: Set by the runner in every job -- including the one that runs this suite in
#: CI, where they would put these tests inside a real workflow run.
RUNNER_VARIABLES = (
    "GITHUB_ACTIONS",
    "GITHUB_WORKSPACE",
    "GITHUB_REPOSITORY",
    "GITHUB_SERVER_URL",
    "GITHUB_RUN_ID",
    "GITHUB_EVENT_NAME",
    "GITHUB_EVENT_PATH",
    "GITHUB_STEP_SUMMARY",
)


#: What Homebrew's mvn does before it starts Maven: run on the JDK JAVA_HOME
#: names, and on Homebrew's own newest openjdk when nothing names one. Each
#: fake JDK below is a directory holding the version it reports.
HOMEBREW_MVN = """#!/bin/sh
echo "$@" >> "{root}/mvn-calls"
JAVA_HOME="${{JAVA_HOME:-{root}/openjdk}}"
if [ ! -f "$JAVA_HOME/version" ]; then
  echo "The JAVA_HOME environment variable is not defined correctly," >&2
  exit 1
fi
echo "Apache Maven 3.10.0"
echo "Java version: $(cat "$JAVA_HOME/version"), vendor: Homebrew, runtime: $JAVA_HOME"
"""


def jdk(machine: Path, name: str, version: str) -> Path:
    """A JDK on the fake machine, installed (or upgraded) under ``name``."""

    home = machine / name
    home.mkdir(exist_ok=True)
    (home / "version").write_text(version, encoding="utf-8")
    return home


@pytest.fixture
def machine(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This machine's Maven, as doctor asks it: Homebrew's, its newest openjdk at 21.

    Outside the workspace, which is often ``tmp_path`` itself. A test about a
    newer JDK upgrades ``openjdk`` with ``jdk(machine, "openjdk", ...)``.
    """

    root = tmp_path_factory.mktemp("machine")
    jdk(root, "openjdk", "21.0.13")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    mvn = bin_dir / "mvn"
    mvn.write_text(HOMEBREW_MVN.format(root=root), encoding="utf-8")
    mvn.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("JAVA_HOME", raising=False)
    return root


@pytest.fixture(autouse=True)
def _process_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, machine: Path) -> None:
    # Keep the CLI from probing docker, and from picking up a developer's own
    # provider choice or credentials while asserting on the doctor output.
    monkeypatch.setenv("HONEST_CI_SANDBOX_MODE", "process")
    # Nor this machine's ~/.m2, which a process-mode Java run reads: what it
    # holds is no test's business. Its mvn and JDK are ``machine``'s.
    monkeypatch.setenv("HONEST_CI_MAVEN_HOST_REPOSITORY", str(tmp_path / "host-m2"))
    for name in (
        "HONEST_CI_MODEL_PROVIDER",
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
        "QWEN_API_KEY",
        "ZAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    # Nor from finding itself in the workflow run that runs this suite in CI:
    # the tests about a workflow set up the one they need.
    for name in RUNNER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    # Most of these tests verify a patch against a tree that does not hold it,
    # which is exactly what the regression gate refuses to test. The tests that
    # are about that gate switch it back on and build a tree that does.
    monkeypatch.setenv("HONEST_CI_DISABLED_GATES", "regression")


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "honest-ci" in result.stdout


def test_doctor_reports_configuration_without_leaking_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_abcdefghijklmnopqrstuvwxyz012345")
    monkeypatch.setenv("HONEST_CI_FORGE", "github")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "ghp_" not in result.stdout
    assert field(result.stdout, "Model provider") == MODEL_NOT_ENABLED  # off by default
    assert "fail-closed" in field(result.stdout, "Tool unavailable")
    # Says whether the model may speak: the switch alone would read "on" here.
    assert field(result.stdout, "Model explanation").startswith(f"off ({MODEL_NOT_ENABLED})")
    # spec does not ship yet, and doctor says so plainly.
    assert field(result.stdout, "Not deployed").startswith("spec ")
    assert "review" in field(result.stdout, "Deployed gates").split(", ")


def test_doctor_keeps_the_model_off_when_only_a_key_is_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A key lying in the environment -- exported for some other tool -- is not
    # a decision to send this repository's changes to that vendor.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abcdefghijklmnopqrstuvwx")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "sk-abcdefghijklmnopqrstuvwx" not in result.stdout
    assert field(result.stdout, "Model provider") == MODEL_NOT_ENABLED
    assert field(result.stdout, "Model") == "none"
    assert field(result.stdout, "Model judgment").startswith(f"off ({MODEL_NOT_ENABLED})")
    # The generated regression tests are off for the same reason, and say so.
    assert f"off ({MODEL_NOT_ENABLED})" in field(result.stdout, "Regression gate")


def test_doctor_names_the_missing_key_once_a_provider_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_MODEL_PROVIDER", "qwen")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    provider = field(result.stdout, "Model provider")
    assert provider.startswith("qwen (") and "DASHSCOPE_API_KEY" in provider
    assert field(result.stdout, "Model") == "none"
    explanation = field(result.stdout, "Model explanation")
    assert explanation.startswith("off (") and "DASHSCOPE_API_KEY" in explanation


def test_doctor_reports_a_model_once_both_switches_are_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abcdefghijklmnopqrstuvwx")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "sk-abcdefghijklmnopqrstuvwx" not in result.stdout
    assert field(result.stdout, "Model provider") == "deepseek"
    assert field(result.stdout, "Model").startswith("deepseek-chat")
    assert field(result.stdout, "Model explanation").startswith("on ")


def test_doctor_says_which_host_an_openai_compatible_key_goes_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With ``openai`` pointed at a local Ollama, the provider's name alone
    # would suggest the key and the diffs go to OpenAI.
    monkeypatch.setenv("HONEST_CI_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abcdefghijklmnopqrstuvwx")
    monkeypatch.setenv("HONEST_CI_OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("HONEST_CI_OPENAI_MODEL", "qwen3:8b")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "sk-abcdefghijklmnopqrstuvwx" not in result.stdout
    assert field(result.stdout, "Model") == "qwen3:8b @ http://localhost:11434/v1"


def test_doctor_refuses_a_provider_it_does_not_know(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A misspelt name must not quietly read as "off": whoever set it meant on.
    monkeypatch.setenv("HONEST_CI_MODEL_PROVIDER", "deepsek")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 2
    assert "HONEST_CI_MODEL_PROVIDER=deepsek" in result.stdout
    assert "Model provider" not in result.stdout


def test_doctor_warns_when_the_policy_requires_a_gate_nobody_deployed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HONEST_CI_DISABLED_GATES")
    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    # The default policy requires spec, which does not ship yet. An operator
    # should learn that here, not from a fortnight of needs_human verdicts.
    unsatisfiable = field(result.stdout, "⚠ Unsatisfiable")
    assert unsatisfiable.startswith("spec ")
    assert "needs_human" in unsatisfiable


def test_doctor_reports_the_regression_gate_and_what_is_switched_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_COVERAGE_THRESHOLD", "0.8")
    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert field(result.stdout, "Deployed gates") == "review, regression"
    # Switched off is its own line, and it says what switched-off means.
    switched_off = field(result.stdout, "Switched off")
    assert switched_off.startswith("regression (HONEST_CI_DISABLED_GATES")
    assert "skipped" in switched_off
    regression = field(result.stdout, "Regression gate")
    assert "Python pytest -q" in regression
    assert "80%" in regression
    # A required gate that is switched off is as unsatisfiable as one nobody deployed.
    assert field(result.stdout, "⚠ Unsatisfiable").startswith("spec, regression ")


def test_doctor_reports_the_rule_library(tmp_path: Path) -> None:
    (tmp_path / "honest-ci-rules.toml").write_text(
        """
        [[rule]]
        id = "T-01"
        title = "Static checks must pass"
        severity = "high"
        kind = "tool"
        tool = "ruff"
        """,
        encoding="utf-8",
    )

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    library = field(result.stdout, "Rule library")
    assert re.search(r"\b1 rule\b", library)
    assert "ruff" in library  # the scanner the rule needs


def test_doctor_says_whose_the_license_key_is_and_never_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unset = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])
    key = _signed(monkeypatch, expires=_today() + timedelta(days=365))
    monkeypatch.setenv(KEY_VARIABLE, key)

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert field(unset.stdout, "License key") == "none (HONEST_CI_LICENSE_KEY is not set)"
    assert result.exit_code == 0
    assert field(result.stdout, "License key").startswith(
        "signature verified · L-2026-0007 for octo-org (id 9001) · plan small, 3 seats · "
        "expires on "
    )
    assert key.split(".")[2] not in result.stdout


def test_a_license_key_doctor_refuses_is_said_and_leaves_its_exit_code_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configuration, not readiness: the key stops a run only where one is due.

    Whether one is due turns on the repository and the pull request, which
    doctor doesn't know; where none is, the key isn't even read.
    """

    monkeypatch.setenv(KEY_VARIABLE, "not a key")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert field(result.stdout, "License key").startswith(
        "refused: it isn't an honest-ci license key"
    )
    assert "not a key" not in result.stdout


# ── doctor: everything this machine lacks, said at once ──────────────────────


def test_doctor_separates_what_the_machine_lacks_from_what_it_was_configured_to_do(
    tmp_path: Path,
) -> None:
    """Readiness is its own block, and the exit code follows only it.

    The default policy here requires ``spec``, which nobody deployed — doctor
    says so, and still exits 0: a deployment that runs one gate is configured
    exactly like that on purpose, and an installer that refused over it would
    be refusing over somebody's decision rather than over a broken machine.
    """

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    checks = ready(result.stdout)
    assert any(str(tmp_path / ".honest-ci" / "ledger.sqlite3") in text for text in checks)
    assert any(str(tmp_path / ".honest-ci" / "artifacts") in text for text in checks)
    # Process mode needs no image, and the check says which switch made it so.
    assert any("HONEST_CI_SANDBOX_MODE" in text for text in checks)
    assert not_ready(result.stdout) == []


def test_doctor_fails_when_the_ledger_directory_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    read_only.chmod(0o500)
    monkeypatch.setenv("HONEST_CI_LEDGER_PATH", "ledger/ledger.sqlite3")
    try:
        result = runner.invoke(app, ["doctor", "--workspace", str(read_only)])
    finally:
        read_only.chmod(0o700)

    # The one an installer script can branch on: nothing this deployment does
    # can be written down, so nothing it concludes can be evidence.
    assert result.exit_code == 1
    failures = not_ready(result.stdout)
    assert any(str(read_only / "ledger" / "ledger.sqlite3") in text for text in failures)
    assert any(str(read_only / ".honest-ci" / "artifacts") in text for text in failures)
    # Every blocker at once, which is the whole point of the section: an
    # operator fixes the list, not the first line and then runs again.
    assert len(failures) == 2
    assert re.search(r"\b2 checks\b", result.stdout)


def test_doctor_passes_in_a_workspace_a_run_has_already_used(tmp_path: Path) -> None:
    # Every run after the first finds the ledger there. A doctor that took the
    # file for a directory that cannot be written failed every workspace in
    # which anyone had ever run verify.
    ledger = tmp_path / ".honest-ci" / "ledger.sqlite3"
    SQLiteLedger(ledger)

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert not_ready(result.stdout) == []
    assert any(str(ledger) in text for text in ready(result.stdout))


@pytest.mark.parametrize("read_only", ["ledger", "directory"])
def test_doctor_fails_when_a_run_could_read_the_ledger_but_not_write_to_it(
    tmp_path: Path, read_only: str
) -> None:
    # Writing to SQLite takes both the file and its directory, where every
    # write puts the rollback journal. With either one read-only, a run finds
    # the ledger and cannot record a thing in it.
    ledger = tmp_path / ".honest-ci" / "ledger.sqlite3"
    SQLiteLedger(ledger)
    (tmp_path / ".honest-ci" / "artifacts").mkdir()
    locked, mode = (ledger, 0o400) if read_only == "ledger" else (ledger.parent, 0o500)
    locked.chmod(mode)
    try:
        result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])
    finally:
        locked.chmod(0o700)

    assert result.exit_code == 1
    [failure] = not_ready(result.stdout)
    assert str(ledger) in failure


def test_doctor_says_docker_is_configured_but_not_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_SANDBOX_MODE", "docker")
    monkeypatch.setattr(cli, "docker_available", lambda: False)

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 1
    [failure] = not_ready(result.stdout)
    assert "Docker" in failure
    # And says what that silently becomes, rather than only that it failed.
    assert "process mode" in failure


def test_doctor_names_the_image_to_pull_before_the_first_run_pulls_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first-run failure this exists for: an image that is not on the machine.

    The first run would go and pull it, which on an air-gapped machine
    is a half hour nobody budgeted for, or never finishes. Doctor asks
    with ``image inspect`` and hands back the exact ``docker pull`` — it never
    pulls, because a diagnostic that changes the machine it is diagnosing is
    not one.
    """

    monkeypatch.setenv("HONEST_CI_SANDBOX_MODE", "docker")
    monkeypatch.setenv("HONEST_CI_DOCKER_IMAGE", "python:3.12-slim")
    monkeypatch.setattr(cli, "docker_available", lambda: True)
    monkeypatch.setattr(cli, "docker_image_present", lambda image: False)

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 1
    [failure] = not_ready(result.stdout)
    assert "docker pull python:3.12-slim" in failure


def test_doctor_says_a_java_deployment_has_a_cold_maven_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mvn -o`` on an empty local repository resolves nothing, not slowly.

    Worth its own line because the failure reads like a broken POM: Maven
    reports a plugin it cannot resolve, and the cause is that nobody warmed
    the cache this deployment is contractually offline against.
    """

    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    monkeypatch.setenv("HONEST_CI_SANDBOX_MODE", "docker")
    monkeypatch.setattr(cli, "docker_available", lambda: True)
    monkeypatch.setattr(cli, "docker_image_present", lambda image: True)

    cold = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert cold.exit_code == 1
    repository = str(tmp_path / ".honest-ci" / "m2")
    [failure] = not_ready(cold.stdout)
    assert repository in failure
    assert "mvn -o" in failure

    stamp = tmp_path / MAVEN_STAMP
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("warm", encoding="utf-8")
    warm = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert warm.exit_code == 0
    assert any(repository in text for text in ready(warm.stdout))


def test_doctor_counts_how_much_of_the_rule_library_records_its_samples(
    tmp_path: Path,
) -> None:
    # Rule mining goes from samples to a draft to a version a person signs off.
    # A rule that doesn't record its samples can't say why it exists, so the
    # share is worth putting in front of an operator.
    (tmp_path / "honest-ci-rules.toml").write_text(
        """
        [[rule]]
        id = "R-01"
        title = "Money is never computed with float"
        severity = "high"
        kind = "regex"
        pattern = "float"
        source_samples = ["INC-EXAMPLE-1"]

        [[rule]]
        id = "R-02"
        title = "No bare except"
        severity = "low"
        kind = "regex"
        pattern = "except"
        """,
        encoding="utf-8",
    )

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert "1/2" in field(result.stdout, "Rule library")


def test_doctor_names_a_broken_rule_file_instead_of_crashing(tmp_path: Path) -> None:
    (tmp_path / "honest-ci-rules.toml").write_text(
        '[[rule]]\nid = "R-1"\ntitle = "x"\nkind = "regex"\npattern = "(unclosed"\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "invalid pattern" in field(result.stdout, "Rule library")


def test_status_reports_a_missing_run(tmp_path: Path) -> None:
    result = runner.invoke(app, ["status", "nope", "--workspace", str(tmp_path)])
    assert result.exit_code == 1
    assert "nope" in result.stdout


def test_runs_lists_the_ledger(tmp_path: Path) -> None:
    ledger = SQLiteLedger(tmp_path / Settings().ledger_path)
    ledger.save_run_record(
        RunRecord(
            run_id="r1",
            change_ref="team/orders#482@aaaaaaaaaaaa",
            outcome="blocked",
            metrics=RunMetrics(gates_blocked=1, findings_total=2),
        )
    )

    result = runner.invoke(app, ["runs", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "team/orders#482" in result.stdout
    assert "blocked" in result.stdout


def test_runs_on_an_empty_ledger(tmp_path: Path) -> None:
    result = runner.invoke(app, ["runs", "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    # One line that says so, rather than an empty screen that could be a hang.
    assert len(result.stdout.strip().splitlines()) == 1


# ── running a change: honest-ci verify / rerun / approve ─────────────────────

RULES = r"""
version = "v1"

[[rule]]
id = "R-01"
title = "Money is never computed with float"
severity = "high"
kind = "regex"
pattern = 'float\s*\('
paths = ["**/*.py"]
fix = "Use decimal.Decimal."
"""

#: A rule library whose text happens to carry something shaped like a token.
#: Redaction has to hold all the way onto disk.
RULES_WITH_TOKEN = RULES + r"""
[[rule]]
id = "R-02"
title = "Never commit a token like glpat-abcdefghijklmnopqrst"
severity = "high"
kind = "regex"
pattern = 'glpat-'
paths = ["**/*.py"]
"""

BLOCKING_PATCH = """diff --git a/src/refund.py b/src/refund.py
--- a/src/refund.py
+++ b/src/refund.py
@@ -1,2 +1,3 @@
 def split_refund(order, amount):
+    share = float(amount) / len(order.items)
     return 0
"""

CLEAN_PATCH = """diff --git a/src/refund.py b/src/refund.py
--- a/src/refund.py
+++ b/src/refund.py
@@ -1,2 +1,3 @@
 def split_refund(order, amount):
+    # split by line item
     return 0
"""


def _workspace(
    tmp_path: Path,
    patch: str = BLOCKING_PATCH,
    rules: str | None = RULES,
    required_checks: str | None = None,
) -> Path:
    """A repo-shaped directory: the changed file, a patch, and the two configs."""

    source = tmp_path / "src" / "refund.py"
    source.parent.mkdir(exist_ok=True)
    source.write_text("def split_refund(order, amount):\n    return 0\n", encoding="utf-8")
    (tmp_path / "change.patch").write_text(patch, encoding="utf-8")
    if rules is not None:
        (tmp_path / "honest-ci-rules.toml").write_text(rules, encoding="utf-8")
    if required_checks is not None:
        (tmp_path / "honest-ci-policy.toml").write_text(
            f'[policy]\nversion = "v1"\nrequired_checks = {required_checks}\n', encoding="utf-8"
        )
    return tmp_path


def _verify(tmp_path: Path, *extra: str) -> Result:
    return runner.invoke(
        app,
        [
            "verify",
            "--workspace",
            str(tmp_path),
            "--diff",
            str(tmp_path / "change.patch"),
            *extra,
        ],
    )


def _run_id(tmp_path: Path) -> str:
    listing = runner.invoke(app, ["runs", "--workspace", str(tmp_path), "--limit", "1"])
    return listing.stdout.split()[0]


def test_verify_blocks_and_names_the_rule_that_fired(tmp_path: Path) -> None:
    result = _verify(_workspace(tmp_path))

    assert result.exit_code == 1
    assert "[R-01]" in result.stdout
    assert "refund.py:2" in result.stdout
    assert "Use decimal.Decimal." in result.stdout  # the rule's own fix, under it


def test_the_pack_names_a_patch_inside_the_workspace_relatively(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    # A CI agent passes the patch by absolute path; the pack must not inherit
    # the agent's build directory just because of how it was invoked.
    assert _verify(workspace).exit_code == 1

    pack = json.loads(_evidence_pack(workspace).read_text(encoding="utf-8"))
    assert pack["change"]["title"] == "Local change · patch file change.patch"
    assert pack["change"]["description"].startswith("Source: patch file change.patch")


def test_a_patch_from_outside_the_workspace_is_marked_as_such(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    workspace = _workspace(repo)
    outside = tmp_path / "patches"
    outside.mkdir()
    patch = outside / "change.patch"
    patch.write_text(BLOCKING_PATCH, encoding="utf-8")

    result = runner.invoke(
        app, ["verify", "--workspace", str(workspace), "--diff", str(patch)]
    )

    assert result.exit_code == 1, result.stdout
    pack = json.loads(_evidence_pack(workspace).read_text(encoding="utf-8"))
    # Shortening it to the bare filename would read as a patch that lives in
    # the repository. Where a change came from is not a claim to blur, so the
    # pack says the name it can verify and marks the rest as outside.
    assert pack["change"]["title"] == (
        "Local change · patch file change.patch (outside the repository)"
    )
    assert str(outside) not in pack["change"]["description"]


def test_verify_points_at_an_evidence_pack_that_exists(tmp_path: Path) -> None:
    result = _verify(_workspace(tmp_path))

    evidence = Path(field(result.stdout, "Evidence"))
    assert evidence.is_file()
    assert "R-01" in evidence.read_text(encoding="utf-8")


def test_verify_points_at_a_readable_rendering_of_that_pack(tmp_path: Path) -> None:
    result = _verify(_workspace(tmp_path))

    summary = Path(field(result.stdout, "Readable"))
    text = summary.read_text(encoding="utf-8")
    assert headings(text)[0].startswith("Evidence pack")
    # The blocked developer's next step, under the finding that blocked them.
    assert "Use decimal.Decimal." in section(text, "Blocking findings")
    # The regression gate is switched off here, and the reading says so.
    assert any("regression" in item for item in unverified(text))


def test_evidence_renders_a_run_from_the_ledger(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _verify(workspace)

    result = runner.invoke(app, ["evidence", _run_id(workspace), "--workspace", str(workspace)])

    assert result.exit_code == 0
    assert headings(result.stdout)[0].startswith("Evidence pack")
    assert "[R-01]" in result.stdout
    # No model was enabled, and it says so -- in both the places that would name one.
    assert "no model is enabled (`HONEST_CI_MODEL_PROVIDER`)" in result.stdout
    configuration = section(result.stdout, "Gate configuration")
    assert table_row(configuration, "Model") == ["not enabled"]


def test_evidence_renders_the_human_decision_alongside_the_gates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, rules=None)  # no rule library → needs_human
    _verify(workspace)
    run_id = _run_id(workspace)
    runner.invoke(
        app,
        [
            "approve", run_id, "--reason", "The rounding rule is confirmed", "--reviewer", "chris",
            "--workspace", str(workspace),
        ],
    )

    result = runner.invoke(app, ["evidence", run_id, "--workspace", str(workspace)])

    decision = section(result.stdout, "Human decision")
    assert "- Decision: **approved by a person**" in decision.splitlines()
    assert "- Reviewer: chris" in decision.splitlines()
    assert "once the fingerprint changes, this sign-off no longer applies" in decision


def test_evidence_reports_an_unknown_run(tmp_path: Path) -> None:
    result = runner.invoke(app, ["evidence", "nope", "--workspace", str(tmp_path)])

    assert result.exit_code == 1
    assert "nope" in result.stdout


def test_verify_sends_a_change_to_a_human_when_there_is_no_rule_library(tmp_path: Path) -> None:
    result = _verify(_workspace(tmp_path, rules=None))

    # "No rules" is not "no problems": the run ends at a human, with its own code.
    assert result.exit_code == 3
    assert "awaiting_human" in result.stdout


def test_verify_only_reaches_mergeable_when_the_policy_asks_for_deployed_gates(
    tmp_path: Path,
) -> None:
    default_policy = _verify(_workspace(tmp_path, patch=CLEAN_PATCH))
    assert default_policy.exit_code == 3  # spec is not deployed; regression is switched off
    # The run names the gates it is waiting on, and why that is not a pass.
    missing = next(
        line for line in default_policy.stdout.splitlines() if "spec, regression" in line
    )
    assert "needs_human" in missing

    narrowed = _verify(_workspace(tmp_path, patch=CLEAN_PATCH, required_checks='["review"]'))
    assert narrowed.exit_code == 0
    assert "mergeable" in narrowed.stdout


def test_verify_reads_a_patch_from_stdin(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result = runner.invoke(
        app,
        ["verify", "--workspace", str(workspace), "--diff", "-"],
        input=BLOCKING_PATCH,
    )

    assert result.exit_code == 1
    assert "[R-01]" in result.stdout


def test_verify_refuses_a_diff_and_a_base_at_once(tmp_path: Path) -> None:
    result = _verify(_workspace(tmp_path), "--base", "main")

    assert result.exit_code == 2
    [message] = result.stdout.strip().splitlines()
    assert "--diff" in message and "--base" in message  # both named, nothing run


def test_verify_reports_an_unreadable_change_instead_of_a_clean_verdict(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["verify", "--workspace", str(tmp_path), "--diff", str(tmp_path / "nope.patch")],
    )

    assert result.exit_code == 2
    assert "nope.patch" in result.stdout
    assert "mergeable" not in result.stdout


def test_verify_says_when_the_workspace_does_not_hold_the_changed_files(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "src" / "refund.py").unlink()

    result = _verify(workspace)

    # The scanners read the workspace, not the patch. Silence here would read
    # like a clean bill of health on files nobody scanned.
    assert any(
        line.startswith("⚠") and "src/refund.py" in line for line in result.stdout.splitlines()
    )


DELETED_TEST_PATCH = """diff --git a/tests/test_refund.py b/tests/test_refund.py
--- a/tests/test_refund.py
+++ b/tests/test_refund.py
@@ -1,4 +1,2 @@
-def test_split_rounds_down():
-    assert split_refund(None, 3) == 0
 def test_split():
     assert split_refund(None, 1) == 0
"""

INTEGRITY_RULES = """
version = "v1"

[[rule]]
id = "I-01"
title = "Tests do not disappear"
kind = "integrity"
check = "deleted_test"
"""


def test_verify_names_the_deleted_test_it_is_waiting_on_a_person_for(tmp_path: Path) -> None:
    workspace = _workspace(
        tmp_path, patch=DELETED_TEST_PATCH, rules=INTEGRITY_RULES, required_checks='["review"]'
    )
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_refund.py").write_text(
        "def test_split():\n    assert split_refund(None, 1) == 0\n", encoding="utf-8"
    )

    result = _verify(workspace)

    assert result.exit_code == 3, result.stdout
    assert "Test integrity (needs a human):" in result.stdout
    assert "[I-01] medium   tests/test_refund.py:1 (base)  Tests do not disappear" in result.stdout
    assert "Test `test_split_rounds_down` is removed" in result.stdout
    # It is why the run waits, so it is not also filed under "does not block".
    assert not any(line.startswith("Other") for line in result.stdout.splitlines())


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="the working-tree entry needs git")
@pytest.mark.parametrize(
    "char", ["\x9b", "‮", " "], ids=["c1-csi", "right-to-left-override", "line-separator"]
)
def test_a_working_tree_file_whose_name_cannot_be_shown_holds_the_change_for_a_person(
    tmp_path: Path, char: str
) -> None:
    workspace = _workspace(tmp_path, patch=CLEAN_PATCH, required_checks='["review"]')
    _git(workspace, "init", "--quiet")
    _git(workspace, "-c", "user.name=Dev", "-c", "user.email=dev@example.com", "add", "--all")
    _git(workspace, "-c", "user.name=Dev", "-c", "user.email=dev@example.com", "commit", "-qm", "base")
    (workspace / "src" / "refund.py").write_text(
        "def split_refund(order, amount):\n    # split by line item\n    return 0\n",
        encoding="utf-8",
    )
    assert runner.invoke(app, ["verify", "--workspace", str(workspace)]).exit_code == 0
    # Staged, so its name reaches intake through git's -z file list and its
    # lines through a diff header git quotes in octal.
    hostile = f"src/refund{char}.py"
    (workspace / hostile).write_text("import os\n", encoding="utf-8")
    _git(workspace, "add", "--all")

    result = runner.invoke(app, ["verify", "--workspace", str(workspace)])

    assert result.exit_code == 3, result.stdout
    assert f"unsafe path: `{hostile!r}`" in result.stdout
    summary = Path(field(result.stdout, "Readable")).read_text(encoding="utf-8")
    assert repr(hostile) in summary
    for text in (result.stdout, summary):
        assert char not in text


# ── the regression gate, end to end ─────────────────────────────────────────

PASSING_TEST = """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refund import split_refund


def test_split():
    assert split_refund(None, 1) == 0
"""

FAILING_TEST = PASSING_TEST.replace("== 0", "== 1")


def _tree_holding_the_change(tmp_path: Path, test_body: str) -> Path:
    """A repo whose working tree already has the clean patch applied, plus a suite."""

    workspace = _workspace(tmp_path, patch=CLEAN_PATCH, required_checks='["review", "regression"]')
    (workspace / "src" / "refund.py").write_text(
        "def split_refund(order, amount):\n    # split by line item\n    return 0\n",
        encoding="utf-8",
    )
    tests = workspace / "tests"
    tests.mkdir()
    (tests / "test_refund.py").write_text(test_body, encoding="utf-8")
    (workspace / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n'
        '[tool.coverage.run]\nsource = ["src"]\n',
        encoding="utf-8",
    )
    return workspace


def test_verify_runs_the_suite_measures_coverage_and_reaches_mergeable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HONEST_CI_DISABLED_GATES")
    workspace = _tree_holding_the_change(tmp_path, PASSING_TEST)

    result = _verify(workspace)

    assert result.exit_code == 0, result.stdout
    assert "mergeable" in result.stdout
    assert gate_row(result.stdout, "regression")[0] == "pass"
    evidence = runner.invoke(app, ["evidence", _run_id(workspace), "--workspace", str(workspace)])
    # The suite, its command and the coverage it measured are in the reading.
    tests = section(evidence.stdout, "Regression tests")
    assert "pytest" in table_row(tests, "Command")[0]
    assert table_row(tests, "Results")[0].startswith("1 passed · 0 failed")
    assert table_row(tests, "Line coverage") == ["100% (threshold 70%)"]


def test_the_pack_names_no_path_on_the_machine_that_produced_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evidence pack is handed to an auditor, so it names no machine.

    Asserted over every string in the pack rather than field by field: the
    point is that a field added later cannot quietly put the operator's disk
    back into the pack.
    """

    monkeypatch.delenv("HONEST_CI_DISABLED_GATES")
    workspace = _tree_holding_the_change(tmp_path, PASSING_TEST)
    assert _verify(workspace).exit_code == 0

    pack = json.loads(_evidence_pack(workspace).read_text(encoding="utf-8"))
    for where, value in _strings_in(pack):
        assert not value.startswith("/"), f"{where} is an absolute path: {value}"
        assert str(workspace) not in value, f"{where} names the host: {value}"

    reports = {
        gate["gate"]: gate["verifier_report"]
        for gate in pack["gate_reports"]
        if gate["verifier_report"]
    }
    assert set(reports) == {"review", "regression"}
    # What the workspace path was carrying survives as one of two words.
    assert [report["workspace_kind"] for report in reports.values()] == [
        "deployment-root",
        "deployment-root",
    ]
    # And the rules that judged the change are still named, repo-relative.
    assert reports["review"]["rule_source"] == "honest-ci-rules.toml"
    assert reports["regression"]["tree_check"]["status"] == "present"
    # The suite still says what it ran, interpreter version and all -- only the
    # directory that interpreter happened to live in is gone.
    assert reports["regression"]["test_command"].endswith("-m coverage run -m pytest -q")
    assert reports["regression"]["coverage"]["command"].startswith("python")


def _evidence_pack(workspace: Path) -> Path:
    manifest = runner.invoke(app, ["inspect", _run_id(workspace), "--workspace", str(workspace)])
    record = json.loads(manifest.stdout)["run_record"]
    return Path(record["evidence_path"])


def _strings_in(node: object, field: str = "") -> list[tuple[str, str]]:
    """Every string in a nested report, paired with where it sits."""

    if isinstance(node, dict):
        return [item for k, v in node.items() for item in _strings_in(v, f"{field}.{k}")]
    if isinstance(node, list):
        return [item for i, v in enumerate(node) for item in _strings_in(v, f"{field}[{i}]")]
    return [(field, node)] if isinstance(node, str) else []


def test_verify_blocks_on_a_failing_test_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HONEST_CI_DISABLED_GATES")
    monkeypatch.setenv("HONEST_CI_COVERAGE_THRESHOLD", "0")
    workspace = _tree_holding_the_change(tmp_path, FAILING_TEST)

    result = _verify(workspace)

    assert result.exit_code == 1, result.stdout
    assert gate_row(result.stdout, "regression")[0] == "blocked"
    # The test that failed, by name, and the file it is in.
    assert "`test_split`" in result.stdout
    assert "tests/test_refund.py" in result.stdout


def test_verify_hands_over_when_the_tree_does_not_hold_the_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HONEST_CI_DISABLED_GATES")
    # The stock workspace: the patch is beside the tree, not applied to it.
    workspace = _workspace(tmp_path, patch=CLEAN_PATCH, required_checks='["review", "regression"]')

    result = _verify(workspace)

    # Testing a tree without the change would say nothing about the change.
    # The gate runs nothing and says why, and the run waits for a person.
    assert result.exit_code == 3, result.stdout
    verdict, summary = gate_row(result.stdout, "regression")
    assert verdict == "needs_human"
    assert "src/refund.py" in summary  # the file the tree does not hold


def test_verify_stops_on_a_rule_library_it_cannot_parse(tmp_path: Path) -> None:
    result = _verify(_workspace(tmp_path, rules='[[rule]]\nid = "R-1"\nkind = "regex"\n'))

    assert result.exit_code == 2
    # The file and the rule to fix, and no verdict reached without them.
    assert "honest-ci-rules.toml" in result.stdout and "R-1" in result.stdout
    assert "Outcome" not in result.stdout


def test_approve_records_a_human_decision_against_the_fingerprint(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, rules=None)
    _verify(workspace)
    run_id = _run_id(workspace)
    waiting = SQLiteLedger(workspace / Settings().ledger_path).load_state(run_id)
    assert waiting is not None and waiting.approval_ticket is not None

    result = runner.invoke(
        app,
        [
            "approve",
            run_id,
            "--workspace",
            str(workspace),
            "--reason",
            "Read the diff myself",
            "--reviewer",
            "chris",
            "--minutes",
            "6",
        ],
    )

    assert result.exit_code == 0
    # What is being signed for, shown before the signature is taken.
    assert waiting.approval_ticket.fingerprint in result.stdout
    # Mergeable because a person signed for it, and the line has to say so: the
    # review gate's own conclusion did not change.
    outcome = field(result.stdout, "Outcome")
    assert outcome.startswith("mergeable") and "chris" in outcome
    assert STATUS_HINT["mergeable"] not in outcome
    assert gate_row(result.stdout, "review")[0] == "needs_human"


def test_approve_can_record_a_rejection(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, rules=None)
    _verify(workspace)

    result = runner.invoke(
        app,
        [
            "approve", _run_id(workspace), "--workspace", str(workspace),
            "--reason", "Add the tests first", "--reject",
        ],
    )

    assert result.exit_code == 1
    # Blocked by a person's decision, and said as one rather than as a gate's.
    outcome = field(result.stdout, "Outcome")
    assert outcome.startswith("blocked")
    assert STATUS_HINT["blocked"] not in outcome


def test_approve_refuses_a_run_that_is_not_waiting_for_a_human(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _verify(workspace)
    run_id = _run_id(workspace)

    result = runner.invoke(
        app,
        ["approve", run_id, "--workspace", str(workspace), "--reason", "Let it through"],
    )

    # A blocked run is not approvable: the fix is a new change, not a signature.
    assert result.exit_code == 2
    assert run_id in result.stdout
    assert "Outcome" not in result.stdout


def test_approve_reports_an_unknown_run(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["approve", "nope", "--workspace", str(tmp_path), "--reason", "x"]
    )

    assert result.exit_code == 2
    assert "nope" in result.stdout


def test_rerun_reaches_a_verdict_once_the_rule_library_is_there(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, rules=None)
    assert _verify(workspace).exit_code == 3
    run_id = _run_id(workspace)
    (workspace / "honest-ci-rules.toml").write_text(RULES, encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "rerun", run_id, "review", "--workspace", str(workspace),
            "--reason", "The rule library is in place now",
        ],
    )

    assert result.exit_code == 1
    assert "[R-01]" in result.stdout


def test_rerun_rejects_a_gate_that_does_not_exist(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _verify(workspace)

    result = runner.invoke(
        app, ["rerun", _run_id(workspace), "nosuchgate", "--workspace", str(workspace)]
    )

    assert result.exit_code == 2
    assert "nosuchgate" in result.stdout
    assert "review" in result.stdout  # the gates that do exist
    assert "Outcome" not in result.stdout


# ── run-pr: the platform entry point ─────────────────────────────────────────


def _run_pr(tmp_path: Path, *extra: str) -> Result:
    return runner.invoke(
        app, ["run-pr", "octo-org/orders", "7", "--workspace", str(tmp_path), *extra]
    )


def _forge(monkeypatch: pytest.MonkeyPatch, stub: StubForge) -> StubForge:
    monkeypatch.setattr(cli, "build_forge_adapter", lambda settings: stub)
    return stub


def _mr_change() -> ChangeRequest:
    return make_change(diff=BLOCKING_PATCH, files_changed=[], number=7)


def test_run_pr_runs_the_gates_and_writes_the_verdict_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _forge(monkeypatch, StubForge(_mr_change()))

    result = _run_pr(workspace)

    assert result.exit_code == 1  # blocked
    assert stub.fetched == [("octo-org/orders", 7)]
    assert "[R-01]" in result.stdout
    # What was written, with the verdict it carried.
    assert "blocked" in field(result.stdout, "Written")
    # The exact verdict the gates reached, not a softened version of it.
    assert [verdict for verdict, _ in stub.statuses] == [Verdict.BLOCKED]
    assert len(stub.reports) == 1 and stub.reports[0][1] == "blocked"
    assert stub.events == ["progress", "report", "status", "inline"]
    assert stub.progress[0].startswith("Verifying · run ")
    assert stub.inline
    assert field(result.stdout, "Inline comments").startswith(f"{len(stub.inline)} ")


@pytest.mark.parametrize("failure", ["publish_progress", "publish_inline"])
def test_optional_forge_failures_do_not_prevent_the_verdict_being_delivered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    workspace = _workspace(tmp_path)
    stub = _forge(monkeypatch, StubForge(_mr_change(), fails_on=failure))
    result = _run_pr(workspace)
    assert result.exit_code == 1 and stub.reports and stub.statuses
    if failure == "publish_inline":
        assert "publish_inline" in field(result.stdout, "Inline comments")


def test_progress_is_sent_before_the_gate_runner_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    stub = _forge(monkeypatch, StubForge(_mr_change()))
    original = cli.GateRunner.run

    def run(self: cli.GateRunner, state: RunState) -> RunState:
        assert stub.progress == [f"Verifying · run {state.run_id}"]
        return original(self, state)

    monkeypatch.setattr(cli.GateRunner, "run", run)
    assert _run_pr(workspace).exit_code == 1


OPEN_THREAD = PlatformReview(
    author="wangwei",
    body="Should this split use Decimal?",
    path="src/refund.py",
    line=2,
    resolved=False,
)


def test_an_unanswered_review_thread_keeps_a_clean_change_with_a_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, CLEAN_PATCH, required_checks='["review"]')
    change = make_change(diff=CLEAN_PATCH, files_changed=[], number=7)
    stub = _forge(monkeypatch, StubForge(change, reviews=[OPEN_THREAD]))

    result = _run_pr(workspace)

    # Nothing in the code is wrong. Somebody is still waiting for an answer.
    assert result.exit_code == 3  # awaiting_human
    assert stub.reviews_fetched == [change.model_copy(update={"repo": "octo-org/orders"}).ref]
    assert [verdict for verdict, _ in stub.statuses] == [Verdict.NEEDS_HUMAN]
    assert "1 unresolved, which the policy sends to a person" in result.stdout

    evidence = runner.invoke(
        app, ["evidence", _run_id(workspace), "--workspace", str(workspace)]
    )
    reviews = section(evidence.stdout, "Review comments already on the change")
    assert reviews.startswith("This change already has 1 review comment, 1 of them unresolved.")
    assert table_row(reviews, "wangwei")[1] == "**unresolved**"


def test_a_platform_that_will_not_share_its_discussion_still_gets_a_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, required_checks='["review"]')
    stub = _forge(monkeypatch, StubForge(_mr_change(), fails_on="fetch_reviews"))

    result = _run_pr(workspace)

    # Fail-soft on the read, and the pack says which part went unread.
    assert result.exit_code == 1 and stub.reports and stub.statuses
    evidence = runner.invoke(
        app, ["evidence", _run_id(workspace), "--workspace", str(workspace)]
    )
    assert any(
        item.startswith("The review comments already on the change were not read")
        for item in unverified(evidence.stdout)
    )


def test_a_dry_run_reads_the_discussion_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, required_checks='["review"]')
    stub = _forge(monkeypatch, StubForge(_mr_change(), reviews=[OPEN_THREAD]))

    assert _run_pr(workspace, "--dry-run").exit_code == 1
    assert stub.reviews_fetched and not stub.events


def test_a_local_change_has_no_platform_reviews_and_says_nothing_about_them(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    assert _verify(workspace).exit_code == 1

    evidence = runner.invoke(
        app, ["evidence", _run_id(workspace), "--workspace", str(workspace)]
    )
    # A local diff has no platform. The section must not appear at all, rather
    # than appear saying "0 review comments" about something that could never
    # have existed.
    assert "Review comments already on the change" not in headings(evidence.stdout)


def test_doctor_reports_java_image_and_pinned_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_JAVA_DOCKER_IMAGE", "maven:private")
    # Process mode resolves from this machine's repository; one that holds
    # the whole pinned toolchain is ready.
    fill_repository(tmp_path / "host-m2", offline_coordinates())
    (tmp_path / "pom.xml").write_text("<project/>")
    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    assert "java (detected from pom.xml)" in result.stdout
    assert "maven:private" in result.stdout and "checkstyle 3.6.0" in result.stdout


def test_doctor_lists_what_a_process_mode_java_run_cannot_resolve_all_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What a developer's own builds leave: the project's plugins, not the
    # versions this toolchain pins. A run would find those missing one per run.
    clean = MAVEN_PLUGINS["clean"]
    fill_repository(tmp_path / "host-m2", [clean])
    (tmp_path / "pom.xml").write_text("<project/>")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 1
    pinned = offline_coordinates()
    missing = [item for item in pinned if item != clean]
    assert f"lacks {len(missing)} of the {len(pinned)}" in result.stdout
    assert all(item in result.stdout for item in missing)
    # The line that fetches every one of them into the repository a run reads.
    assert f"-Dmaven.repo.local={tmp_path / 'host-m2'}" in result.stdout
    assert f"{MAVEN_PLUGINS['dependency']}:get" in result.stdout


def test_doctor_says_when_homebrews_maven_would_run_on_a_jdk_the_pinned_tools_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: Path
) -> None:
    """Asked in the environment a run gets, which is where JAVA_HOME decides.

    Homebrew's mvn runs on Homebrew's newest openjdk unless JAVA_HOME says
    otherwise, and the newest one is past what SpotBugs and PMD read: a run
    there cannot check anything, so the machine is not ready.
    """

    jdk(machine, "openjdk", "27")
    fill_repository(tmp_path / "host-m2", offline_coordinates())
    (tmp_path / "pom.xml").write_text("<project/>")

    newest = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert newest.exit_code == 1
    [failure] = not_ready(newest.stdout)
    assert "Java 27" in failure and str(machine / "openjdk") in failure
    assert f"JDK {MAVEN_JDK_CEILING} or earlier" in failure and "JAVA_HOME" in failure
    # Every pinned tool's own bound, so the operator sees which one stops it.
    for name, ceiling in MAVEN_CLASS_FILE_CEILINGS.items():
        assert f"{name} {MAVEN_PLUGINS[name].rsplit(':', 1)[-1]} up to Java {ceiling}" in failure

    monkeypatch.setenv("JAVA_HOME", str(jdk(machine, "openjdk@21", "21.0.13")))
    chosen = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert chosen.exit_code == 0, chosen.stdout
    assert any(
        "Java 21.0.13" in text and str(machine / "openjdk@21") in text
        for text in ready(chosen.stdout)
    )


@pytest.mark.parametrize(
    "broken,said",
    [
        # Left over from a JDK since uninstalled. Runs never saw JAVA_HOME
        # before, so this used to be harmless; now every run's mvn stops on it.
        ("java_home", "not defined correctly"),
        ("no_mvn", "No mvn"),
    ],
)
def test_doctor_fails_when_maven_cannot_say_which_jdk_a_run_gets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: Path, broken: str, said: str
) -> None:
    if broken == "java_home":
        monkeypatch.setenv("JAVA_HOME", str(machine / "uninstalled-jdk"))
    else:
        empty = machine / "empty"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
    fill_repository(tmp_path / "host-m2", offline_coordinates())
    (tmp_path / "pom.xml").write_text("<project/>")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 1
    [failure] = not_ready(result.stdout)
    assert said in failure


@pytest.mark.parametrize("where", ["python", "docker"])
def test_doctor_asks_about_the_jdk_only_where_a_run_uses_this_machines_maven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: Path, where: str
) -> None:
    # Docker mode runs the image's JDK, and a Python workspace runs no Maven:
    # this machine's JDK, however new, is nothing either run would use.
    jdk(machine, "openjdk", "27")
    if where == "docker":
        (tmp_path / "pom.xml").write_text("<project/>")
        monkeypatch.setenv("HONEST_CI_SANDBOX_MODE", "docker")
        monkeypatch.setattr(cli, "docker_available", lambda: True)
        monkeypatch.setattr(cli, "docker_image_present", lambda image: True)
        stamp = tmp_path / MAVEN_STAMP
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text("warm", encoding="utf-8")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert not (machine / "mvn-calls").exists()


def test_doctor_counts_what_the_workspaces_own_repository_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Copied over whole from a machine with a network, and read first.
    fill_repository(tmp_path / ".honest-ci" / "m2", offline_coordinates())
    (tmp_path / "pom.xml").write_text("<project/>")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "Every plugin and artifact the toolchain pins is in" in result.stdout


def test_run_pr_scans_the_checkout_rather_than_the_directory_it_was_run_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    _forge(monkeypatch, StubForge(_mr_change()))

    result = _run_pr(workspace)
    inspected = runner.invoke(app, ["inspect", _run_id(workspace), "--workspace", str(workspace)])
    state = json.loads(inspected.stdout)["state"]

    assert field(result.stdout, "Checkout") == state["workspace_path"]
    assert state["workspace_path"].startswith(str(workspace / ".honest-ci" / "workspaces"))


def test_run_pr_dry_run_shows_the_note_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _forge(monkeypatch, StubForge(_mr_change()))

    result = _run_pr(workspace, "--dry-run")

    assert result.exit_code == 1
    # What a dry run prints is the note itself, not a description of it.
    note = "\n".join(
        line.removeprefix("  | ") for line in result.stdout.splitlines() if line.startswith("  | ")
    )
    assert headings(note)[0] == "honest-ci · blocked"
    assert "What this run could not verify" in headings(note)
    assert "honest-ci evidence" in result.stdout
    assert stub.reports == [] and stub.statuses == []
    assert stub.progress == [] and stub.inline == [] and stub.events == []


def test_run_pr_records_a_failed_write_as_a_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    _forge(monkeypatch, StubForge(_mr_change(), fails_on="publish_report"))

    result = _run_pr(workspace)

    # The gates concluded "blocked", but nobody on the platform can see it.
    # Reporting exit 1 would claim a write that never happened.
    assert result.exit_code == 4
    # Said, with what the platform answered, and not only in the exit code.
    assert any(
        line.startswith("⚠") and "publish_report" in line for line in result.stdout.splitlines()
    )
    # The verdict itself is still there to read.
    assert Path(field(result.stdout, "Evidence")).is_file()


def test_run_pr_reports_a_platform_it_could_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    _forge(monkeypatch, StubForge(_mr_change(), fails_on="fetch_change"))

    result = _run_pr(workspace)

    assert result.exit_code == 4
    [message] = result.stdout.strip().splitlines()
    assert "octo-org/orders" in message and "fetch_change" in message


def test_run_pr_says_the_local_forge_has_no_pull_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_FORGE", "local")

    result = _run_pr(tmp_path)

    assert result.exit_code == 2
    assert "honest-ci verify" in result.stdout


def test_run_pr_says_when_a_forge_has_no_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_FORGE", "gitlab")

    result = _run_pr(tmp_path)

    assert result.exit_code == 2
    # The platform asked for, and the one there is.
    assert "no adapter for gitlab" in result.stdout
    assert "Implemented: github." in result.stdout


# ── run-pr inside a GitHub Actions workflow ──────────────────────────────────

BASE_REPOSITORY = {"id": 1001, "full_name": "octo-org/orders"}
FORK_REPOSITORY = {"id": 2002, "full_name": "octo-contributor/orders"}


def _in_a_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    head_repository: dict[str, object] | None,
) -> Path:
    """Become a step of a pull_request workflow run; returns the job summary file."""

    runner_files = tmp_path / "runner"
    runner_files.mkdir()
    event = runner_files / "event.json"
    event.write_text(
        json.dumps(
            {
                "pull_request": {
                    "number": 7,
                    "head": {"repo": head_repository},
                    "base": {"repo": BASE_REPOSITORY},
                }
            }
        ),
        encoding="utf-8",
    )
    summary = runner_files / "step_summary.md"
    for name, value in {
        "HONEST_CI_FORGE": "github",
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": "octo-org/orders",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "4242",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_STEP_SUMMARY": str(summary),
    }.items():
        monkeypatch.setenv(name, value)
    return summary


def test_a_pull_request_from_a_fork_is_judged_but_nothing_is_written_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _forge(monkeypatch, StubForge(_mr_change()))
    summary = _in_a_workflow(monkeypatch, tmp_path, FORK_REPOSITORY)

    result = _run_pr(workspace)

    # The token GitHub gives a fork's run is read-only: every write would fail
    # on a 403 and take the verdict down with it.
    assert stub.events == []
    assert stub.reviews_fetched  # reading is still allowed, and still done
    # The exit code is still the verdict: a fork's change is not waved
    # through because its token cannot write.
    assert result.exit_code == 1
    assert "read-only token" in result.stdout
    text = summary.read_text(encoding="utf-8")
    assert "[R-01]" in text
    assert "octo-contributor/orders" in text


def test_a_dry_run_inside_a_workflow_still_reports_to_the_job_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _forge(monkeypatch, StubForge(_mr_change()))
    summary = _in_a_workflow(monkeypatch, tmp_path, BASE_REPOSITORY)

    result = _run_pr(workspace, "--dry-run")

    assert stub.events == []
    assert result.exit_code == 1
    # The job summary is the workflow run's page, not the pull request.
    assert "[R-01]" in summary.read_text(encoding="utf-8")


def test_a_job_summary_that_cannot_be_written_does_not_change_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _forge(monkeypatch, StubForge(_mr_change()))
    _in_a_workflow(monkeypatch, tmp_path, BASE_REPOSITORY)
    # A directory where the file should be.
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "runner"))

    result = _run_pr(workspace)

    assert result.exit_code == 1
    assert "Could not write the job summary" in result.stdout
    assert "status" in stub.events


def test_the_verifying_status_links_to_the_workflow_run_as_the_verdict_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _forge(monkeypatch, StubForge(_mr_change()))
    _in_a_workflow(monkeypatch, tmp_path, BASE_REPOSITORY)

    assert _run_pr(workspace).exit_code == 1

    # While the gates run, the "verifying" status is the only way from the pull
    # request to the run, and the run's log is what someone waiting wants.
    run = "https://github.com/octo-org/orders/actions/runs/4242"
    assert stub.links == [("progress", run), ("status", run)]


def test_the_command_that_signs_for_a_run_is_named_only_where_its_ledger_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No rule library, so both runs wait on a person.
    local = tmp_path / "local"
    local.mkdir()
    verified = _verify(_workspace(local, rules=None))

    assert verified.exit_code == 3, verified.stdout
    assert f"honest-ci approve {_run_id(local)} --reason" in verified.stdout

    workflow = tmp_path / "workflow"
    workflow.mkdir()
    _forge(monkeypatch, StubForge(_mr_change()))
    summary = _in_a_workflow(monkeypatch, workflow, BASE_REPOSITORY)
    result = _run_pr(_workspace(workflow, rules=None))

    # The runner, and the ledger that command would read, go away with the job.
    assert result.exit_code == 3, result.stdout
    assert "honest-ci approve" not in result.stdout
    assert "honest-ci approve" not in summary.read_text(encoding="utf-8")


# ── approvals given on the pull request ──────────────────────────────────────


def _approval(**overrides: object) -> PlatformApproval:
    payload: dict[str, object] = {
        "reviewer": "octo-lead",
        "state": "approved",
        "commit_sha": _mr_change().head_sha,
        "reason": "Checked the refund split with finance.",
        "can_push": True,
        "submitted_at": datetime(2026, 9, 1, 10, 5, tzinfo=UTC),
        "url": "https://github.com/octo-org/orders/pull/7#pullrequestreview-1",
    }
    payload.update(overrides)
    return PlatformApproval.model_validate(payload)


def _waiting_on_a_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *reviews: PlatformApproval
) -> tuple[Path, StubForge]:
    """A pull request the gates hand to a person, then ``reviews`` given on it.

    Given after its first conclusion was published: in the window that
    conclusion recorded, as a person reading it would give them.
    """

    workspace = _workspace(tmp_path, rules=None)
    stub = _forge(monkeypatch, StubForge(_mr_change(), approvals=PlatformApprovals()))
    first = _run_pr(workspace)
    assert first.exit_code == 3, first.stdout
    assert "nobody has approved this commit on the pull request yet" in first.stdout
    [reading] = stub.readings
    assert reading is not None and reading.window is not None
    stub.approvals = PlatformApprovals(reviews=list(reviews), window=reading.window)
    return workspace, stub


def test_an_approve_review_with_a_reason_lets_a_waiting_pull_request_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, stub = _waiting_on_a_person(tmp_path, monkeypatch, _approval())

    result = _run_pr(workspace)

    assert result.exit_code == 0, result.stdout
    assert stub.reports[-1][1] == "mergeable"
    verdict, summary = stub.statuses[-1]
    assert verdict is Verdict.PASS
    # Green, but never "mergeable" on its own: that reads as every gate passing.
    assert summary.startswith("approved · ")
    reading = stub.readings[-1]
    assert reading is not None and reading.approval == _approval()
    assert "Approval  : approved by octo-lead on the pull request" in result.stdout
    assert "Reason  : Checked the refund split with finance." in result.stdout


@pytest.mark.parametrize(
    ("review", "why"),
    [
        (_approval(reason=""), "octo-lead: it gives no reason"),
        (_approval(can_push=False), "octo-lead: the reviewer cannot push to this repository"),
        (_approval(is_bot=True), "octo-lead: a bot gave it"),
        (
            _approval(submitted_at=datetime(2026, 9, 1, 9, 0, tzinfo=UTC)),
            "octo-lead: it was given before this conclusion was published",
        ),
    ],
    ids=["no-reason", "cannot-push", "bot", "before-the-conclusion"],
)
def test_an_approve_review_that_misses_a_condition_leaves_the_pull_request_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, review: PlatformApproval, why: str
) -> None:
    workspace, stub = _waiting_on_a_person(tmp_path, monkeypatch)
    assert stub.approvals is not None and stub.approvals.window is not None
    # As if a review had already been there when the first conclusion went up.
    published = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    window = stub.approvals.window.model_copy(update={"after": published})
    stub.approvals = PlatformApprovals(reviews=[review], window=window)

    result = _run_pr(workspace)

    assert result.exit_code == 3, result.stdout
    assert stub.statuses[-1][0] is Verdict.NEEDS_HUMAN
    assert "Approval  : none counts, so the change keeps waiting." in result.stdout
    assert why in result.stdout


def test_a_standing_request_for_changes_holds_an_approval_and_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objection = _approval(
        reviewer="octo-qa",
        state="changes_requested",
        reason="Add a rounding test first.",
        submitted_at=datetime(2026, 9, 1, 10, 2, tzinfo=UTC),
    )
    workspace, stub = _waiting_on_a_person(tmp_path, monkeypatch, objection, _approval())

    result = _run_pr(workspace)

    assert result.exit_code == 3, result.stdout
    assert "octo-lead: a request for changes stands" in result.stdout
    assert "octo-qa requested changes, and that stands." in result.stdout


def test_an_approval_stops_counting_once_a_new_commit_is_pushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, stub = _waiting_on_a_person(tmp_path, monkeypatch, _approval())
    assert _run_pr(workspace).exit_code == 0

    stub.change = _mr_change().model_copy(update={"head_sha": "c" * 40})
    result = _run_pr(workspace)

    assert result.exit_code == 3, result.stdout
    assert stub.statuses[-1][0] is Verdict.NEEDS_HUMAN
    assert "octo-lead: it was given on another commit (aaaaaaaaaaaa)" in result.stdout


def test_a_blocked_pull_request_is_never_asked_about_approvals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _forge(
        monkeypatch, StubForge(_mr_change(), approvals=PlatformApprovals(reviews=[_approval()]))
    )

    result = _run_pr(workspace)

    # The fix for a blocking finding is a new commit, not a signature.
    assert result.exit_code == 1
    assert stub.approvals_fetched == []
    assert stub.readings == [None]
    assert "Approval  :" not in result.stdout


def test_approvals_that_cannot_be_read_leave_the_pull_request_waiting_and_say_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, rules=None)
    stub = _forge(monkeypatch, StubForge(_mr_change(), fails_on="fetch_approvals"))

    result = _run_pr(workspace)

    assert result.exit_code == 3, result.stdout
    [reading] = stub.readings
    assert reading is not None and reading.error is not None
    assert "fetch_approvals" in reading.error
    # Nothing to read the next run's approvals against: the strict reading.
    assert reading.window is None
    assert "Approval  : none applied, the change keeps waiting" in result.stdout


def test_a_pull_request_from_a_fork_never_reads_its_approvals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, rules=None)
    stub = _forge(
        monkeypatch, StubForge(_mr_change(), approvals=PlatformApprovals(reviews=[_approval()]))
    )
    summary = _in_a_workflow(monkeypatch, tmp_path, FORK_REPOSITORY)

    result = _run_pr(workspace)

    # Its run cannot record a window, so no approval could ever be read in one.
    assert result.exit_code == 3, result.stdout
    assert stub.approvals_fetched == []
    assert "an Approve review on the pull request cannot take effect" in summary.read_text(
        encoding="utf-8"
    )


def test_a_dry_run_counts_an_approval_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, stub = _waiting_on_a_person(tmp_path, monkeypatch, _approval())
    written = list(stub.events)

    result = _run_pr(workspace, "--dry-run")

    assert result.exit_code == 0, result.stdout
    assert stub.events == written
    assert "## honest-ci · approved" in result.stdout


def test_where_the_platform_hands_over_no_approvals_nothing_is_said_about_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, rules=None)
    stub = _forge(monkeypatch, StubForge(_mr_change()))

    result = _run_pr(workspace)

    assert result.exit_code == 3
    assert stub.approvals_fetched == [
        _mr_change().model_copy(update={"repo": "octo-org/orders"}).ref
    ]
    assert stub.readings == [None]
    assert "Approval  :" not in result.stdout


def test_run_pr_says_which_token_is_missing_and_what_the_job_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_FORGE", "github")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result = _run_pr(tmp_path)

    assert result.exit_code == 2
    assert "GITHUB_TOKEN is not set" in result.stdout
    assert "pull-requests: write" in result.stdout and "statuses: write" in result.stdout


def test_doctor_says_whether_the_platform_entry_point_would_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_FORGE", "github")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_abcdefghijklmnopqrstuvwxyz012345")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    platform = field(result.stdout, "Platform")
    assert platform.startswith("github (")
    assert "GITHUB_TOKEN" not in platform  # a token is configured, so none is asked for
    assert "honest-ci/verify" in platform  # the status name a verdict would be written under
    assert "ghp_" not in result.stdout


# ── honest-ci resume ─────────────────────────────────────────────────────────


def _interrupted(workspace: Path, **overrides: object) -> str:
    """Write the checkpoint a process killed mid-gate would have left behind."""

    fields: dict[str, object] = {
        "run_id": "dead0beef001",
        "change": _mr_change(),
        "workspace_path": str(workspace),
        "current_gate": "review",
        "status": "running",
    }
    fields.update(overrides)
    state = RunState(**fields)
    SQLiteLedger(workspace / Settings().ledger_path).save_state(state)
    return state.run_id


def test_resume_finishes_a_run_that_was_interrupted_and_writes_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, required_checks='["review"]')
    stub = _forge(monkeypatch, StubForge(_mr_change()))
    run_id = _interrupted(workspace)

    result = runner.invoke(app, ["resume", run_id, "--workspace", str(workspace)])

    assert result.exit_code == 1  # blocked, the verdict the gates reached
    assert "blocked" in field(result.stdout, "Written")
    assert [verdict for verdict, _ in stub.statuses] == [Verdict.BLOCKED]


def test_resume_will_not_drive_a_run_that_is_waiting_for_a_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming one of these would finish a change nobody approved."""

    workspace = _workspace(tmp_path)
    stub = _forge(monkeypatch, StubForge(_mr_change()))
    run_id = _interrupted(workspace, status="awaiting_human")

    result = runner.invoke(app, ["resume", run_id, "--workspace", str(workspace)])

    assert result.exit_code == 2
    assert "honest-ci approve" in result.stdout  # the command an operator actually wants
    assert stub.statuses == []


def test_resume_reports_a_run_it_has_never_heard_of(tmp_path: Path) -> None:
    result = runner.invoke(app, ["resume", "nosuchrun", "--workspace", str(tmp_path)])

    assert result.exit_code == 2
    assert "nosuchrun" in result.stdout


def test_resume_of_a_local_run_has_no_platform_to_tell(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, required_checks='["review"]')
    run_id = _interrupted(workspace, change=make_change(forge="local", number=None))

    result = runner.invoke(app, ["resume", run_id, "--workspace", str(workspace)])

    assert result.exit_code == 1  # the terminal state, not a write-back failure
    assert "Written" not in result.stdout and "Write-back" not in result.stdout


# ── the license check: a key due, and none that works ────────────────────────


def _in_ci(monkeypatch: pytest.MonkeyPatch, key: str | None = None) -> None:
    """A job in a CI system other than GitHub Actions, given ``key`` as its license key."""

    monkeypatch.setenv("CI", "true")
    if key is not None:
        monkeypatch.setenv(KEY_VARIABLE, key)


def _in_actions(monkeypatch: pytest.MonkeyPatch, job: Path, key: str | None = None) -> Path:
    """A workflow step on a pull request onto octo-org's private repository; returns its summary.

    ``job`` is where the runner keeps the event and the job summary.
    """

    job.mkdir()
    event = job / "event.json"
    event.write_text(
        json.dumps(pull_request_event(repository(ORG), SOMEONE_ELSE)), encoding="utf-8"
    )
    summary = job / "step-summary.md"
    for name, value in {
        "GITHUB_ACTIONS": "true",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_STEP_SUMMARY": str(summary),
    }.items():
        monkeypatch.setenv(name, value)
    _in_ci(monkeypatch, key)
    return summary


def _org_forge(monkeypatch: pytest.MonkeyPatch, fails_on: str | None = None) -> StubForge:
    """A platform whose pull requests are onto an organization's private repository."""

    return _forge(
        monkeypatch,
        StubForge(_mr_change(), ownership=AN_ORGANIZATIONS_PRIVATE_REPOSITORY, fails_on=fails_on),
    )


def _signed(monkeypatch: pytest.MonkeyPatch, *, expires: date) -> str:
    """A key for octo-org that expires on ``expires``, from a key pair the check is made to trust."""

    monkeypatch.setattr(license_check, "PUBLIC_KEY", TEST)
    return issue(org_id=9001, expires=expires.isoformat())


def _today() -> date:
    return datetime.now(UTC).date()


def _no_runs(workspace: Path) -> bool:
    listing = runner.invoke(app, ["runs", "--workspace", str(workspace)])
    return listing.stdout.strip() == "(no runs yet)"


NO_KEY = "No gate ran. No license key, and this repository needs one in CI"
NOT_A_KEY = (
    "No gate ran. License key refused: it isn't an honest-ci license key, which is "
    "hci1.<payload>.<signature> on one line"
)


def test_run_pr_in_ci_with_no_key_where_one_is_due_runs_no_gate_and_says_why_on_the_pull_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _org_forge(monkeypatch)
    _in_ci(monkeypatch)

    result = _run_pr(workspace)

    assert result.exit_code == 4
    # A comment and an error status, and none of what a run of the gates writes.
    assert stub.events == ["note", "status"]
    assert stub.errors == [NO_KEY]
    [note] = stub.notes
    assert note.startswith(f"{VERDICT_MARKER}\n## honest-ci · not run\n\n**{NO_KEY}.**")
    assert field(result.stdout, "Written") == "comment and commit status error (honest-ci/verify)"
    # The output says what the comment says.
    assert (
        f"⚠ {NO_KEY}. This is not a verdict: honest-ci checked nothing in this change."
        in result.stdout
    )
    assert "  - Get a key at https://honest-ci.dev/pricing." in result.stdout
    # Nothing the gates would have said, and no run to look the change up by.
    assert "[R-01]" not in result.stdout
    assert _no_runs(workspace)


def test_a_dry_run_the_license_check_stops_shows_what_it_would_write_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _org_forge(monkeypatch)
    _in_ci(monkeypatch)

    result = _run_pr(workspace, "--dry-run")

    assert result.exit_code == 4
    assert stub.events == []
    assert (
        "Write-back: --dry-run, nothing below was sent (the commit status would be an error)"
        in result.stdout
    )
    assert field(result.stdout, "Status") == NO_KEY
    assert "  | ## honest-ci · not run" in result.stdout.splitlines()


@pytest.mark.parametrize("failure", ["publish_note", "publish_status"])
def test_a_stop_that_cant_be_written_back_says_nobody_on_the_pull_request_can_see_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    workspace = _workspace(tmp_path)
    _org_forge(monkeypatch, fails_on=failure)
    _in_ci(monkeypatch)

    result = _run_pr(workspace)

    assert result.exit_code == 4
    assert f"⚠ Write-back failed: the platform refused {failure}" in result.stdout
    assert "  Nobody on the pull request can see why no gate ran." in result.stdout.splitlines()


def test_run_pr_in_actions_says_why_no_gate_ran_in_the_job_summary_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _org_forge(monkeypatch)
    summary = _in_actions(monkeypatch, tmp_path / "job", key="not a key")

    result = _run_pr(workspace)

    assert result.exit_code == 4
    assert stub.errors == [NOT_A_KEY]
    [note] = stub.notes
    assert "save the key as an organization secret named `HONEST_CI_LICENSE_KEY`" in note
    assert note.strip() in summary.read_text(encoding="utf-8")


def test_run_pr_with_a_key_that_works_runs_the_gates_as_ever_and_says_nothing_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _org_forge(monkeypatch)
    key = _signed(monkeypatch, expires=_today() + timedelta(days=365))
    _in_ci(monkeypatch, key)

    result = _run_pr(workspace)

    assert result.exit_code == 1  # blocked, the verdict the gates reached
    assert stub.events == ["progress", "report", "status", "inline"]
    assert stub.license_notes == [None]
    assert "License   :" not in result.stdout
    assert key not in result.stdout


def test_run_pr_on_a_key_past_its_expiry_date_says_how_long_it_has_left_here_and_on_the_pull_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    stub = _org_forge(monkeypatch)
    expires = _today() - timedelta(days=10)
    key = _signed(monkeypatch, expires=expires)
    _in_ci(monkeypatch, key)

    result = _run_pr(workspace)

    assert result.exit_code == 1
    [note] = stub.license_notes
    assert note is not None
    assert note.startswith(f"The license key for octo-org expired on {expires.isoformat()} ")
    assert field(result.stdout, "License") == note
    assert key not in result.stdout


def test_verify_in_ci_with_no_key_that_works_where_one_is_due_runs_no_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    _in_actions(monkeypatch, tmp_path / "job", key="not a key")

    result = _verify(workspace)

    assert result.exit_code == 4
    assert f"⚠ {NOT_A_KEY}. This is not a verdict" in result.stdout
    assert "[R-01]" not in result.stdout
    assert _no_runs(workspace)


def test_verify_on_a_key_past_its_expiry_date_runs_the_gates_and_says_how_long_it_has_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    key = _signed(monkeypatch, expires=_today() - timedelta(days=10))
    _in_actions(monkeypatch, tmp_path / "job", key=key)

    result = _verify(workspace)

    assert result.exit_code == 1
    assert field(result.stdout, "License").startswith("The license key for octo-org expired on ")


def test_resume_in_ci_with_no_key_that_works_leaves_the_run_for_one_with_a_key_to_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, required_checks='["review"]')
    stub = _forge(monkeypatch, StubForge(_mr_change()))
    run_id = _interrupted(workspace)
    _in_actions(monkeypatch, tmp_path / "job", key="not a key")

    result = runner.invoke(app, ["resume", run_id, "--workspace", str(workspace)])

    assert result.exit_code == 4
    assert f"⚠ {NOT_A_KEY}. This is not a verdict" in result.stdout
    assert stub.events == []
    state = SQLiteLedger(workspace / Settings().ledger_path).load_state(run_id)
    assert state is not None and (state.status, state.current_gate) == ("running", "review")


def test_resume_on_a_key_past_its_expiry_date_ends_the_comment_with_how_long_it_has_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, required_checks='["review"]')
    stub = _forge(monkeypatch, StubForge(_mr_change()))
    run_id = _interrupted(workspace)
    key = _signed(monkeypatch, expires=_today() - timedelta(days=10))
    _in_actions(monkeypatch, tmp_path / "job", key=key)

    result = runner.invoke(app, ["resume", run_id, "--workspace", str(workspace)])

    assert result.exit_code == 1
    [note] = stub.license_notes
    assert note is not None and note.startswith("The license key for octo-org expired on ")
    assert field(result.stdout, "License") == note


# ── annotate: a person's reading of a finding ────────────────────────────────


def _annotate(workspace: Path, run_id: str, *extra: str) -> Result:
    return runner.invoke(app, ["annotate", run_id, "--workspace", str(workspace), *extra])


def test_annotate_marks_a_false_positive_without_changing_the_verdict(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _verify(workspace)
    run_id = _run_id(workspace)

    result = _annotate(
        workspace,
        run_id,
        "--finding", "review.R-01.1",
        "--false-positive",
        "--reason", "This float only reaches a log line",
        "--annotator", "chris",
        "--minutes", "4",
    )

    assert result.exit_code == 0
    annotation = field(result.stdout, "Annotation")
    assert annotation.startswith("review.R-01.1 → ")
    assert "false positive" in annotation and "chris" in annotation
    assert field(result.stdout, "Review").startswith("4 ")
    # The run it belongs to, still at the outcome the gates reached.
    run = field(result.stdout, "Run")
    assert run.startswith(run_id) and re.search(r"\bblocked\b", run)
    evidence = runner.invoke(app, ["evidence", run_id, "--workspace", str(workspace)])
    assert any(
        line.startswith("- Marked a false positive by a person")
        for line in section(evidence.stdout, "Blocking findings").splitlines()
    )
    assert table_row(section(evidence.stdout, "Metrics"), "Human review") == ["4 min"]


def test_annotate_refuses_a_false_positive_without_a_reason(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _verify(workspace)

    result = _annotate(workspace, _run_id(workspace), "--finding", "review.R-01.1", "--false-positive")

    assert result.exit_code == 2
    assert "--reason" in result.stdout  # the option that is missing


def test_annotate_needs_something_to_record(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _verify(workspace)

    assert _annotate(workspace, _run_id(workspace)).exit_code == 2
    assert _annotate(workspace, _run_id(workspace), "--false-positive").exit_code == 2
    both = _annotate(
        workspace, _run_id(workspace), "--finding", "review.R-01.1", "--false-positive", "--confirmed"
    )
    assert both.exit_code == 2
    assert "--false-positive" in both.stdout and "--confirmed" in both.stdout


def test_annotate_names_the_findings_when_the_id_is_wrong(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _verify(workspace)

    result = _annotate(workspace, _run_id(workspace), "--finding", "review.R-99.1", "--confirmed")

    assert result.exit_code == 2
    assert "review.R-99.1" in result.stdout  # the id that was given
    assert "review.R-01.1" in result.stdout  # the ids that do exist


# ── which toolchain ──────────────────────────────────────────────────────────


def test_doctor_says_which_toolchain_this_directory_gets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    line = field(result.stdout, "Toolchain")
    # No marker here: the fallback is named as a default, never as a recognition.
    assert "python (default:" in line
    assert "python (ruff, mypy, semgrep)" in line  # every toolchain this build has

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    recognised = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)]).stdout
    assert "python (detected from pyproject.toml)" in recognised

    monkeypatch.setenv("HONEST_CI_LANGUAGE", "klingon")
    misconfigured = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)]).stdout
    assert "⚠ HONEST_CI_LANGUAGE='klingon'" in misconfigured


def test_verify_refuses_to_start_on_a_toolchain_nobody_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_LANGUAGE", "klingon")

    result = _verify(_workspace(tmp_path))

    assert result.exit_code == 2
    assert "klingon" in result.stdout
    assert "Outcome" not in result.stdout


def test_doctor_says_how_a_custom_test_command_will_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_TEST_COMMAND", "make test")
    only_exit_code = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)]).stdout
    # Read by its exit code only, and the line names the switch that changes that.
    assert "HONEST_CI_TEST_REPORT" in field(only_exit_code, "Regression gate")

    monkeypatch.setenv("HONEST_CI_TEST_REPORT", "reports/junit.xml")
    with_report = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)]).stdout
    regression = field(with_report, "Regression gate")
    assert "JUnit XML reports/junit.xml" in regression
    assert "HONEST_CI_TEST_REPORT" not in regression


def test_doctor_does_not_promise_impacted_tests_a_run_will_not_narrow_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_TEST_SELECTION", "impacted")
    # The default overall bar is on, and a narrowed run cannot measure it.
    under_a_bar = field(
        runner.invoke(app, ["doctor", "--workspace", str(tmp_path)]).stdout, "Regression gate"
    )
    assert "runs the whole suite" in under_a_bar
    assert "HONEST_CI_COVERAGE_THRESHOLD" in under_a_bar

    monkeypatch.setenv("HONEST_CI_COVERAGE_THRESHOLD", "0")
    narrowed = field(
        runner.invoke(app, ["doctor", "--workspace", str(tmp_path)]).stdout, "Regression gate"
    )
    assert "runs the impacted tests" in narrowed
    # With no bar on, no toolchain collects coverage, and the line does not say it does.
    assert "coverage run" not in narrowed
    assert "JaCoCo" not in narrowed

    monkeypatch.setenv("HONEST_CI_TEST_COMMAND", "make test")
    as_written = field(
        runner.invoke(app, ["doctor", "--workspace", str(tmp_path)]).stdout, "Regression gate"
    )
    assert "runs the whole suite" in as_written
    assert "HONEST_CI_TEST_COMMAND" in as_written


def test_a_threshold_outside_the_range_is_a_message_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dogfooding typed 75 for 75% and got a gate that blocked on "coverage 92%
    # is below 7500%". An operator should meet that in doctor, before a run.
    monkeypatch.setenv("HONEST_CI_COVERAGE_THRESHOLD", "75")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 2  # a usage error, not a blocked change
    # The line to fix, and the value that was meant.
    assert "HONEST_CI_COVERAGE_THRESHOLD=75" in result.stdout
    assert "0.75" in result.stdout
    assert "Traceback" not in result.stdout


def test_a_refused_threshold_stops_verify_before_it_judges_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD", "80")

    result = _verify(_workspace(tmp_path))

    assert result.exit_code == 2
    assert "HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD=80" in result.stdout
    assert "Outcome" not in result.stdout


# ── evidence: what a pack says, and what it never carries ────────────────────


def test_the_rendered_pack_says_what_configuration_produced_it(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _verify(workspace)

    result = runner.invoke(app, ["evidence", _run_id(workspace), "--workspace", str(workspace)])

    configuration = section(result.stdout, "Gate configuration")
    assert table_row(configuration, "Rules")[0].startswith("`honest-ci-rules.toml` · ")
    assert "This section is a record and takes no part in the verdict" in configuration


def test_schema_prints_the_contract_a_verifier_checks_against(tmp_path: Path) -> None:
    # The format is public: anyone's own install hands over the contract, so a verifier
    # never has to import this software to know what shape a pack is.
    result = runner.invoke(app, ["schema"])

    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["x-schema-version"] == "v1"
    assert "schema_version" in schema["properties"]


def test_no_credential_and_no_host_path_reaches_the_evidence_on_disk(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, rules=RULES_WITH_TOKEN)
    _verify(workspace)
    run_dir = workspace / ".honest-ci" / "artifacts" / _run_id(workspace)

    blob = "\n".join(
        item.read_text(encoding="utf-8") for item in sorted(run_dir.iterdir()) if item.is_file()
    )

    assert "glpat-" not in blob
    # The rule that mentioned it is still there; only the credential is gone.
    assert "R-02" in blob
    # A configuration file is named the way the team named it, not the way
    # this machine happens to spell its temp directory.
    assert "honest-ci-rules.toml" in blob
    assert f"{tmp_path}/honest-ci-rules.toml" not in blob
    # And nothing else in the pack names this machine either: the tree that
    # was verified, the commands that ran and the patch it came from are all
    # recorded by what they were, not by where they sat.
    assert str(tmp_path) not in blob
    assert "workspace_kind" in blob


def test_a_deletion_still_counts_as_a_changed_file_in_the_scope_line() -> None:
    """The scope line and the blast-radius limits must count the same files.

    A change that only deletes a file printed "0 files, 43 lines" -- a scope
    line denying the change whose missing tests the run then went to a human
    about, while the policy engine and the evidence pack both counted it as 1.
    """

    change = make_change(
        files_changed=[
            ChangedFile(path="src/app/refund.py", change_type="modified", added_lines=3),
            ChangedFile(path="tests/test_refund.py", change_type="deleted", removed_lines=43),
        ]
    )

    assert cli._scope_paths(change) == ["src/app/refund.py", "tests/test_refund.py"]
    # The other reading stays as it was: a deleted file has nothing to scan,
    # and listing it as absent from the tree would be noise, not news.
    assert cli._declared_paths(change) == ["src/app/refund.py"]


# ── what the public build leaves out ─────────────────────────────────────────

#: The team edition's commands (AGENTS.md, The public edition's boundary). Absent
#: rather than hidden: a command left out of the help screen is still one anybody
#: can run.
TEAM_EDITION_COMMANDS = ("report", "replay", "evalset")


def _names(help_text: str, pattern: str) -> set[str]:
    return set(re.findall(pattern, help_text, flags=re.MULTILINE))


def test_the_team_editions_commands_are_not_in_the_public_build() -> None:
    """The cross-run report, the evaluation set and its replay stay out of this build.

    Held by CI rather than by memory: any of them is one careless merge away
    from coming back, and a help screen is where nobody would notice.
    """

    help_text = runner.invoke(app, ["--help"]).stdout
    # One command per row, whether the screen is drawn in boxes or in plain text.
    listed = _names(help_text, r"^(?:│ | {2})([a-z][a-z0-9-]*)(?= {2})")

    # The screen was read at all, so an empty parse cannot pass for "absent".
    assert {"verify", "approve", "annotate", "evidence"} <= listed
    for name in TEAM_EDITION_COMMANDS:
        assert name not in listed
        result = runner.invoke(app, [name, "--help"])
        assert result.exit_code == 2
        assert f"No such command '{name}'" in result.output


def test_the_gitlab_entry_points_are_not_in_the_public_build() -> None:
    """``run-mr`` and the webhook service ``serve`` are left out for now."""

    for name in ("run-mr", "serve"):
        result = runner.invoke(app, [name, "--help"])
        assert result.exit_code == 2
        assert f"No such command '{name}'" in result.output


def test_evidence_renders_a_pack_and_neither_exports_nor_verifies_it() -> None:
    """An export for auditors and its standalone verifier are the team edition's."""

    options = _names(runner.invoke(app, ["evidence", "--help"]).stdout, r"(--[a-z][a-z-]*)")

    assert "--workspace" in options
    for option in ("--export", "--verify"):
        assert option not in options
        result = runner.invoke(app, ["evidence", "r1", option])
        assert result.exit_code == 2
        assert f"No such option: {option}" in result.output


def test_a_rename_counts_under_both_names_in_the_scope_line() -> None:
    change = make_change(
        files_changed=[
            ChangedFile(
                path="docs/deploy.yml",
                change_type="renamed",
                previous_path=".github/workflows/deploy.yml",
            ),
        ]
    )

    # The count the limits read; the old name is not there to scan.
    assert cli._scope_paths(change) == [".github/workflows/deploy.yml", "docs/deploy.yml"]
    assert cli._declared_paths(change) == ["docs/deploy.yml"]


# ── the language it speaks ───────────────────────────────────────────────────


def test_every_help_screen_is_in_english() -> None:
    """The top-level screen and every command's own, option help included."""

    group = typer.main.get_command(app)
    assert isinstance(group, typer.core.TyperGroup)
    names = sorted(group.commands)
    # The commands were found at all, so an empty list cannot pass for "all English".
    assert {"verify", "doctor", "evidence", "run-pr", "approve", "resume"} <= set(names)

    screens = {"honest-ci --help": runner.invoke(app, ["--help"])}
    screens |= {f"honest-ci {name} --help": runner.invoke(app, [name, "--help"]) for name in names}

    for where, result in screens.items():
        assert result.exit_code == 0, (where, result.output)
    shown = {where: cjk_lines(result.output) for where, result in screens.items()}
    assert {where: lines for where, lines in shown.items() if lines} == {}


def test_no_help_text_is_broken_mid_sentence_in_an_80_column_terminal() -> None:
    """Help keeps the line breaks it was written with, except where Typer joins them.

    The command list shows each summary with its line breaks, so a summary
    written over two lines is cut in the middle there. On a command's own
    screen the paragraphs after the first keep theirs too, so a line wider than
    the terminal is broken twice: once where the terminal ends, and again where
    the text did.
    """

    group = typer.main.get_command(app)
    assert isinstance(group, typer.core.TyperGroup)
    texts = {"honest-ci": group.help or ""}
    texts |= {f"honest-ci {name}": command.help or "" for name, command in group.commands.items()}

    cut_summaries: list[str] = []
    too_wide: list[tuple[str, str]] = []
    for where, text in texts.items():
        summary, _, body = inspect.cleandoc(text).partition("\n\n")
        if "\n" in summary:
            cut_summaries.append(where)
        # One column of the 80 is Typer's margin.
        too_wide += [(where, line) for line in body.splitlines() if len(line) > 78]

    assert "honest-ci run-pr" in texts
    assert cut_summaries == []
    assert too_wide == []
