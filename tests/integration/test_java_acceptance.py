"""The Java toolchain against a real Docker daemon and a real Maven repository.

Everything else about the Java path is tested against canned Maven output: the
XML readers, the diagnostics, the gate wiring. Canned output cannot answer the
only question that matters for a team's first afternoon -- does ``mvn -o``
inside the sandbox image actually produce those files -- and it has no opinion
at all about the two failures that only appear on a real machine: a cold
dependency cache that outruns the scanner timeout, and the artifacts Maven
resolves while a goal runs, which ``go-offline`` cannot pre-fetch.

So this file drives the same example a team would copy, through the same
CLI they would type, and asserts the three terminal states the toolchain owes:

* a clean change is ``mergeable``,
* a change that breaks a deterministic rule is ``blocked``,
* a change that deletes the tests is ``needs_human`` -- nothing *failed*, and
  that is exactly why a person has to look.

Then it runs all three again with provisioning switched off, which is what
"offline" means here in a form the kernel enforces: provisioning is the only
step ``ToolRunner`` ever grants a network to, so with it off every container in
the pass runs under ``--network none``. Same verdicts, or Java is not offline.

Marked ``docker`` and skipped without a daemon. ``make acceptance`` pulls the
image first, so a skip there means the machine has no Docker, never that the
suite quietly declined to check.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from honest_ci.tools import docker_available

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "java-junit5"
SCENARIOS = EXAMPLE / "scenarios"

#: Maven picks these while a goal runs, so no amount of ``go-offline`` finds
#: them. Kept beside the example's README, which explains each one; a change
#: here is a change there.
EXTRA_ARTIFACTS = (
    "org.apache.maven.surefire:surefire-junit-platform:3.5.2",
    "org.junit.platform:junit-platform-commons:1.9.3",
)

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not docker_available(), reason="没有 Docker 守护进程，跳过真机验收"),
]


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=acceptance@honest-ci.test", "-c", "user.name=acceptance", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )


def _env(**overrides: str) -> dict[str, str]:
    """A deployment with no model and no inherited credentials.

    The three model switches are off on purpose rather than by accident of an
    unset key: the acceptance asserts terminal states, and a judgment-class
    call or a generated regression case would make two of them depend on what
    a model said this afternoon. ``HOME`` is forwarded because that is where
    the Docker CLI keeps the context that names the daemon socket.
    """

    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": str(REPO / "src"),
        "HONEST_CI_MODEL_EXPLAIN": "0",
        "HONEST_CI_MODEL_JUDGMENT": "0",
        "HONEST_CI_MODEL_TESTGEN": "0",
        "HONEST_CI_MAVEN_EXTRA_ARTIFACTS": ",".join(EXTRA_ARTIFACTS),
    }
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "LANG", "LC_ALL"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    env.update(overrides)
    return env


@pytest.fixture(scope="session")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One checkout of the example, shared by every scenario.

    Shared because ``.honest-ci/m2`` is: provisioning a Maven cache from cold costs
    minutes and a hundred megabytes, and doing it once per scenario would make
    the acceptance too expensive to run, which is the same as not having one.
    Each scenario resets the tracked tree and leaves the ignored one alone.
    """

    root = tmp_path_factory.mktemp("java-acceptance") / "repo"
    shutil.copytree(EXAMPLE, root)
    shutil.rmtree(root / "scenarios")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "baseline")
    yield root


def _reset(workspace: Path) -> None:
    _git(workspace, "checkout", "--", ".")
    # No -x: target/ and .honest-ci/ are ignored, and the Maven cache under the
    # latter is the whole reason these scenarios share one checkout.
    _git(workspace, "clean", "-fdq")


def _verify(
    workspace: Path, scenario: str, env: Mapping[str, str]
) -> tuple[int, str, dict[str, Any]]:
    """Apply one scenario patch and run ``honest-ci verify`` over the result."""

    _reset(workspace)
    _git(workspace, "apply", str(SCENARIOS / f"{scenario}.patch"))
    proc = subprocess.run(
        [sys.executable, "-m", "honest_ci.cli", "verify", "--workspace", str(workspace)],
        capture_output=True,
        text=True,
        env=dict(env),
        # Generous: the first scenario in a session pays for provisioning the
        # whole Maven toolchain, which is the thing being verified.
        timeout=3600,
    )
    return proc.returncode, proc.stdout + proc.stderr, _latest_pack(workspace)


def _latest_pack(workspace: Path) -> dict[str, Any]:
    packs = sorted(
        (workspace / ".honest-ci" / "artifacts").glob("*/evidence.json"),
        key=lambda path: path.stat().st_mtime,
    )
    assert packs, "没有写出证据包"
    payload = json.loads(packs[-1].read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _gate(pack: Mapping[str, Any], name: str) -> dict[str, Any]:
    reports = pack["gate_reports"]
    assert isinstance(reports, list)
    for report in reports:
        if report.get("gate") == name:
            assert isinstance(report, dict)
            return report
    raise AssertionError(f"证据包里没有 {name} 门禁的结论")


def _findings(pack: Mapping[str, Any]) -> list[dict[str, Any]]:
    reports = pack["gate_reports"]
    assert isinstance(reports, list)
    return [finding for report in reports for finding in report.get("findings", [])]


# ── the three terminal states ────────────────────────────────────────────────


def test_a_clean_change_is_mergeable(workspace: Path) -> None:
    code, output, pack = _verify(workspace, "clean", _env())

    assert code == 0, output
    assert pack["verdict"] == "pass"
    regression = _gate(pack, "regression")
    assert regression["verdict"] == "pass"
    # Seven, not "some": the baseline's five plus the two the patch adds. A
    # suite that silently stopped collecting is the failure this catches.
    assert regression["test_results"]["passed"] == 7
    assert regression["test_results"]["runner"] == "junit"
    # JaCoCo produced a number and the gate compared it. "Measured" is the
    # claim: a threshold nobody could check is a block, not a pass.
    metrics = regression["metrics"]
    assert metrics["tests_collected"] == 7
    assert metrics["coverage"] >= metrics["coverage_threshold"] > 0


def test_a_rule_violation_is_blocked(workspace: Path) -> None:
    code, output, pack = _verify(workspace, "violation", _env())

    assert code == 1, output
    assert pack["verdict"] == "blocked"
    blocking = [item for item in _findings(pack) if item.get("rule_id") == "R-01"]
    assert blocking, _findings(pack)
    # The deterministic rule blocked it, on a line the patch added. Both halves
    # matter: a model finding may not block, and a rule may not fire on code
    # the change never touched.
    assert blocking[0]["decided_by"] == "deterministic"
    assert blocking[0]["file"] == "src/main/java/com/example/RefundSplitter.java"
    assert blocking[0]["severity"] == "high"


def test_deleting_the_tests_goes_to_a_human(workspace: Path) -> None:
    # Arm the trap first, through the product's own path: one ordinary run that
    # builds and runs the suite, exactly the state a developer's workspace is
    # always in. `_reset` keeps target/, so its compiled test class is still
    # there when the next run starts -- and Maven will not remove a class whose
    # source has disappeared. Without the per-run clean, Surefire re-runs that
    # class and the suite reports passes for a file no reviewer can open.
    # Armed here rather than inherited from the test above it: which scenario
    # ran last, and whether it got as far as compiling tests, is not this
    # test's business.
    _verify(workspace, "clean", _env())
    assert list(workspace.glob("target/test-classes/**/*.class")), "陷阱没装上，这条测不到东西"

    code, output, pack = _verify(workspace, "drop-test", _env())

    # Nothing failed here. `mvn test` exits 0, the review gate has no added
    # lines to judge, and reading that as "mergeable" takes only not asking
    # whether anything was verified.
    assert code == 3, output
    assert pack["verdict"] == "needs_human"
    regression = _gate(pack, "regression")
    assert regression["verdict"] == "needs_human"
    results = regression["test_results"]
    assert results["available"] is True
    assert results["passed"] + results["failed"] + results["errors"] + results["skipped"] == 0


# ── the same three, offline ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [("clean", 0), ("violation", 1), ("drop-test", 3)],
)
def test_the_same_verdicts_without_a_network(
    workspace: Path, scenario: str, expected: int
) -> None:
    """Java is offline or it is not; the verdicts may not depend on which.

    ``HONEST_CI_SANDBOX_BUILD_DEPS=0`` removes the only step the runner ever grants
    a network to, so every container in this pass is started with
    ``--network none``. It is a stronger statement than unplugging the host:
    the isolation is the kernel's, not the test's.
    """

    code, output, _ = _verify(workspace, scenario, _env(HONEST_CI_SANDBOX_BUILD_DEPS="0"))

    assert code == expected, output
