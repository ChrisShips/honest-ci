"""The shipped Java example, checked without a Docker daemon.

``make acceptance`` proves the example produces the three terminal states on a
real machine. It needs Docker, minutes and a network, so it does not run in
CI -- which leaves the example free to rot between acceptances: a rule file
that no longer parses, a plugin version that drifted away from ``maven.py``, a
scenario patch that stopped applying after somebody reformatted the baseline.

Those are all readable off the files. This is where they are read, in the suite
that runs on every commit, so the acceptance fails over the toolchain rather
than over a stale fixture.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from honest_ci.maven import MAVEN_PLUGINS
from honest_ci.policy import PolicyConfig
from honest_ci.rules import RuleSet

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "java-junit5"
SCENARIOS = ("clean", "violation", "drop-test")


def test_the_example_rule_library_loads() -> None:
    rules = RuleSet.load(EXAMPLE, Path("honest-ci-rules.toml"), "v0")

    assert rules.version == "v1"
    assert {rule.kind for rule in rules.rules} == {"regex", "path", "tool"}
    # R-01 is the one the violation scenario trips, and the acceptance asserts
    # on that id. Blocking severity is what makes it a block rather than a note.
    violated = next(rule for rule in rules.rules if rule.id == "R-01")
    assert violated.blocking
    assert violated.source_samples


def test_the_example_has_no_judgment_rules() -> None:
    """Deliberate, and the reason belongs next to the assertion.

    A judgment-class rule needs a model credential to reach a verdict at all;
    without one it is recorded as 未判定 and the change goes to a human. The
    acceptance would then never see ``mergeable``, for a reason that has
    nothing to do with the Java toolchain it is there to verify.
    """

    rules = RuleSet.load(EXAMPLE, Path("honest-ci-rules.toml"), "v0")

    assert not rules.of_kind("judgment")


def test_the_example_policy_requires_only_the_deployed_gates() -> None:
    policy = PolicyConfig.load(EXAMPLE, Path("honest-ci-policy.toml"), "v0")

    # `spec` is not implemented, so requiring it would pin every terminal state
    # to needs_human -- correctly, and uselessly for an acceptance.
    assert policy.required_checks == ["review", "regression"]


def test_the_example_pom_pins_the_versions_the_gates_invoke() -> None:
    """The pom and ``maven.py`` have to name the same plugin versions.

    The gates invoke fully qualified goals. A pom that pins a different version
    configures one plugin and runs another, so the checkstyle.xml and ruleset
    the example ships would quietly stop applying. The dependency plugin is
    left out: it provisions, and is never configured by the repository under
    verification. The clean plugin stays in -- a developer types ``mvn clean``
    in this directory too, and that should be the plugin the gate ran.
    """

    pom = (EXAMPLE / "pom.xml").read_text(encoding="utf-8")

    for name, coordinate in MAVEN_PLUGINS.items():
        if name == "dependency":
            continue
        _, artifact, version = coordinate.split(":")
        assert f"<artifactId>{artifact}</artifactId>" in pom, artifact
        assert f"<version>{version}</version>" in pom, coordinate


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_patch_still_applies(scenario: str, tmp_path: Path) -> None:
    """A patch that stopped applying is an acceptance that cannot run at all."""

    if shutil.which("git") is None:  # pragma: no cover - git is a hard dependency
        pytest.skip("没有 git")
    root = tmp_path / "repo"
    shutil.copytree(EXAMPLE, root)
    shutil.rmtree(root / "scenarios")

    proc = subprocess.run(
        ["git", "apply", "--check", str(EXAMPLE / "scenarios" / f"{scenario}.patch")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
