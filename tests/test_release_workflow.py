"""The workflow that publishes honest-ci to PyPI.

It holds the one credential in this repository that can put code on other
people's machines: the OIDC token PyPI trusts. So what makes it safe is read
off the parsed file on every commit -- the one event that starts it, the one
job that may mint the token, the commit every action it uses is pinned to --
and its tag check runs here for real, because reading a shell script proves
nothing about what it does.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from honest_ci import __version__

from .workflow_yaml import load

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / ".github" / "workflows" / "release.yml"

#: ``owner/repo@<full commit SHA>``, as the Action's own workflow pins them.
PINNED = re.compile(r"[\w.-]+/[\w.-]+(?:/[\w./-]+)?@[0-9a-f]{40}")

TAG_CHECK = "Check the tag against the package version"


def _workflow() -> dict[str, Any]:
    workflow = load(RELEASE.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow


def _steps(job: str) -> list[dict[str, Any]]:
    steps = _workflow()["jobs"][job]["steps"]
    assert isinstance(steps, list)
    return steps


def test_only_a_published_release_publishes() -> None:
    """Not a push, not a tag, not a pull request: a release somebody published."""

    assert _workflow()["on"] == {"release": {"types": ["published"]}}


def test_only_the_publish_job_can_mint_the_token_pypi_trusts() -> None:
    """The build runs setuptools and whatever it pulls in; none of that may publish."""

    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}
    assert "permissions" not in workflow["jobs"]["build"]
    assert workflow["jobs"]["publish"]["permissions"] == {"id-token": "write"}
    assert workflow["jobs"]["publish"]["environment"]["name"] == "pypi"
    assert workflow["jobs"]["publish"]["needs"] == "build"


def test_every_action_is_pinned_to_a_commit() -> None:
    """A tag can be moved by whoever controls the action; a commit cannot."""

    uses = [step["uses"] for job in ("build", "publish") for step in _steps(job) if "uses" in step]
    assert uses
    assert [ref for ref in uses if not PINNED.fullmatch(ref)] == []


def test_the_checkout_leaves_no_credential_behind() -> None:
    checkout = next(s for s in _steps("build") if s.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["persist-credentials"] == "false"


@pytest.mark.parametrize(
    ("tag", "published"),
    [(f"v{__version__}", True), ("v99.0.0", False), (__version__, False)],
)
def test_a_release_whose_tag_is_not_the_package_version_stops(tag: str, published: bool) -> None:
    """The workflow init writes pins ``v<version>``; PyPI must get that version."""

    step = next(s for s in _steps("build") if s.get("name") == TAG_CHECK)
    # The step runs on a runner whose `python` is the one setup-python put on
    # PATH; here it is the interpreter running the tests.
    path = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=REPO,
        env={"PATH": path, "TAG": tag},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is published, result.stdout + result.stderr
    if not published:
        assert f"Tag the release v{__version__}." in result.stdout
