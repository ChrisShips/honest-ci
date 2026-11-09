"""What ``honest-ci init`` writes into a repository.

A first install has to give the very next pull request a verdict that means
something, with nothing else set up: no model, no scanner, no test sandbox. So
the files run the review gate alone, with rules decided from the diff, under a
policy that requires that gate and no other. A policy that required a gate
nobody runs would hold every pull request for a person
(``policy.unsatisfied_checks``), which a new user would read as honest-ci not
working.

Every file is package data, one copy of each, and the workflow is the template
``actions.workflow_template`` hands out: what init writes is the text the tests
read, not a second copy of it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import resources
from pathlib import Path
from typing import Literal

from .actions import workflow_template


def _resource(name: str) -> str:
    return resources.files("honest_ci").joinpath("resources", name).read_text(encoding="utf-8")


@dataclass(frozen=True)
class ScaffoldFile:
    #: Where the file goes, relative to the repository root, with "/" between parts.
    path: str
    content: Callable[[], str]


#: For the GitHub Action. The two configuration files go where the Action reads
#: them when HONEST_CI_RULES_PATH and HONEST_CI_POLICY_PATH are unset, and
#: GitHub only runs a workflow from .github/workflows at the repository root.
GITHUB_FILES: tuple[ScaffoldFile, ...] = (
    ScaffoldFile("honest-ci-rules.toml", partial(_resource, "honest-ci-rules.toml")),
    ScaffoldFile("honest-ci-policy.toml", partial(_resource, "honest-ci-policy.toml")),
    ScaffoldFile(".github/workflows/honest-ci.yml", workflow_template),
)

Outcome = Literal["created", "exists"]


def write_file(root: Path, item: ScaffoldFile) -> Outcome:
    """Create the file, or leave alone whatever is already at its path.

    A file that exists is somebody's configuration, possibly one they wrote by
    hand, and init has no way to tell an edit from an accident. Created with
    ``x`` so that the check and the write are one step: nothing written between
    them gets overwritten, and a symbolic link at the path is never followed.

    Not through ``ArtifactStore``, which redacts what a run writes: these are
    the package's own templates, written into the repository as source, and
    nothing in them came from a run or from a credential.
    """

    target = root.joinpath(*item.path.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            # Bytes, so the file is the template byte for byte on any platform.
            handle.write(item.content().encode("utf-8"))
    except FileExistsError:
        return "exists"
    return "created"


def git_toplevel(path: Path) -> Path | None:
    """The root of the git work tree ``path`` is in; ``None`` when git cannot say.

    Not being able to tell is not an error here: a directory that is not a
    repository yet is a fine place to write the files before ``git init``.
    """

    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    top = proc.stdout.strip()
    return Path(top) if proc.returncode == 0 and top else None


__all__ = ["GITHUB_FILES", "Outcome", "ScaffoldFile", "git_toplevel", "write_file"]
