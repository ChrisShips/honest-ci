"""GitHub Actions: what a run can learn about the workflow it is a step of.

``honest-ci run-pr`` usually runs inside somebody's workflow, and three things
it has to get right depend on facts only the runner knows:

* **which tree to verify.** The workflow has already checked the repository
  out; the GitHub adapter verifies that checkout instead of fetching a second
  copy with the same token.
* **where a reader lands.** The commit status links to this workflow run, which
  is where the log and any uploaded evidence are, and the conclusion names the
  artifact the evidence is uploaded as: the ledger ``honest-ci evidence`` reads
  is thrown away with the job.
* **whether writing back is possible at all.** A pull request from a fork runs
  with a read-only token, so the conclusion goes to the job summary instead of
  failing on the first write.

Every fact here comes from a variable or a file the runner documents, or one
the Action sets for its own step, never from the repository under
verification: that tree is the one thing that must not get a say in how it is
judged.

The workflow that puts ``run-pr`` there in the first place is handed out from
here too, by ``workflow_template``.

https://docs.github.com/en/actions/reference/workflows-and-actions/variables#default-environment-variables
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .redaction import redact_text

#: The events whose token GitHub makes read-only when the pull request comes
#: from a fork: "The ``GITHUB_TOKEN`` has read-only permissions in pull requests
#: from forked repositories."
#: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request
#:
#: ``pull_request_target`` is absent on purpose. It runs in the base
#: repository's context with a token that can write, which is exactly why a
#: workflow must not use it to run a fork's code -- but when one does, this
#: module has no business pretending the token is read-only.
FORK_READ_ONLY_EVENTS = frozenset(
    {"pull_request", "pull_request_review", "pull_request_review_comment"}
)

#: What an artifact name may be for the comment to quote it: the characters
#: the Action's own name is made of. Anything else is left unquoted rather than
#: put between backticks in a comment.
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9._-]{1,128}")


@dataclass(frozen=True)
class ActionsContext:
    """The runner's description of the job this process runs in."""

    #: ``GITHUB_WORKSPACE``: where actions/checkout put the repository.
    workspace: Path | None = None
    #: ``GITHUB_REPOSITORY``: ``owner/name`` of the repository the workflow
    #: runs in, which is the base repository for a pull request event.
    repository: str | None = None
    server_url: str | None = None
    run_id: str | None = None
    event_name: str | None = None
    event_path: Path | None = None
    step_summary: Path | None = None
    #: ``HONEST_CI_EVIDENCE_ARTIFACT``: set by the Action, not by the runner --
    #: the name its upload step gives the evidence directory. A workflow that
    #: runs ``run-pr`` without the Action uploads nothing it has declared.
    evidence_artifact: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ActionsContext | None:
        """The current job, or ``None`` when GitHub Actions did not start this process.

        ``GITHUB_ACTIONS`` is the documented switch ("Always set to ``true``
        when GitHub Actions is running the workflow"). Without it every other
        variable is somebody's shell, and a stray ``GITHUB_WORKSPACE`` must not
        make a local run verify whatever directory it names.
        """

        env = os.environ if environ is None else environ
        if env.get("GITHUB_ACTIONS") != "true":
            return None
        return cls(
            workspace=_path(env.get("GITHUB_WORKSPACE")),
            repository=env.get("GITHUB_REPOSITORY") or None,
            server_url=(env.get("GITHUB_SERVER_URL") or "").rstrip("/") or None,
            run_id=env.get("GITHUB_RUN_ID") or None,
            event_name=env.get("GITHUB_EVENT_NAME") or None,
            event_path=_path(env.get("GITHUB_EVENT_PATH")),
            step_summary=_path(env.get("GITHUB_STEP_SUMMARY")),
            evidence_artifact=_artifact_name(env.get("HONEST_CI_EVIDENCE_ARTIFACT")),
        )

    @property
    def run_url(self) -> str | None:
        """This workflow run's page, for the commit status to link to.

        Assembled the way the variables reference spells it:
        ``$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID``.
        ``None`` when any part is missing or malformed, because a status whose
        link points somewhere invented is worse than a status without one.
        """

        if not (self.server_url and self.repository and self.run_id):
            return None
        if not self.server_url.startswith(("https://", "http://")) or not self.run_id.isdigit():
            return None
        return f"{self.server_url}/{self.repository}/actions/runs/{self.run_id}"

    @property
    def evidence_at(self) -> str:
        """Where the conclusion comment says the full evidence pack is, in Markdown.

        Not ``honest-ci evidence``: the ledger it reads is on the runner, and
        the runner is gone by the time anyone reads the comment. What outlives
        the job is the artifact the Action uploads, on the workflow run's page.
        """

        run = f"[this workflow run]({self.run_url})" if self.run_url else "this workflow run"
        if self.evidence_artifact is None:
            return f"written on the runner of {run}, and kept only if the workflow uploads it"
        return f"the `{self.evidence_artifact}` artifact of {run}"

    def owns(self, repo: str) -> bool:
        """Whether the workflow runs in ``repo`` -- the only case where its checkout is the change's.

        GitHub treats ``owner/name`` case-insensitively, and so does this.
        """

        return self.repository is not None and self.repository.lower() == repo.strip("/").lower()

    def read_only_reason(self) -> str | None:
        """Why this job's token cannot write to the pull request, or ``None`` when it can.

        Decided from the event the runner recorded, not from the pull request
        the API describes: the token's permissions follow the trigger. A fork's
        pull request verified from a ``workflow_dispatch`` run has a token that
        can write; the same pull request's own ``pull_request`` run does not.

        A payload that cannot be read while the event is one of the read-only
        kinds counts as a fork. The two ways of being wrong are not alike:
        skipping the write on a same-repository pull request loses a comment
        and leaves the required status missing, which still blocks the merge;
        attempting it from a fork fails the whole run on a 403.
        """

        if self.event_name not in FORK_READ_ONLY_EVENTS:
            return None
        payload = self.event_payload()
        pull = payload.get("pull_request") if payload is not None else None
        if not isinstance(pull, dict):
            return (
                f"the {self.event_name} event payload could not be read, "
                "so a pull request from a fork cannot be ruled out"
            )
        head = _mapping(_mapping(pull.get("head")).get("repo"))
        base = _mapping(_mapping(pull.get("base")).get("repo"))
        if not head:
            # GitHub reports a head repository that no longer exists as null,
            # and only a fork can be deleted out from under its pull request.
            return "the pull request's head repository no longer exists (a deleted fork)"
        if _same_repository(head, base):
            return None
        return f"the pull request comes from the fork {head.get('full_name') or '(unnamed)'}"

    def write_step_summary(self, markdown: str) -> bool:
        """Append to this step's job summary; ``False`` when there is nowhere to write.

        Appended rather than overwritten, because the file belongs to the step
        and another tool in the same step may already have written to it. Not
        routed through ``ArtifactStore``: this is output handed to the runner,
        like standard output, not an artifact the evidence pack names. It is
        redacted like every other thing this deployment says in public.

        https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands#adding-a-job-summary
        """

        if self.step_summary is None:
            return False
        text = redact_text(markdown)
        try:
            with self.step_summary.open("a", encoding="utf-8") as handle:
                handle.write(text if text.endswith("\n") else f"{text}\n")
        except OSError:
            return False
        return True

    def event_payload(self) -> dict[str, Any] | None:
        """The webhook payload of the event that started the workflow, as the runner wrote it.

        ``None`` when there is no file or it isn't a JSON object; a caller
        decides what not knowing means for it.
        """

        if self.event_path is None:
            return None
        try:
            payload = json.loads(self.event_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None


def _same_repository(head: Mapping[str, Any], base: Mapping[str, Any]) -> bool:
    """Compare by id when the payload has one; a rename changes the name, not the id."""

    head_id, base_id = head.get("id"), base.get("id")
    if isinstance(head_id, int) and isinstance(base_id, int):
        return head_id == base_id
    head_name, base_name = head.get("full_name"), base.get("full_name")
    return (
        isinstance(head_name, str)
        and isinstance(base_name, str)
        and head_name.lower() == base_name.lower()
    )


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _artifact_name(value: str | None) -> str | None:
    return value if value and _ARTIFACT_NAME.fullmatch(value) else None


def workflow_template() -> str:
    """The workflow that runs honest-ci's Action on every pull request.

    Exactly one copy exists, shipped as package data, so that what a
    repository installs and what its documentation quotes cannot drift apart.
    It pins the Action to this package's own release.
    """

    template = resources.files("honest_ci").joinpath("resources", "github-workflow.yml")
    return template.read_text(encoding="utf-8")


__all__ = ["FORK_READ_ONLY_EVENTS", "ActionsContext", "workflow_template"]
