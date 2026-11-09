"""GitHub: where pull requests come from, and where verdicts go back.

Held to four properties, each of which GitHub gives its own way to fail:

* **A diff nobody could fetch is never rendered as a clean file.** GitHub lists
  at most 3000 files per pull request and leaves ``patch`` out of an entry whose
  diff is binary or too large. A pull request past the cap, a listing that does
  not match the pull request's own ``changed_files``, or a head that moves while
  the pages are read raises instead of judging part of a change. A file with no
  patch -- or with a patch that no longer adds up to the file's own line counts
  -- stays in ``files_changed`` with no line counts, which is what puts it in
  the evidence pack's "not verified" section. So does a file whose name is not
  safe to use as a path, and the structural guardrails refuse the change for it.
* **The scanners run on the commit under review, or on nothing.** Inside GitHub
  Actions the workflow has already checked the repository out, and that tree is
  reused only if its ``HEAD`` is the pull request head, its tracked files are
  untouched and no credential was left in its git config. Outside Actions,
  ``refs/pull/N/head`` is fetched into a fresh tree and checked the same way.
* **The token stays in headers.** REST and GraphQL calls carry it in
  ``Authorization`` through ``open_validated_http_request``; git gets it through
  ``GIT_CONFIG_*`` in the child environment. It is never in a URL, in argv, in
  ``.git/config`` or in anything posted back.
* **Only a pass is a pass.** ``needs_human`` and ``skipped`` become a *pending*
  commit status, a run that fell over becomes ``error``, and nothing but
  ``pass`` becomes ``success``.

Identity is the one thing GitHub makes harder. The token a workflow gets cannot
call ``GET /user``, so "the conclusion comment this deployment wrote" is found
through GraphQL's ``viewerDidAuthor`` instead of by comparing logins -- which
also keeps a comment anyone else wrote, marker or not, from ever being edited.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ..actions import ActionsContext
from ..contracts import (
    MAX_PLATFORM_REVIEWS,
    ApprovalReading,
    ChangedFile,
    ChangeRequest,
    EvidencePack,
    PlatformApproval,
    PlatformApprovals,
    PlatformReview,
    PlatformReviewState,
    Verdict,
)
from ..diffs import diff_label, file_line_counts, is_safe_relative_path, printable_path
from ..license_check import Ownership, read_ownership
from ..redaction import redact_text
from ..report import (
    MARKER_PREFIX,
    VERDICT_MARKER,
    read_approval_window,
    render_forge_comment,
)
from ..settings import Settings
from ..state import RunStatus
from ..utils import open_validated_http_request
from .base import ForgeAdapter, ForgeError

#: The REST API version this adapter was written against.
#: https://docs.github.com/en/rest/about-the-rest-api/api-versions
API_VERSION = "2022-11-28"

#: ``per_page`` tops out at 100. "Responses include a maximum of 3000 files":
#: past that the endpoint cannot hand the change over whole, so the cap is the
#: point at which this adapter refuses rather than a tuning knob.
#: https://docs.github.com/en/rest/pulls/pulls#list-pull-requests-files
FILE_PAGE_SIZE = 100
MAX_FILES = 3000

#: GraphQL connections are read 100 nodes at a time, and a conversation longer
#: than this many pages is refused rather than half-read.
GRAPHQL_PAGE_SIZE = 100
MAX_COMMENT_PAGES = 10

#: GitHub answers 422 to a longer status description. The REST reference only
#: says "a short description of the status"
#: (https://docs.github.com/en/rest/commits/statuses#create-a-commit-status),
#: so the number is the API's own validation error, not the documentation's.
MAX_STATUS_DESCRIPTION = 140

#: A checkout is a network operation on someone else's server; it should
#: surface as an error rather than stall a job until the runner kills it.
GIT_TIMEOUT_SECONDS = 300

#: Verdict → commit-status state. ``needs_human`` and ``skipped`` are
#: ``pending`` on purpose: a required status that is pending blocks the merge,
#: which is the correct reading of "nobody has verified this yet".
#: https://docs.github.com/en/rest/commits/statuses#create-a-commit-status
STATUS_STATE: dict[Verdict, str] = {
    Verdict.PASS: "success",
    Verdict.BLOCKED: "failure",
    Verdict.NEEDS_HUMAN: "pending",
    Verdict.SKIPPED: "pending",
}
#: A run that fell over reached no verdict, and GitHub has a state for exactly
#: that. It still blocks a required check, like every state but ``success``.
ERROR_STATE = "error"
PROGRESS_STATE = "pending"

_ChangeType = Literal["added", "modified", "deleted", "renamed"]
#: GitHub's ``status`` of a changed file, in the contract's words. A copy is a
#: new file as far as the tree is concerned; the source it was copied from is
#: still there. ``changed`` (mode only) and anything GitHub adds later read as
#: modified, the reading that claims the least.
_CHANGE_TYPES: dict[str, _ChangeType] = {
    "added": "added",
    "copied": "added",
    "removed": "deleted",
    "renamed": "renamed",
}

#: What GitHub accepts in an owner or repository name. Checked before a name
#: becomes part of a URL, so a path segment can never be smuggled in.
_REPO_PART = re.compile(r"[A-Za-z0-9._-]+")

#: The fields every comment-like node is read with.
_COMMENT_FIELDS = "author { __typename login } body createdAt"
_THREAD_FIELDS = (
    f"isResolved path line comments(first: {GRAPHQL_PAGE_SIZE}) "
    f"{{ totalCount nodes {{ {_COMMENT_FIELDS} }} }}"
)
#: ``fullDatabaseId`` rather than ``databaseId``: the REST id a comment is
#: edited by has outgrown a GraphQL ``Int``, and the BigInt arrives as a string.
#: ``editor`` is whoever edited it last, null until somebody does; with
#: ``author`` it tells whether the text is still this deployment's alone.
#: https://docs.github.com/en/graphql/reference/interfaces#comment
_OWN_COMMENT_FIELDS = "fullDatabaseId viewerDidAuthor body author { login } editor { login }"
#: The review states that grant, refuse or withdraw an approval, as GraphQL
#: filters them, and in the contract's words. ``COMMENTED`` and ``PENDING`` say
#: neither, so they are not read.
#: https://docs.github.com/en/graphql/reference/enums#pullrequestreviewstate
_DECISIVE_STATES: dict[str, PlatformReviewState] = {
    "APPROVED": "approved",
    "CHANGES_REQUESTED": "changes_requested",
    "DISMISSED": "dismissed",
}
#: What an approval is judged by. ``authorCanPushToRepository`` -- "whether the
#: author of this review has push access to the repository" -- is the write
#: permission, read with the review itself rather than per reviewer. ``editor``
#: says whether the text is still the reviewer's, as for the comment above.
#: https://docs.github.com/en/graphql/reference/objects#pullrequestreview
_APPROVAL_FIELDS = (
    "author { __typename login } authorCanPushToRepository body commit { oid } "
    "editor { login } state submittedAt url"
)


class GitHubError(ForgeError):
    """A GitHub call failed, or answered with something that cannot be judged."""


class GitHubForgeAdapter(ForgeAdapter):
    def __init__(self, settings: Settings, actions: ActionsContext | None = None) -> None:
        self.settings = settings
        self.base_url = settings.github_base_url.rstrip("/")
        self.graphql_url = graphql_url(self.base_url)
        self._token = settings.github_token
        #: The job this process runs in, when GitHub Actions started it.
        self._actions = actions if actions is not None else ActionsContext.from_env()
        #: ``clone_url`` per repository, so a checkout costs one extra call the
        #: first time and none after that.
        self._clone_urls: dict[str, str] = {}
        #: What each pull request ``fetch_change`` read said about the
        #: repository it merges into and who opened it, for ``ownership``. Kept
        #: here rather than on the ``ChangeRequest``, which goes into the
        #: evidence pack.
        self._ownership: dict[tuple[str, int | None], tuple[Any, Any]] = {}

    # ── inbound ──────────────────────────────────────────────────────────

    def fetch_change(self, repo: str, number: int) -> ChangeRequest:
        owner, name = _split_repo(repo)
        path = f"/repos/{owner}/{name}/pulls/{number}"
        pull = self._pull_request(path, repo, number)
        head_sha = _sha(pull, "head")
        if not head_sha:
            # Without it there is no fingerprint, so an approval could not be
            # bound to anything -- which is worse than refusing to run.
            raise GitHubError(
                f"{repo}#{number} has no head commit: cannot tell which commit is under review."
            )
        expected = _count(pull.get("changed_files"))
        if expected is None:
            raise GitHubError(f"GitHub did not say how many files {repo}#{number} changes.")
        if expected > MAX_FILES:
            raise GitHubError(
                f"{repo}#{number} changes {expected} files; GitHub lists at most {MAX_FILES}. "
                "Judging the files it does list would declare the rest clean unread."
            )
        entries = self._file_entries(path, expected)

        # The listing is several requests. A push in between would mix two
        # heads' files into one diff, so the pull request is read once more and
        # must not have moved.
        again = self._pull_request(path, repo, number)
        if _sha(again, "head") != head_sha or _count(again.get("changed_files")) != expected:
            raise GitHubError(
                f"{repo}#{number} moved while its files were being read; "
                "the run that the new push triggers will verify it."
            )
        if len(entries) != expected:
            raise GitHubError(
                f"{repo}#{number} changes {expected} files, but GitHub listed {len(entries)}. "
                "Judging a list that does not match the change would call the difference clean."
            )

        assembled = _assemble_diff(entries)
        head = _mapping(pull.get("head"))
        base = _mapping(pull.get("base"))
        self._ownership[(repo, number)] = (base.get("repo"), pull.get("user"))
        return ChangeRequest(
            forge="github",
            repo=repo,
            number=number,
            url=_optional_str(pull.get("html_url")),
            title=str(pull.get("title") or ""),
            description=_description(pull, assembled),
            author=_optional_str(_mapping(pull.get("user")).get("login")),
            source_branch=_optional_str(head.get("ref")),
            target_branch=_optional_str(base.get("ref")),
            head_sha=head_sha,
            base_sha=_optional_str(base.get("sha")),
            diff=assembled.diff,
            files_changed=assembled.files,
            labels=[
                str(label["name"])
                for label in pull.get("labels") or []
                if isinstance(label, dict) and label.get("name")
            ],
        )

    def _pull_request(self, path: str, repo: str, number: int) -> dict[str, Any]:
        pull = self._get(path)
        if not isinstance(pull, dict):
            raise GitHubError(f"GitHub did not return a pull request object for {repo}#{number}.")
        return pull

    def _file_entries(self, path: str, expected: int) -> list[dict[str, Any]]:
        """The file list, read until it ends, reaches the pull request's own count, or the cap.

        No more than the cap is ever asked for, so a pull request of exactly
        3000 files is read in the 30 pages GitHub serves for it. Whether the
        list matches the change is the caller's check: a list that ends early
        or runs long is refused there, never trusted here.
        """

        entries: list[dict[str, Any]] = []
        for page in range(1, MAX_FILES // FILE_PAGE_SIZE + 1):
            payload = self._get(
                f"{path}/files", {"page": str(page), "per_page": str(FILE_PAGE_SIZE)}
            )
            if not isinstance(payload, list):
                raise GitHubError("GitHub's pull request files endpoint did not return a list.")
            entries += [item for item in payload if isinstance(item, dict)]
            # A full page can be the last one; the count says so without an
            # extra request for the empty page after it.
            if len(payload) < FILE_PAGE_SIZE or len(entries) >= expected:
                break
        return entries

    def materialize(self, change: ChangeRequest, workspace: Path) -> Path:
        """The tree at the pull request head, proven to be that head."""

        actions = self._actions
        if actions is not None and actions.workspace is not None and actions.owns(change.repo):
            return self._verified_actions_checkout(actions.workspace, change)
        return self._fetched_checkout(change, workspace)

    def _verified_actions_checkout(self, checkout: Path, change: ChangeRequest) -> Path:
        """Reuse the workflow's checkout, but only as exactly the commit under review.

        Nothing here repairs a checkout. A wrong commit, a modified tree or a
        leftover credential is the workflow's configuration talking, and the
        run stops with the line to change rather than quietly verifying
        something else.
        """

        # Through ``_git`` so that a tree git will not read -- a container job
        # whose user does not own the checkout trips "dubious ownership" --
        # stops the run in git's own words.
        head = self._git(checkout, "rev-parse", "HEAD").strip()
        if head != change.head_sha:
            # On pull_request events GITHUB_SHA is "Last merge commit on the
            # GITHUB_REF branch", which is what actions/checkout checks out by
            # default -- a tree that exists nowhere in the pull request.
            # https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request
            raise GitHubError(
                f"The workflow's checkout ({checkout}) is at {head[:12]}, not at the pull request "
                f"head {change.head_sha[:12]}. On pull_request events actions/checkout checks "
                "out a merge commit by default; check out the head instead with "
                "`ref: ${{ github.event.pull_request.head.sha }}`. If the pull request moved "
                "after this run started, the run for the new push verifies it."
            )
        # ``persist-credentials`` defaults to true: actions/checkout v4 writes
        # the token into .git/config and v6 into a file under $RUNNER_TEMP that
        # .git/config includes. Either way the token sits where the pull
        # request's own code can read it once the sandbox runs that code over
        # this tree. ``--name-only`` because the value must not be read here
        # either. https://github.com/actions/checkout#usage
        persisted = self._git(
            checkout,
            "config",
            "--local",
            "--includes",
            "--name-only",
            "--get-regexp",
            r"^http\..*\.extraheader$",
            accept_exit=(0, 1),
        )
        if persisted.strip():
            raise GitHubError(
                "actions/checkout left a credential in this checkout's git config "
                "(persist-credentials defaults to true). The pull request's code runs "
                "against this tree, so set `persist-credentials: false` on the checkout step."
            )
        if self._git(checkout, "status", "--porcelain", "--untracked-files=no").strip():
            raise GitHubError(
                f"Tracked files in the workflow's checkout ({checkout}) differ from "
                f"{change.head_sha[:12]}: a step before honest-ci changed them, and the scanners "
                "must read the commit under review, not a modified copy of it."
            )
        return checkout

    def _fetched_checkout(self, change: ChangeRequest, workspace: Path) -> Path:
        target = workspace / f"{_slug(change.repo)}-{change.head_sha[:12]}"
        if _head_commit(target) == change.head_sha:
            # Same commit, same tree: re-running a gate should not re-clone.
            return target
        target.mkdir(parents=True, exist_ok=True)
        clone_url = self._clone_url(change.repo)
        # GitHub publishes every pull request head in the base repository as
        # refs/pull/N/head -- for a fork's pull request too, which is why it is
        # fetched from here and never from the fork.
        # https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/checking-out-pull-requests-locally
        ref = f"refs/pull/{change.number}/head" if change.number is not None else change.head_sha
        self._git(target, "init", "--quiet")
        self._git(
            target,
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "--no-tags",
            clone_url,
            ref,
            clone_url=clone_url,
        )
        self._git(target, "checkout", "--quiet", "--detach", "FETCH_HEAD")

        checked_out = self._git(target, "rev-parse", "HEAD").strip()
        if checked_out != change.head_sha:
            raise GitHubError(
                f"Checked out {checked_out[:12]}, not the commit under review "
                f"{change.head_sha[:12]}: the scanners cannot judge this change on another tree."
            )
        return target

    def _clone_url(self, repo: str) -> str:
        cached = self._clone_urls.get(repo)
        if cached is not None:
            return cached
        owner, name = _split_repo(repo)
        payload = self._get(f"/repos/{owner}/{name}")
        url = str(payload.get("clone_url") or "") if isinstance(payload, dict) else ""
        if not url:
            raise GitHubError(f"GitHub gave no clone URL for {repo}; cannot check the head out.")
        self._clone_urls[repo] = url
        return url

    def fetch_reviews(self, change: ChangeRequest) -> list[PlatformReview]:
        """Read what people have already said on this pull request.

        Review threads come first because they are the only comments GitHub
        tracks as resolvable, and an open one is what routes a change to a
        person; each comment in a thread carries the thread's ``isResolved``.
        Then the conversation and the reviews' own summaries, which are
        opinions but never open questions.

        This deployment's own comments are dropped: reading them back would let
        the product cite itself as a human who raised the concern. A thread or
        a conversation that cannot be read whole raises instead of returning
        the part that was read.
        https://docs.github.com/en/graphql/reference/objects#pullrequestreviewthread
        """

        reviews: list[PlatformReview] = []
        for thread in self._nodes(change, "reviewThreads", _THREAD_FIELDS):
            resolved = thread.get("isResolved")
            comments = _mapping(thread.get("comments"))
            nodes = comments.get("nodes")
            total = comments.get("totalCount")
            if not isinstance(resolved, bool) or not isinstance(nodes, list):
                raise GitHubError("GitHub returned a review thread without its state or comments.")
            if not isinstance(total, int) or total > len(nodes):
                raise GitHubError(
                    f"A review thread has more than {GRAPHQL_PAGE_SIZE} comments; reading only "
                    "the first ones could miss the reply that left it open."
                )
            for comment in nodes:
                reviews += _reviews(
                    comment, path=thread.get("path"), line=thread.get("line"), resolved=resolved
                )
            _check_review_count(reviews)
        for comment in self._nodes(change, "comments", _COMMENT_FIELDS):
            reviews += _reviews(comment)
            _check_review_count(reviews)
        for review in self._nodes(change, "reviews", _COMMENT_FIELDS):
            # A review submitted with no text ("Approve", or inline comments
            # only) says nothing here; its inline comments are in the threads.
            if str(review.get("body") or "").strip():
                reviews += _reviews(review)
            _check_review_count(reviews)
        return reviews

    def fetch_approvals(self, change: ChangeRequest) -> PlatformApprovals:
        """Every decisive review on this pull request, and the window last recorded on it.

        The window comes out of this deployment's own conclusion comment, found
        the way ``publish_report`` finds it, so a comment anyone else wrote
        cannot supply one -- nor can this one once somebody else edited it
        last. Anyone who can push can edit it, and one who moved the window
        back would revive approvals given under other rules or another policy;
        the strict window this run writes back instead is this deployment's
        again. It is read before the reviews: a review given in between then
        falls inside what this run sees as already given, which is the strict
        side to err on.

        Every review is read, not only approvals, because an approval is
        superseded by a later request for changes from the same person, and a
        dismissed approval comes back as ``DISMISSED`` rather than vanishing.
        https://docs.github.com/en/graphql/reference/objects#pullrequest
        """

        own = self._own_verdict_comment(change)
        window = (
            read_approval_window(own.body)
            if own is not None and not own.edited_by_another
            else None
        )
        states = ", ".join(_DECISIVE_STATES)
        reviews = [
            approval
            for node in self._nodes(
                change, "reviews", _APPROVAL_FIELDS, arguments=f"states: [{states}]"
            )
            for approval in _approval(node)
        ]
        return PlatformApprovals(reviews=reviews, window=window)

    def ownership(self, change: ChangeRequest) -> Ownership | None:
        """Read off the pull request ``fetch_change`` received: ``base.repo`` and ``user``.

        No request: GitHub already said, in the answer the change was built from.
        """

        kept = self._ownership.get((change.repo, change.number))
        if kept is None:
            return None
        repository, author = kept
        return read_ownership(repository, author)

    # ── outbound ─────────────────────────────────────────────────────────

    def publish_report(
        self,
        change: ChangeRequest,
        evidence: EvidencePack,
        status: RunStatus,
        approvals: ApprovalReading | None = None,
        evidence_at: str | None = None,
        license_note: str | None = None,
    ) -> None:
        """Update this deployment's conclusion comment in place, or post the first one."""

        body = render_forge_comment(
            evidence,
            status=status,
            approvals=approvals,
            evidence_at=evidence_at,
            license_note=license_note,
        )
        self.publish_note(change, body)

    def publish_note(self, change: ChangeRequest, body: str) -> None:
        """Write ``body`` over this deployment's conclusion comment, or post it as the first one.

        A pull request's conversation is its issue's, so both writes use the
        issue-comments endpoints: "Every pull request is an issue, but not
        every issue is a pull request."
        https://docs.github.com/en/rest/issues/comments
        """

        owner, name = _split_repo(change.repo)
        number = _number(change)
        existing = self._own_verdict_comment(change)
        if existing is None:
            self._post(f"/repos/{owner}/{name}/issues/{number}/comments", {"body": body})
        else:
            self._patch(f"/repos/{owner}/{name}/issues/comments/{existing.id}", {"body": body})

    def _own_verdict_comment(self, change: ChangeRequest) -> _OwnComment | None:
        """The conclusion comment this token's owner wrote, if any.

        ``viewerDidAuthor`` ("Did the viewer author this comment") answers "is
        this mine" for a personal token and for the Actions token alike, where
        ``GET /user`` answers only for the first. The marker alone is not
        enough: anyone can type it, and only a comment this deployment wrote
        may be overwritten.
        https://docs.github.com/en/graphql/reference/objects#issuecomment
        """

        for page in self._pages(change, "comments", _OWN_COMMENT_FIELDS):
            for comment in page:
                if comment.get("viewerDidAuthor") is not True:
                    continue
                if not str(comment.get("body") or "").startswith(VERDICT_MARKER):
                    continue
                identifier = str(comment.get("fullDatabaseId") or "")
                if not identifier.isdigit():
                    raise GitHubError("The existing conclusion comment has no usable id.")
                return _OwnComment(
                    id=int(identifier),
                    body=str(comment.get("body") or ""),
                    edited_by_another=_edited_by_another(comment),
                )
        return None

    def publish_progress(
        self,
        change: ChangeRequest,
        summary: str,
        url: str | None = None,
    ) -> None:
        self._publish_status(change, PROGRESS_STATE, summary, url)

    def publish_status(
        self,
        change: ChangeRequest,
        verdict: Verdict,
        summary: str,
        url: str | None = None,
    ) -> None:
        self._publish_status(change, STATUS_STATE[verdict], summary, url)

    def publish_error(self, change: ChangeRequest, summary: str, url: str | None = None) -> None:
        self._publish_status(change, ERROR_STATE, summary, url)

    def _publish_status(
        self,
        change: ChangeRequest,
        state: str,
        summary: str,
        url: str | None = None,
    ) -> None:
        owner, name = _split_repo(change.repo)
        body: dict[str, Any] = {
            "state": state,
            # "A string label to differentiate this status from the status of
            # other systems" -- the name a branch protection rule requires.
            "context": self.settings.forge_status_name,
            "description": redact_text(" ".join(summary.split()))[:MAX_STATUS_DESCRIPTION],
        }
        if url:
            body["target_url"] = url
        self._post(f"/repos/{owner}/{name}/statuses/{change.head_sha}", body)

    # ── HTTP ─────────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        return self._request(url, method="GET")

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        return self._request(f"{self.base_url}{path}", method="POST", body=body)

    def _patch(self, path: str, body: dict[str, Any]) -> Any:
        return self._request(f"{self.base_url}{path}", method="PATCH", body=body)

    def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = self._request(
            self.graphql_url,
            method="POST",
            body={"query": query, "variables": variables},
            # The GraphQL endpoint is not under the REST base on GitHub
            # Enterprise Server, so the boundary is the endpoint itself.
            allowed_base=self.graphql_url,
        )
        if not isinstance(payload, dict):
            raise GitHubError("GitHub GraphQL did not return an object.")
        if payload.get("errors"):
            # GraphQL reports a refusal inside a 200. Treated as the refusal it
            # is: the data beside it, if any, is partial by definition.
            detail = redact_text(_graphql_errors(payload["errors"]))
            raise GitHubError(f"GitHub GraphQL refused the query: {detail[:400]}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise GitHubError("GitHub GraphQL returned no data.")
        return data

    def _nodes(
        self, change: ChangeRequest, connection: str, fields: str, *, arguments: str = ""
    ) -> list[dict[str, Any]]:
        """Every node of one pull request connection, or an error -- never a prefix."""

        return [
            node
            for page in self._pages(change, connection, fields, arguments=arguments)
            for node in page
        ]

    def _pages(
        self, change: ChangeRequest, connection: str, fields: str, *, arguments: str = ""
    ) -> Iterator[list[dict[str, Any]]]:
        """One pull request connection, page by page, until GitHub says it ended.

        A caller that finds what it looked for may stop early; one that reads
        to the end gets an error rather than a prefix when the end is out of
        reach. ``arguments`` are the connection's own filters, written by this
        module and never taken from the change.
        """

        owner, name = _split_repo(change.repo)
        filters = f"{arguments}, " if arguments else ""
        query = (
            "query($owner: String!, $name: String!, $number: Int!, $cursor: String) {"
            " repository(owner: $owner, name: $name) { pullRequest(number: $number) {"
            f" {connection}({filters}first: {GRAPHQL_PAGE_SIZE}, after: $cursor) {{"
            f" pageInfo {{ hasNextPage endCursor }} nodes {{ {fields} }} }} }} }} }}"
        )
        variables: dict[str, Any] = {
            "owner": owner,
            "name": name,
            "number": _number(change),
            "cursor": None,
        }
        for _ in range(MAX_COMMENT_PAGES):
            data = self._graphql(query, variables)
            pull = _mapping(_mapping(data.get("repository")).get("pullRequest"))
            page = _mapping(pull.get(connection))
            items = page.get("nodes")
            if not isinstance(items, list):
                raise GitHubError(
                    f"GitHub GraphQL returned no {connection} for {change.repo}#{change.number}."
                )
            yield [item for item in items if isinstance(item, dict)]
            info = _mapping(page.get("pageInfo"))
            if info.get("hasNextPage") is not True:
                return
            cursor = info.get("endCursor")
            if not isinstance(cursor, str) or not cursor:
                raise GitHubError(f"GitHub GraphQL said there are more {connection} but not where.")
            variables["cursor"] = cursor
        # Same reasoning as the file pagination: an unfinished read cannot
        # establish what is not there -- an open thread, or a comment of ours.
        raise GitHubError(
            f"More than {MAX_COMMENT_PAGES * GRAPHQL_PAGE_SIZE} {connection} on "
            f"{change.repo}#{change.number}; not read in full, so not read at all."
        )

    def _request(
        self,
        url: str,
        *,
        method: str,
        body: dict[str, Any] | None = None,
        allowed_base: str | None = None,
    ) -> Any:
        if not self._token:
            raise GitHubError("GITHUB_TOKEN is not set: the GitHub adapter needs a token.")
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                # https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                # GitHub rejects requests without a User-Agent.
                "User-Agent": "honest-ci",
                "X-GitHub-Api-Version": API_VERSION,
            },
        )
        try:
            # The initial URL and every redirect stay under the configured API
            # base, so a redirect cannot carry the token to another host.
            with open_validated_http_request(
                request,
                allowed_base=allowed_base or self.base_url,
                timeout_seconds=self.settings.forge_timeout_seconds,
                allow_loopback_http=True,
            ) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = redact_text(exc.read().decode("utf-8", errors="replace")).strip()
            raise GitHubError(
                f"GitHub {method} {_printable(url)} returned {exc.code}: "
                f"{detail[:400] or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise GitHubError(f"Cannot reach GitHub ({self.base_url}): {exc.reason}") from exc
        except ValueError as exc:
            # The URL validator refused it. Treated as a failure rather than
            # relaxed, because the thing it is protecting is the token.
            raise GitHubError(f"Outbound URL refused: {exc}") from exc
        if not raw.strip():
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GitHubError(
                f"GitHub {method} {_printable(url)} did not return JSON "
                "(a proxy or a sign-in page may have answered instead)."
            ) from exc

    # ── git ──────────────────────────────────────────────────────────────

    def _git(
        self,
        cwd: Path,
        *args: str,
        clone_url: str | None = None,
        accept_exit: tuple[int, ...] = (0,),
    ) -> str:
        try:
            proc = subprocess.run(
                ["git", "-C", str(cwd), *args],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=GIT_TIMEOUT_SECONDS,
                check=False,
                env=self._git_env(clone_url),
            )
        except FileNotFoundError as exc:
            raise GitHubError("git not found: checking the head out needs git.") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise GitHubError(f"git {args[0]} could not run: {redact_text(str(exc))}") from exc
        if proc.returncode not in accept_exit:
            detail = redact_text((proc.stderr or proc.stdout).strip()).splitlines()
            reason = detail[-1] if detail else f"exit code {proc.returncode}"
            raise GitHubError(f"git {args[0]} failed: {reason}")
        return proc.stdout

    def _git_env(self, clone_url: str | None) -> dict[str, str]:
        """Environment for a git call, with the token as a scoped header when fetching.

        ``GIT_CONFIG_*`` rather than ``-c`` or a credential in the URL: argv is
        readable from any process list, and a remote URL or a written config
        would leave the token inside the tree under verification. Basic with
        ``x-access-token`` because that is how GitHub documents a token as a
        git password.
        https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app-installation
        """

        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_OPTIONAL_LOCKS"] = "0"
        env["GIT_PAGER"] = "cat"
        origin = _http_origin(clone_url) if clone_url else None
        if origin and self._token:
            basic = base64.b64encode(f"x-access-token:{self._token}".encode()).decode("ascii")
            env["GIT_CONFIG_COUNT"] = "1"
            env["GIT_CONFIG_KEY_0"] = f"http.{origin}/.extraheader"
            env["GIT_CONFIG_VALUE_0"] = f"Authorization: Basic {basic}"
        return env


@dataclass(frozen=True)
class _OwnComment:
    """This deployment's conclusion comment: where to edit it, and what it says now."""

    id: int
    body: str
    #: Whether somebody other than this deployment edited it last, so that
    #: what it says now may not be what this deployment wrote.
    edited_by_another: bool


def _edited_by_another(comment: dict[str, Any]) -> bool:
    """Whether a comment's last edit was by someone other than its author.

    An editor GitHub names without a login is taken as someone else: this
    decides whether the text can be trusted, so not knowing counts as no.
    """

    editor = comment.get("editor")
    if editor is None:
        return False
    login = str(_mapping(editor).get("login") or "")
    author = str(_mapping(comment.get("author")).get("login") or "")
    # Logins are case-insensitive.
    return not login or login.lower() != author.lower()


def graphql_url(base_url: str) -> str:
    """Where GraphQL lives for a REST base.

    GitHub.com serves it beside REST (``https://api.github.com/graphql``);
    GitHub Enterprise Server serves REST under ``/api/v3`` and GraphQL at
    ``/api/graphql``.
    https://docs.github.com/en/enterprise-server@latest/graphql/guides/forming-calls-with-graphql#the-graphql-endpoint
    """

    base = base_url.rstrip("/")
    if base.endswith("/api/v3"):
        return f"{base.removesuffix('/v3')}/graphql"
    return f"{base}/graphql"


# ── the diff ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Assembled:
    """One unified diff, its file inventory, and what did not make it in."""

    diff: str
    files: list[ChangedFile]
    #: In the change, but with no reviewable diff text: binary, too large, or
    #: cut short by the platform.
    unreadable: list[str]
    #: Entries with a name ``is_safe_relative_path`` refuses, as the description
    #: names them. Their files are in ``files`` too, never in ``diff``.
    rejected: list[str]


def _assemble_diff(entries: Sequence[dict[str, Any]]) -> _Assembled:
    """Rebuild one unified diff from GitHub's per-file entries.

    GitHub hands over each file's hunks without the ``diff --git`` / ``---`` /
    ``+++`` headers, and ``diffs.py`` stays the only parser: the headers are
    synthesised here so every gate reads the same text as a local diff.
    """

    chunks: list[str] = []
    inventory: list[tuple[str, _ChangeType, str | None]] = []
    unreadable: list[str] = []
    rejected: list[str] = []

    for entry in entries:
        path = str(entry.get("filename") or "")
        status = str(entry.get("status") or "")
        change_type = _CHANGE_TYPES.get(status, "modified")
        source = str(entry.get("previous_filename") or "")
        old_path = source if status in {"renamed", "copied"} and source else path

        if not path:
            # Nothing to list it under, and a change judged without one of its
            # files would call that file clean.
            raise GitHubError("GitHub listed a changed file without a name.")
        if not _paths_are_safe(old_path, path):
            # Neither name goes into the diff, which every gate reads and which
            # would make it a tool argument and a report location. Dropping the
            # file, though, would let the change pass without it. So its names
            # stay in the file list with no line counts, where the structural
            # guardrails refuse the change -- as they do when local intake
            # finds the same file.
            inventory += _refused_files(old_path, path, status, change_type)
            rejected.append(
                printable_path(path)
                if old_path == path
                else f"{printable_path(old_path)} -> {printable_path(path)}"
            )
            continue

        inventory.append((path, change_type, old_path if change_type == "renamed" else None))
        patch = entry.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            # GitHub leaves ``patch`` out for a binary or oversized diff, and
            # for a change with no lines at all (a pure rename, a mode change),
            # and reports zero counts for the first as well as the last, so the
            # two cannot be told apart here. It stays in the file list with no
            # line counts, which is what puts it in the evidence pack's "what
            # this run could not verify" section.
            unreadable.append(path)
            continue
        chunk = _file_header(old_path, path, status) + _terminated(patch)
        counted = file_line_counts(chunk).get(path, (0, 0))
        if counted != (_count(entry.get("additions")), _count(entry.get("deletions"))):
            # A patch that does not add up to the file's own counts was cut
            # short somewhere. Judging the part that arrived would call the
            # rest of the file clean.
            unreadable.append(path)
            continue
        chunks.append(chunk)

    diff = "".join(chunks)
    counts = file_line_counts(diff)
    files = [
        ChangedFile(
            path=path,
            change_type=change_type,
            added_lines=counts.get(path, (0, 0))[0],
            removed_lines=counts.get(path, (0, 0))[1],
            previous_path=previous,
        )
        for path, change_type, previous in inventory
    ]
    return _Assembled(diff=diff, files=files, unreadable=unreadable, rejected=rejected)


def _paths_are_safe(old_path: str, new_path: str) -> bool:
    """Both sides, because the diff header names both and either can escape."""

    return all(is_safe_relative_path(path) for path in {old_path, new_path} if path)


def _refused_files(
    old_path: str, new_path: str, status: str, change_type: _ChangeType
) -> list[tuple[str, _ChangeType, str | None]]:
    """The files a refused entry stands for, as inventory rows with no diff.

    A rename reads the way local intake reads every rename (``--no-renames``):
    a deletion and an addition. Neither name becomes a ``previous_path``, and
    both are in front of the guardrails. A copy leaves its source where it was,
    so the source is listed only when it is the name refused -- as modified,
    the reading that claims the least -- since nothing else would put it in
    front of the guardrails.
    """

    if old_path == new_path:
        return [(new_path, change_type, None)]
    if status == "renamed":
        return [(old_path, "deleted", None), (new_path, "added", None)]
    source: list[tuple[str, _ChangeType, str | None]] = (
        [] if is_safe_relative_path(old_path) else [(old_path, "modified", None)]
    )
    return [(new_path, change_type, None), *source]


def _file_header(old_path: str, new_path: str, status: str) -> str:
    left = "/dev/null" if status == "added" else f"a/{old_path}"
    right = "/dev/null" if status == "removed" else f"b/{new_path}"
    return (
        f"diff --git a/{old_path} b/{new_path}\n"
        f"--- {diff_label(left)}\n"
        f"+++ {diff_label(right)}\n"
    )


def _terminated(body: str) -> str:
    return body if body.endswith("\n") else f"{body}\n"


def _description(pull: dict[str, Any], assembled: _Assembled) -> str:
    """The author's description, plus what this adapter could not read.

    Both omissions go here as well as into the evidence pack, because this
    string is what a reviewer and the reading-aid model see first.
    """

    parts = [str(pull.get("body") or "").strip()]
    if assembled.unreadable:
        parts.append(
            "⚠ No reviewable diff for these files (binary, too large, cut short, or no line "
            "changes at all); they are listed but not reviewed: "
            + ", ".join(sorted(set(assembled.unreadable)))
        )
    if assembled.rejected:
        parts.append(
            "⚠ These file names are not safe to use as paths (they point outside the "
            "repository, or contain a control character or another character paths may "
            "not carry); they are listed with no diff and were not reviewed: "
            + ", ".join(sorted(set(assembled.rejected)))
        )
    return "\n\n".join(part for part in parts if part)


# ── small helpers ────────────────────────────────────────────────────────────


def _reviews(
    node: Any,
    *,
    path: Any = None,
    line: Any = None,
    resolved: bool | None = None,
) -> list[PlatformReview]:
    """One comment-like GraphQL node as a review, or nothing if it is our own."""

    if not isinstance(node, dict):
        return []
    body = str(node.get("body") or "")
    if MARKER_PREFIX in body:
        return []
    author = _mapping(node.get("author"))
    return [
        PlatformReview(
            # A deleted account comes back as a null author.
            author=str(author.get("login") or ""),
            body=body,
            path=str(path) if path else None,
            # ``line`` is null once the thread is outdated; the path alone is
            # still more locating than nothing.
            line=line if isinstance(line, int) and line > 0 else None,
            resolved=resolved,
            is_bot=author.get("__typename") == "Bot",
            created_at=_timestamp(node.get("createdAt")),
        )
    ]


def _approval(node: Any) -> list[PlatformApproval]:
    """One review node as a decisive review, or nothing if it is not one."""

    if not isinstance(node, dict):
        return []
    state = _DECISIVE_STATES.get(str(node.get("state") or ""))
    if state is None:
        return []
    author = _mapping(node.get("author"))
    return [
        PlatformApproval(
            # A deleted account comes back as a null author, and its review
            # then belongs to nobody.
            reviewer=str(author.get("login") or ""),
            state=state,
            commit_sha=str(_mapping(node.get("commit")).get("oid") or ""),
            reason=str(node.get("body") or ""),
            edited_by=str(_mapping(node.get("editor")).get("login") or ""),
            # Anything but an explicit true is no push access: the field is
            # what makes an approval count, so its absence must not.
            can_push=node.get("authorCanPushToRepository") is True,
            is_bot=author.get("__typename") == "Bot",
            submitted_at=_timestamp(node.get("submittedAt")),
            url=_optional_str(node.get("url")),
        )
    ]


def _graphql_errors(errors: Any) -> str:
    if not isinstance(errors, list):
        return str(errors)
    messages = [str(error.get("message") or "") for error in errors if isinstance(error, dict)]
    return "; ".join(message for message in messages if message) or "no message"


def _check_review_count(reviews: Sequence[PlatformReview]) -> None:
    if len(reviews) > MAX_PLATFORM_REVIEWS:
        raise GitHubError(
            f"This pull request carries more than {MAX_PLATFORM_REVIEWS} review comments; "
            "reading only the first ones could miss the open thread that routes it to a person."
        )


def _split_repo(repo: str) -> tuple[str, str]:
    owner, _, name = repo.strip("/").partition("/")
    if not all(_REPO_PART.fullmatch(part) and part not in {".", ".."} for part in (owner, name)):
        raise GitHubError(f"{repo!r} is not an owner/name repository.")
    return owner, name


def _number(change: ChangeRequest) -> int:
    if change.number is None:
        raise GitHubError(f"{change.repo} has no pull request number to write back to.")
    return change.number


def _sha(pull: dict[str, Any], side: str) -> str:
    return str(_mapping(pull.get(side)).get("sha") or "")


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _mapping(value: Any) -> dict[str, Any]:
    """A nested object GitHub may send as null, as something safe to read."""

    return value if isinstance(value, dict) else {}


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        # A timestamp this adapter cannot read is not a reason to discard the
        # comment it belongs to.
        return None


def _slug(repo: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in repo).strip("-")


def _head_commit(path: Path) -> str | None:
    """The commit a checkout is at, or ``None`` if there is none."""

    if not (path / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _http_origin(url: str) -> str | None:
    """Scheme and host of a clone URL, for scoping the git auth header.

    ``None`` for anything that is not HTTP(S) -- a local path in a test, or an
    ssh remote -- so the header is only ever attached to a request that would
    carry it anyway.
    """

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _printable(url: str) -> str:
    """A URL for an error message. The token is a header, so the path is safe."""

    parsed = urllib.parse.urlsplit(url)
    return parsed.path or url


def _optional_str(value: Any) -> str | None:
    return str(value) if value else None


__all__ = [
    "ERROR_STATE",
    "FILE_PAGE_SIZE",
    "MAX_FILES",
    "MAX_STATUS_DESCRIPTION",
    "STATUS_STATE",
    "GitHubError",
    "GitHubForgeAdapter",
    "graphql_url",
]
