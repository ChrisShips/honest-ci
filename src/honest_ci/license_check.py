"""The license key check: where honest-ci asks for a key, from when, and what a run says.

Under the Fair Core License the check is the line between free and paid: where
it asks for no key, anyone may use honest-ci free of charge. It asks only when a
command that decides a change -- ``run-pr``, ``verify`` or ``resume`` -- runs in
CI, and then only on two kinds of repository that aren't public: one an
organization owns, and one a personal account owns, on a pull request someone
other than its owner opened. In GitHub Actions a run that needs a key goes
ahead without one for a 14-day trial first. LICENSE, the README's License
section and this module say the same, and change together.

A run that needs a key and has none that works stops before any gate: it
says why, and where to get one, and never what it would have said about the
change. A run on its trial, or on a key past its expiry date, goes ahead and
says how long it has left.

What it knows about a repository it reads where the run already is: for
``run-pr``, the pull request it fetched anyway; for ``verify`` and ``resume``,
the event GitHub Actions recorded, or outside Actions, GitHub's own description
of the github.com repository ``origin`` names. The trial's first day is the
day the workflow file was first committed, as GitHub lists the file's commits.
It calls no server of honest-ci's. When it can't tell, it asks for no key and
says so in the log: it runs in a team's own CI, where a check that goes wrong
would stop every merge.

Whether a key that is due is there, and good, is ``licensing.py``'s. What the
check finds stays out of the evidence pack and the fingerprint, as the key does.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .actions import ActionsContext
from .licensing import (
    PUBLIC_KEY,
    LicenseKey,
    LicenseKeyError,
    PublicKey,
    check_key,
    expiry_refusal,
    read_key,
)
from .redaction import redact_text
from .report import VERDICT_MARKER
from .settings import Settings
from .utils import open_validated_http_request

if TYPE_CHECKING:
    from .adapters.base import ForgeAdapter
    from .contracts import ChangeRequest

log = logging.getLogger(__name__)

#: Set in every job by the CI systems that don't set ``CI``: Jenkins, Azure
#: Pipelines, TeamCity and AWS CodeBuild.
CI_VARIABLES = ("JENKINS_URL", "TF_BUILD", "TEAMCITY_VERSION", "CODEBUILD_BUILD_ID")

#: github.com's REST API, where ``verify`` and ``resume`` look a repository up
#: outside Actions. Only there: this edition supports GitHub, and a repository
#: on GitHub Enterprise Server isn't on github.com.
GITHUB_API = "https://api.github.com"
#: The REST API version these lookups were written against, as the adapter's.
API_VERSION = "2022-11-28"

#: GitHub's ``type`` for the two kinds of account a repository can belong to.
#: A person's account is ``User``, a machine account made for an agent
#: included; a GitHub App, such as Dependabot, opens pull requests as a ``Bot``.
ORGANIZATION = "Organization"
PERSON = "User"

#: The hosts an ``origin`` on github.com names. ``ssh.github.com`` is where
#: GitHub serves SSH on port 443, for networks that close port 22.
_GITHUB_HOSTS = frozenset({"github.com", "ssh.github.com"})
#: The transports git names in a remote's URL. Any of them to github.com names
#: a repository there, whether or not GitHub still serves it that way.
_GIT_SCHEMES = frozenset({"https", "http", "ssh", "git", "git+ssh", "ssh+git"})
#: What an owner's or a repository's name may hold, as the adapter checks it:
#: the name becomes part of a URL, so no path segment can be smuggled in.
_NAME = re.compile(r"[A-Za-z0-9._-]+")
#: git's scp-like syntax, ``[user@]host:path``, which git reads as such only
#: when no slash comes before the first colon.
_SCP_LIKE = re.compile(r"(?:[^@/:]+@)?(?P<host>[^@/:]+):(?P<path>.+)")
#: ``git remote get-url`` only reads a config file; a run must not wait on it.
_GIT_TIMEOUT_SECONDS = 30
#: Passed when the run has no pull request, so there is nobody who opened one.
_NO_PULL_REQUEST: Any = object()

#: How many days a run that needs a key goes ahead without one in GitHub
#: Actions, the day its workflow file was first committed included: the trial
#: the terms of service, the README and the pricing page promise.
TRIAL_DAYS = 14
#: ``GITHUB_WORKFLOW_REF``: ``owner/name/.github/workflows/<file>@<ref>``.
#: GitHub runs workflow files from ``.github/workflows`` itself, and only ones
#: that end in ``.yml`` or ``.yaml``. The first ``@`` after that ending begins
#: the ref, which may hold an ``@`` of its own.
_WORKFLOW_REF = re.compile(
    r"(?P<owner>[^/]+)/(?P<name>[^/]+)/(?P<path>\.github/workflows/[^/]+?(?i:\.ya?ml))@.+"
)
#: A commit's id: SHA-1, or SHA-256 in a repository that uses it.
_COMMIT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
#: One link of a ``Link`` header: ``<url>; rel="last"``.
_LINK = re.compile(r'<(?P<url>[^<>]*)>\s*;\s*rel="(?P<rel>[^"]*)"')
#: A page number as a link gives it. Ten digits are more pages than any file's
#: history has, and keep ``int`` well inside the digits it will read.
_PAGE = re.compile(r"[1-9][0-9]{0,9}")

#: The environment variable a key is read from, in GitHub Actions as in any other CI.
KEY_VARIABLE = "HONEST_CI_LICENSE_KEY"
#: Where a key is bought.
PRICING_PAGE = "https://honest-ci.dev/pricing"
#: Where a trial key outside GitHub Actions, and a free key for a machine
#: account, are asked for.
SUPPORT_ADDRESS = "support@honest-ci.dev"


class Unclear(ValueError):
    """What the check reads doesn't say what a key depends on; the message says what is missing."""


@dataclass(frozen=True)
class Opener:
    """The account that opened a pull request."""

    login: str
    #: Its numeric id, when the payload gives one.
    id: int | None
    #: Whether it is a person's account (GitHub's type ``User``) rather than,
    #: say, a GitHub App's.
    person: bool


@dataclass(frozen=True)
class Ownership:
    """Whose repository a run is on, whether it is public, and who opened its pull request."""

    #: Whether anyone can see the repository. A private one isn't public, and
    #: neither is an internal one, which only an enterprise's members can see.
    public: bool
    #: The owner's numeric id, which a key for it names as its ``org_id``.
    owner_id: int
    owner: str
    #: Whether an organization owns the repository; otherwise a personal account does.
    organization: bool
    #: Who opened the pull request, when the run is on one.
    opener: Opener | None = None


@dataclass(frozen=True)
class KeyDue:
    """A run that needs a license key, and whose key it needs."""

    #: The id a key has to name as its ``org_id`` to let the gates run here.
    owner_id: int
    owner: str
    #: Whether a personal account owns the repository, so that the key is due
    #: because someone else opened the pull request. A machine account made for
    #: an agent counts as someone else, and is what a free key is for.
    personal: bool


@dataclass(frozen=True)
class Trial:
    """The trial of a run that needs a key and has none: counted from which file, and since when."""

    #: The workflow file the run is a step in, such as
    #: ``.github/workflows/honest-ci.yml``. The trial counts from its first commit.
    workflow: str
    #: The day, in UTC, that commit was made.
    started: date

    @property
    def works_until(self) -> date:
        """The trial's last day."""

        return self.started + timedelta(days=TRIAL_DAYS - 1)

    def days_left(self, today: date) -> int:
        """How many days the trial has left, ``today`` included; 0 once it is over."""

        # Never more than a whole trial: a commit's date is whatever the
        # committer's clock said, and a clock can be ahead.
        return max(0, min(TRIAL_DAYS, (self.works_until - today).days + 1))

    def over(self, today: date) -> bool:
        """Whether the trial ended before ``today``."""

        return today > self.works_until


def in_ci(environ: Mapping[str, str]) -> bool:
    """Whether this process runs in CI, as the variables CI systems set say.

    Outside CI, such as on someone's own computer, a key is never due: that is
    a person using honest-ci. A CI job that hides these variables to go
    unchecked is getting around the check, which the license forbids, and
    nothing here tries to catch it.
    """

    if environ.get("GITHUB_ACTIONS") == "true":
        return True
    # GitHub Actions, GitLab CI, CircleCI, Buildkite, Travis, Bitbucket
    # Pipelines and Drone all set it, to ``true`` or to ``1``.
    ci = environ.get("CI", "").strip().lower()
    if ci and ci not in {"false", "0"}:
        return True
    return any(environ.get(name, "").strip() for name in CI_VARIABLES)


def key_due(ownership: Ownership) -> KeyDue | None:
    """Whether a run on this repository needs a key, and whose; ``None`` when none is due.

    A public repository never needs one. A repository that isn't public does
    when an organization owns it, and, when a personal account does, on a pull
    request another person opened. The owner's own pull requests, and the ones
    a GitHub App such as Dependabot opens, are one person using honest-ci.
    """

    if ownership.public:
        return None
    if ownership.organization:
        return KeyDue(owner_id=ownership.owner_id, owner=ownership.owner, personal=False)
    opener = ownership.opener
    if opener is None or not opener.person or _opened_by_owner(opener, ownership):
        return None
    return KeyDue(owner_id=ownership.owner_id, owner=ownership.owner, personal=True)


def _opened_by_owner(opener: Opener, ownership: Ownership) -> bool:
    # By id when there is one, as ``actions.py`` tells repositories apart: a
    # rename changes the login, not the id. Logins are case-insensitive.
    if opener.id is not None:
        return opener.id == ownership.owner_id
    return opener.login.lower() == ownership.owner.lower()


def read_ownership(repository: Any, opener: Any = _NO_PULL_REQUEST) -> Ownership:
    """``repository``, and the pull request's ``opener``, as GitHub describes them.

    The objects are the same in an event, in a pull request (``base.repo`` and
    ``user``) and in ``GET /repos``, so each is read the same way. ``opener``
    is left out when the run has no pull request. Raises ``Unclear``, in words
    that follow the name of what was read, when something the check needs is
    missing.
    """

    if not isinstance(repository, dict):
        raise Unclear("doesn't describe its repository")
    owner_id, owner, organization = _owner(repository.get("owner"), "its repository's owner")
    return Ownership(
        public=_public(repository),
        owner_id=owner_id,
        owner=owner,
        organization=organization,
        opener=None if opener is _NO_PULL_REQUEST else _opener(opener),
    )


def _public(repository: Mapping[str, Any]) -> bool:
    private, visibility = repository.get("private"), repository.get("visibility")
    if not isinstance(visibility, str):
        visibility = None
    # Either one saying so is enough: an internal repository is private to
    # everyone outside the enterprise, and ``private`` says so of it too.
    if private is True or visibility in {"private", "internal"}:
        return False
    if private is False or visibility == "public":
        return True
    raise Unclear("doesn't say whether its repository is public")


def _owner(value: Any, whose: str) -> tuple[int, str, bool]:
    """The id, the login and whether it is an organization, of the account ``value`` describes."""

    identifier, login, kind = _account(value, whose)
    if identifier is None:
        raise Unclear(f"has no id for {whose}")
    if kind not in {ORGANIZATION, PERSON}:
        raise Unclear(
            f"says {whose} is of type {_shown(kind)}: neither an organization nor a person"
        )
    return identifier, login, kind == ORGANIZATION


def _opener(value: Any) -> Opener:
    identifier, login, kind = _account(value, "the pull request's author")
    return Opener(login=login, id=identifier, person=kind == PERSON)


def _account(value: Any, whose: str) -> tuple[int | None, str, str]:
    if not isinstance(value, dict):
        raise Unclear(f"doesn't describe {whose}")
    identifier, login, kind = value.get("id"), value.get("login"), value.get("type")
    if not isinstance(login, str) or not login:
        raise Unclear(f"has no login for {whose}")
    if not isinstance(kind, str) or not kind:
        raise Unclear(f"has no type for {whose}")
    # ``type`` rather than ``isinstance``: JSON's ``true`` is a ``bool``, which
    # Python counts as an ``int``.
    return (identifier if type(identifier) is int and identifier > 0 else None), login, kind


def _shown(value: str) -> str:
    """``value`` as a log line quotes it, cut short: it is whatever the payload holds."""

    shown = json.dumps(value)
    return shown if len(shown) <= 40 else shown[:37] + "..."


# ── where the run is ─────────────────────────────────────────────────────────


def key_due_on_pull_request(
    environ: Mapping[str, str],
    actions: ActionsContext | None,
    platform: ForgeAdapter,
    change: ChangeRequest,
) -> KeyDue | None:
    """Whether ``run-pr`` needs a key for ``change``, and whose.

    The pull request ``run-pr`` fetched says, from the platform's own answer:
    whose repository it merges into, and who opened it. In Actions too, where
    the event describes what started the workflow, not the pull request
    ``run-pr`` was given: a workflow started by hand, by a comment or on a
    schedule, or one in another repository, can run it on any pull request
    its token can read. Of the event, only whether the run is a fork's counts.
    """

    if not in_ci(environ):
        return None
    if actions is not None and _from_a_fork(actions):
        return None
    try:
        ownership = platform.ownership(change)
    except Unclear as exc:
        _unclear(f"the pull request {exc}")
        return None
    if ownership is None:
        _unclear("the platform didn't say whose repository the pull request is on")
        return None
    return key_due(ownership)


def key_due_on_workspace(
    environ: Mapping[str, str],
    actions: ActionsContext | None,
    workspace: Path,
    settings: Settings,
    *,
    api: str = GITHUB_API,
) -> KeyDue | None:
    """Whether ``verify`` or ``resume`` needs a key in ``workspace``, and whose.

    In Actions, the event says. Elsewhere there is no pull request to read, so
    the repository is the github.com one ``origin`` names, and a personal
    account's never needs a key: nothing says who opened a pull request.
    ``api`` is where GitHub's REST API is: github.com's, but in a test.
    """

    if not in_ci(environ):
        return None
    if actions is not None:
        return _key_due_in_actions(actions)
    try:
        repository = origin_repository(workspace)
        if repository is None:
            return None
        # The token goes only to the API it is for: a token for a GitHub
        # Enterprise Server must not reach github.com.
        token = settings.github_token if _same_api(settings.github_base_url, api) else None
        ownership = _look_up(
            *repository, api=api, token=token, timeout_seconds=settings.forge_timeout_seconds
        )
    except Unclear as exc:
        _unclear(str(exc))
        return None
    return key_due(ownership)


def _key_due_in_actions(actions: ActionsContext) -> KeyDue | None:
    """Whether ``verify`` or ``resume`` in GitHub Actions needs a key, read off the event.

    It reads the file ``actions.py`` reads to tell a fork's pull request, and
    asks GitHub nothing. A run whose event has no pull request, such as one
    started by hand, says nothing of who opened one, so a personal account's
    repository needs no key there.
    """

    event = f"{actions.event_name or 'workflow'} event"
    payload = actions.event_payload()
    if payload is None:
        _unclear(f"the {event} couldn't be read")
        return None
    if _from_a_fork(actions):
        return None
    try:
        if "pull_request" in payload:
            pull = payload["pull_request"]
            opener = pull.get("user") if isinstance(pull, dict) else None
            ownership = read_ownership(payload.get("repository"), opener)
        else:
            ownership = read_ownership(payload.get("repository"))
    except Unclear as exc:
        _unclear(f"the {event} {exc}")
        return None
    return key_due(ownership)


def _from_a_fork(actions: ActionsContext) -> bool:
    """Whether the run is for a pull request from a fork, which no command asks for a key.

    Such a run gets none of the repository's secrets, so it couldn't have the
    key, and can't write a verdict back either: it runs read-only, as it did
    before there were keys. So does a run whose event can't rule a fork out.
    """

    return actions.read_only_reason() is not None


def origin_repository(workspace: Path) -> tuple[str, str] | None:
    """The owner and name of the github.com repository ``origin`` names in ``workspace``.

    ``None`` when there is none to name: no repository, no ``origin``, or one
    on another host. None of those is a repository this edition can find on
    github.com, so none is checked. Raises ``Unclear`` when git can't be run
    to ask. The URL itself is never repeated: a CI system may have put a
    credential in it.
    """

    try:
        done = subprocess.run(
            ["git", "-C", str(workspace), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Unclear(f"git couldn't be run to read origin ({exc})") from exc
    if done.returncode != 0:
        return None
    return github_repository(done.stdout.strip())


def github_repository(url: str) -> tuple[str, str] | None:
    """The owner and name of the github.com repository ``url`` names, or ``None``.

    Read the ways a remote names one: a URL such as
    ``https://github.com/owner/name`` or ``ssh://git@github.com/owner/name``,
    and git's short form for SSH, ``git@github.com:owner/name``, each with
    ``.git`` after it or not. Anything else -- another platform, GitHub
    Enterprise Server, a path on disk -- isn't a repository on github.com.
    """

    if "://" in url:
        try:
            parts = urllib.parse.urlsplit(url)
            # A port that isn't a number makes the URL unreadable, not github.com's.
            _ = parts.port
        except ValueError:
            return None
        if parts.scheme.lower() not in _GIT_SCHEMES or parts.hostname not in _GITHUB_HOSTS:
            return None
        if parts.query or parts.fragment:
            return None
        path = parts.path
    else:
        match = _SCP_LIKE.fullmatch(url)
        if match is None or match["host"].lower() not in _GITHUB_HOSTS:
            return None
        path = match["path"]
    owner, slash, name = path.strip("/").removesuffix(".git").partition("/")
    if not slash or not (_is_name(owner) and _is_name(name)):
        return None
    return owner, name


def _is_name(part: str) -> bool:
    """Whether ``part`` can be an owner's or a repository's name in a URL's path."""

    return bool(_NAME.fullmatch(part)) and part not in {".", ".."}


def _look_up(
    owner: str, name: str, *, api: str, token: str | None, timeout_seconds: float
) -> Ownership:
    """The repository ``owner/name`` as GitHub describes it to this run.

    ``GET /repos`` answers 200 with whether the repository is private and who
    owns it. To a run whose token can't see the repository, or that has no
    token, it answers 404, which it never does for a public repository; then
    ``GET /users``, which anyone may ask, says who the owner is.
    """

    path = f"/repos/{owner}/{name}"
    status, payload, _ = _get(api, path, token=token, timeout_seconds=timeout_seconds)
    if status == 200:
        try:
            return read_ownership(payload)
        except Unclear as exc:
            raise Unclear(f"GitHub's answer to GET {path} {exc}") from exc
    if status != 404:
        raise Unclear(f"GitHub answered {status} to GET {path}")
    path = f"/users/{owner}"
    # Without the token: anyone may read an account, and a token that can't
    # see the repository has nothing to add here.
    status, payload, _ = _get(api, path, token=None, timeout_seconds=timeout_seconds)
    if status != 200:
        raise Unclear(f"GitHub answered {status} to GET {path}")
    try:
        owner_id, login, organization = _owner(payload, "the account")
    except Unclear as exc:
        raise Unclear(f"GitHub's answer to GET {path} {exc}") from exc
    return Ownership(public=False, owner_id=owner_id, owner=login, organization=organization)


def _get(
    api: str,
    path: str,
    *,
    token: str | None,
    timeout_seconds: float,
    params: Mapping[str, str] | None = None,
) -> tuple[int, Any, str]:
    """GET ``path`` of the REST API at ``api``, with ``params`` as its query.

    Returns the status, the JSON that came with a 200, and the ``Link`` header,
    which says where a list's other pages are. Through
    ``open_validated_http_request``, as every outbound request is, so that the
    request and any redirect stay under ``api``, and the token, when there is
    one, stays in a header. Raises ``Unclear`` when GitHub can't be reached,
    or answers a 200 that isn't JSON.
    """

    headers = {
        "Accept": "application/vnd.github+json",
        # GitHub rejects requests without a User-Agent.
        "User-Agent": "honest-ci",
        "X-GitHub-Api-Version": API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{api.rstrip('/')}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with open_validated_http_request(
            request, allowed_base=api, timeout_seconds=timeout_seconds, allow_loopback_http=True
        ) as response:
            status, raw = response.status, response.read()
            link = response.headers.get("Link") or ""
    except urllib.error.HTTPError as exc:
        exc.close()
        return exc.code, None, ""
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise Unclear(f"GitHub couldn't be reached for GET {path} ({exc})") from exc
    if status != 200:
        return status, None, link
    try:
        return status, json.loads(raw), link
    except ValueError as exc:
        raise Unclear(f"GitHub's answer to GET {path} isn't JSON") from exc


def _same_api(configured: str, api: str) -> bool:
    return configured.rstrip("/").lower() == api.rstrip("/").lower()


def _unclear(reason: str) -> None:
    """Say in the log that the check couldn't tell; the caller then asks for no key.

    The run goes on as it would without the check: a check that fails in a
    team's CI must not stop every merge.
    """

    log.warning(
        "The license check couldn't tell whether this run needs a license key, so it asks "
        "for none: %s.",
        redact_text(reason),
    )


# ── the trial ────────────────────────────────────────────────────────────────


def workflow_trial(environ: Mapping[str, str], settings: Settings) -> Trial | None:
    """The trial of a run in GitHub Actions that needs a key and has none.

    It starts the day the workflow file the run is a step in was first
    committed, as GitHub lists the commits that touched the file, looking back
    from the one the run took it from. The check sees no further than this
    repository and keeps nothing between runs, so that is as near as it comes
    to the day honest-ci first ran here. The terms count the trial per
    organization or account: starting it again, under another file or in
    another repository, is getting around the check, which the license
    forbids, and nothing here tries to catch it.

    ``None`` when it can't tell, which the log says; the run then goes ahead.
    Outside Actions there is no workflow file to ask about, so no trial: a
    trial key comes by email.
    """

    try:
        owner, name, workflow = _workflow_file(environ)
        commit = environ.get("GITHUB_WORKFLOW_SHA", "")
        if not _COMMIT_ID.fullmatch(commit):
            raise Unclear("GITHUB_WORKFLOW_SHA doesn't name the commit the workflow file is from")
        api = environ.get("GITHUB_API_URL", "")
        if not api:
            raise Unclear("GITHUB_API_URL doesn't say where GitHub's API is")
        # The token goes only to the API it was given for, as for ``origin``.
        token = settings.github_token if _same_api(settings.github_base_url, api) else None
        started = _first_committed(
            owner,
            name,
            workflow,
            commit,
            api=api,
            token=token,
            timeout_seconds=settings.forge_timeout_seconds,
        )
    except Unclear as exc:
        _no_trial_start(str(exc))
        return None
    return Trial(workflow=workflow, started=started)


def _workflow_file(environ: Mapping[str, str]) -> tuple[str, str, str]:
    """The owner, the repository and the path of the workflow file the run is a step in."""

    match = _WORKFLOW_REF.fullmatch(environ.get("GITHUB_WORKFLOW_REF", ""))
    if match is None or not (_is_name(match["owner"]) and _is_name(match["name"])):
        raise Unclear("GITHUB_WORKFLOW_REF doesn't name a workflow file in .github/workflows")
    return match["owner"], match["name"], match["path"]


def _first_committed(
    owner: str,
    name: str,
    path: str,
    commit: str,
    *,
    api: str,
    token: str | None,
    timeout_seconds: float,
) -> date:
    """The day, in UTC, of the first commit that touched ``path``, looking back from ``commit``.

    The commits endpoint lists them newest first. Asked for one a page, it
    says in ``Link`` which page is the last, and that page holds the first
    commit: two requests, however long the file's history. From ``commit``
    rather than the default branch: on the pull request that adds the
    workflow file, the default branch doesn't have it yet.
    """

    endpoint = f"/repos/{owner}/{name}/commits"
    query = {"path": path, "sha": commit, "per_page": "1"}
    commits, link = _commits(api, endpoint, query, token=token, timeout_seconds=timeout_seconds)
    last = _last_page(link, endpoint)
    if last > 1:
        commits, _ = _commits(
            api,
            endpoint,
            {**query, "page": str(last)},
            token=token,
            timeout_seconds=timeout_seconds,
        )
    if not commits:
        raise Unclear(f"GitHub's answer to GET {endpoint} lists no commit that touched {path}")
    return _committed_on(commits[-1], endpoint)


def _commits(
    api: str,
    endpoint: str,
    query: Mapping[str, str],
    *,
    token: str | None,
    timeout_seconds: float,
) -> tuple[list[Any], str]:
    """One page of the commits endpoint, and its ``Link`` header."""

    status, payload, link = _get(
        api, endpoint, token=token, timeout_seconds=timeout_seconds, params=query
    )
    if status != 200:
        raise Unclear(f"GitHub answered {status} to GET {endpoint}")
    if not isinstance(payload, list):
        raise Unclear(f"GitHub's answer to GET {endpoint} isn't a list of commits")
    return payload, link


def _last_page(link: str, endpoint: str) -> int:
    """The number of the last page, as ``link`` names it; 1 when it names no other page.

    GitHub names the last page whenever there is more than one, and on the
    commits endpoint even when there isn't. Only the number is taken: the
    request for that page is this module's own, to the endpoint it asked, so
    a link can't lead it anywhere else.
    """

    urls: dict[str, str] = {}
    for match in _LINK.finditer(link):
        for relation in match["rel"].split():
            urls.setdefault(relation, match["url"])
    if "last" not in urls:
        # A next page with no last one leaves the first commit out of reach:
        # the first page holds the newest.
        if "next" in urls:
            raise Unclear(
                f"GitHub's answer to GET {endpoint} has more pages but doesn't say which is the last"
            )
        return 1
    try:
        pages = urllib.parse.parse_qs(urllib.parse.urlsplit(urls["last"]).query).get("page", [])
    except ValueError:
        pages = []
    if len(pages) != 1 or not _PAGE.fullmatch(pages[0]):
        raise Unclear(f"GitHub's answer to GET {endpoint} doesn't say which page is the last")
    return int(pages[0])


def _committed_on(commit: Any, endpoint: str) -> date:
    """The day, in UTC, ``commit`` was made, as the commits endpoint describes it.

    The committer's date, not the author's: when the commit was made, rather
    than when its change was first written, which for a change picked from
    elsewhere can be long before.
    """

    details = commit.get("commit") if isinstance(commit, dict) else None
    committer = details.get("committer") if isinstance(details, dict) else None
    value = committer.get("date") if isinstance(committer, dict) else None
    if not isinstance(value, str):
        raise Unclear(f"GitHub's answer to GET {endpoint} doesn't date the commit")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise Unclear(
            f"GitHub's answer to GET {endpoint} dates the commit {_shown(value)}, "
            "which isn't a date"
        ) from exc
    # GitHub writes its dates in UTC, with a ``Z``; one without a zone is read so too.
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).date()


def _no_trial_start(reason: str) -> None:
    """Say in the log that the trial's first day couldn't be read; the run then goes ahead.

    For ``_unclear``'s reason: a check that fails in a team's CI must not stop
    every merge.
    """

    log.warning(
        "The license check couldn't tell when this repository's %d-day trial started, so it "
        "lets the run go ahead without a license key: %s.",
        TRIAL_DAYS,
        redact_text(reason),
    )


# ── whether the gates run ────────────────────────────────────────────────────

#: What a run the check stops says first, after why: it judged nothing.
_NOT_A_VERDICT = "This is not a verdict: honest-ci checked nothing in this change."
#: Said wherever a key is due because someone else opened the pull request: a
#: machine account made for an agent is someone else to the check, and the
#: terms give its owner a key for it free of charge.
_MACHINE_ACCOUNT = (
    "If a machine account you use for an AI agent opened this pull request, email "
    f"{SUPPORT_ADDRESS} for a free key."
)
#: How a workflow hands the secret to the step, in the workflow's own syntax.
_SECRET_EXPRESSION = f"${{{{ secrets.{KEY_VARIABLE} }}}}"
#: ``GITHUB_ACTOR`` in a run Dependabot started. GitHub gives such a run
#: Dependabot's secrets and none of the repository's or the organization's
#: Actions secrets, so a key saved only there never reaches it.
#: https://docs.github.com/en/code-security/dependabot/troubleshooting-dependabot/troubleshooting-dependabot-on-github-actions
DEPENDABOT_ACTOR = "dependabot[bot]"


@dataclass(frozen=True)
class Cleared:
    """A run whose gates the license check lets run, and what it says about the run's license."""

    #: For a run on its trial, or on a key past its expiry date, how long it
    #: has left, in one paragraph: the run's output says it, and its pull
    #: request comment ends with it. ``None`` when there is nothing to say, as
    #: on a repository that needs no key.
    note: str | None = None


@dataclass(frozen=True)
class Stopped:
    """A run the license check stops before any gate: whose key it needs, and what is wrong.

    What it says stands where a verdict would: in the run's output, and on a
    pull request in a comment and an error status. None of it is about the
    change, which nothing checked.
    """

    due: KeyDue
    #: Whether the run is in GitHub Actions, where a key is kept as a secret.
    in_actions: bool
    #: Why the key the run was given doesn't let the gates run, in
    #: ``licensing``'s words, which never repeat the key. ``None`` when it was
    #: given none.
    refusal: str | None = None
    #: The trial that is over, when the run had no key in GitHub Actions.
    trial: Trial | None = None
    #: Whether Dependabot started the run, which then had Dependabot's secrets only.
    by_dependabot: bool = False

    @property
    def status(self) -> str:
        """That no gate ran, and why, in one line a commit status can carry."""

        if self.refusal is not None:
            return f"No gate ran. License key refused: {_first_clause(self.refusal)}"
        if self.trial is not None:
            return (
                f"No gate ran. No license key, and the {TRIAL_DAYS}-day trial ended on "
                f"{self.trial.works_until.isoformat()}"
            )
        return "No gate ran. No license key, and this repository needs one in CI"

    @property
    def explanation(self) -> list[str]:
        """What the run says in place of a verdict, a line each: that no gate ran, why, and what to do.

        Markdown that reads as plain text, so that the output can print what
        a pull request comment and a job summary show.
        """

        return [f"{self.status}. {_NOT_A_VERDICT}", "", *self._what_to_do()]

    @property
    def comment(self) -> str:
        """The comment that stands on the pull request in place of a verdict.

        It starts as ``render_forge_comment``'s does, so that the next run,
        with a key or without, writes over it rather than adding another.
        """

        lines = [
            VERDICT_MARKER,
            "## honest-ci · not run",
            "",
            f"**{self.status}.** {_NOT_A_VERDICT}",
            "",
            *self._what_to_do(),
        ]
        return redact_text("\n".join(lines) + "\n")

    def _what_to_do(self) -> list[str]:
        return [
            f"{_why_due(self.due)} {self._cause()}",
            "",
            f"- Get a key at {PRICING_PAGE}. It comes by email, and a renewal comes with a "
            "new one.",
            f"- Set it: {self._setting()}",
            f"- Or stop running honest-ci here: {self._removing()}",
        ]

    def _cause(self) -> str:
        if self.refusal is not None:
            return f"The key in `{KEY_VARIABLE}` was refused because {self.refusal}."
        if self.trial is not None:
            # A workflow file that is older than its honest-ci step dates the
            # trial from before honest-ci ran, possibly over before it began:
            # the check sees when the file began, not when the step did.
            return (
                f"Without one it ran on a {TRIAL_DAYS}-day trial, which counts from the day "
                f"`{self.trial.workflow}` was first committed: "
                f"{self.trial.started.isoformat()}. If honest-ci was added to a workflow file "
                "that already existed, the trial started before honest-ci did: email "
                f"{SUPPORT_ADDRESS} for a trial key."
            )
        return (
            f"Outside GitHub Actions there is no automatic trial: email {SUPPORT_ADDRESS} for "
            f"a {TRIAL_DAYS}-day trial key."
        )

    def _setting(self) -> str:
        if not self.in_actions:
            return (
                f"in the CI job that runs honest-ci, set the environment variable "
                f"`{KEY_VARIABLE}` to the key, the way the job gets its other secrets."
            )
        # An organization's secret reaches the repositories it is shared with,
        # except on GitHub Free, which passes it to no private one: a run there
        # reads it as empty, as if no key were set. The check asks GitHub
        # about the repository only, never about its owner's plan, so it names
        # both places. A personal account has no secrets of its own, only its
        # repositories do.
        # https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets
        secret = (
            f"a secret of this repository named `{KEY_VARIABLE}` (a personal account has no "
            "account-wide secrets)"
            if self.due.personal
            else (
                f"an organization secret named `{KEY_VARIABLE}` on GitHub Team or Enterprise, or "
                "as a secret of this repository on GitHub Free, which doesn't pass organization "
                "secrets to private repositories"
            )
        )
        setting = (
            f"save the key as {secret}. The honest-ci step passes it on as "
            f"`license-key: {_SECRET_EXPRESSION}` under its `with:`, as the workflow "
            "`honest-ci init` writes does."
        )
        if self.by_dependabot:
            setting += (
                " Dependabot started this run, and GitHub gives such runs Dependabot secrets "
                "only: save the key as a Dependabot secret too, at the same level."
            )
        return setting

    def _removing(self) -> str:
        step = (
            f"its step from `{self.trial.workflow}`"
            if self.trial is not None
            else "the step that runs it"
        )
        return (
            f"remove {step}, and its status from the branch's required status checks if it is one."
        )


def check_license(
    due: KeyDue | None,
    environ: Mapping[str, str],
    settings: Settings,
    *,
    today: date,
    public_key: PublicKey | None = None,
) -> Cleared | Stopped:
    """Whether a run's gates run, given the key ``due`` and the one in ``HONEST_CI_LICENSE_KEY``.

    Where no key is due, the variable isn't read. Where one is, a key given
    is checked, during the trial too: one that doesn't work is better found
    on the first day than on the fifteenth. An empty one is none, as a
    workflow passes a secret nobody set. With none, a run in GitHub Actions
    goes ahead for its trial, and one that can't tell when the trial started
    goes ahead too, which the log says; anywhere else, or once the trial is
    over, it stops. ``today`` is the day in UTC; ``public_key`` is the one a
    key is verified against, the shipped one but in a test.
    """

    if due is None:
        return Cleared()
    in_actions = environ.get("GITHUB_ACTIONS") == "true"
    by_dependabot = in_actions and environ.get("GITHUB_ACTOR") == DEPENDABOT_ACTOR
    text = environ.get(KEY_VARIABLE, "").strip()
    if text:
        try:
            key = check_key(
                text, owner_id=due.owner_id, today=today, public_key=public_key or PUBLIC_KEY
            )
        except LicenseKeyError as exc:
            return Stopped(
                due=due, in_actions=in_actions, refusal=str(exc), by_dependabot=by_dependabot
            )
        # A trial key stops the day after its expiry date, so every run before
        # then says how long is left, as a run on the trial in Actions does.
        if key.is_trial:
            return Cleared(note=_trial_key_note(key, today))
        return Cleared(note=_grace_note(key, today) if key.in_grace(today) else None)
    if not in_actions:
        return Stopped(due=due, in_actions=False)
    trial = workflow_trial(environ, settings)
    if trial is None:
        return Cleared()
    if trial.over(today):
        return Stopped(due=due, in_actions=True, trial=trial, by_dependabot=by_dependabot)
    return Cleared(note=_trial_note(trial, due, today))


def describe_key(
    environ: Mapping[str, str], *, today: date, public_key: PublicKey | None = None
) -> str:
    """The key in ``HONEST_CI_LICENSE_KEY`` in one line, for ``doctor``: whose, until when, or why not.

    It says what a run would make of the key wherever one is due, and nothing
    about whether one is: that turns on the repository and on who opened the
    pull request, which a run looks up and ``doctor`` doesn't. For the same
    reason a key for another organization reads here as it is. As in a
    refusal, the key itself is never repeated.
    """

    text = environ.get(KEY_VARIABLE, "").strip()
    if not text:
        return f"none ({KEY_VARIABLE} is not set)"
    try:
        key = read_key(text, public_key=public_key or PUBLIC_KEY)
    except LicenseKeyError as exc:
        return f"refused: {exc}"
    whose = f"{key.id} for {key.org} (id {key.org_id}) · plan {key.plan}"
    if key.seats:
        whose += f", {key.seats} seat{'' if key.seats == 1 else 's'}"
    refused = expiry_refusal(key, today)
    if refused is not None:
        return f"refused: {refused} · {whose}"
    expired = key.expires.isoformat()
    if key.in_grace(today):
        return (
            f"signature verified · {whose} · expired on {expired}, works through "
            f"{key.works_until.isoformat()}: {_days(_days_left(key, today))} left"
        )
    return f"signature verified · {whose} · expires on {expired}"


def _why_due(due: KeyDue) -> str:
    if not due.personal:
        return (
            f"This repository belongs to the organization {due.owner} and isn't public, so "
            "honest-ci needs a license key here when it runs in CI."
        )
    return (
        f"This repository belongs to {due.owner} and isn't public, and someone other than "
        f"{due.owner} opened this pull request, so honest-ci needs a license key here when it "
        f"runs in CI. {_MACHINE_ACCOUNT}"
    )


def _trial_note(trial: Trial, due: KeyDue, today: date) -> str:
    note = (
        f"honest-ci is on its {TRIAL_DAYS}-day trial here: {_days(trial.days_left(today))} "
        f"left, through {trial.works_until.isoformat()}, counted from "
        f"{trial.started.isoformat()}, when `{trial.workflow}` was first committed. After "
        f"that it needs a license key: {PRICING_PAGE}."
    )
    return f"{note} {_MACHINE_ACCOUNT}" if due.personal else note


def _trial_key_note(key: LicenseKey, today: date) -> str:
    return (
        f"honest-ci is on a trial key here: {_days(_days_left(key, today))} left, through "
        f"{key.works_until.isoformat()}. After that it needs a subscription, which comes with "
        f"a new key for `{KEY_VARIABLE}`: {PRICING_PAGE}."
    )


def _grace_note(key: LicenseKey, today: date) -> str:
    return (
        f"The license key for {key.org} expired on {key.expires.isoformat()} and keeps working "
        f"through {key.works_until.isoformat()}: {_days(_days_left(key, today))} left. A "
        f"renewal comes with a new key for `{KEY_VARIABLE}`; to renew, see {PRICING_PAGE}."
    )


def _days_left(key: LicenseKey, today: date) -> int:
    """How many more days ``key`` works, ``today`` included: a trial key's, or the grace days'."""

    return (key.works_until - today).days + 1


def _days(count: int) -> str:
    return f"{count} day" if count == 1 else f"{count} days"


def _first_clause(refusal: str) -> str:
    """What is wrong with a key, without what ``licensing`` adds after it: for a status line.

    A refusal says first what is wrong, then, after a colon, a semicolon or
    ", and", what may have caused it or what it was compared with. A commit
    status has room for the first part only: the adapter cuts a description
    at 140 characters, which for an organization with a long name would cut
    the id it was compared with in two.
    """

    return re.split(r"[:;] |, and ", refusal, maxsplit=1)[0]


__all__ = [
    "CI_VARIABLES",
    "DEPENDABOT_ACTOR",
    "GITHUB_API",
    "KEY_VARIABLE",
    "ORGANIZATION",
    "PERSON",
    "PRICING_PAGE",
    "SUPPORT_ADDRESS",
    "TRIAL_DAYS",
    "Cleared",
    "KeyDue",
    "Opener",
    "Ownership",
    "Stopped",
    "Trial",
    "Unclear",
    "check_license",
    "describe_key",
    "github_repository",
    "in_ci",
    "key_due",
    "key_due_on_pull_request",
    "key_due_on_workspace",
    "origin_repository",
    "read_ownership",
    "workflow_trial",
]
