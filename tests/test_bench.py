"""The benchmark in bench/: who opened a pull request, how GitHub is read and
cached, how the groups are drawn, and the numbers written out.

GitHub itself is played by ``FakeGitHub``: a fixed set of pull requests
behind the same GraphQL queries and diff requests the benchmark makes, so the
real ``Source`` and ``Cache`` run over it unchanged.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from bench import cli, github, report, sample, source
from bench.agents import AGENTS, AMBIGUOUS, ASSISTED, BOT, HUMAN, identify, plainly_not_human
from bench.cache import Cache, Cached, cache_key
from bench.github import (
    DIFF_MEDIA_TYPE,
    MAX_ATTEMPTS,
    OVERSIZED,
    GitHub,
    GitHubError,
    PullDiff,
    Response,
    ResponseTooLarge,
    TransportError,
    urllib_send,
)
from bench.sample import (
    Config,
    Group,
    PullRequest,
    Scanned,
    interleave,
    match_controls,
    sample_agent,
    time_slices,
)
from bench.source import OfflineMiss, Source
from honest_ci.integrity import IntegrityCheck, IntegrityHit

JSON = dict[str, Any]

#: The fake credential the gitleaks config allows; it must never reach a file.
TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz012345"

# ── diffs ────────────────────────────────────────────────────────────────────

DELETES_A_TEST = """\
diff --git a/tests/test_refund.py b/tests/test_refund.py
--- a/tests/test_refund.py
+++ b/tests/test_refund.py
@@ -1,6 +1,2 @@
 def test_whole_amounts():
     assert refund(9) == 3
-
-
-def test_rounding():
-    assert refund(10) == 3.33
"""

SKIPS_A_TEST = """\
diff --git a/tests/test_refund.py b/tests/test_refund.py
--- a/tests/test_refund.py
+++ b/tests/test_refund.py
@@ -1,2 +1,3 @@
+@pytest.mark.skip(reason="flaky")
 def test_rounding():
     assert refund(10) == 3.33
"""

EDITS_CODE = """\
diff --git a/src/refund.py b/src/refund.py
--- a/src/refund.py
+++ b/src/refund.py
@@ -1,2 +1,2 @@
 def refund(amount):
-    return amount / 3
+    return round(amount / 3, 2)
"""

#: Three test-directory files deleted: one binary, which GitHub renders with
#: no lines; an empty package marker, which no runner collects; and one whose
#: lines are all there.
DELETES_TEST_FILES = """\
diff --git a/tests/test_totals.py b/tests/test_totals.py
deleted file mode 100644
index 3b18e51..0000000
Binary files a/tests/test_totals.py and /dev/null differ
diff --git a/tests/__init__.py b/tests/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/tests/test_old.py b/tests/test_old.py
deleted file mode 100644
index 5d3b1c2..0000000
--- a/tests/test_old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def test_old():
-    assert old() == 1
"""

# ── pull requests ────────────────────────────────────────────────────────────


def pull(
    number: int,
    created: str,
    *,
    repo: str = "acme/widgets",
    login: str = "alice",
    bot: bool = False,
    branch: str = "",
    body: str = "",
    #: ``(message, author)``, the author as ``Name <email>`` or a bare name.
    commits: Sequence[tuple[str, str]] = (),
    stars: int = 50,
    fork: bool = False,
    private: bool = False,
    merged: bool = False,
    diff: str | PullDiff = EDITS_CODE,
    found_by: Sequence[str] = (),
) -> JSON:
    """A pull request as the benchmark's full query returns it.

    Keys starting with ``_`` are ``FakeGitHub``'s: which agents' searches
    find it, and its diff.
    """

    return {
        "id": f"PR_{repo}#{number}",
        "url": f"https://github.com/{repo}/pull/{number}",
        "number": number,
        "createdAt": created,
        "mergedAt": "2026-07-03T00:00:00Z" if merged else None,
        "closedAt": "2026-07-03T00:00:00Z" if merged else None,
        "state": "MERGED" if merged else "OPEN",
        "isDraft": False,
        "headRefName": branch or f"change-{number}",
        "title": f"Change {number}",
        "body": body,
        "additions": 2,
        "deletions": 1,
        "changedFiles": 1,
        "author": {"__typename": "Bot" if bot else "User", "login": login},
        "repository": {
            "nameWithOwner": repo,
            "stargazerCount": stars,
            "isFork": fork,
            "isPrivate": private,
        },
        "commits": {
            "totalCount": len(commits),
            "nodes": [
                {"commit": {"message": message, "author": _commit_author(author)}}
                for message, author in commits
            ],
        },
        "_found_by": set(found_by),
        "_diff": diff,
    }


def _commit_author(author: str) -> dict[str, str]:
    name, _, email = author.partition(" <")
    return {"name": name, "email": email.removesuffix(">")}


def by_agent(name: str, number: int, created: str, **fields: Any) -> JSON:
    """A pull request the agent's GitHub App opened."""

    (app,) = AGENTS[name].apps
    return pull(number, created, login=app, bot=True, found_by=[name], **fields)


def identity_of(**fields: Any) -> tuple[str, tuple[str, ...]]:
    found = identify(pull(1, "2026-07-01T10:00:00Z", **fields))
    return found.group, found.via


def at(hour: int, minute: int = 0) -> str:
    return f"2026-07-01T{hour:02d}:{minute:02d}:00Z"


def moment(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 1, hour, minute, tzinfo=UTC)


def config(**fields: Any) -> Config:
    """Four hours of 1 July, in hour-long slices: the worlds below are laid out by the hour."""

    values: dict[str, Any] = {
        "since": moment(0),
        "until": moment(4),
        "per_group": 50,
        "slice_length": timedelta(hours=1),
    }
    return Config(**{**values, **fields})


# ── a GitHub to read from ────────────────────────────────────────────────────


class FakeGitHub:
    """The benchmark's GraphQL queries and diff requests, over fixed pull requests.

    Search understands what the benchmark's searches say: ``repo:``,
    ``created:`` bounds, ``sort:created-asc``/``-desc``, and an agent's search
    terms, which find the pull requests that list that agent in ``_found_by``.
    ``fail`` decides which queries fail as GitHub's timeouts do.
    """

    def __init__(
        self,
        pulls: Sequence[JSON],
        *,
        cap: int = source.SEARCH_CAP,
        fail: Callable[[str, Mapping[str, Any]], bool] | None = None,
    ) -> None:
        self.pulls = list(pulls)
        self.cap = cap
        self.fail = fail
        self.requests = 0
        self.searched: list[str] = []
        #: Each query's kind ("search", "searches", "nodes") and its attempts.
        self.asked: list[tuple[str, int]] = []
        #: How many searches each query of several carried.
        self.batches: list[int] = []
        #: How many pull requests each ``nodes`` lookup asked for.
        self.lookups: list[int] = []
        self.diffs: list[tuple[str, int]] = []

    def graphql(
        self, query: str, variables: Mapping[str, Any], *, attempts: int = MAX_ATTEMPTS
    ) -> JSON:
        self.requests += 1
        kind = "nodes" if "nodes(ids:" in query else "searches" if "$q0" in query else "search"
        self.asked.append((kind, attempts))
        if kind == "nodes":
            self.lookups.append(len(variables["ids"]))
        if kind == "searches":
            self.batches.append(sum(1 for name in variables if re.fullmatch(r"q\d+", name)))
        if self.fail is not None and self.fail(query, variables):
            raise GitHubError("GraphQL returned no data: timed out")
        if kind == "nodes":
            return {"nodes": [self._full(node_id) for node_id in variables["ids"]]}
        if kind == "searches":
            return {
                f"s{index}": self._search(variables[f"q{index}"], variables["first"], None)
                for index in range(self.batches[-1])
            }
        return {"search": self._search(variables["q"], variables["first"], variables["after"])}

    def pull_diff(self, repo: str, number: int) -> PullDiff:
        self.requests += 1
        self.diffs.append((repo, number))
        for item in self.pulls:
            if item["repository"]["nameWithOwner"] == repo and item["number"] == number:
                diff = item["_diff"]
                return diff if isinstance(diff, PullDiff) else PullDiff(diff)
        return PullDiff(None, "http_404")

    def _full(self, node_id: str) -> JSON | None:
        for item in self.pulls:
            if item["id"] == node_id and not item.get("_gone"):
                return {key: value for key, value in item.items() if not key.startswith("_")}
        return None

    def _search(self, query: str, first: int, after: str | None) -> JSON:
        self.searched.append(query)
        found = sorted(
            (item for item in self.pulls if _matches(item, query)),
            key=lambda item: str(item["createdAt"]),
            reverse="sort:created-desc" in query,
        )
        start = int(after or 0)
        stop = min(start + first, len(found), self.cap)
        fields = ("id", "url", "number", "createdAt", "headRefName", "author", "repository")
        return {
            "issueCount": len(found),
            "pageInfo": {"hasNextPage": stop < min(len(found), self.cap), "endCursor": str(stop)},
            "nodes": [{key: item[key] for key in fields} for item in found[start:stop]],
        }


def _matches(item: JSON, query: str) -> bool:
    repo = re.search(r"\brepo:(\S+)", query)
    if repo is not None:
        if item["repository"]["nameWithOwner"] != repo.group(1):
            return False
    elif not any(AGENTS[name].search in query for name in item["_found_by"]):
        return False
    if item.get("_ignores_dates"):
        return True
    created = str(item["createdAt"])
    for operator, value in re.findall(r"\bcreated:(>=|>|<)(\S+)", query):
        if not {">=": created >= value, ">": created > value, "<": created < value}[operator]:
            return False
    return True


def reader(world: FakeGitHub, root: Path) -> Source:
    return Source(Cache(root), world)


def quiet(_message: str) -> None:
    pass


# ── who opened it ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"login": "copilot-swe-agent", "bot": True}, ("copilot", ("app",))),
        ({"login": "chatgpt-codex-connector", "bot": True}, ("codex", ("app",))),
        ({"login": "claude", "bot": True}, ("claude", ("app",))),
        ({"login": "cursor", "bot": True}, ("cursor", ("app",))),
        ({"login": "devin-ai-integration", "bot": True}, ("devin", ("app",))),
        ({"login": "google-labs-jules", "bot": True}, ("jules", ("app",))),
        (
            {"body": "https://chatgpt.com/codex/tasks/task_e_0123456789abcdef"},
            ("codex", ("body",)),
        ),
        ({"branch": "codex/round-refunds"}, ("codex", ("branch",))),
        (
            {"body": "🤖 Generated with [Claude Code](https://claude.com/claude-code)"},
            ("claude", ("body",)),
        ),
        ({"body": "Session: https://claude.ai/code/session_0123abcd"}, ("claude", ("body",))),
        (
            {"branch": "claude/issue-7-20260701", "body": "Generated with Claude Code"},
            ("claude", ("body", "branch")),
        ),
        ({"body": "Fix\n<!-- CURSOR_AGENT_PR_BODY_END -->"}, ("cursor", ("body",))),
        ({"branch": "cursor/round-refunds-1a2b"}, ("cursor", ("branch",))),
    ],
)
def test_an_agent_is_recognised_by_its_own_signatures(
    fields: dict[str, Any], expected: tuple[str, tuple[str, ...]]
) -> None:
    assert identity_of(**fields) == expected


@pytest.mark.parametrize(
    ("fields", "agents"),
    [
        # A person working on the branch Copilot's agent opened, or Copilot in an editor.
        ({"branch": "copilot/round-refunds"}, ("copilot",)),
        (
            {"commits": [("Round refunds\n\nCo-Authored-By: Claude <noreply@anthropic.com>", "")]},
            ("claude",),
        ),
        ({"commits": [("Round refunds", "copilot-swe-agent[bot]")]}, ("copilot",)),
        (
            {"commits": [("Round refunds", "GitHub Copilot <1+Copilot@users.noreply.github.com>")]},
            ("copilot",),
        ),
        ({"body": "Started in https://app.devin.ai/sessions/0123abcd"}, ("devin",)),
        ({"branch": "devin/1751364000-round-refunds"}, ("devin",)),
        ({"body": "From https://jules.google.com/task/123456"}, ("jules",)),
        ({"branch": "jules-1234567890-round-refunds"}, ("jules",)),
        (
            {"commits": [("Co-authored-by: Cursor Agent <cursoragent@cursor.com>", "Alice")]},
            ("cursor",),
        ),
        # A person opening what Cursor's agent committed on a branch of their own.
        ({"commits": [("Round refunds", "Cursor Agent <cursoragent@cursor.com>")]}, ("cursor",)),
    ],
)
def test_a_weaker_trace_puts_a_pull_request_in_neither_group(
    fields: dict[str, Any], agents: tuple[str, ...]
) -> None:
    assert identity_of(**fields) == (ASSISTED, agents)


def test_two_agents_opening_one_pull_request_is_ambiguous() -> None:
    assert identity_of(branch="codex/round-refunds", body="Generated with Claude Code") == (
        AMBIGUOUS,
        ("claude", "codex"),
    )


def test_the_app_as_author_outweighs_what_the_body_quotes() -> None:
    assert identity_of(login="copilot-swe-agent", bot=True, body="Generated with Claude Code") == (
        "copilot",
        ("app",),
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"login": "dependabot", "bot": True},
        # A workflow that ran an agent opens as github-actions, not as the agent.
        {"login": "github-actions", "bot": True, "body": "Generated with Claude Code"},
        {"login": "k8s-ci-robot"},
        {"login": "renovate-bot"},
        {"login": "release_bot"},
    ],
)
def test_other_bots_are_in_neither_group(fields: dict[str, Any]) -> None:
    assert identity_of(**fields) == (BOT, ())


@pytest.mark.parametrize("login", ["alice", "abbot", "botanist", "robotics-lab"])
def test_a_person_is_whoever_has_no_agent_or_bot_signature(login: str) -> None:
    assert identity_of(login=login, body="Rounds refunds to cents.") == (HUMAN, ())


@pytest.mark.parametrize(
    ("fields", "ruled_out"),
    [
        ({"login": "dependabot", "bot": True}, True),
        ({"login": "renovate-bot"}, True),
        ({"branch": "codex/round-refunds"}, True),
        ({"branch": "copilot/round-refunds"}, True),
        ({"branch": "jules-1234567890-round-refunds"}, True),
        # Only the full pull request shows a body or a commit trailer.
        ({"body": "Generated with Claude Code"}, False),
        ({}, False),
    ],
)
def test_search_results_rule_out_only_what_they_show(
    fields: dict[str, Any], ruled_out: bool
) -> None:
    assert plainly_not_human(pull(1, at(10), **fields)) is ruled_out


def test_every_agent_can_be_found_and_recognised() -> None:
    for agent in AGENTS.values():
        assert agent.apps, agent.name
        assert agent.search.strip(), agent.name
        for app in agent.apps:
            assert f"author:app/{app}" in agent.search, agent.name


# ── the client ───────────────────────────────────────────────────────────────


class Wire:
    """GitHub's end of the client's requests: canned answers, in order."""

    def __init__(self, *answers: Response | Exception) -> None:
        self.answers = list(answers)
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request) -> Response:
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class Clock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def answer(status: int = 200, body: Any = None, **headers: str) -> Response:
    raw = body if isinstance(body, bytes) else json.dumps({} if body is None else body).encode()
    return Response(status, {name.replace("_", "-"): value for name, value in headers.items()}, raw)


def data(**fields: Any) -> Response:
    return answer(200, {"data": fields})


def client(
    wire: Wire, clock: Clock | None = None, logs: list[str] | None = None, pace: float = 0.0
) -> GitHub:
    clock = clock or Clock()
    return GitHub(
        TOKEN,
        send=wire,
        sleep=clock.sleep,
        clock=clock,
        log=logs.append if logs is not None else quiet,
        min_interval=pace,
    )


def test_the_token_goes_in_the_authorization_header_and_nowhere_else() -> None:
    wire = Wire(data(viewer={"login": "alice"}), answer(200, EDITS_CODE.encode()))
    api = client(wire)

    assert api.graphql("query { viewer { login } }", {}) == {"viewer": {"login": "alice"}}
    assert api.pull_diff("acme/widgets", 7) == PullDiff(EDITS_CODE)

    for request in wire.requests:
        body = request.data
        assert body is None or isinstance(body, bytes)
        assert request.get_header("Authorization") == f"Bearer {TOKEN}"
        assert TOKEN not in request.full_url
        assert TOKEN.encode() not in (body or b"")
        assert all(
            TOKEN not in value for name, value in request.header_items() if name != "Authorization"
        )
    assert TOKEN not in repr(api)
    graphql, diff = wire.requests
    assert (graphql.get_method(), graphql.full_url) == ("POST", "https://api.github.com/graphql")
    assert (diff.get_method(), diff.full_url) == (
        "GET",
        "https://api.github.com/repos/acme/widgets/pulls/7",
    )
    assert diff.get_header("Accept") == DIFF_MEDIA_TYPE


def test_a_rate_limit_that_says_when_is_waited_out() -> None:
    clock = Clock()
    logs: list[str] = []
    wire = Wire(answer(403, {"message": "secondary rate limit"}, retry_after="30"), data(ok=True))

    assert client(wire, clock, logs).graphql("query", {}) == {"ok": True}
    assert clock.slept == [30.0]
    assert len(logs) == 1 and "waiting 30s" in logs[0]


def test_a_secondary_rate_limit_without_retry_after_waits_longer_each_time() -> None:
    clock = Clock()
    limited = answer(403, {"message": "You have exceeded a secondary rate limit."})
    wire = Wire(limited, limited, data(ok=True))

    assert client(wire, clock).graphql("query", {}) == {"ok": True}
    assert clock.slept == [60.0, 120.0]


def test_a_spent_quota_waits_until_it_resets() -> None:
    clock = Clock(now=1000.0)
    spent = answer(
        403,
        {"message": "API rate limit exceeded"},
        x_ratelimit_remaining="0",
        x_ratelimit_reset="1600",
    )
    wire = Wire(spent, data(ok=True))

    assert client(wire, clock).graphql("query", {}) == {"ok": True}
    assert clock.slept == [601.0]


def test_the_last_request_of_a_window_holds_back_the_next_one_on_that_quota() -> None:
    clock = Clock(now=1000.0)
    last = answer(200, {"data": {"a": 1}}, x_ratelimit_remaining="0", x_ratelimit_reset="1600")
    wire = Wire(last, answer(200, EDITS_CODE.encode()), data(b=2))
    api = client(wire, clock)

    assert api.graphql("query", {}) == {"a": 1}
    # Diffs count against the REST quota, which has not run out.
    assert api.pull_diff("acme/widgets", 7).text == EDITS_CODE
    assert clock.slept == []
    assert api.graphql("query", {}) == {"b": 2}
    assert clock.slept == [601.0]


def test_a_graphql_rate_limit_error_waits_for_the_reset() -> None:
    # Said in the body alone: the headers need not show the quota as spent.
    clock = Clock(now=1000.0)
    limited = answer(
        200,
        {"errors": [{"type": "RATE_LIMITED", "message": "API rate limit exceeded"}]},
        x_ratelimit_reset="1300",
    )
    wire = Wire(limited, answer(200, {"errors": [{"type": "RATE_LIMITED"}]}), data(ok=True))

    assert client(wire, clock).graphql("query", {}) == {"ok": True}
    # Without a reset time, a minute.
    assert clock.slept == [301.0, 60.0]


def test_server_errors_are_retried_with_backoff() -> None:
    clock = Clock()
    wire = Wire(answer(502, b"<html>Bad gateway</html>"), answer(503, b""), data(ok=True))

    assert client(wire, clock).graphql("query", {}) == {"ok": True}
    assert clock.slept == [5.0, 10.0]


def test_the_html_page_of_a_timed_out_search_is_retried() -> None:
    clock = Clock()
    wire = Wire(answer(200, b"<html>We couldn't respond in time</html>"), data(ok=True))

    assert client(wire, clock).graphql("query", {}) == {"ok": True}
    assert clock.slept == [5.0]


def test_a_search_that_failed_inside_a_query_is_asked_again() -> None:
    clock = Clock()
    partial = answer(
        200, {"data": {"s0": {"issueCount": 1}, "s1": None}, "errors": [{"message": "timeout"}]}
    )
    wire = Wire(partial, data(s0={"issueCount": 1}, s1={"issueCount": 2}))

    assert client(wire, clock).graphql("query", {}) == {
        "s0": {"issueCount": 1},
        "s1": {"issueCount": 2},
    }
    assert clock.slept == [5.0]


def test_a_dropped_connection_is_retried() -> None:
    clock = Clock()
    wire = Wire(TransportError("ConnectionResetError: reset by peer"), data(ok=True))

    assert client(wire, clock).graphql("query", {}) == {"ok": True}
    assert clock.slept == [5.0]


def test_the_client_gives_up_on_what_keeps_failing() -> None:
    clock = Clock()
    wire = Wire(*[answer(502, b"")] * MAX_ATTEMPTS)

    with pytest.raises(GitHubError, match=f"giving up after {MAX_ATTEMPTS} attempts"):
        client(wire, clock).graphql("query", {})
    assert len(wire.requests) == MAX_ATTEMPTS
    assert clock.slept == [5.0, 10.0, 20.0, 40.0, 80.0]


@pytest.mark.parametrize(
    "refusal",
    [
        answer(401, {"message": f"Bad credentials: {TOKEN}"}),
        answer(403, {"message": "Resource not accessible by personal access token"}),
    ],
)
def test_a_refusal_fails_at_once_and_its_message_carries_no_token(refusal: Response) -> None:
    wire = Wire(refusal)

    with pytest.raises(GitHubError) as caught:
        client(wire).graphql("query", {})
    assert f"HTTP {refusal.status}" in str(caught.value)
    assert TOKEN not in str(caught.value)
    assert len(wire.requests) == 1


def test_a_diff_github_will_not_render_is_unavailable_rather_than_an_error() -> None:
    wire = Wire(
        answer(406, {"message": "diff too large", "errors": [{"code": "too_large"}]}),
        answer(404, {"message": "Not Found"}),
        answer(500, b""),
        answer(200, SKIPS_A_TEST.encode()),
    )
    api = client(wire)

    assert api.pull_diff("acme/widgets", 1) == PullDiff(None, "too_large")
    assert api.pull_diff("acme/widgets", 2) == PullDiff(None, "http_404")
    assert api.pull_diff("acme/widgets", 3) == PullDiff(SKIPS_A_TEST)


def test_a_diff_github_fails_on_every_time_is_unavailable_while_github_answers() -> None:
    wire = Wire(
        *[answer(500, b"")] * MAX_ATTEMPTS,
        answer(200, {"resources": {}}),
        answer(200, SKIPS_A_TEST.encode()),
    )
    api = client(wire)

    assert api.pull_diff("acme/widgets", 1) == PullDiff(None, "http_500")
    assert wire.requests[-1].full_url == "https://api.github.com/rate_limit"
    assert api.pull_diff("acme/widgets", 2) == PullDiff(SKIPS_A_TEST)


def test_a_diff_that_fails_while_github_fails_everywhere_stops_the_run() -> None:
    wire = Wire(*[answer(502, b"")] * (2 * MAX_ATTEMPTS))

    with pytest.raises(GitHubError, match=f"giving up after {MAX_ATTEMPTS} attempts"):
        client(wire).pull_diff("acme/widgets", 1)
    assert wire.answers == []


def test_a_diff_whose_attempts_did_not_all_reach_github_stops_the_run() -> None:
    reset = TransportError("ConnectionResetError: reset by peer")
    wire = Wire(reset, *[answer(500, b"")] * (MAX_ATTEMPTS - 1))

    with pytest.raises(GitHubError, match=f"giving up after {MAX_ATTEMPTS} attempts"):
        client(wire).pull_diff("acme/widgets", 1)
    assert len(wire.requests) == MAX_ATTEMPTS


@pytest.mark.parametrize("repo", ["../user", "acme/..", "acme/widgets/pulls", "acme widgets", ""])
def test_a_diff_is_asked_for_a_repository_name_only(repo: str) -> None:
    with pytest.raises(ValueError, match="not a repository name"):
        client(Wire()).pull_diff(repo, 1)


def test_a_query_with_a_cheaper_way_to_be_asked_gives_up_sooner() -> None:
    clock = Clock()
    wire = Wire(answer(502, b""), answer(502, b""))

    with pytest.raises(GitHubError, match="giving up after 2 attempts"):
        client(wire, clock).graphql("query", {}, attempts=2)
    assert len(wire.requests) == 2 and clock.slept == [5.0]


def test_requests_are_spaced_out() -> None:
    clock = Clock()
    api = client(Wire(data(a=1), data(b=2)), clock, pace=0.25)

    api.graphql("query", {})
    api.graphql("query", {})
    assert clock.slept == [0.25]


def test_short_waits_are_not_announced() -> None:
    logs: list[str] = []
    client(Wire(answer(502, b""), data(ok=True)), logs=logs).graphql("query", {})
    assert logs == []


class _Opened:
    def __init__(self, status: int, headers: Mapping[str, str], body: bytes) -> None:
        self.status = status
        self.headers = Message()
        for name, value in headers.items():
            self.headers[name] = value
        self._body = body
        self.reads: list[int] = []

    def __enter__(self) -> _Opened:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def read(self, amount: int = -1) -> bytes:
        self.reads.append(amount)
        return self._body if amount < 0 else self._body[:amount]


def _request(url: str = "https://api.github.com/graphql") -> urllib.request.Request:
    return urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})


def test_urllib_send_goes_through_the_products_outbound_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def opened(request: urllib.request.Request, *, allowed_base: str, **_: Any) -> _Opened:
        calls.append((request.full_url, allowed_base))
        return _Opened(200, {"X-RateLimit-Remaining": "4999"}, b'{"data": {}}')

    monkeypatch.setattr(github, "open_validated_http_request", opened)

    assert urllib_send(_request()) == Response(
        200, {"x-ratelimit-remaining": "4999"}, b'{"data": {}}'
    )
    assert calls == [("https://api.github.com/graphql", "https://api.github.com")]


def test_urllib_send_turns_an_http_error_into_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(request: urllib.request.Request, **_: Any) -> _Opened:
        headers = Message()
        headers["Retry-After"] = "30"
        raise urllib.error.HTTPError(
            request.full_url, 403, "Forbidden", headers, io.BytesIO(b'{"message": "slow down"}')
        )

    monkeypatch.setattr(github, "open_validated_http_request", refused)

    assert urllib_send(_request()) == Response(
        403, {"retry-after": "30"}, b'{"message": "slow down"}'
    )


def test_urllib_send_turns_a_network_failure_into_a_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreachable(request: urllib.request.Request, **_: Any) -> _Opened:
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(github, "open_validated_http_request", unreachable)

    with pytest.raises(TransportError, match="timed out") as caught:
        urllib_send(_request())
    assert TOKEN not in str(caught.value)


def test_an_answer_too_big_to_read_is_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    declared = _Opened(200, {"Content-Length": "495000000"}, b"")
    undeclared = _Opened(200, {}, b"x" * 11)
    fits = _Opened(200, {}, b"x" * 10)
    answers = iter([declared, undeclared, fits])
    monkeypatch.setattr(github, "open_validated_http_request", lambda *_a, **_k: next(answers))

    with pytest.raises(ResponseTooLarge):
        urllib_send(_request(), max_bytes=10)
    assert declared.reads == []
    with pytest.raises(ResponseTooLarge):
        urllib_send(_request(), max_bytes=10)
    assert urllib_send(_request(), max_bytes=10).body == b"x" * 10


def test_an_oversized_diff_is_unavailable_and_asked_for_once() -> None:
    wire = Wire(ResponseTooLarge("too big"), ResponseTooLarge("too big"))
    api = client(wire)

    assert api.pull_diff("acme/widgets", 1) == PullDiff(None, OVERSIZED)
    # Anywhere else it is a failure, and the caller's to route around.
    with pytest.raises(GitHubError):
        api.graphql("query", {})
    assert len(wire.requests) == 2


def test_urllib_send_refuses_any_host_but_githubs_api() -> None:
    with pytest.raises(ValueError):
        urllib_send(_request("https://example.com/graphql"))


# ── the cache ────────────────────────────────────────────────────────────────


def test_the_cache_keeps_answers_including_none(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.put("search", "ab12", {"issueCount": 3})
    cache.put("pull", "cd34", None)

    assert cache.get("search", "ab12") == Cached({"issueCount": 3})
    # A pull request that no longer resolves is an answer too, not a miss.
    assert cache.get("pull", "cd34") == Cached(None)
    assert cache.get("pull", "ef56") is None


def test_a_damaged_entry_is_a_miss(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.put("diff", "ab12", {"text": EDITS_CODE})
    (entry,) = tmp_path.rglob("*.json.gz")
    entry.write_bytes(b"not gzip")

    assert cache.get("diff", "ab12") is None


def test_an_interrupted_write_leaves_no_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(_source: str, _target: str) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("bench.cache.os.replace", interrupted)
    cache = Cache(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        cache.put("diff", "ab12", {"text": EDITS_CODE})
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []
    assert cache.get("diff", "ab12") is None


def test_a_cache_key_is_the_question() -> None:
    assert cache_key("acme/widgets", 7) == cache_key("acme/widgets", 7)
    assert cache_key("acme/widgets", 7) != cache_key("acme/widgets", "7")
    assert cache_key({"b": 1, "a": 2}) == cache_key({"a": 2, "b": 1})


# ── the source ───────────────────────────────────────────────────────────────


def _world() -> FakeGitHub:
    return FakeGitHub(
        [
            pull(1, at(9), repo="acme/widgets"),
            pull(2, at(10), repo="acme/widgets"),
            pull(3, at(11), repo="acme/gadgets", diff=PullDiff(None, "too_large")),
        ]
    )


def test_answers_are_filed_one_by_one_however_they_were_asked(tmp_path: Path) -> None:
    world = _world()
    online = reader(world, tmp_path)
    queries = ["repo:acme/widgets is:pr", "repo:acme/gadgets is:pr"]

    pages = online.searches(queries, 20)
    pulls = online.pull_requests(["PR_acme/widgets#1", "PR_acme/widgets#2"])
    online.diff("acme/gadgets", 3)
    assert world.requests == 3

    offline = Source(Cache(tmp_path), None)
    assert [offline.search(query, 20) for query in reversed(queries)] == pages[::-1]
    assert offline.pull_requests(["PR_acme/widgets#2"]) == {
        "PR_acme/widgets#2": pulls["PR_acme/widgets#2"]
    }
    assert offline.diff("acme/gadgets", 3) == PullDiff(None, "too_large")
    assert [page["issueCount"] for page in pages] == [2, 1]


def test_a_prefetch_asks_only_for_what_is_missing(tmp_path: Path) -> None:
    world = _world()
    online = reader(world, tmp_path)
    queries = ["repo:acme/widgets is:pr", "repo:acme/gadgets is:pr"]

    # One search is not worth a batch; it is asked when it is needed.
    assert online.prefetch(queries[:1], 20) and world.requests == 0
    assert online.prefetch(queries, 20) and world.batches == [2]
    assert online.prefetch(queries, 20) and world.batches == [2]
    assert [online.search(query, 20)["issueCount"] for query in queries] == [2, 1]
    assert world.requests == 1
    # Offline there is nothing to ask, and that is not a failure.
    assert Source(Cache(tmp_path / "empty"), None).prefetch(queries, 20)


def test_asking_for_other_fields_is_asking_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world()
    online = reader(world, tmp_path)
    online.search("repo:acme/widgets is:pr", 20)
    online.pull_requests(["PR_acme/widgets#1"])
    assert world.requests == 2

    # Answers of the old shape stay where they are, and are not taken for the new.
    monkeypatch.setattr(source, "SEARCH_FIELDS", f"{source.SEARCH_FIELDS} title")
    monkeypatch.setattr(
        source, "PULL_QUERY", source.PULL_QUERY.replace("isDraft", "isDraft locked")
    )
    online.search("repo:acme/widgets is:pr", 20)
    online.pull_requests(["PR_acme/widgets#1"])
    assert world.requests == 4


def test_an_offline_source_names_what_it_is_missing(tmp_path: Path) -> None:
    offline = Source(Cache(tmp_path), None)

    with pytest.raises(OfflineMiss, match="GITHUB_TOKEN"):
        offline.search("repo:acme/widgets is:pr")
    with pytest.raises(OfflineMiss, match="acme/widgets#7"):
        offline.diff("acme/widgets", 7)


def test_a_pull_request_that_no_longer_resolves_is_remembered(tmp_path: Path) -> None:
    world = _world()
    world.pulls[0]["_gone"] = True

    assert reader(world, tmp_path).pull_requests(["PR_acme/widgets#1"]) == {
        "PR_acme/widgets#1": None
    }
    assert Source(Cache(tmp_path), None).pull_requests(["PR_acme/widgets#1"]) == {
        "PR_acme/widgets#1": None
    }


def test_a_batch_of_pull_requests_too_big_to_answer_is_halved(tmp_path: Path) -> None:
    world = FakeGitHub(
        [pull(number, at(10)) for number in range(1, 6)],
        fail=lambda _query, variables: len(variables.get("ids", ())) > 2,
    )

    found = reader(world, tmp_path).pull_requests([f"PR_acme/widgets#{n}" for n in range(1, 6)])
    assert all(found[f"PR_acme/widgets#{n}"] is not None for n in range(1, 6))


def test_searches_that_fail_together_are_asked_one_at_a_time(tmp_path: Path) -> None:
    world = FakeGitHub(
        [pull(1, at(10)), pull(2, at(11), repo="acme/gadgets")],
        fail=lambda query, _variables: "$q0" in query,
    )

    pages = reader(world, tmp_path).searches(["repo:acme/widgets", "repo:acme/gadgets"], 20)
    assert [page["issueCount"] for page in pages] == [1, 1]


# ── agent groups ─────────────────────────────────────────────────────────────


def _busy_world() -> FakeGitHub:
    """Copilot pull requests in each of four hours, in repositories of their own."""

    return FakeGitHub(
        [
            by_agent(
                "copilot", hour * 10 + index, at(hour, 10 * index + 5), repo=f"acme/r{hour}{index}"
            )
            for hour in range(4)
            for index in range(3)
        ]
    )


def _urls(group: Group) -> list[str]:
    return [item.pr.url for item in group.scanned]


def test_an_agent_group_is_reproducible_and_a_larger_one_extends_it(tmp_path: Path) -> None:
    small = sample_agent(
        reader(_busy_world(), tmp_path / "a"), AGENTS["copilot"], config(per_group=4), quiet
    )
    again = sample_agent(
        reader(_busy_world(), tmp_path / "b"), AGENTS["copilot"], config(per_group=4), quiet
    )
    large = sample_agent(
        reader(_busy_world(), tmp_path / "c"), AGENTS["copilot"], config(per_group=10), quiet
    )

    assert _urls(small) == _urls(again)
    assert len(_urls(small)) == 4 and len(_urls(large)) == 10
    assert _urls(large)[:4] == _urls(small)


def test_a_slice_gives_a_group_at_most_its_share(tmp_path: Path) -> None:
    group = sample_agent(
        reader(_busy_world(), tmp_path),
        AGENTS["copilot"],
        config(per_group=8, per_slice=2),
        quiet,
    )

    assert Counter(item.pr.created_at.hour for item in group.scanned) == {0: 2, 1: 2, 2: 2, 3: 2}
    # The last slice's share filled the group too, and a full group is the reason.
    assert group.left_out == Counter({"over the per-slice cap": 3, "after the group was full": 1})
    assert group.slices == 4 and group.seen == 12


def test_a_slices_share_is_any_of_it_not_its_first(tmp_path: Path) -> None:
    hour = [by_agent("copilot", n, at(1, 5 * n), repo=f"acme/r{n}") for n in range(10)]

    chosen = {
        sample_agent(
            reader(FakeGitHub(hour), tmp_path / str(seed)),
            AGENTS["copilot"],
            config(since=moment(1), until=moment(2), per_group=1, seed=seed),
            quiet,
        )
        .scanned[0]
        .pr.number
        for seed in range(1, 9)
    }
    assert len(chosen) > 1


def test_the_seed_decides_which_slices_come_first(tmp_path: Path) -> None:
    firsts = {
        sample_agent(
            reader(_busy_world(), tmp_path / str(seed)),
            AGENTS["copilot"],
            config(per_group=1, seed=seed),
            quiet,
        )
        .scanned[0]
        .pr.created_at.hour
        for seed in range(1, 9)
    }
    assert len(firsts) > 1


def test_the_first_pages_of_ten_slices_are_asked_as_one_query(tmp_path: Path) -> None:
    # One pull request in each ten minutes of four hours.
    world = FakeGitHub(
        [by_agent("copilot", n, at(n // 6, 10 * (n % 6) + 5), repo=f"acme/r{n}") for n in range(24)]
    )

    group = sample_agent(
        reader(world, tmp_path),
        AGENTS["copilot"],
        config(slice_length=timedelta(minutes=10)),
        quiet,
    )

    assert len(group.scanned) == 24 and group.slices == 24
    assert world.batches == [10, 10, 4]
    # Each slice had a single page, so no search went out on its own.
    assert len(world.searched) == 24
    assert {tries for kind, tries in world.asked if kind == "searches"} == {source.BATCH_ATTEMPTS}


def test_a_batch_too_heavy_to_answer_makes_the_next_one_smaller(tmp_path: Path) -> None:
    world = FakeGitHub(
        [
            by_agent("copilot", n, at(n // 6, 10 * (n % 6) + 5), repo=f"acme/r{n}")
            for n in range(24)
        ],
        # GitHub cannot answer more than five searches at once.
        fail=lambda query, _variables: "$q5" in query,
    )

    group = sample_agent(
        reader(world, tmp_path),
        AGENTS["copilot"],
        config(slice_length=timedelta(minutes=10)),
        quiet,
    )

    assert len(group.scanned) == 24
    # Ten fail, so five; five pass, so ten again, and the last nine fail.
    assert world.batches == [10, 5, 9]


@pytest.mark.parametrize(
    ("per_group", "lookups", "left"),
    [(10, [20], 110), (50, [50], 70), (60, [50, 20], 60)],
)
def test_candidates_are_read_in_full_only_when_their_turn_comes(
    tmp_path: Path, per_group: int, lookups: list[int], left: int
) -> None:
    world = FakeGitHub(
        [
            by_agent(
                "copilot", n, f"2026-07-01T01:{n // 2:02d}:{30 * (n % 2):02d}Z", repo=f"acme/r{n}"
            )
            for n in range(120)
        ]
    )

    group = sample_agent(
        reader(world, tmp_path),
        AGENTS["copilot"],
        config(since=moment(1), until=moment(2), per_group=per_group, per_slice=1000),
        quiet,
    )

    assert len(group.scanned) == per_group
    # Twice as many as are still needed, up to fifty at a time.
    assert world.lookups == lookups
    assert group.left_out == Counter({"after the group was full": left})
    assert group.seen == 120


def test_every_candidate_not_taken_is_counted_under_its_reason(tmp_path: Path) -> None:
    def copilot(number: int, **fields: Any) -> JSON:
        return by_agent("copilot", number, at(1, number), **fields)

    world = FakeGitHub(
        [
            copilot(1, fork=True),
            copilot(2, stars=9),
            copilot(3, private=True),
            # Search is loose: a person on a copilot/ branch can come back too.
            pull(4, at(1, 4), branch="copilot/round-refunds", found_by=["copilot"]),
            copilot(5, diff=PullDiff(None, "too_large")),
            copilot(6, diff=""),
            copilot(7, repo="acme/busy"),
            copilot(8, repo="acme/busy"),
            copilot(9, repo="acme/busy"),
            copilot(10, repo="acme/busy"),
            copilot(11),
            {**copilot(12), "_ignores_dates": True, "createdAt": "2026-06-30T12:00:00Z"},
        ]
    )
    world.pulls[10]["_gone"] = True

    group = sample_agent(
        reader(world, tmp_path),
        AGENTS["copilot"],
        config(since=moment(1), until=moment(2), per_group=50),
        quiet,
    )

    busy = {f"https://github.com/acme/busy/pull/{n}" for n in (7, 8, 9, 10)}
    assert len(group.scanned) == 3 and set(_urls(group)) < busy
    assert group.left_out == Counter(
        {
            "fork": 1,
            "under 10 stars": 1,
            "private repository": 1,
            "identified as assisted": 1,
            "diff unavailable (too_large)": 1,
            "no changes": 1,
            "over the per-repository cap": 1,
            "no longer resolves": 1,
            "outside the slice": 1,
        }
    )
    assert group.seen == 12 == len(group.scanned) + sum(group.left_out.values())


def test_forks_and_fewer_stars_can_be_asked_for(tmp_path: Path) -> None:
    world = FakeGitHub(
        [
            by_agent("copilot", 1, at(1, 1), fork=True, repo="acme/fork"),
            by_agent("copilot", 2, at(1, 2), stars=0, repo="acme/new"),
        ]
    )

    group = sample_agent(
        reader(world, tmp_path),
        AGENTS["copilot"],
        config(since=moment(1), until=moment(2), include_forks=True, min_stars=0),
        quiet,
    )
    assert len(group.scanned) == 2


def test_the_detector_runs_on_each_diff(tmp_path: Path) -> None:
    world = FakeGitHub(
        [
            by_agent("claude", 1, at(1, 1), diff=DELETES_A_TEST, merged=True),
            by_agent("claude", 2, at(1, 2), diff=SKIPS_A_TEST, repo="acme/gadgets"),
            by_agent("claude", 3, at(1, 3), repo="acme/tools"),
        ]
    )

    group = sample_agent(
        reader(world, tmp_path), AGENTS["claude"], config(since=moment(1), until=moment(2)), quiet
    )

    hits = {
        item.pr.number: [(hit.check, hit.file, hit.line) for hit in item.hits]
        for item in group.scanned
    }
    assert hits == {
        1: [("deleted_test", "tests/test_refund.py", 5)],
        2: [("skip_or_focus", "tests/test_refund.py", 1)],
        3: [],
    }
    assert {item.pr.number: item.pr.merged for item in group.scanned} == {
        1: True,
        2: False,
        3: False,
    }


def test_a_test_file_deleted_with_none_of_its_lines_shown_is_a_hit(tmp_path: Path) -> None:
    world = FakeGitHub([by_agent("claude", 1, at(1, 1), diff=DELETES_TEST_FILES)])

    group = sample_agent(
        reader(world, tmp_path), AGENTS["claude"], config(since=moment(1), until=moment(2)), quiet
    )

    assert len(group.scanned) == 1
    assert [(hit.check, hit.file, hit.line, hit.side) for hit in group.scanned[0].hits] == [
        ("deleted_test", "tests/test_totals.py", None, "base"),
        ("deleted_test", "tests/test_old.py", 1, "base"),
    ]


def test_a_slice_over_the_search_cap_is_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sample, "SEARCH_CAP", 3)
    minutes = (5, 10, 40, 45, 50)
    world = FakeGitHub(
        [by_agent("copilot", m, at(1, m), repo=f"acme/r{m}") for m in minutes], cap=3
    )

    group = sample_agent(
        reader(world, tmp_path),
        AGENTS["copilot"],
        config(since=moment(1), until=moment(2)),
        quiet,
    )

    assert sorted(item.pr.number for item in group.scanned) == list(minutes)
    assert "beyond the search cap" not in group.left_out
    assert any(f"created:<{sample.iso(moment(1, 30))}" in query for query in world.searched)


def test_what_a_minute_holds_beyond_the_search_cap_is_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sample, "SEARCH_CAP", 3)
    world = FakeGitHub(
        [
            by_agent("copilot", n, f"2026-07-01T01:00:{10 * n:02d}Z", repo=f"acme/r{n}")
            for n in range(5)
        ],
        cap=3,
    )

    group = sample_agent(
        reader(world, tmp_path),
        AGENTS["copilot"],
        config(since=moment(1), until=moment(1, 1), slice_length=timedelta(minutes=1)),
        quiet,
    )

    assert len(group.scanned) == 3
    assert group.left_out["beyond the search cap"] == 2


def test_a_search_github_cannot_finish_is_split(tmp_path: Path) -> None:
    whole_hour = f"created:>={sample.iso(moment(1))} created:<{sample.iso(moment(2))}"
    world = FakeGitHub(
        [by_agent("copilot", m, at(1, m), repo=f"acme/r{m}") for m in (5, 50)],
        fail=lambda _query, variables: whole_hour in str(variables.get("q", "")),
    )

    group = sample_agent(
        reader(world, tmp_path),
        AGENTS["copilot"],
        config(since=moment(1), until=moment(2)),
        quiet,
    )
    assert sorted(item.pr.number for item in group.scanned) == [5, 50]


def test_an_offline_run_replays_one_whose_searches_failed(tmp_path: Path) -> None:
    whole_hour = f"created:>={sample.iso(moment(1))} created:<{sample.iso(moment(2))}"
    world = FakeGitHub(
        [by_agent("copilot", m, at(h, m), repo=f"acme/r{h}{m}") for h in (1, 2) for m in (5, 50)],
        # Batches of searches fail, and so does the whole of the second hour.
        fail=lambda query, variables: "$q0" in query or whole_hour in str(variables.get("q", "")),
    )
    per_group = config(per_group=3)

    online = sample_agent(reader(world, tmp_path), AGENTS["copilot"], per_group, quiet)
    offline = sample_agent(Source(Cache(tmp_path), None), AGENTS["copilot"], per_group, quiet)

    assert len(online.scanned) == 3
    assert _urls(offline) == _urls(online)


def test_a_minute_github_cannot_search_ends_the_run(tmp_path: Path) -> None:
    world = FakeGitHub([], fail=lambda _query, _variables: True)

    with pytest.raises(GitHubError):
        sample_agent(
            reader(world, tmp_path),
            AGENTS["copilot"],
            config(since=moment(1), until=moment(1, 1), slice_length=timedelta(minutes=1)),
            quiet,
        )


def test_a_detector_error_is_counted_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def detect(diff: str) -> list[IntegrityHit]:
        raise RuntimeError("unexpected hunk")

    monkeypatch.setattr(sample, "detect", detect)
    logs: list[str] = []
    world = FakeGitHub([by_agent("copilot", 1, at(1, 1))])

    group = sample_agent(
        reader(world, tmp_path),
        AGENTS["copilot"],
        config(since=moment(1), until=moment(2)),
        logs.append,
    )

    assert group.scanned == [] and group.left_out == Counter({"detector error": 1})
    assert any("https://github.com/acme/widgets/pull/1" in line for line in logs)
    assert "(the window ran out first)" in logs[-1]


# ── the human group ──────────────────────────────────────────────────────────


def _repository_world() -> FakeGitHub:
    """One repository: two agent pull requests among people's, bots' and assisted ones."""

    return FakeGitHub(
        [
            pull(1, at(9), merged=True),
            by_agent("copilot", 2, at(10), diff=DELETES_A_TEST),
            pull(3, at(10, 5), login="dependabot", bot=True),
            pull(4, at(10, 10), branch="copilot/round-refunds"),
            pull(5, at(10, 15), commits=[("Co-Authored-By: Claude <noreply@anthropic.com>", "")]),
            pull(6, at(10, 20), branch="codex/round-refunds", found_by=["codex"]),
            pull(7, at(10, 30), diff=SKIPS_A_TEST),
            pull(8, at(12)),
        ]
    )


def _controls(world: FakeGitHub, root: Path, agents: Sequence[str]) -> tuple[list[Group], Group]:
    """The agents' groups and their controls, over the whole of 1 July."""

    shared = reader(world, root)
    day = config(until=moment(0) + timedelta(days=1))
    groups = [sample_agent(shared, AGENTS[name], day, quiet) for name in agents]
    return groups, match_controls(shared, groups, day, quiet)


def test_each_agent_pull_request_is_matched_to_the_nearest_unused_human_one(
    tmp_path: Path,
) -> None:
    (copilot, codex), human = _controls(_repository_world(), tmp_path, ["copilot", "codex"])

    assert [item.pr.number for item in copilot.scanned] == [2]
    assert [item.pr.number for item in codex.scanned] == [6]
    # Copilot's at 10:00 takes 10:30; Codex's at 10:20 then has 9:00 and 12:00
    # left, and 9:00 is nearer.
    assert [
        (item.pr.number, item.matched_to and item.matched_to.number) for item in human.scanned
    ] == [
        (7, 2),
        (1, 6),
    ]
    assert human.left_out["identified as assisted"] == 2
    assert [item.hits[0].check for item in human.scanned if item.hits] == ["skip_or_focus"]


def test_an_agent_pull_request_with_no_human_one_nearby_is_counted(tmp_path: Path) -> None:
    world = FakeGitHub([by_agent("copilot", 1, at(10)), pull(2, at(11), login="renovate-bot")])

    _, human = _controls(world, tmp_path, ["copilot"])

    assert human.scanned == []
    assert human.left_out == Counter({"agent pull requests with no human one nearby": 1})
    assert human.seen == 1


def test_a_human_pull_request_without_a_diff_is_passed_over(tmp_path: Path) -> None:
    world = FakeGitHub(
        [
            by_agent("copilot", 1, at(10)),
            pull(2, at(10, 5), diff=PullDiff(None, "too_large")),
            pull(3, at(11)),
        ]
    )

    _, human = _controls(world, tmp_path, ["copilot"])

    assert [item.pr.number for item in human.scanned] == [3]
    assert human.left_out == Counter({"diff unavailable (too_large)": 1})


def _scanned(group: str, number: int) -> Scanned:
    return Scanned(
        group,
        PullRequest.from_node(pull(number, at(10))),
        (),
        (),
    )


def test_controls_are_matched_one_agent_at_a_time() -> None:
    a1, a2, a3, b1, c1, c2 = (
        _scanned("a", 1),
        _scanned("a", 2),
        _scanned("a", 3),
        _scanned("b", 4),
        _scanned("c", 5),
        _scanned("c", 6),
    )
    assert interleave([[a1, a2, a3], [b1], [c1, c2]]) == [a1, b1, c1, a2, c2, a3]
    assert interleave([]) == []


def test_the_window_is_cut_into_slices() -> None:
    assert time_slices(moment(0), moment(2, 30), timedelta(hours=1)) == [
        (moment(0), moment(1)),
        (moment(1), moment(2)),
        (moment(2), moment(2, 30)),
    ]
    with pytest.raises(ValueError):
        time_slices(moment(0), moment(1), timedelta(0))


# ── the numbers ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("count", "total", "low", "high"),
    [(0, 10, 0.0, 0.2775), (5, 10, 0.2366, 0.7634), (10, 10, 0.7225, 1.0), (1, 2, 0.0945, 0.9055)],
)
def test_wilson_intervals(count: int, total: int, low: float, high: float) -> None:
    interval = report.wilson(count, total)
    assert interval is not None
    assert interval[0] == pytest.approx(low, abs=1e-4)
    assert interval[1] == pytest.approx(high, abs=1e-4)


def test_wilson_needs_a_proportion() -> None:
    assert report.wilson(0, 0) is None
    with pytest.raises(ValueError):
        report.wilson(3, 2)


def _hit(
    check: IntegrityCheck, file: str = "tests/test_refund.py", line: int | None = 5
) -> IntegrityHit:
    return IntegrityHit(
        check=check, file=file, line=line, side="base", text="def test_rounding():", detail="gone"
    )


def _group(name: str, *items: tuple[bool, Sequence[IntegrityHit]]) -> Group:
    group = Group(name)
    for number, (merged, hits) in enumerate(items, start=1):
        node = pull(number, at(10), repo=f"acme/r{number}", merged=merged)
        group.scanned.append(Scanned(name, PullRequest.from_node(node), ("app",), tuple(hits)))
    return group


def test_a_group_summary_counts_pull_requests_hits_and_merges() -> None:
    group = _group(
        "copilot",
        (True, [_hit("deleted_test")]),
        (False, [_hit("deleted_test"), _hit("skip_or_focus")]),
        (True, []),
    )

    summary = report.summarize_group(group)

    assert summary["scanned"] == 3
    assert (summary["merged"]["count"], summary["merged"]["of"]) == (2, 3)
    assert summary["any_check"]["hits"] == 3
    assert (summary["any_check"]["prs"]["count"], summary["any_check"]["merged"]["count"]) == (2, 1)
    deleted = summary["checks"]["deleted_test"]
    assert (deleted["prs"]["count"], deleted["hits"], deleted["merged"]["of"]) == (2, 2, 2)
    unused = summary["checks"]["fewer_assertions"]
    assert unused["prs"]["count"] == 0 and unused["merged"]["share"] is None
    assert summary["identified_via"] == {"app": 3}
    assert summary["repositories"] == 3


def test_the_human_summary_reports_how_far_controls_are_from_their_match() -> None:
    agent_pr = PullRequest.from_node(pull(1, at(10)))
    group = Group(HUMAN)
    for number, hour in ((2, 11), (3, 13), (4, 10)):
        node = pull(number, at(hour))
        group.scanned.append(Scanned(HUMAN, PullRequest.from_node(node), (), (), agent_pr))

    assert report.summarize_group(group)["median_gap_hours"] == 1.0


def test_a_hit_links_to_its_line_in_the_pull_request() -> None:
    anchor = hashlib.sha256(b"tests/test_refund.py").hexdigest()
    url = "https://github.com/acme/widgets/pull/7"

    assert report.line_link(url, _hit("deleted_test")) == f"{url}/files#diff-{anchor}L5"
    head = IntegrityHit(
        check="skip_or_focus", file="tests/test_refund.py", line=1, side="head", text="x", detail=""
    )
    assert report.line_link(url, head) == f"{url}/files#diff-{anchor}R1"
    assert report.line_link(url, _hit("deleted_test", line=None)) == f"{url}/files#diff-{anchor}"


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        ('=HYPERLINK("https://example.com")', '\'=HYPERLINK("https://example.com")'),
        ("-1", "'-1"),
        ("+x", "'+x"),
        ("@sum", "'@sum"),
        ("assert x == 1", "assert x == 1"),
        ("", ""),
        (5, 5),
    ],
)
def test_csv_cells_are_text_to_a_spreadsheet(value: object, shown: object) -> None:
    assert report.spreadsheet_safe(value) == shown


def _summary(groups: Sequence[Group]) -> JSON:
    return report.summarize(groups, config(), datetime(2026, 7, 5, tzinfo=UTC))


def test_the_markdown_has_a_row_per_group_and_a_dash_for_nothing_to_divide() -> None:
    groups = [_group("copilot", (True, [_hit("deleted_test")]), (False, [])), Group(HUMAN)]

    markdown = report.render_markdown(_summary(groups))

    rows = [line for line in markdown.splitlines() if line.startswith("| GitHub Copilot")]
    assert len(rows) == 3
    assert "| 1 · 50.0% (9.5–90.5) |" in rows[0]
    human = next(line for line in markdown.splitlines() if line.startswith("| Human"))
    assert "| 0 | – |" in human


def test_results_are_written_as_markdown_json_and_two_csv_files(tmp_path: Path) -> None:
    groups = [
        _group("copilot", (True, [_hit("deleted_test"), _hit("skip_or_focus")]), (False, [])),
        Group(HUMAN),
    ]

    paths = report.write_results(tmp_path, groups, _summary(groups))

    assert sorted(path.name for path in paths) == [
        "hits.csv",
        "prs.csv",
        "summary.json",
        "summary.md",
    ]
    hits = list(csv.reader((tmp_path / "hits.csv").open(encoding="utf-8")))
    assert tuple(hits[0]) == report.HITS_COLUMNS and len(hits) == 3
    assert hits[1][report.HITS_COLUMNS.index("label")] == ""
    prs = list(csv.reader((tmp_path / "prs.csv").open(encoding="utf-8")))
    assert len(prs) == 3
    assert json.loads((tmp_path / "summary.json").read_text())["groups"]["copilot"]["scanned"] == 2


# ── the command ──────────────────────────────────────────────────────────────


def _world_for_the_command() -> FakeGitHub:
    return FakeGitHub(
        [
            by_agent("copilot", 1, at(0, 20), diff=DELETES_A_TEST),
            pull(2, at(0, 40), merged=True),
            by_agent("claude", 3, at(2, 10), repo="acme/gadgets", diff=SKIPS_A_TEST),
            pull(4, at(2, 30), repo="acme/gadgets"),
            by_agent("claude", 5, at(3, 10), repo="acme/tools"),
            pull(6, at(1, 10), repo="acme/tools"),
        ]
    )


ARGS = ["--since", "2026-07-01", "--until", "2026-07-01T04:00:00Z", "--agents", "copilot,claude"]


def _files(root: Path) -> Iterator[tuple[Path, str]]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            yield path, (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8")


def test_the_command_writes_results_and_can_rebuild_them_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _world_for_the_command()
    tokens: list[str] = []

    def connect(token: str) -> FakeGitHub:
        tokens.append(token)
        return world

    monkeypatch.setattr(cli, "GitHub", connect)
    monkeypatch.setenv("GITHUB_TOKEN", f"  {TOKEN}\n")

    assert cli.main([*ARGS, "--out", str(tmp_path)]) == 0
    online = capsys.readouterr()
    results = tmp_path / "20260701-20260701_n50_stars10_cap3_seed1_copilot-claude"

    assert tokens == [TOKEN]
    assert "| GitHub Copilot coding agent | 1 |" in online.out
    assert "| Claude Code | 2 |" in online.out
    assert f"wrote {results / 'hits.csv'}" in online.err
    hits = list(csv.DictReader((results / "hits.csv").open(encoding="utf-8")))
    assert [(row["group"], row["check"]) for row in hits] == [
        ("copilot", "deleted_test"),
        ("claude", "skip_or_focus"),
    ]

    def offline(token: str) -> FakeGitHub:
        raise AssertionError("an offline run must not connect")

    monkeypatch.setattr(cli, "GitHub", offline)
    monkeypatch.delenv("GITHUB_TOKEN")
    before = {name: (results / name).read_bytes() for name in ("hits.csv", "prs.csv")}

    assert cli.main([*ARGS, "--out", str(tmp_path), "--offline"]) == 0
    assert {name: (results / name).read_bytes() for name in before} == before
    # Online, the answers are as of the run; offline, older than it.
    assert "Read from GitHub up to" in online.out
    rebuilt = (results / "summary.md").read_text(encoding="utf-8")
    assert "Rebuilt offline from the cache at" in rebuilt and "Read from GitHub" not in rebuilt
    assert json.loads((results / "summary.json").read_text(encoding="utf-8"))["offline"] is True


def test_the_token_is_in_no_file_and_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _world_for_the_command()
    monkeypatch.setattr(cli, "GitHub", lambda token: world)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)

    assert cli.main([*ARGS, "--out", str(tmp_path)]) == 0
    output = capsys.readouterr()

    assert TOKEN not in output.out and TOKEN not in output.err
    written = list(_files(tmp_path))
    assert any(path.suffix == ".gz" for path, _ in written)
    assert all(TOKEN not in text for _, text in written)


def test_the_command_needs_a_token_unless_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    assert cli.main([*ARGS, "--out", str(tmp_path)]) == 2
    assert "GITHUB_TOKEN is not set" in capsys.readouterr().err
    assert cli.main([*ARGS, "--out", str(tmp_path), "--offline"]) == 1
    assert "not in the cache" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--since", "2026-07-02", "--until", "2026-07-01"],
        ["--since", "2026-07-01", "--until", "2026-07-02", "--per-group", "0"],
        ["--since", "2026-07-01", "--until", "2026-07-02", "--per-repo", "0"],
        ["--since", "2026-07-01", "--until", "2026-07-02", "--slice-minutes", "0.5"],
        ["--since", "2026-07-01", "--until", "2026-07-02", "--per-slice", "0"],
        ["--since", "2026-07-01", "--until", "2026-07-02", "--agents", "copilot,hal"],
        ["--since", "2026-07-01", "--until", "2026-07-02", "--agents", ","],
    ],
)
def test_the_command_refuses_parameters_that_make_no_sample(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main([*arguments, "--offline"]) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_the_command_refuses_a_date_it_cannot_read() -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["--since", "July", "--until", "2026-07-02"])
    assert caught.value.code == 2


def test_a_github_failure_says_the_run_can_be_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = FakeGitHub([], fail=lambda _query, _variables: True)
    monkeypatch.setattr(cli, "GitHub", lambda token: world)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)

    assert cli.main([*ARGS, "--out", str(tmp_path), "--slice-minutes", "1"]) == 1
    assert "run the same command to resume" in capsys.readouterr().err


def test_a_run_is_named_after_what_it_samples() -> None:
    assert (
        cli.run_name(
            Config(
                since=datetime(2026, 7, 1, tzinfo=UTC),
                until=datetime(2026, 8, 27, tzinfo=UTC),
                per_group=50,
            ),
            list(AGENTS.values()),
        )
        == "20260701-20260827_n50_stars10_cap3_seed1"
    )
    assert (
        cli.run_name(
            config(
                include_forks=True,
                slice_length=timedelta(minutes=30),
                per_slice=2,
                per_group=1000,
            ),
            [AGENTS["codex"], AGENTS["jules"]],
        )
        == "20260701-20260701_n1000_stars10_cap3_seed1_forks_slice30m_perslice2_codex-jules"
    )
