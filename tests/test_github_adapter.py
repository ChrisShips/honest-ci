"""The GitHub port, exercised against a real HTTP server on loopback.

Whether the token stays in a header, whether a redirect off the configured
base is refused, and whether a request really goes where we think it does are
questions only a server can answer. This one is laid out like GitHub Enterprise Server -- REST
under ``/api/v3``, GraphQL at ``/api/graphql`` -- because that is the layout
where GraphQL is *not* under the REST base, the harder case for the check that
keeps the token on its configured endpoints.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
import threading
import urllib.parse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from honest_ci.actions import ActionsContext
from honest_ci.adapters import build_forge_adapter
from honest_ci.adapters.github import (
    ERROR_STATE,
    FILE_PAGE_SIZE,
    MAX_STATUS_DESCRIPTION,
    STATUS_STATE,
    GitHubError,
    GitHubForgeAdapter,
    graphql_url,
)
from honest_ci.cli import app
from honest_ci.contracts import (
    MAX_PLATFORM_REVIEWS,
    ApprovalReading,
    ApprovalWindow,
    ChangeRequest,
    EvidencePack,
    Finding,
    GateReport,
    Verdict,
)
from honest_ci.diffs import added_lines, changed_line_locations, extract_changed_files
from honest_ci.integrity import detect
from honest_ci.merge_requests import fetch_merge_request, publish_conclusion
from honest_ci.policy import PolicyConfig, PolicyEngine
from honest_ci.report import (
    APPROVAL_WINDOW_PREFIX,
    VERDICT_MARKER,
    approval_window_marker,
    finding_marker,
)
from honest_ci.settings import Settings
from honest_ci.state import RunState, RunStatus

from .factories import REFUSED_NAME_CHARACTERS, cjk_lines, scanner_arguments

# The placeholder the gitleaks config already allows by exact value.
TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz012345"
#: The token as git sends it: GitHub documents it as the password of
#: ``x-access-token``. Built here rather than written out, so there is no second
#: credential-shaped literal in the tree.
BASIC = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode("ascii")

#: The runner sets these in every job -- including the one that runs this suite
#: in CI, where they would make every adapter here believe it was a step in a
#: workflow of this repository.
RUNNER_VARIABLES = (
    "GITHUB_ACTIONS",
    "GITHUB_WORKSPACE",
    "GITHUB_REPOSITORY",
    "GITHUB_SERVER_URL",
    "GITHUB_RUN_ID",
    "GITHUB_EVENT_NAME",
    "GITHUB_EVENT_PATH",
    "GITHUB_STEP_SUMMARY",
    "GITHUB_TOKEN",
)


@pytest.fixture(autouse=True)
def outside_any_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in RUNNER_VARIABLES:
        monkeypatch.delenv(name, raising=False)


# ── a GitHub that answers on loopback ────────────────────────────────────────

REST_PREFIX = "/api/v3"
GRAPHQL_PATH = "/api/graphql"
#: Which pull request connection a query reads: the field under ``pullRequest``.
_CONNECTION = re.compile(r"pullRequest\(number: \$number\) \{ (\w+)\(")


@dataclass
class Recorded:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: Any

    @property
    def connection(self) -> str | None:
        """The pull request connection this call read, when it was a GraphQL query."""

        if self.path != GRAPHQL_PATH or not isinstance(self.body, dict):
            return None
        match = _CONNECTION.search(str(self.body.get("query") or ""))
        return match.group(1) if match else None

    @property
    def is_query(self) -> bool:
        return self.connection is not None and str(
            self.body.get("query") or ""
        ).lstrip().startswith("query")


Route = Callable[[Recorded], tuple[int, Any]]


@dataclass
class Headed:
    """A payload a route serves with headers of its own, such as a page's ``Link``."""

    payload: Any
    headers: dict[str, str]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_PATCH(self) -> None:
        self._handle("PATCH")

    def log_message(self, *args: Any) -> None:
        """Keep the test output readable."""

    def _handle(self, method: str) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        recorded = Recorded(
            method=method,
            path=parsed.path,
            query=urllib.parse.parse_qs(parsed.query),
            headers={key.lower(): value for key, value in self.headers.items()},
            body=json.loads(raw) if raw else None,
        )
        server = self.server
        assert isinstance(server, _Server)
        server.recorded.append(recorded)

        route = server.routes.get((method, parsed.path))
        if route is None:
            self._respond(404, {"message": "Not Found"})
            return
        status, payload = route(recorded)
        if status in {301, 302, 307} and isinstance(payload, str):
            self.send_response(status)
            self.send_header("Location", payload)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        headers: dict[str, str] = {}
        if isinstance(payload, Headed):
            payload, headers = payload.payload, payload.headers
        self._respond(status, payload, headers)

    def _respond(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
        # Bytes go out as they are: what a proxy's sign-in page looks like to a
        # client that expected JSON.
        if isinstance(payload, bytes):
            body, content_type = payload, "text/html"
        else:
            body, content_type = json.dumps(payload).encode("utf-8"), "application/json"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recorded: list[Recorded] = []
        self.routes: dict[tuple[str, str], Route] = {}
        #: Pull request connection → the pages GraphQL hands out, in order.
        self.connections: dict[str, list[list[dict[str, Any]]]] = {}


@dataclass
class FakeGitHub:
    server: _Server

    @property
    def origin(self) -> str:
        host, port = self.server.server_address[:2]
        assert isinstance(host, str)
        return f"http://{host}:{port}"

    @property
    def base_url(self) -> str:
        return f"{self.origin}{REST_PREFIX}"

    @property
    def recorded(self) -> list[Recorded]:
        return list(self.server.recorded)

    @property
    def writes(self) -> list[Recorded]:
        """Every request that could have changed something on the platform."""

        return [item for item in self.recorded if item.method != "GET" and not item.is_query]

    def on(self, method: str, path: str, status: int = 200, payload: Any = None) -> None:
        self.server.routes[(method, f"{REST_PREFIX}{path}")] = lambda _: (status, payload)

    def on_call(self, method: str, path: str, route: Route) -> None:
        self.server.routes[(method, f"{REST_PREFIX}{path}")] = route

    def on_graphql(self, route: Route) -> None:
        self.server.routes[("POST", GRAPHQL_PATH)] = route

    def connection(self, name: str, *pages: list[dict[str, Any]]) -> None:
        """Serve one pull request connection as these pages, linked by cursors."""

        self.server.connections[name] = [list(page) for page in pages]

    def calls(self, method: str, suffix: str) -> list[Recorded]:
        return [
            item for item in self.recorded if item.method == method and item.path.endswith(suffix)
        ]

    def queries(self, connection: str) -> list[Recorded]:
        return [item for item in self.recorded if item.connection == connection]


def _answer_graphql(server: _Server) -> Route:
    """Page through whichever connection a query asks for; empty unless served."""

    def route(call: Recorded) -> tuple[int, Any]:
        name = call.connection
        if name is None:
            return 200, {"errors": [{"message": "This fake only answers connection queries."}]}
        pages = server.connections.get(name, [[]])
        cursor = call.body["variables"].get("cursor")
        index = int(cursor.removeprefix("cursor-")) if cursor else 0
        more = index + 1 < len(pages)
        page = {
            "pageInfo": {"hasNextPage": more, "endCursor": f"cursor-{index + 1}" if more else None},
            "nodes": pages[index],
        }
        return 200, {"data": {"repository": {"pullRequest": {name: page}}}}

    return route


@contextmanager
def fake_github() -> Iterator[FakeGitHub]:
    """A GitHub on loopback for as long as the block lasts; other test modules use it too."""

    server = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    api = FakeGitHub(server=server)
    api.on_graphql(_answer_graphql(server))
    try:
        yield api
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def github() -> Iterator[FakeGitHub]:
    with fake_github() as api:
        yield api


def make_adapter(
    github: FakeGitHub, *, actions: ActionsContext | None = None, **overrides: Any
) -> GitHubForgeAdapter:
    settings = Settings(
        forge="github",
        github_base_url=github.base_url,
        github_token=TOKEN,
        forge_timeout_seconds=10,
        **overrides,
    )
    return GitHubForgeAdapter(settings, actions)


# ── fixtures for one pull request ────────────────────────────────────────────

REPO = "octo-org/orders"
HEAD_SHA = "c" * 40
BASE_SHA = "d" * 40
PULL_PATH = "/repos/octo-org/orders/pulls/7"
COMMENTS_PATH = "/repos/octo-org/orders/issues/7/comments"
STATUS_PATH = f"/repos/octo-org/orders/statuses/{HEAD_SHA}"
RUN_URL = "https://github.com/octo-org/orders/actions/runs/4242"

PULL_REQUEST: dict[str, Any] = {
    "number": 7,
    "title": "feat: refund each line item",
    "body": "Split the whole-order refund into one refund per line item.",
    "user": {"login": "octo-dev"},
    "html_url": "https://github.com/octo-org/orders/pull/7",
    "head": {"ref": "feat/refund-split", "sha": HEAD_SHA},
    "base": {"ref": "main", "sha": BASE_SHA},
    "labels": [{"name": "refund"}, {"name": "ai-assisted"}],
    "changed_files": 1,
}


def file_entry(
    filename: str,
    status: str,
    patch: str | None,
    *,
    additions: int = 0,
    deletions: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    """One entry of the pull request files endpoint; ``patch=None`` leaves it out."""

    entry: dict[str, Any] = {
        "filename": filename,
        "status": status,
        "additions": additions,
        "deletions": deletions,
        "changes": additions + deletions,
        **extra,
    }
    if patch is not None:
        entry["patch"] = patch
    return entry


# GitHub's patches end without a newline; left that way on purpose.
FILE_ENTRY = file_entry(
    "src/app/refund.py",
    "modified",
    "@@ -10,2 +10,5 @@ def split_refund(order, amount):\n"
    "     total = 0\n"
    "-    return total\n"
    "+    share = float(amount) / len(order.items)\n"
    "+    for item in order.items:\n"
    "+        item.refund = share\n"
    "+    return total",
    additions=4,
    deletions=1,
)


def serve_pull_request(github: FakeGitHub, entries: list[dict[str, Any]], **overrides: Any) -> None:
    github.on(
        "GET", PULL_PATH, payload={**PULL_REQUEST, "changed_files": len(entries), **overrides}
    )
    github.on_call(
        "GET",
        PULL_PATH + "/files",
        lambda call: (200, entries if call.query.get("page") == ["1"] else []),
    )


# ── fetching ─────────────────────────────────────────────────────────────────


def test_a_pull_request_becomes_a_change_the_gates_can_read(github: FakeGitHub) -> None:
    serve_pull_request(github, [FILE_ENTRY])

    change = make_adapter(github).fetch_change(REPO, 7)

    assert change.forge == "github"
    assert (change.repo, change.number) == (REPO, 7)
    assert (change.head_sha, change.base_sha) == (HEAD_SHA, BASE_SHA)
    assert change.author == "octo-dev"
    assert (change.source_branch, change.target_branch) == ("feat/refund-split", "main")
    assert change.url == "https://github.com/octo-org/orders/pull/7"
    assert change.labels == ["refund", "ai-assisted"]
    assert change.description == PULL_REQUEST["body"]
    # The headers GitHub omits are the ones every other layer parses the diff by.
    assert "diff --git a/src/app/refund.py b/src/app/refund.py\n" in change.diff
    assert "--- a/src/app/refund.py\n+++ b/src/app/refund.py\n" in change.diff
    [item] = change.files_changed
    assert (item.path, item.change_type) == ("src/app/refund.py", "modified")
    assert (item.added_lines, item.removed_lines) == (4, 1)


def test_the_added_lines_a_gate_will_judge_come_out_at_their_real_line_numbers(
    github: FakeGitHub,
) -> None:
    serve_pull_request(github, [FILE_ENTRY])

    change = make_adapter(github).fetch_change(REPO, 7)

    # 10 is the hunk's first new line; the deletion does not consume one.
    assert changed_line_locations(change.diff) == {"src/app/refund.py": {11, 12, 13, 14}}


def test_added_deleted_renamed_and_copied_files_each_keep_their_place(
    github: FakeGitHub,
) -> None:
    serve_pull_request(
        github,
        [
            file_entry(
                "src/app/audit.py",
                "added",
                "@@ -0,0 +1,2 @@\n+import os\n+VALUE = os.environ",
                additions=2,
            ),
            file_entry(
                "src/app/legacy.py",
                "removed",
                "@@ -1,2 +0,0 @@\n-import os\n-VALUE = 1",
                deletions=2,
            ),
            file_entry(
                "src/app/refunds.py",
                "renamed",
                "@@ -1 +1 @@\n-total = 0\n+total = Decimal(0)",
                additions=1,
                deletions=1,
                previous_filename="src/app/refund.py",
            ),
            file_entry(
                "src/app/refund_v2.py",
                "copied",
                "@@ -1 +1 @@\n-total = 0\n+total = Decimal(1)",
                additions=1,
                deletions=1,
                previous_filename="src/app/refund.py",
            ),
        ],
    )

    change = make_adapter(github).fetch_change(REPO, 7)

    inventory = {item.path: item for item in change.files_changed}
    audit = inventory["src/app/audit.py"]
    assert (audit.change_type, audit.added_lines) == ("added", 2)
    # A deleted file that fell out of the list is a sensitive path the
    # structural guardrails would stop seeing.
    assert inventory["src/app/legacy.py"].change_type == "deleted"
    assert inventory["src/app/legacy.py"].removed_lines == 2
    assert inventory["src/app/refunds.py"].change_type == "renamed"
    assert inventory["src/app/refunds.py"].previous_path == "src/app/refund.py"
    # A copy is a new file; its source is still there.
    assert inventory["src/app/refund_v2.py"].change_type == "added"
    assert inventory["src/app/refund_v2.py"].previous_path is None
    assert "--- /dev/null\n+++ b/src/app/audit.py\n" in change.diff
    assert "--- a/src/app/legacy.py\n+++ /dev/null\n" in change.diff
    assert "diff --git a/src/app/refund.py b/src/app/refunds.py\n" in change.diff
    # Patches arrive without a final newline. Glued to the next file's header,
    # a file's last line would reach every rule as text nobody wrote.
    added = {
        path: [line.text for line in lines] for path, lines in added_lines(change.diff).items()
    }
    assert added["src/app/audit.py"] == ["import os", "VALUE = os.environ"]
    assert added["src/app/refunds.py"] == ["total = Decimal(0)"]
    assert added["src/app/refund_v2.py"] == ["total = Decimal(1)"]


@pytest.mark.parametrize(
    ("patch", "counts"),
    [("@@ -1 +1 @@\n-on: push\n+on: workflow_dispatch", (1, 1)), (None, (0, 0))],
    ids=["edited", "pure-rename"],
)
def test_a_file_renamed_out_of_a_sensitive_path_still_goes_to_a_person(
    github: FakeGitHub, patch: str | None, counts: tuple[int, int]
) -> None:
    additions, deletions = counts
    serve_pull_request(
        github,
        [
            FILE_ENTRY,
            file_entry(
                "docs/deploy.yml",
                "renamed",
                patch,
                additions=additions,
                deletions=deletions,
                previous_filename=".github/workflows/deploy.yml",
            ),
        ],
    )

    change = make_adapter(github).fetch_change(REPO, 7)
    decision = PolicyEngine(PolicyConfig(version="v1")).evaluate_change(change)

    # Moving the deploy workflow out of .github/ is a change to .github/: the
    # workflow stops running. Local intake reads the same rename as a deletion
    # and an addition and holds the change for it; judged by its new name
    # alone, the platform's rename walked past the guardrail.
    assert not decision.allowed
    assert decision.risk_level == "high"
    assert decision.violations == [
        "sensitive paths require human review: `.github/workflows/deploy.yml`"
    ]
    # Both names count toward the blast radius, as they do locally.
    assert decision.changed_files == [
        ".github/workflows/deploy.yml",
        "docs/deploy.yml",
        "src/app/refund.py",
    ]


def test_a_test_file_renamed_out_of_collection_is_seen_by_its_names(github: FakeGitHub) -> None:
    serve_pull_request(
        github,
        [
            FILE_ENTRY,
            file_entry(
                "tests/disabled/refund.py",
                "renamed",
                None,
                previous_filename="tests/test_refund.py",
            ),
        ],
    )

    change = make_adapter(github).fetch_change(REPO, 7)
    hits = detect(change.diff, change.renames)

    # A pure rename comes with no patch, so the diff alone never shows that
    # pytest stopped collecting the file. Its two names do.
    assert detect(change.diff) == []
    assert [(hit.check, hit.file, hit.line) for hit in hits] == [
        ("deleted_test", "tests/test_refund.py", None)
    ]
    # It moved no lines, which is also what a local move without edits counts.
    assert change.changed_lines == 5


def test_a_file_github_sends_without_a_patch_is_listed_but_never_reviewed(
    github: FakeGitHub,
) -> None:
    serve_pull_request(github, [FILE_ENTRY, file_entry("assets/logo.png", "added", None)])

    change = make_adapter(github).fetch_change(REPO, 7)

    [logo] = [item for item in change.files_changed if item.path == "assets/logo.png"]
    assert (logo.added_lines, logo.removed_lines) == (0, 0)
    assert "assets/logo.png" not in change.diff
    # Zero line counts is what the evidence pack reads as "nothing to review
    # here", so the description has to say so out loud too.
    assert "listed but not reviewed: assets/logo.png" in change.description


def test_a_patch_that_does_not_add_up_to_the_files_own_counts_is_not_judged(
    github: FakeGitHub,
) -> None:
    # GitHub says 500 lines were added; the patch it sent carries four.
    serve_pull_request(github, [{**FILE_ENTRY, "additions": 500, "changes": 501}])

    change = make_adapter(github).fetch_change(REPO, 7)

    # Judging the four would call the other 496 clean.
    assert change.diff == ""
    [item] = change.files_changed
    assert (item.added_lines, item.removed_lines) == (0, 0)
    assert "listed but not reviewed: src/app/refund.py" in change.description


@pytest.mark.parametrize(
    "entry",
    [
        file_entry(
            "../../etc/passwd",
            "modified",
            "@@ -1 +1 @@\n-root\n+intruder",
            additions=1,
            deletions=1,
        ),
        file_entry(
            "src/app/passwd.py",
            "renamed",
            "@@ -1 +1 @@\n-root\n+intruder",
            additions=1,
            deletions=1,
            previous_filename="../../etc/passwd",
        ),
    ],
    ids=["new-path", "renamed-from"],
)
def test_a_path_that_points_out_of_the_tree_never_becomes_a_reviewable_file(
    github: FakeGitHub, entry: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_pull_request(github, [FILE_ENTRY, entry])
    workspace = tmp_path / "repo"
    (workspace / "src" / "app").mkdir(parents=True)
    (workspace / "src" / "app" / "refund.py").write_text("VALUE = 1\n", encoding="utf-8")

    change = make_adapter(github).fetch_change(REPO, 7)

    assert "etc/passwd" not in change.diff
    [refused] = [item for item in change.files_changed if item.path == "../../etc/passwd"]
    assert (refused.added_lines, refused.removed_lines) == (0, 0)
    # Not a rename's previous path either, which is a path a write-back uses.
    assert all(item.previous_path is None for item in change.files_changed)
    # Listed rather than dropped, so the guardrails refuse the change for it...
    decision = PolicyEngine(PolicyConfig(version="v1")).evaluate_change(change)
    assert "unsafe path: `../../etc/passwd`" in decision.violations
    # ...while the tools are still never handed it.
    arguments = scanner_arguments(change, workspace, monkeypatch)
    assert "src/app/refund.py" in arguments
    assert not any("etc/passwd" in argument for argument in arguments)
    # Named, and in its own words: "GitHub sent no diff" and "this name cannot
    # be used as a path" are not the same news.
    assert "not safe to use as paths" in change.description
    assert "../../etc/passwd" in change.description


@pytest.mark.parametrize(
    "char", REFUSED_NAME_CHARACTERS.values(), ids=REFUSED_NAME_CHARACTERS.keys()
)
def test_a_file_whose_name_holds_a_control_character_keeps_the_change_from_passing(
    github: FakeGitHub, char: str
) -> None:
    hostile = f"src/app/refund{char}.py"
    serve_pull_request(
        github,
        [FILE_ENTRY, file_entry(hostile, "added", "@@ -0,0 +1 @@\n+import os", additions=1)],
    )

    change = make_adapter(github).fetch_change(REPO, 7)
    decision = PolicyEngine(PolicyConfig(version="v1")).evaluate_change(change)

    # Left out of the change, the file would leave one clean file for the
    # guardrails to judge, and nothing of it but a line in the description.
    assert not decision.allowed
    assert decision.risk_level == "high"
    assert decision.violations == [f"unsafe path: `{hostile!r}`"]
    # The description names it too, escaped: the reading aid gets it as a prompt.
    assert repr(hostile) in change.description
    assert hostile not in change.description


@pytest.mark.parametrize(
    "char", REFUSED_NAME_CHARACTERS.values(), ids=REFUSED_NAME_CHARACTERS.keys()
)
def test_a_file_whose_name_holds_a_control_character_reaches_neither_the_diff_nor_a_tool(
    github: FakeGitHub, char: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hostile = f"src/app/refund{char}.py"
    serve_pull_request(
        github,
        [FILE_ENTRY, file_entry(hostile, "added", "@@ -0,0 +1 @@\n+import os", additions=1)],
    )
    # In the tree, and a Python file: only its name keeps it off a command line.
    workspace = tmp_path / "repo"
    (workspace / "src" / "app").mkdir(parents=True)
    for name in ("src/app/refund.py", hostile):
        (workspace / name).write_text("VALUE = 1\n", encoding="utf-8")

    change = make_adapter(github).fetch_change(REPO, 7)

    [listed] = [item for item in change.files_changed if item.path == hostile]
    assert (listed.added_lines, listed.removed_lines) == (0, 0)
    assert extract_changed_files(change.diff) == ["src/app/refund.py"]
    arguments = scanner_arguments(change, workspace, monkeypatch)
    assert "src/app/refund.py" in arguments
    assert not any(hostile in argument for argument in arguments)


@pytest.mark.parametrize(
    ("source", "copy", "listed"),
    [
        (
            "src/app/refund\t.py",
            "src/app/refund_v2.py",
            {"src/app/refund_v2.py": "added", "src/app/refund\t.py": "modified"},
        ),
        ("src/app/refund.py", "src/app/refund\t_v2.py", {"src/app/refund\t_v2.py": "added"}),
    ],
    ids=["from-a-refused-name", "to-a-refused-name"],
)
def test_a_copy_is_held_for_whichever_of_its_names_cannot_be_used(
    github: FakeGitHub, source: str, copy: str, listed: dict[str, str]
) -> None:
    serve_pull_request(
        github,
        [
            file_entry(
                copy,
                "copied",
                "@@ -1 +1 @@\n-total = 0\n+total = 1",
                additions=1,
                deletions=1,
                previous_filename=source,
            )
        ],
    )

    change = make_adapter(github).fetch_change(REPO, 7)
    decision = PolicyEngine(PolicyConfig(version="v1")).evaluate_change(change)

    assert change.diff == ""
    # A copy leaves its source where it was, so the source is listed only when
    # it is the name the guardrails would otherwise never see.
    assert {item.path: item.change_type for item in change.files_changed} == listed
    assert not decision.allowed


@pytest.mark.parametrize(
    ("name", "header"),
    [
        # Git ends a label that holds a space with a tab; the header written
        # here has to read back the way git's own would.
        ("src/app/refund.py ", "+++ b/src/app/refund.py \t\n"),
        ("src/app/refund.py\u00a0", "+++ b/src/app/refund.py\u00a0\n"),
    ],
    ids=["space", "no-break-space"],
)
def test_a_name_ending_in_whitespace_keeps_its_own_lines(
    github: FakeGitHub, name: str, header: str
) -> None:
    serve_pull_request(
        github, [FILE_ENTRY, file_entry(name, "added", "@@ -0,0 +1 @@\n+import os", additions=1)]
    )

    change = make_adapter(github).fetch_change(REPO, 7)

    added = {
        path: [line.text for line in lines] for path, lines in added_lines(change.diff).items()
    }
    # Trimmed, the name would lend its lines to refund.py, or -- not adding up
    # to its own counts under a name it no longer has -- go unreviewed.
    assert added[name] == ["import os"]
    assert "import os" not in added["src/app/refund.py"]
    counts = {item.path: (item.added_lines, item.removed_lines) for item in change.files_changed}
    assert counts == {"src/app/refund.py": (4, 1), name: (1, 0)}
    assert change.description == PULL_REQUEST["body"]
    assert f"--- /dev/null\n{header}" in change.diff


def test_a_file_github_lists_without_a_name_stops_the_run(github: FakeGitHub) -> None:
    serve_pull_request(github, [FILE_ENTRY, file_entry("", "added", "@@ -0,0 +1 @@\n+x = 1")])

    with pytest.raises(GitHubError, match="without a name"):
        make_adapter(github).fetch_change(REPO, 7)


def test_a_pull_request_with_no_head_commit_is_refused(github: FakeGitHub) -> None:
    serve_pull_request(github, [FILE_ENTRY], head={"ref": "feat/refund-split", "sha": None})

    # No head, no fingerprint: an approval would have nothing to bind to.
    with pytest.raises(GitHubError, match="no head commit"):
        make_adapter(github).fetch_change(REPO, 7)


def test_a_pull_request_past_githubs_file_cap_is_refused_before_a_file_is_read(
    github: FakeGitHub,
) -> None:
    github.on("GET", PULL_PATH, payload={**PULL_REQUEST, "changed_files": 3001})

    # Judging the 3000 files GitHub lists and reporting "no findings" for the
    # rest is exactly the failure this product exists to prevent.
    with pytest.raises(GitHubError, match="at most 3000"):
        make_adapter(github).fetch_change(REPO, 7)

    assert github.calls("GET", "/files") == []


@pytest.mark.parametrize("changed_files", [2, 0], ids=["fewer-listed", "more-listed"])
def test_a_file_list_that_does_not_match_the_pull_request_is_refused(
    github: FakeGitHub, changed_files: int
) -> None:
    serve_pull_request(github, [FILE_ENTRY], changed_files=changed_files)

    with pytest.raises(GitHubError, match=f"changes {changed_files} files, but GitHub listed 1"):
        make_adapter(github).fetch_change(REPO, 7)


def test_a_pull_request_that_moves_while_its_files_are_read_is_refused(
    github: FakeGitHub,
) -> None:
    reads: list[str] = []

    def pull(_: Recorded) -> tuple[int, Any]:
        reads.append("read")
        sha = HEAD_SHA if len(reads) == 1 else "e" * 40
        return 200, {**PULL_REQUEST, "head": {**PULL_REQUEST["head"], "sha": sha}}

    github.on_call("GET", PULL_PATH, pull)
    github.on("GET", PULL_PATH + "/files", payload=[FILE_ENTRY])

    # A push between two pages would mix two heads' files into one diff.
    with pytest.raises(GitHubError, match="moved while its files were being read"):
        make_adapter(github).fetch_change(REPO, 7)


def test_the_file_list_is_read_across_pages(github: FakeGitHub) -> None:
    first = [{**FILE_ENTRY, "filename": f"src/f{index}.py"} for index in range(FILE_PAGE_SIZE)]
    github.on("GET", PULL_PATH, payload={**PULL_REQUEST, "changed_files": FILE_PAGE_SIZE + 1})
    github.on_call(
        "GET",
        PULL_PATH + "/files",
        lambda call: (
            200,
            first if call.query["page"] == ["1"] else [{**FILE_ENTRY, "filename": "src/last.py"}],
        ),
    )

    change = make_adapter(github).fetch_change(REPO, 7)

    assert len(change.files_changed) == FILE_PAGE_SIZE + 1
    assert change.files_changed[-1].path == "src/last.py"
    pages = github.calls("GET", "/files")
    assert [call.query["page"] for call in pages] == [["1"], ["2"]]
    assert all(call.query["per_page"] == [str(FILE_PAGE_SIZE)] for call in pages)


def test_a_pull_request_at_exactly_the_cap_is_read_whole(
    github: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("honest_ci.adapters.github.MAX_FILES", 2 * FILE_PAGE_SIZE)
    github.on("GET", PULL_PATH, payload={**PULL_REQUEST, "changed_files": 2 * FILE_PAGE_SIZE})
    github.on_call(
        "GET",
        PULL_PATH + "/files",
        lambda call: (
            200,
            [
                {**FILE_ENTRY, "filename": f"src/p{call.query['page'][0]}_{index}.py"}
                for index in range(FILE_PAGE_SIZE)
            ],
        ),
    )

    change = make_adapter(github).fetch_change(REPO, 7)

    # Every page was full, so nothing but the count and the cap say the list
    # is over -- and a complete list is not a reason to refuse the change.
    assert len(change.files_changed) == 2 * FILE_PAGE_SIZE
    assert len(github.calls("GET", "/files")) == 2


def test_a_files_response_that_is_not_a_list_stops_the_run(github: FakeGitHub) -> None:
    github.on("GET", PULL_PATH, payload=PULL_REQUEST)
    github.on("GET", PULL_PATH + "/files", payload={"message": "unexpected"})

    with pytest.raises(GitHubError, match="did not return a list"):
        make_adapter(github).fetch_change(REPO, 7)


@pytest.mark.parametrize(
    "repo", ["octo-org", "octo-org/orders/pulls", "../orders", "octo-org/..", "octo org/orders"]
)
def test_a_repository_that_is_not_owner_and_name_is_refused_before_any_request(
    github: FakeGitHub, repo: str
) -> None:
    # The name becomes part of a URL path; a slash or a dot segment in it
    # would point the token at an endpoint nobody configured.
    with pytest.raises(GitHubError, match="owner/name"):
        make_adapter(github).fetch_change(repo, 7)

    assert github.recorded == []


# ── the credential boundary ──────────────────────────────────────────────────


def test_the_token_travels_as_a_bearer_header_and_never_in_the_url(github: FakeGitHub) -> None:
    serve_pull_request(github, [FILE_ENTRY])
    adapter = make_adapter(github)

    adapter.fetch_reviews(adapter.fetch_change(REPO, 7))

    calls = github.recorded
    assert {REST_PREFIX + PULL_PATH, GRAPHQL_PATH} <= {call.path for call in calls}
    assert all(call.headers.get("authorization") == f"Bearer {TOKEN}" for call in calls)
    assert all(call.headers.get("x-github-api-version") == "2022-11-28" for call in calls)
    assert all(
        TOKEN not in call.path
        and TOKEN not in str(call.query)
        and TOKEN not in json.dumps(call.body)
        for call in calls
    )


def test_a_redirect_off_the_configured_base_is_refused(github: FakeGitHub) -> None:
    # Same host, but outside /api/v3: a proxy's sign-in page, say. Followed,
    # it would receive the token.
    github.server.routes[("GET", "/login")] = lambda _: (200, PULL_REQUEST)
    github.on_call("GET", PULL_PATH, lambda _: (302, f"{github.origin}/login"))

    with pytest.raises(GitHubError, match="Outbound URL refused"):
        make_adapter(github).fetch_change(REPO, 7)

    assert [call.path for call in github.recorded] == [REST_PREFIX + PULL_PATH]


def test_a_graphql_redirect_is_held_to_the_graphql_endpoint(github: FakeGitHub) -> None:
    github.on("GET", "/graphql", payload={"data": {}})
    github.on_graphql(lambda _: (302, f"{github.base_url}/graphql"))

    # GraphQL is not under the REST base on Enterprise Server, so its boundary
    # is its own endpoint -- the REST base is outside it too.
    with pytest.raises(GitHubError, match="Outbound URL refused"):
        make_adapter(github).fetch_reviews(make_change())

    assert github.calls("GET", "/graphql") == []


def test_an_error_body_that_echoes_a_credential_is_redacted_before_it_is_raised(
    github: FakeGitHub,
) -> None:
    github.on("GET", PULL_PATH, status=401, payload={"message": f"Bad credentials: {TOKEN}"})

    with pytest.raises(GitHubError) as excinfo:
        make_adapter(github).fetch_change(REPO, 7)

    assert TOKEN not in str(excinfo.value)
    assert "401" in str(excinfo.value)


def test_a_sign_in_page_answering_instead_of_the_api_stops_the_run(github: FakeGitHub) -> None:
    github.on("GET", PULL_PATH, payload=b"<html><body>Sign in to continue</body></html>")

    with pytest.raises(GitHubError, match="did not return JSON"):
        make_adapter(github).fetch_change(REPO, 7)


def test_a_missing_token_fails_before_any_request_is_made(github: FakeGitHub) -> None:
    adapter = GitHubForgeAdapter(
        Settings(forge="github", github_base_url=github.base_url, github_token=None)
    )

    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        adapter.fetch_change(REPO, 7)

    assert github.recorded == []


def test_the_token_reaches_git_as_a_scoped_header_rather_than_an_argument(
    github: FakeGitHub,
) -> None:
    # Reached through the private helper because there is no public surface for
    # "what environment did the child process get", and that environment is the
    # whole property: argv is readable from any process list.
    env = make_adapter(github)._git_env("https://github.example.com/octo-org/orders.git")

    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.example.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_0"] == f"Authorization: Basic {BASIC}"
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_a_clone_url_that_would_not_carry_the_header_does_not_get_one(
    github: FakeGitHub,
) -> None:
    env = make_adapter(github)._git_env("git@github.com:octo-org/orders.git")

    assert "GIT_CONFIG_VALUE_0" not in env


def test_git_sends_the_token_to_the_clone_origin_and_writes_it_nowhere(
    github: FakeGitHub, tmp_path: Path
) -> None:
    # The fake answers git's first request with a 404, which is enough: the
    # request it received is the one that shows how the token travelled.
    github.on(
        "GET",
        "/repos/octo-org/orders",
        payload={"clone_url": f"{github.origin}/octo-org/orders.git"},
    )

    with pytest.raises(GitHubError, match="git fetch failed") as excinfo:
        make_adapter(github).materialize(make_change(), tmp_path / "workspaces")

    fetches = [call for call in github.recorded if call.path.startswith("/octo-org/orders.git/")]
    assert fetches
    assert all(call.headers.get("authorization") == f"Basic {BASIC}" for call in fetches)
    assert all(TOKEN not in call.path and TOKEN not in str(call.query) for call in fetches)
    assert TOKEN not in str(excinfo.value)
    # The scanners copy this tree into a sandbox. A token written into its git
    # config would ride along with it.
    config = tmp_path / "workspaces" / f"octo-org-orders-{HEAD_SHA[:12]}" / ".git" / "config"
    assert BASIC not in config.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://api.github.com", "https://api.github.com/graphql"),
        ("https://api.github.com/", "https://api.github.com/graphql"),
        ("https://github.example.com/api/v3", "https://github.example.com/api/graphql"),
        ("https://github.example.com/api/v3/", "https://github.example.com/api/graphql"),
    ],
)
def test_graphql_is_found_beside_rest_on_github_com_and_under_api_on_enterprise_server(
    base: str, expected: str
) -> None:
    assert graphql_url(base) == expected


def test_the_registry_builds_a_github_adapter_only_when_there_is_a_token() -> None:
    adapter = build_forge_adapter(Settings(forge="github", github_token=TOKEN))

    assert isinstance(adapter, GitHubForgeAdapter)
    assert build_forge_adapter(Settings(forge="github", github_token=None)) is None


# ── writing back ─────────────────────────────────────────────────────────────


def make_change() -> ChangeRequest:
    return ChangeRequest(
        forge="github",
        repo=REPO,
        number=7,
        head_sha=HEAD_SHA,
        source_branch="feat/refund-split",
        target_branch="main",
        diff=FILE_ENTRY["patch"],
    )


def make_evidence(*, verdict: Verdict = Verdict.BLOCKED, detail: str = "") -> EvidencePack:
    finding = Finding(
        id="review.R-01.1",
        gate="review",
        rule_id="R-01",
        severity="high",
        title="Money amounts must not use float",
        detail=detail,
        file="src/app/refund.py",
        line_start=11,
        suggested_fix="Use decimal.Decimal.",
    )
    return EvidencePack(
        run_id="run0001",
        change=make_change(),
        gate_reports=[
            GateReport(
                gate="review",
                verdict=verdict,
                summary="1 rule (v1) checked; 1 hit, 1 of them blocking.",
                findings=[finding],
            )
        ],
        verdict=verdict,
        fingerprint="f" * 64,
    )


def state_with(evidence: EvidencePack, status: RunStatus = "blocked") -> RunState:
    return RunState(
        run_id=evidence.run_id,
        change=evidence.change,
        status=status,
        evidence=evidence,
        gate_reports={report.gate: report for report in evidence.gate_reports},
    )


#: Past 2^31: the size of id a GraphQL ``Int`` can no longer carry, which is
#: why the comment is edited by ``fullDatabaseId``.
OWN_COMMENT_ID = "2147483648123"


def own(body: str, identifier: str = OWN_COMMENT_ID) -> dict[str, Any]:
    return {"fullDatabaseId": identifier, "viewerDidAuthor": True, "body": body}


def someone_elses(body: str) -> dict[str, Any]:
    return {"fullDatabaseId": "11", "viewerDidAuthor": False, "body": body}


def test_the_first_conclusion_is_posted_as_a_comment_on_the_pull_request(
    github: FakeGitHub,
) -> None:
    github.on("POST", COMMENTS_PATH, status=201, payload={"id": 1})

    make_adapter(github).publish_report(make_change(), make_evidence(), "blocked")

    [post] = github.calls("POST", COMMENTS_PATH)
    body = post.body["body"]
    assert body.startswith(VERDICT_MARKER)
    assert "[R-01] Money amounts must not use float" in body
    assert "Use decimal.Decimal." in body
    # The comment is a summary; it has to say where the full record is.
    assert "honest-ci evidence run0001" in body


def test_a_rerun_edits_the_conclusion_this_deployment_wrote_instead_of_adding_one(
    github: FakeGitHub,
) -> None:
    github.connection("comments", [someone_elses("LGTM"), own(f"{VERDICT_MARKER}\nold")])
    github.on("PATCH", f"/repos/octo-org/orders/issues/comments/{OWN_COMMENT_ID}", payload={})
    evidence = make_evidence().model_copy(update={"run_id": "run0002"})

    make_adapter(github).publish_report(make_change(), evidence, "blocked")

    assert github.calls("POST", COMMENTS_PATH) == []
    [edit] = github.calls("PATCH", f"/issues/comments/{OWN_COMMENT_ID}")
    assert edit.body["body"].startswith(VERDICT_MARKER)
    assert "run0002" in edit.body["body"]


#: What a run with no evidence pack to report writes, such as one the license
#: check stopped.
NOTE = f"{VERDICT_MARKER}\n## honest-ci · not run\n\nNo gate ran.\n"


def test_a_note_is_posted_where_the_conclusion_would_be_when_there_is_none_yet(
    github: FakeGitHub,
) -> None:
    github.on("POST", COMMENTS_PATH, status=201, payload={"id": 1})

    make_adapter(github).publish_note(make_change(), NOTE)

    [post] = github.calls("POST", COMMENTS_PATH)
    assert post.body == {"body": NOTE}


def test_a_note_takes_the_place_of_the_conclusion_this_deployment_wrote(
    github: FakeGitHub,
) -> None:
    github.connection("comments", [someone_elses("LGTM"), own(f"{VERDICT_MARKER}\nmergeable")])
    github.on("PATCH", f"/repos/octo-org/orders/issues/comments/{OWN_COMMENT_ID}", payload={})

    make_adapter(github).publish_note(make_change(), NOTE)

    # The last conclusion was about another run; left up, it would read as this one's.
    assert github.calls("POST", COMMENTS_PATH) == []
    [edit] = github.calls("PATCH", f"/issues/comments/{OWN_COMMENT_ID}")
    assert edit.body == {"body": NOTE}


@pytest.mark.parametrize(
    "comment",
    [
        someone_elses(f"{VERDICT_MARKER}\nA reviewer quoting the last conclusion."),
        own("A note this token's owner wrote by hand."),
    ],
    ids=["marker-by-someone-else", "ours-without-marker"],
)
def test_only_a_marked_comment_this_deployment_wrote_is_ever_edited(
    github: FakeGitHub, comment: dict[str, Any]
) -> None:
    github.connection("comments", [comment])
    github.on("POST", COMMENTS_PATH, status=201, payload={"id": 1})

    make_adapter(github).publish_report(make_change(), make_evidence(), "blocked")

    # Anyone can type the marker. Overwriting their comment because of it
    # would put this deployment's words under someone else's name.
    assert github.calls("PATCH", "/issues/comments/11") == []
    assert github.calls("PATCH", f"/issues/comments/{OWN_COMMENT_ID}") == []
    assert len(github.calls("POST", COMMENTS_PATH)) == 1


def test_the_conclusion_search_reaches_later_pages(github: FakeGitHub) -> None:
    github.connection("comments", [someone_elses("LGTM")] * 100, [own(f"{VERDICT_MARKER}\nold")])
    github.on("PATCH", f"/repos/octo-org/orders/issues/comments/{OWN_COMMENT_ID}", payload={})

    make_adapter(github).publish_report(make_change(), make_evidence(), "blocked")

    assert len(github.queries("comments")) == 2
    assert github.calls("POST", COMMENTS_PATH) == []
    assert len(github.calls("PATCH", f"/issues/comments/{OWN_COMMENT_ID}")) == 1


def test_a_conversation_too_long_to_search_cannot_create_a_duplicate(
    github: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("honest_ci.adapters.github.MAX_COMMENT_PAGES", 1)
    github.connection("comments", [someone_elses("LGTM")] * 100, [own(VERDICT_MARKER)])

    # "Not found in the part we read" is not "not there": posting now could
    # put a second conclusion under the one on page two.
    with pytest.raises(GitHubError, match="not read in full"):
        make_adapter(github).publish_report(make_change(), make_evidence(), "blocked")

    assert github.writes == []


def test_a_conclusion_comment_without_a_usable_id_is_not_duplicated(
    github: FakeGitHub,
) -> None:
    github.connection("comments", [own(VERDICT_MARKER, identifier="")])

    with pytest.raises(GitHubError, match="no usable id"):
        make_adapter(github).publish_report(make_change(), make_evidence(), "blocked")

    assert github.writes == []


def test_a_credential_quoted_by_a_tool_never_reaches_the_pull_request(
    github: FakeGitHub,
) -> None:
    github.on("POST", COMMENTS_PATH, status=201, payload={"id": 1})
    evidence = make_evidence(detail=f"ruff: hardcoded token {TOKEN} in settings")

    make_adapter(github).publish_report(make_change(), evidence, "blocked")

    assert TOKEN not in github.calls("POST", COMMENTS_PATH)[0].body["body"]


def test_a_change_without_a_pull_request_number_cannot_be_written_back(
    github: FakeGitHub,
) -> None:
    change = make_change().model_copy(update={"number": None})

    with pytest.raises(GitHubError, match="no pull request number"):
        make_adapter(github).publish_report(change, make_evidence(), "blocked")

    assert github.recorded == []


@pytest.mark.parametrize(
    ("verdict", "state"),
    [
        (Verdict.PASS, "success"),
        (Verdict.BLOCKED, "failure"),
        (Verdict.NEEDS_HUMAN, "pending"),
        (Verdict.SKIPPED, "pending"),
    ],
)
def test_only_a_pass_becomes_a_green_commit_status(
    github: FakeGitHub, verdict: Verdict, state: str
) -> None:
    github.on("POST", STATUS_PATH, status=201, payload={"id": 2})

    make_adapter(github).publish_status(make_change(), verdict, "summary")

    [call] = github.calls("POST", STATUS_PATH)
    assert call.body["state"] == state == STATUS_STATE[verdict]
    # The name a branch protection rule requires.
    assert call.body["context"] == "honest-ci/verify"
    assert "target_url" not in call.body


def test_a_run_that_fell_over_is_reported_as_an_error_not_as_a_failure(
    github: FakeGitHub,
) -> None:
    github.on("POST", STATUS_PATH, status=201, payload={"id": 2})

    make_adapter(github).publish_error(make_change(), "failed · run run0001", RUN_URL)

    [call] = github.calls("POST", STATUS_PATH)
    # "The gates blocked this" and "nothing concluded anything" are different
    # news; both still block a required check.
    assert call.body["state"] == ERROR_STATE == "error"
    assert call.body["target_url"] == RUN_URL


def test_progress_is_a_pending_status_under_the_same_name(github: FakeGitHub) -> None:
    github.on("POST", STATUS_PATH, status=201, payload={"id": 2})

    make_adapter(github).publish_progress(make_change(), "verifying · run run0001")

    [call] = github.calls("POST", STATUS_PATH)
    assert (call.body["state"], call.body["context"]) == ("pending", "honest-ci/verify")


def test_progress_links_to_the_run_it_announces(github: FakeGitHub) -> None:
    github.on("POST", STATUS_PATH, status=201, payload={"id": 2})

    make_adapter(github).publish_progress(make_change(), "Verifying · run run0001", RUN_URL)

    [call] = github.calls("POST", STATUS_PATH)
    assert call.body["target_url"] == RUN_URL


def test_the_status_carries_the_name_this_deployment_configured(github: FakeGitHub) -> None:
    github.on("POST", STATUS_PATH, status=201, payload={"id": 2})

    make_adapter(github, forge_status_name="ci/honest").publish_status(
        make_change(), Verdict.PASS, "summary"
    )

    assert github.calls("POST", STATUS_PATH)[0].body["context"] == "ci/honest"


def test_a_long_status_description_is_cut_to_what_github_accepts(github: FakeGitHub) -> None:
    github.on("POST", STATUS_PATH, status=201, payload={"id": 2})

    make_adapter(github).publish_status(
        make_change(), Verdict.BLOCKED, "blocked ·\n" + "x" * 400 + f" {TOKEN}"
    )

    description = github.calls("POST", STATUS_PATH)[0].body["description"]
    # GitHub answers 422 to a longer one, and a status that never lands leaves
    # the required check waiting.
    assert len(description) == MAX_STATUS_DESCRIPTION
    assert "\n" not in description


def test_a_status_description_never_carries_a_credential(github: FakeGitHub) -> None:
    github.on("POST", STATUS_PATH, status=201, payload={"id": 2})

    make_adapter(github).publish_status(make_change(), Verdict.BLOCKED, f"blocked · {TOKEN}")

    assert TOKEN not in github.calls("POST", STATUS_PATH)[0].body["description"]


def test_a_failed_run_reaches_github_as_an_error_linked_to_the_run(github: FakeGitHub) -> None:
    github.on("POST", COMMENTS_PATH, status=201, payload={"id": 1})
    github.on("POST", STATUS_PATH, status=201, payload={"id": 2})

    published = publish_conclusion(
        make_adapter(github), state_with(make_evidence(), "failed"), target_url=RUN_URL
    )

    assert published.ok and published.sent
    [status] = github.calls("POST", STATUS_PATH)
    assert (status.body["state"], status.body["target_url"]) == ("error", RUN_URL)


def test_inline_comments_are_not_offered_yet(github: FakeGitHub) -> None:
    adapter = make_adapter(github)
    evidence = make_evidence()

    # Empty is the port's honest "nothing was posted"; the run's output reports
    # it as such rather than as a success.
    assert adapter.publish_inline(make_change(), evidence, evidence.findings) == []
    assert github.recorded == []


def test_a_dry_run_sends_nothing(github: FakeGitHub) -> None:
    published = publish_conclusion(make_adapter(github), state_with(make_evidence()), dry_run=True)

    assert published.ok and not published.sent
    assert github.recorded == []


# ── review comments already on the pull request ──────────────────────────────


def comment(
    body: str = "Should this split use Decimal?",
    *,
    login: str = "octo-reviewer",
    bot: bool = False,
) -> dict[str, Any]:
    return {
        "author": {"__typename": "Bot" if bot else "User", "login": login},
        "body": body,
        "createdAt": "2026-09-01T10:00:00Z",
    }


def thread(
    *comments: dict[str, Any],
    resolved: bool = False,
    path: str = "src/app/refund.py",
    line: int | None = 11,
) -> dict[str, Any]:
    return {
        "isResolved": resolved,
        "path": path,
        "line": line,
        "comments": {"totalCount": len(comments), "nodes": list(comments)},
    }


def test_an_open_review_thread_comes_back_located_and_unresolved(github: FakeGitHub) -> None:
    github.connection("reviewThreads", [thread(comment())])

    [review] = make_adapter(github).fetch_reviews(make_change())

    assert (review.author, review.path, review.line) == ("octo-reviewer", "src/app/refund.py", 11)
    assert review.unresolved and not review.is_bot
    assert review.created_at is not None and review.created_at.year == 2026


def test_every_reply_in_a_resolved_thread_is_resolved(github: FakeGitHub) -> None:
    github.connection(
        "reviewThreads",
        [thread(comment(), comment("Fixed in the next push.", login="octo-dev"), resolved=True)],
    )

    reviews = make_adapter(github).fetch_reviews(make_change())

    assert [review.author for review in reviews] == ["octo-reviewer", "octo-dev"]
    assert all(review.resolved is True and not review.unresolved for review in reviews)


def test_an_outdated_thread_keeps_its_file_without_a_line(github: FakeGitHub) -> None:
    github.connection("reviewThreads", [thread(comment(), line=None)])

    [review] = make_adapter(github).fetch_reviews(make_change())

    # The line moved away under a later push; the file is still worth knowing,
    # and the question is still open.
    assert (review.path, review.line) == ("src/app/refund.py", None)
    assert review.unresolved


def test_conversation_comments_and_review_summaries_are_opinions_not_open_questions(
    github: FakeGitHub,
) -> None:
    github.connection("comments", [comment("Looks fine overall.", login="octo-lead")])
    github.connection(
        "reviews",
        [comment("Please add a rounding test.", login="octo-qa"), comment("", login="octo-lead")],
    )

    reviews = make_adapter(github).fetch_reviews(make_change())

    # A review submitted with no text (a bare "Approve") says nothing to carry.
    assert [review.author for review in reviews] == ["octo-lead", "octo-qa"]
    # GitHub cannot resolve either kind, so neither is a question left open.
    assert all(review.resolved is None and not review.unresolved for review in reviews)
    assert all(review.path is None and review.line is None for review in reviews)


def test_the_platforms_own_assistant_is_recorded_rather_than_discarded(
    github: FakeGitHub,
) -> None:
    github.connection("reviewThreads", [thread(comment(login="octo-assistant", bot=True))])

    assert make_adapter(github).fetch_reviews(make_change())[0].is_bot


def test_this_deployments_own_comments_are_never_read_back_as_a_review(
    github: FakeGitHub,
) -> None:
    github.connection(
        "reviewThreads",
        [
            thread(
                comment(
                    f"{finding_marker('review.R-01', HEAD_SHA)}\n**Money must not use float**",
                    login="github-actions",
                    bot=True,
                )
            ),
            thread(comment()),
        ],
    )
    github.connection(
        "comments", [comment(f"{VERDICT_MARKER}\n## honest-ci", login="github-actions", bot=True)]
    )

    reviews = make_adapter(github).fetch_reviews(make_change())

    # Reading its own output back would let the product cite itself as a human
    # who raised the concern -- and its own open inline thread would then route
    # every rerun to a person forever.
    assert [review.author for review in reviews] == ["octo-reviewer"]


def test_a_comment_from_a_deleted_account_still_counts(github: FakeGitHub) -> None:
    github.connection("reviewThreads", [thread({**comment(), "author": None})])

    [review] = make_adapter(github).fetch_reviews(make_change())

    # Its author is gone; the question it asked is not.
    assert review.author == "" and review.unresolved


def test_a_comment_quoting_a_credential_is_redacted_before_it_is_stored(
    github: FakeGitHub,
) -> None:
    github.connection("comments", [comment(f"CI uses {TOKEN}, do not change it")])

    assert TOKEN not in make_adapter(github).fetch_reviews(make_change())[0].body


def test_review_threads_are_read_across_pages(github: FakeGitHub) -> None:
    github.connection(
        "reviewThreads", [thread(comment())] * 100, [thread(comment(login="octo-late"))]
    )

    reviews = make_adapter(github).fetch_reviews(make_change())

    assert len(reviews) == 101 and reviews[-1].author == "octo-late"
    cursors = [call.body["variables"]["cursor"] for call in github.queries("reviewThreads")]
    assert cursors == [None, "cursor-1"]


def test_a_thread_longer_than_one_page_of_comments_is_refused(github: FakeGitHub) -> None:
    long_thread = thread(*[comment()] * 100)
    long_thread["comments"]["totalCount"] = 101
    github.connection("reviewThreads", [long_thread])

    # The reply that left the thread open may be the 101st.
    with pytest.raises(GitHubError, match="more than 100 comments"):
        make_adapter(github).fetch_reviews(make_change())


def test_a_thread_without_its_state_is_refused(github: FakeGitHub) -> None:
    github.connection("reviewThreads", [{"path": "src/app/refund.py", "comments": {}}])

    with pytest.raises(GitHubError, match="without its state"):
        make_adapter(github).fetch_reviews(make_change())


def test_more_comments_than_the_cap_are_refused_rather_than_undercounted(
    github: FakeGitHub,
) -> None:
    github.connection("comments", [comment()] * 100, [comment()] * 100, [comment()])

    # Keeping the first N would drop whichever open thread happened to land in
    # the tail, and nothing in the pack would contradict the undercount.
    with pytest.raises(GitHubError, match=f"more than {MAX_PLATFORM_REVIEWS}"):
        make_adapter(github).fetch_reviews(make_change())


def test_a_conversation_too_long_to_page_through_is_refused(
    github: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("honest_ci.adapters.github.MAX_COMMENT_PAGES", 1)
    github.connection("reviewThreads", [thread(comment())], [thread(comment())])

    with pytest.raises(GitHubError, match="not read in full"):
        make_adapter(github).fetch_reviews(make_change())


def test_a_page_that_says_there_is_more_but_not_where_is_refused(github: FakeGitHub) -> None:
    github.on_graphql(
        lambda call: (
            200,
            {
                "data": {
                    "repository": {
                        "pullRequest": {
                            str(call.connection): {
                                "pageInfo": {"hasNextPage": True, "endCursor": None},
                                "nodes": [],
                            }
                        }
                    }
                }
            },
        )
    )

    with pytest.raises(GitHubError, match="but not where"):
        make_adapter(github).fetch_reviews(make_change())


def test_a_graphql_refusal_inside_a_200_is_an_error_not_an_empty_conversation(
    github: FakeGitHub,
) -> None:
    github.on_graphql(
        lambda call: (
            200,
            {
                # Partial data beside the error is still partial.
                "data": {
                    "repository": {
                        "pullRequest": {
                            str(call.connection): {
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                                "nodes": [],
                            }
                        }
                    }
                },
                "errors": [{"message": f"Resource not accessible by integration ({TOKEN})"}],
            },
        )
    )

    with pytest.raises(GitHubError, match="Resource not accessible by integration") as excinfo:
        make_adapter(github).fetch_reviews(make_change())

    assert TOKEN not in str(excinfo.value)


def test_reading_the_conversation_writes_nothing_to_the_platform(github: FakeGitHub) -> None:
    github.connection("reviewThreads", [thread(comment())])

    make_adapter(github).fetch_reviews(make_change())

    # GraphQL reads are POSTs, so "only GETs" is not the test; "only queries" is.
    assert len(github.recorded) == 3
    assert github.writes == []


# ── approvals given on the pull request ──────────────────────────────────────

REVIEW_URL = "https://github.com/octo-org/orders/pull/7#pullrequestreview-1"
WINDOW = ApprovalWindow(fingerprint="f" * 64, after=datetime(2026, 9, 1, 10, 0, tzinfo=UTC))


def review_node(
    state: str = "APPROVED",
    *,
    login: str = "octo-lead",
    body: str = "Checked the refund split with finance.",
    commit: str = HEAD_SHA,
    can_push: Any = True,
    bot: bool = False,
    submitted: str = "2026-09-01T10:05:00Z",
    editor: str | None = None,
) -> dict[str, Any]:
    """A pull request review as GraphQL returns it with the fields approvals are read by."""

    return {
        "author": {"__typename": "Bot" if bot else "User", "login": login},
        "authorCanPushToRepository": can_push,
        "body": body,
        "commit": {"oid": commit},
        "editor": {"login": editor} if editor is not None else None,
        "state": state,
        "submittedAt": submitted,
        "url": REVIEW_URL,
    }


def test_an_approval_comes_back_with_its_commit_its_reason_and_whether_its_author_can_push(
    github: FakeGitHub,
) -> None:
    github.connection("reviews", [review_node()])

    found = make_adapter(github).fetch_approvals(make_change())

    assert found.window is None
    [approval] = found.reviews
    assert (approval.reviewer, approval.state) == ("octo-lead", "approved")
    assert approval.commit_sha == HEAD_SHA
    assert approval.reason == "Checked the refund split with finance."
    assert approval.can_push and not approval.is_bot
    assert approval.submitted_at == datetime(2026, 9, 1, 10, 5, tzinfo=UTC)
    assert approval.url == REVIEW_URL


def test_approvals_are_read_with_the_decisive_states_and_the_push_permission(
    github: FakeGitHub,
) -> None:
    make_adapter(github).fetch_approvals(make_change())

    [query] = [call.body["query"] for call in github.queries("reviews")]
    # A dismissed approval comes back as DISMISSED rather than vanishing, and a
    # request for changes withdraws an earlier approval: all three are read.
    assert "reviews(states: [APPROVED, CHANGES_REQUESTED, DISMISSED], first: 100" in query
    assert "authorCanPushToRepository" in query and "commit { oid }" in query
    assert "editor { login }" in query
    [comments] = [call.body["query"] for call in github.queries("comments")]
    assert "author { login } editor { login }" in comments
    assert github.writes == []


def test_reviews_that_decide_nothing_are_not_read_as_decisions(github: FakeGitHub) -> None:
    github.connection(
        "reviews",
        [
            review_node("COMMENTED"),
            review_node("PENDING"),
            review_node("CHANGES_REQUESTED", login="octo-qa"),
            review_node("DISMISSED", login="octo-ops"),
        ],
    )

    found = make_adapter(github).fetch_approvals(make_change())

    assert [(item.reviewer, item.state) for item in found.reviews] == [
        ("octo-qa", "changes_requested"),
        ("octo-ops", "dismissed"),
    ]


@pytest.mark.parametrize("can_push", [None, False, "true", 1], ids=["null", "false", "text", "one"])
def test_push_access_github_does_not_state_as_true_is_no_push_access(
    github: FakeGitHub, can_push: Any
) -> None:
    github.connection("reviews", [review_node(can_push=can_push)])

    [approval] = make_adapter(github).fetch_approvals(make_change()).reviews

    # The field is what makes an approval count, so its absence must not.
    assert not approval.can_push


def test_a_review_from_a_deleted_account_belongs_to_nobody(github: FakeGitHub) -> None:
    github.connection("reviews", [{**review_node(), "author": None}])

    [approval] = make_adapter(github).fetch_approvals(make_change()).reviews

    assert approval.reviewer == "" and not approval.is_bot


def test_who_edited_a_review_last_comes_back_with_it(github: FakeGitHub) -> None:
    github.connection(
        "reviews", [review_node(editor="octo-agent"), review_node(login="octo-qa")]
    )

    edited, untouched = make_adapter(github).fetch_approvals(make_change()).reviews

    assert edited.edited_by == "octo-agent"
    assert untouched.edited_by == ""


def test_an_approval_from_a_bot_is_read_as_one(github: FakeGitHub) -> None:
    github.connection("reviews", [review_node(login="octo-release", bot=True)])

    [approval] = make_adapter(github).fetch_approvals(make_change()).reviews

    assert approval.is_bot


def test_a_reason_quoting_a_credential_is_redacted_before_it_is_kept(github: FakeGitHub) -> None:
    github.connection("reviews", [review_node(body=f"Rotated {TOKEN} already, fine to merge.")])

    [approval] = make_adapter(github).fetch_approvals(make_change()).reviews

    assert TOKEN not in approval.reason
    assert "fine to merge" in approval.reason


def test_the_window_is_read_from_this_deployments_own_conclusion_only(
    github: FakeGitHub,
) -> None:
    marked = f"{VERDICT_MARKER}\n{approval_window_marker(WINDOW)}\n## honest-ci"
    github.connection("comments", [someone_elses(marked)])
    assert make_adapter(github).fetch_approvals(make_change()).window is None

    github.connection("comments", [someone_elses(marked), own(marked)])
    assert make_adapter(github).fetch_approvals(make_change()).window == WINDOW


@pytest.mark.parametrize(
    ("editor", "trusted"),
    [
        (None, True),
        ({"login": "GitHub-Actions"}, True),
        ({"login": "octo-agent"}, False),
        ({}, False),
    ],
    ids=["never-edited", "edited-by-this-deployment", "edited-by-someone-else", "editor-unnamed"],
)
def test_the_window_is_read_only_while_this_deployment_edited_its_conclusion_last(
    github: FakeGitHub, editor: dict[str, Any] | None, trusted: bool
) -> None:
    marked = f"{VERDICT_MARKER}\n{approval_window_marker(WINDOW)}\n## honest-ci"
    github.connection(
        "comments", [{**own(marked), "author": {"login": "github-actions"}, "editor": editor}]
    )

    window = make_adapter(github).fetch_approvals(make_change()).window

    # Anyone who can push can edit the comment, and one who moved the window
    # back would revive approvals given under other rules or another policy.
    assert window == (WINDOW if trusted else None)


def test_a_conclusion_someone_else_edited_is_still_the_one_written_over(
    github: FakeGitHub,
) -> None:
    github.connection(
        "comments",
        [
            {
                **own(f"{VERDICT_MARKER}\nold"),
                "author": {"login": "github-actions"},
                "editor": {"login": "octo-agent"},
            }
        ],
    )
    github.on("PATCH", f"/repos/octo-org/orders/issues/comments/{OWN_COMMENT_ID}", payload={})

    make_adapter(github).publish_report(
        make_change(),
        make_evidence(verdict=Verdict.NEEDS_HUMAN),
        "awaiting_human",
        ApprovalReading(window=WINDOW),
    )

    # Written over rather than left, so its window is this deployment's again.
    [edit] = github.calls("PATCH", f"/issues/comments/{OWN_COMMENT_ID}")
    assert edit.body["body"].splitlines()[1] == approval_window_marker(WINDOW)
    assert github.calls("POST", COMMENTS_PATH) == []


def test_approvals_github_refuses_are_an_error_rather_than_nobody_approving(
    github: FakeGitHub,
) -> None:
    github.on_graphql(lambda _: (403, {"message": "Resource not accessible by integration"}))

    with pytest.raises(GitHubError, match="403"):
        make_adapter(github).fetch_approvals(make_change())


def test_the_conclusion_keeps_the_window_on_its_second_line(github: FakeGitHub) -> None:
    github.connection("comments", [own(f"{VERDICT_MARKER}\nold")])
    github.on("PATCH", f"/repos/octo-org/orders/issues/comments/{OWN_COMMENT_ID}", payload={})

    make_adapter(github).publish_report(
        make_change(),
        make_evidence(verdict=Verdict.NEEDS_HUMAN),
        "awaiting_human",
        ApprovalReading(window=WINDOW),
    )

    [edit] = github.calls("PATCH", f"/issues/comments/{OWN_COMMENT_ID}")
    assert edit.body["body"].splitlines()[1] == approval_window_marker(WINDOW)
    assert "submit an Approve review on this pull request" in edit.body["body"]


# ── materialising the tree the scanners read ─────────────────────────────────


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "-C",
            str(repo),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


@pytest.fixture
def origin(tmp_path: Path) -> tuple[Path, str]:
    """A base repository whose pull request head exists only as refs/pull/7/head."""

    repo = tmp_path / "origin"
    (repo / "src").mkdir(parents=True)
    git(repo, "init", "--quiet")
    (repo / "src" / "refund.py").write_text("total = 0\n", encoding="utf-8")
    git(repo, "add", "src/refund.py")
    git(repo, "commit", "--quiet", "-m", "base")
    (repo / "src" / "refund.py").write_text("share = float(amount)\n", encoding="utf-8")
    git(repo, "commit", "--quiet", "-am", "refund")
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/pull/7/head", head)
    # As for a real pull request: the default branch does not contain the
    # head, so reaching it at all shows which ref was fetched.
    git(repo, "reset", "--quiet", "--hard", "HEAD~1")
    return repo, head


@pytest.fixture
def workflow_checkout(tmp_path: Path, origin: tuple[Path, str]) -> tuple[Path, str]:
    """What actions/checkout leaves behind with ``ref: head.sha`` and no persisted token."""

    repo, head = origin
    checkout = tmp_path / "runner-workspace"
    checkout.mkdir()
    git(checkout, "init", "--quiet")
    git(checkout, "fetch", "--quiet", str(repo), "refs/pull/7/head")
    git(checkout, "checkout", "--quiet", "--detach", "FETCH_HEAD")
    return checkout, head


def at(head: str) -> ChangeRequest:
    return make_change().model_copy(update={"head_sha": head})


def serve_clone_url(github: FakeGitHub, repo: Path) -> None:
    github.on("GET", "/repos/octo-org/orders", payload={"clone_url": f"file://{repo}"})


def test_outside_actions_the_head_is_fetched_from_the_base_repository(
    github: FakeGitHub, tmp_path: Path, origin: tuple[Path, str]
) -> None:
    repo, head = origin
    serve_clone_url(github, repo)

    checkout = make_adapter(github).materialize(at(head), tmp_path / "workspaces")

    assert git(checkout, "rev-parse", "HEAD") == head
    assert (checkout / "src" / "refund.py").read_text(encoding="utf-8") == (
        "share = float(amount)\n"
    )


def test_a_second_run_on_the_same_commit_reuses_the_checkout(
    github: FakeGitHub, tmp_path: Path, origin: tuple[Path, str]
) -> None:
    repo, head = origin
    serve_clone_url(github, repo)

    first = make_adapter(github).materialize(at(head), tmp_path / "workspaces")
    calls_after_first = len(github.recorded)
    # A fresh adapter, so the clone URL is not cached: no request means no fetch.
    second = make_adapter(github).materialize(at(head), tmp_path / "workspaces")

    assert first == second
    assert len(github.recorded) == calls_after_first


def test_a_tree_that_is_not_the_commit_under_review_is_refused(
    github: FakeGitHub, tmp_path: Path, origin: tuple[Path, str]
) -> None:
    repo, head = origin
    # The ref now points at another commit than the one the gates were told to
    # judge: a force-push between reading the pull request and checking it out.
    git(repo, "update-ref", "refs/pull/7/head", git(repo, "rev-parse", "HEAD"))
    serve_clone_url(github, repo)

    with pytest.raises(GitHubError, match="not the commit under review"):
        make_adapter(github).materialize(at(head), tmp_path / "workspaces")


def test_a_repository_without_a_clone_url_stops_the_run(github: FakeGitHub, tmp_path: Path) -> None:
    github.on("GET", "/repos/octo-org/orders", payload={"id": 3})

    with pytest.raises(GitHubError, match="no clone URL"):
        make_adapter(github).materialize(make_change(), tmp_path / "workspaces")


def test_a_failed_fetch_names_the_failure_without_echoing_the_token(
    github: FakeGitHub, tmp_path: Path
) -> None:
    serve_clone_url(github, tmp_path / "nowhere")

    with pytest.raises(GitHubError) as excinfo:
        make_adapter(github).materialize(make_change(), tmp_path / "workspaces")

    assert "git fetch failed" in str(excinfo.value)
    assert TOKEN not in str(excinfo.value)


def in_workflow_of(checkout: Path, repository: str = REPO) -> ActionsContext:
    return ActionsContext(workspace=checkout, repository=repository)


def test_inside_actions_the_workflows_own_checkout_is_verified_and_used(
    github: FakeGitHub, tmp_path: Path, workflow_checkout: tuple[Path, str]
) -> None:
    checkout, head = workflow_checkout
    adapter = make_adapter(github, actions=in_workflow_of(checkout))

    assert adapter.materialize(at(head), tmp_path / "workspaces") == checkout
    # A second copy fetched with the same token would be one more place for the
    # token to end up.
    assert github.recorded == []
    assert not (tmp_path / "workspaces").exists()


def test_a_workflow_checkout_at_another_commit_is_refused_with_the_line_to_change(
    github: FakeGitHub, tmp_path: Path, workflow_checkout: tuple[Path, str]
) -> None:
    checkout, _ = workflow_checkout

    # What actions/checkout does by default on a pull_request event: a merge
    # commit that exists nowhere in the pull request.
    with pytest.raises(
        GitHubError, match=re.escape("ref: ${{ github.event.pull_request.head.sha }}")
    ):
        make_adapter(github, actions=in_workflow_of(checkout)).materialize(
            make_change(), tmp_path / "workspaces"
        )


@pytest.mark.parametrize("layout", ["in-git-config", "in-an-included-file"])
def test_a_workflow_checkout_that_kept_the_token_is_refused(
    github: FakeGitHub, tmp_path: Path, workflow_checkout: tuple[Path, str], layout: str
) -> None:
    checkout, head = workflow_checkout
    header = f"AUTHORIZATION: basic {BASIC}"
    if layout == "in-git-config":
        # actions/checkout v4 and earlier write it into .git/config itself.
        git(checkout, "config", "--local", "http.https://github.com/.extraheader", header)
    else:
        # v6 writes it to a file under $RUNNER_TEMP that .git/config includes.
        included = tmp_path / "runner-temp" / "git-credentials.config"
        included.parent.mkdir()
        included.write_text(
            f'[http "https://github.com/"]\n\textraheader = {header}\n', encoding="utf-8"
        )
        git(
            checkout,
            "config",
            "--local",
            f"includeIf.gitdir:{checkout.resolve()}/.git.path",
            str(included),
        )

    # The pull request's own code runs against this tree in the sandbox.
    with pytest.raises(GitHubError, match="persist-credentials: false") as excinfo:
        make_adapter(github, actions=in_workflow_of(checkout)).materialize(
            at(head), tmp_path / "workspaces"
        )

    # Found by name; the value is never read, so it cannot be echoed either.
    assert BASIC not in str(excinfo.value)


def test_a_workflow_checkout_a_previous_step_modified_is_refused(
    github: FakeGitHub, tmp_path: Path, workflow_checkout: tuple[Path, str]
) -> None:
    checkout, head = workflow_checkout
    (checkout / "src" / "refund.py").write_text("share = Decimal(amount)\n", encoding="utf-8")

    # The scanners would judge the modified copy and report on the commit.
    with pytest.raises(GitHubError, match="differ from"):
        make_adapter(github, actions=in_workflow_of(checkout)).materialize(
            at(head), tmp_path / "workspaces"
        )


def test_a_workflow_in_another_repository_fetches_the_pull_request_instead(
    github: FakeGitHub,
    tmp_path: Path,
    origin: tuple[Path, str],
    workflow_checkout: tuple[Path, str],
) -> None:
    repo, head = origin
    checkout, _ = workflow_checkout
    serve_clone_url(github, repo)
    actions = in_workflow_of(checkout, repository="octo-org/dashboards")

    fetched = make_adapter(github, actions=actions).materialize(at(head), tmp_path / "workspaces")

    # That workflow's checkout is some other repository's tree.
    assert fetched != checkout
    assert git(fetched, "rev-parse", "HEAD") == head


def serve_head(github: FakeGitHub, head: str, **overrides: Any) -> None:
    """The pull request whose head is the ``origin`` fixture's head commit.

    ``overrides`` change what GitHub says of it, as ``serve_pull_request``'s do.
    """

    serve_pull_request(
        github,
        [
            file_entry(
                "src/refund.py",
                "modified",
                "@@ -1 +1 @@\n-total = 0\n+share = float(amount)",
                additions=1,
                deletions=1,
            )
        ],
        head={**PULL_REQUEST["head"], "sha": head},
        **overrides,
    )


def test_intake_carries_what_people_already_said_into_the_change(
    github: FakeGitHub, tmp_path: Path, origin: tuple[Path, str]
) -> None:
    repo, head = origin
    serve_head(github, head)
    serve_clone_url(github, repo)
    github.connection("reviewThreads", [thread(comment())])

    source = fetch_merge_request(
        make_adapter(github), Settings(), project=REPO, number=7, workspace=tmp_path
    )

    assert [review.author for review in source.change.platform_reviews] == ["octo-reviewer"]
    assert source.change.platform_reviews_error is None
    assert git(source.checkout, "rev-parse", "HEAD") == head


def test_reviews_github_refuses_do_not_stop_the_run_and_are_recorded(
    github: FakeGitHub, tmp_path: Path, origin: tuple[Path, str]
) -> None:
    repo, head = origin
    serve_head(github, head)
    serve_clone_url(github, repo)
    github.on_graphql(lambda _: (403, {"message": "Resource not accessible by integration"}))

    change = fetch_merge_request(
        make_adapter(github), Settings(), project=REPO, number=7, workspace=tmp_path
    ).change

    # Fail-soft: the change still reaches the gates. But it must not then look
    # like one nobody had commented on.
    assert change.platform_reviews == []
    assert "403" in (change.platform_reviews_error or "")


# ── the whole path, through the CLI ──────────────────────────────────────────

RULES = r"""
version = "v1"

[[rule]]
id = "R-01"
title = "Money amounts must not use float"
severity = "high"
kind = "regex"
pattern = 'float\s*\('
paths = ["**/*.py"]
fix = "Use decimal.Decimal."
"""

BASE_REPOSITORY = {"id": 1001, "full_name": "octo-org/orders"}
FORK_REPOSITORY = {"id": 2002, "full_name": "octo-contributor/orders"}


def serve_write_back(github: FakeGitHub, head: str) -> None:
    github.on("POST", COMMENTS_PATH, status=201, payload={"id": 1})
    github.on("POST", f"/repos/octo-org/orders/statuses/{head}", status=201, payload={"id": 2})


def enter_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkout: Path,
    head_repository: dict[str, Any] | None,
    event_name: str = "pull_request",
) -> Path:
    """Become a step of a workflow run on the pull request; returns the job summary file."""

    event = tmp_path / "event.json"
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
    summary = tmp_path / "step-summary.md"
    for name, value in {
        "GITHUB_ACTIONS": "true",
        "GITHUB_WORKSPACE": str(checkout),
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "4242",
        "GITHUB_EVENT_NAME": event_name,
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_STEP_SUMMARY": str(summary),
    }.items():
        monkeypatch.setenv(name, value)
    return summary


def run_pr(
    github: FakeGitHub,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rules: str | None = RULES,
    token: str | None = TOKEN,
) -> Result:
    """``run-pr`` from a deployment of its own; ``rules=None`` deploys no rule library.

    ``token=None`` runs it the way a workflow that forgot to pass one does.
    """

    deployment = tmp_path / "deployment"
    deployment.mkdir(parents=True)
    if rules is not None:
        (deployment / "honest-ci-rules.toml").write_text(rules, encoding="utf-8")
    # Only the review gate exists, so only the review gate is required; the
    # default policy would otherwise hand every change to a human.
    (deployment / "honest-ci-policy.toml").write_text(
        '[policy]\nversion = "v1"\nrequired_checks = ["review"]\n', encoding="utf-8"
    )
    monkeypatch.setenv("HONEST_CI_FORGE", "github")
    monkeypatch.setenv("HONEST_CI_GITHUB_BASE_URL", github.base_url)
    if token is None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    else:
        monkeypatch.setenv("GITHUB_TOKEN", token)
    monkeypatch.setenv("HONEST_CI_SANDBOX_MODE", "process")
    return CliRunner().invoke(app, ["run-pr", REPO, "7", "--workspace", str(deployment)])


def test_a_pull_request_goes_in_and_a_comment_and_a_status_come_out(
    github: FakeGitHub,
    tmp_path: Path,
    origin: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole path with no stubs between the CLI and the platform."""

    repo, head = origin
    serve_head(github, head)
    serve_clone_url(github, repo)
    serve_write_back(github, head)

    result = run_pr(github, monkeypatch, tmp_path)

    assert result.exit_code == 1, result.output
    assert "[R-01]" in result.output
    [posted] = github.calls("POST", COMMENTS_PATH)
    assert "Money amounts must not use float" in posted.body["body"]
    statuses = github.calls("POST", f"/statuses/{head}")
    assert [call.body["state"] for call in statuses] == ["pending", "failure"]
    # Outside a workflow there is no run page to link to.
    assert all("target_url" not in call.body for call in statuses)


def test_inside_actions_the_status_links_to_the_run_and_the_job_summary_has_the_verdict(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    serve_head(github, head)
    serve_write_back(github, head)
    summary = enter_workflow(monkeypatch, tmp_path, checkout, BASE_REPOSITORY)

    result = run_pr(github, monkeypatch, tmp_path)

    assert result.exit_code == 1, result.output
    final = github.calls("POST", f"/statuses/{head}")[-1]
    assert (final.body["state"], final.body["target_url"]) == ("failure", RUN_URL)
    assert "Money amounts must not use float" in summary.read_text(encoding="utf-8")
    # The workflow's checkout was the tree; nothing was fetched a second time.
    assert github.calls("GET", "/repos/octo-org/orders") == []


def test_inside_actions_the_comment_points_at_the_uploaded_evidence(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not at ``honest-ci evidence``: the ledger it reads goes with the runner."""

    checkout, head = workflow_checkout
    serve_head(github, head)
    serve_write_back(github, head)
    enter_workflow(monkeypatch, tmp_path, checkout, BASE_REPOSITORY)
    monkeypatch.setenv("HONEST_CI_EVIDENCE_ARTIFACT", "honest-ci-evidence-7-2")

    run_pr(github, monkeypatch, tmp_path)

    [posted] = github.calls("POST", COMMENTS_PATH)
    body = posted.body["body"]
    assert (
        f"Full evidence pack: the `honest-ci-evidence-7-2` artifact of [this workflow run]({RUN_URL})."
        in body
    )
    assert "honest-ci evidence " not in body


def test_a_pull_request_from_a_fork_is_judged_but_only_the_job_summary_hears_of_it(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    serve_head(github, head)
    serve_write_back(github, head)
    summary = enter_workflow(monkeypatch, tmp_path, checkout, FORK_REPOSITORY)

    result = run_pr(github, monkeypatch, tmp_path)

    # The token GitHub gives a fork's run cannot write: every attempt would
    # fail, and the verdict would be lost as a failed write-back.
    assert github.writes == []
    # The exit code is still the verdict. A fork's change is not waved
    # through because its token cannot write.
    assert result.exit_code == 1, result.output
    assert "comes from the fork octo-contributor/orders" in result.output
    text = summary.read_text(encoding="utf-8")
    assert "Money amounts must not use float" in text
    assert "Not posted to the pull request" in text


def test_a_write_back_github_refuses_fails_the_run_and_says_so_in_the_job_summary(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    serve_head(github, head)
    serve_write_back(github, head)
    github.on(
        "POST",
        COMMENTS_PATH,
        status=403,
        payload={"message": "Resource not accessible by integration"},
    )
    summary = enter_workflow(monkeypatch, tmp_path, checkout, BASE_REPOSITORY)

    result = run_pr(github, monkeypatch, tmp_path)

    # A verdict nobody on the pull request can see is not a finished run.
    assert result.exit_code == 4, result.output
    text = summary.read_text(encoding="utf-8")
    assert "Money amounts must not use float" in text
    assert "Not written back to the pull request" in text and "403" in text


def test_a_pull_request_that_cannot_be_read_says_so_in_the_job_summary(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = workflow_checkout
    summary = enter_workflow(monkeypatch, tmp_path, checkout, BASE_REPOSITORY)

    result = run_pr(github, monkeypatch, tmp_path)

    assert result.exit_code == 4, result.output
    assert "Could not read octo-org/orders#7" in summary.read_text(encoding="utf-8")


# ── run-pr in a workflow on a repository that needs a license key ────────────

#: What GitHub says of the pull request when a run in CI needs a license key
#: for it: it is onto an organization's repository, and a private one.
ONTO_A_PRIVATE_REPOSITORY: dict[str, Any] = {
    "base": {
        **PULL_REQUEST["base"],
        "repo": {
            **BASE_REPOSITORY,
            "private": True,
            "visibility": "private",
            "owner": {"login": "octo-org", "id": 9001, "type": "Organization"},
        },
    },
    "user": {"login": "octo-dev", "id": 501, "type": "User"},
}
#: The commit the run took its workflow file from.
WORKFLOW_SHA = "5f3c2a9e8d7b6a5f4e3d2c1b0a9f8e7d6c5b4a39"


def on_trial_since(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch, day: str) -> None:
    """A run with no license key, from a workflow file first committed on ``day``."""

    github.on(
        "GET",
        "/repos/octo-org/orders/commits",
        payload=[{"sha": WORKFLOW_SHA, "commit": {"committer": {"date": f"{day}T12:00:00Z"}}}],
    )
    for name, value in {
        "CI": "true",
        "GITHUB_API_URL": github.base_url,
        "GITHUB_WORKFLOW_REF": f"{REPO}/.github/workflows/honest-ci.yml@refs/pull/7/merge",
        "GITHUB_WORKFLOW_SHA": WORKFLOW_SHA,
    }.items():
        monkeypatch.setenv(name, value)


def test_in_a_workflow_once_the_trial_is_over_no_gate_runs_and_the_pull_request_says_why(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    serve_head(github, head, **ONTO_A_PRIVATE_REPOSITORY)
    serve_write_back(github, head)
    summary = enter_workflow(monkeypatch, tmp_path, checkout, BASE_REPOSITORY)
    on_trial_since(github, monkeypatch, "2026-09-01")

    result = run_pr(github, monkeypatch, tmp_path)

    assert result.exit_code == 4, result.output
    # One status, and an error: no pending one before it, and never a success.
    [status] = github.calls("POST", f"/statuses/{head}")
    assert status.body == {
        "state": "error",
        "context": "honest-ci/verify",
        "description": "No gate ran. No license key, and the 14-day trial ended on 2026-09-14",
        "target_url": RUN_URL,
    }
    [posted] = github.calls("POST", COMMENTS_PATH)
    body = posted.body["body"]
    assert body.startswith(f"{VERDICT_MARKER}\n## honest-ci · not run\n")
    assert "`.github/workflows/honest-ci.yml` was first committed: 2026-09-01." in body
    assert "Money amounts must not use float" not in body
    assert body.strip() in summary.read_text(encoding="utf-8")


def test_in_a_workflow_on_its_trial_the_gates_run_and_the_comment_ends_with_how_long_is_left(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    serve_head(github, head, **ONTO_A_PRIVATE_REPOSITORY)
    serve_write_back(github, head)
    summary = enter_workflow(monkeypatch, tmp_path, checkout, BASE_REPOSITORY)
    on_trial_since(github, monkeypatch, datetime.now(UTC).date().isoformat())

    result = run_pr(github, monkeypatch, tmp_path)

    assert result.exit_code == 1, result.output
    [posted] = github.calls("POST", COMMENTS_PATH)
    body = posted.body["body"]
    assert "Money amounts must not use float" in body
    last = body.rstrip("\n").rsplit("\n", 1)[-1]
    assert last.startswith("honest-ci is on its 14-day trial here: ")
    assert last.endswith("After that it needs a license key: https://honest-ci.dev/pricing.")
    assert f"License   : {last}" in result.output.splitlines()
    assert last in summary.read_text(encoding="utf-8")


# ── what run-pr prints, logs and writes is in English ────────────────────────


@dataclass(frozen=True)
class Occasion:
    """One way a run-pr can go: what the platform answers, and the exit code it ends on."""

    in_workflow: bool = True
    head_repository: dict[str, Any] | None = None
    rules: str | None = RULES
    token: str | None = TOKEN
    #: Whether GitHub serves the pull request at all.
    readable: bool = True
    #: Whether GitHub answers the discussion and approval queries.
    conversation: bool = True
    #: The status GitHub answers a new comment with.
    comment_status: int = 201
    #: The status GitHub answers the 'verifying' commit status with.
    progress_status: int = 201
    #: The day the workflow file was first committed, for a run with no
    #: license key on a repository that needs one; ``None``, one that needs none.
    trial_from: str | None = None
    exit_code: int = 1


OCCASIONS = {
    "blocked, outside a workflow": Occasion(in_workflow=False),
    "blocked, inside a workflow": Occasion(),
    "held for a person": Occasion(rules=None, exit_code=3),
    "from a fork": Occasion(head_repository=FORK_REPOSITORY),
    "write-back refused": Occasion(comment_status=403, exit_code=4),
    "progress status refused": Occasion(progress_status=422),
    # The comment to edit is looked up through the same queries, so nothing is written.
    "conversation unreadable": Occasion(rules=None, conversation=False, exit_code=4),
    "pull request unreadable": Occasion(readable=False, exit_code=4),
    "no token": Occasion(token=None, exit_code=2),
    "on its trial": Occasion(trial_from=datetime.now(UTC).date().isoformat()),
    "no gate ran, the trial over": Occasion(trial_from="2026-09-01", exit_code=4),
}


@pytest.mark.parametrize("occasion", OCCASIONS.values(), ids=OCCASIONS.keys())
def test_what_run_pr_prints_logs_and_writes_is_in_english(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    occasion: Occasion,
) -> None:
    """The step's log, the job summary and everything posted, the error paths included."""

    caplog.set_level(logging.DEBUG)
    checkout, head = workflow_checkout
    if occasion.readable:
        serve_head(github, head, **(ONTO_A_PRIVATE_REPOSITORY if occasion.trial_from else {}))
    if occasion.trial_from is not None:
        on_trial_since(github, monkeypatch, occasion.trial_from)
    serve_clone_url(github, tmp_path / "origin")
    github.on("POST", COMMENTS_PATH, status=occasion.comment_status, payload={"id": 1})
    github.on_call(
        "POST",
        f"/repos/octo-org/orders/statuses/{head}",
        lambda call: (
            (occasion.progress_status, {"message": "Validation Failed"})
            if call.body["state"] == "pending"
            else (201, {"id": 2})
        ),
    )
    if not occasion.conversation:
        github.on_graphql(lambda _: (403, {"message": "Resource not accessible by integration"}))
    summary = (
        enter_workflow(
            monkeypatch, tmp_path, checkout, occasion.head_repository or BASE_REPOSITORY
        )
        if occasion.in_workflow
        else None
    )

    result = run_pr(github, monkeypatch, tmp_path, rules=occasion.rules, token=occasion.token)

    assert result.exit_code == occasion.exit_code, result.output
    shown = {
        "terminal": result.output,
        "log": caplog.text,
        "job summary": summary.read_text(encoding="utf-8") if summary and summary.exists() else "",
        "written to GitHub": "\n".join(
            json.dumps(call.body, ensure_ascii=False) for call in github.writes
        ),
    }
    assert result.output.strip()  # it said something, so silence cannot pass for English
    assert {where: cjk_lines(text) for where, text in shown.items() if cjk_lines(text)} == {}


def held_for_a_person(
    github: FakeGitHub,
    tmp_path: Path,
    checkout: Path,
    head: str,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """The pull request's first run, which hands it to a person; returns its comment.

    No rule library is deployed, so the review gate cannot decide on its own.
    The comment is then served back as this deployment's own, as GitHub would.
    """

    serve_head(github, head)
    serve_write_back(github, head)
    github.on("PATCH", f"/repos/octo-org/orders/issues/comments/{OWN_COMMENT_ID}", payload={})
    enter_workflow(monkeypatch, tmp_path, checkout, BASE_REPOSITORY)
    first = run_pr(github, monkeypatch, tmp_path, rules=None)
    assert first.exit_code == 3, first.output
    [posted] = github.calls("POST", COMMENTS_PATH)
    body = str(posted.body["body"])
    github.connection("comments", [own(body)])
    return body


def review_run(
    github: FakeGitHub,
    tmp_path: Path,
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    *reviews: dict[str, Any],
) -> tuple[Result, Path]:
    """The run a review on the pull request starts, once ``reviews`` are on it."""

    github.connection("reviews", list(reviews))
    again = tmp_path / "review-run"
    again.mkdir()
    summary = enter_workflow(
        monkeypatch, again, checkout, BASE_REPOSITORY, event_name="pull_request_review"
    )
    return run_pr(github, monkeypatch, again, rules=None), summary


def test_an_approve_review_with_a_reason_turns_a_held_pull_request_green(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    first = held_for_a_person(github, tmp_path, checkout, head, monkeypatch)
    assert f"{APPROVAL_WINDOW_PREFIX}fingerprint=" in first.splitlines()[1]
    assert github.calls("POST", f"/statuses/{head}")[-1].body["state"] == "pending"

    result, summary = review_run(github, tmp_path, checkout, monkeypatch, review_node(commit=head))

    assert result.exit_code == 0, result.output
    final = github.calls("POST", f"/statuses/{head}")[-1]
    assert final.body["state"] == "success"
    assert final.body["description"].startswith("approved · ")
    [edit] = github.calls("PATCH", f"/issues/comments/{OWN_COMMENT_ID}")
    comment = edit.body["body"]
    assert "## honest-ci · approved" in comment
    assert "Approved by `octo-lead`" in comment
    assert "`Checked the refund split with finance.`" in comment
    # The same evidence, so the same window: it is not moved by being read.
    assert comment.splitlines()[1] == first.splitlines()[1]
    assert "## honest-ci · approved" in summary.read_text(encoding="utf-8")


def test_an_approve_review_without_a_reason_leaves_the_pull_request_held(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    held_for_a_person(github, tmp_path, checkout, head, monkeypatch)

    result, _ = review_run(
        github, tmp_path, checkout, monkeypatch, review_node(commit=head, body="")
    )

    assert result.exit_code == 3, result.output
    assert github.calls("POST", f"/statuses/{head}")[-1].body["state"] == "pending"
    [edit] = github.calls("PATCH", f"/issues/comments/{OWN_COMMENT_ID}")
    assert "## honest-ci · awaiting_human" in edit.body["body"]
    assert "`octo-lead` ([review](" in edit.body["body"]
    assert "it gives no reason" in edit.body["body"]


def test_an_approval_of_an_earlier_commit_does_not_carry_over_to_the_head(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    held_for_a_person(github, tmp_path, checkout, head, monkeypatch)

    result, _ = review_run(
        github, tmp_path, checkout, monkeypatch, review_node(commit="e" * 40)
    )

    assert result.exit_code == 3, result.output
    assert github.calls("POST", f"/statuses/{head}")[-1].body["state"] == "pending"
    [edit] = github.calls("PATCH", f"/issues/comments/{OWN_COMMENT_ID}")
    assert "it was given on another commit (eeeeeeeeeeee)" in edit.body["body"]


def test_after_someone_else_edits_the_conclusion_approvals_already_given_do_not_count(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    first = held_for_a_person(github, tmp_path, checkout, head, monkeypatch)
    github.connection(
        "comments",
        [{**own(first), "author": {"login": "github-actions"}, "editor": {"login": "octo-agent"}}],
    )

    result, _ = review_run(github, tmp_path, checkout, monkeypatch, review_node(commit=head))

    assert result.exit_code == 3, result.output
    assert github.calls("POST", f"/statuses/{head}")[-1].body["state"] == "pending"
    [edit] = github.calls("PATCH", f"/issues/comments/{OWN_COMMENT_ID}")
    comment = edit.body["body"]
    assert "it was given before this conclusion was published" in comment
    # The window written back opens after every review already there, so an
    # approval given from now on counts.
    assert comment.splitlines()[1].endswith(" after=2026-09-01T10:05:00+00:00 -->")


def test_an_approval_whose_text_someone_else_edited_leaves_the_pull_request_held(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    held_for_a_person(github, tmp_path, checkout, head, monkeypatch)

    result, _ = review_run(
        github, tmp_path, checkout, monkeypatch, review_node(commit=head, editor="octo-agent")
    )

    assert result.exit_code == 3, result.output
    assert github.calls("POST", f"/statuses/{head}")[-1].body["state"] == "pending"
    [edit] = github.calls("PATCH", f"/issues/comments/{OWN_COMMENT_ID}")
    assert "someone other than the reviewer edited its text" in edit.body["body"]


def test_an_approval_on_a_blocked_pull_request_changes_nothing(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    serve_head(github, head)
    serve_write_back(github, head)
    github.connection("reviews", [review_node(commit=head)])
    enter_workflow(
        monkeypatch, tmp_path, checkout, BASE_REPOSITORY, event_name="pull_request_review"
    )

    result = run_pr(github, monkeypatch, tmp_path)

    assert result.exit_code == 1, result.output
    assert github.calls("POST", f"/statuses/{head}")[-1].body["state"] == "failure"
    # Not even read: the fix for a blocking finding is a new commit.
    assert not any("states:" in str(call.body["query"]) for call in github.queries("reviews"))
    [posted] = github.calls("POST", COMMENTS_PATH)
    assert "### Approval" not in posted.body["body"]


def test_approvals_on_a_forks_pull_request_are_not_read(
    github: FakeGitHub,
    tmp_path: Path,
    workflow_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, head = workflow_checkout
    serve_head(github, head)
    github.connection("reviews", [review_node(commit=head)])
    summary = enter_workflow(
        monkeypatch, tmp_path, checkout, FORK_REPOSITORY, event_name="pull_request_review"
    )

    result = run_pr(github, monkeypatch, tmp_path, rules=None)

    # Its run cannot write the window an approval is read in, so none is read.
    assert result.exit_code == 3, result.output
    assert github.writes == []
    assert not any("states:" in str(call.body["query"]) for call in github.queries("reviews"))
    assert "an Approve review on the pull request cannot take effect" in summary.read_text(
        encoding="utf-8"
    )


def serve_head_adding(github: FakeGitHub, tmp_path: Path, hostile: str) -> str:
    """A pull request that changes ``src/refund.py`` cleanly and adds ``hostile``.

    Returns the head, which the base repository publishes as refs/pull/7/head.
    """

    repo = tmp_path / "origin"
    (repo / "src").mkdir(parents=True)
    git(repo, "init", "--quiet")
    (repo / "src" / "refund.py").write_text("total = 0\n", encoding="utf-8")
    git(repo, "add", "src/refund.py")
    git(repo, "commit", "--quiet", "-m", "base")
    (repo / "src" / "refund.py").write_text("total = 1\n", encoding="utf-8")
    (repo / hostile).write_text("import os\n", encoding="utf-8")
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "-m", "refund")
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/pull/7/head", head)
    serve_pull_request(
        github,
        [
            file_entry(
                "src/refund.py",
                "modified",
                "@@ -1 +1 @@\n-total = 0\n+total = 1",
                additions=1,
                deletions=1,
            ),
            file_entry(hostile, "added", "@@ -0,0 +1 @@\n+import os", additions=1),
        ],
        head={**PULL_REQUEST["head"], "sha": head},
    )
    serve_clone_url(github, repo)
    serve_write_back(github, head)
    return head


@pytest.mark.parametrize(
    "char", ["\n", "\x1b", "\u202e"], ids=["newline", "escape", "right-to-left-override"]
)
def test_a_file_name_with_a_control_character_holds_the_pull_request_for_a_person(
    github: FakeGitHub,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    char: str,
) -> None:
    hostile = f"src/refund{char}.py"
    head = serve_head_adding(github, tmp_path, hostile)
    # A tree without tests gives regression nothing to run. Switched off, the
    # clean file alone is a mergeable change, so only the other one can stop it.
    monkeypatch.setenv("HONEST_CI_DISABLED_GATES", "regression")

    result = run_pr(github, monkeypatch, tmp_path)

    assert result.exit_code == 3, result.output
    assert github.calls("POST", f"/statuses/{head}")[-1].body["state"] == "pending"
    [posted] = github.calls("POST", COMMENTS_PATH)
    assert f"unsafe path: `{hostile!r}`" in posted.body["body"]
    # Named, but never as the raw character that would split the comment or
    # rewrite the log it is printed to.
    assert hostile not in posted.body["body"]
    assert hostile not in result.output


#: A team rule that fires on a file's name alone, and blocks.
PATH_RULES = (
    RULES
    + r"""
[[rule]]
id = "R-02"
title = "Refund code needs the payments team"
severity = "high"
kind = "path"
paths = ["src/**"]
"""
)


def test_a_rule_that_fires_on_a_refused_name_shows_the_name_escaped(
    github: FakeGitHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ESC [2K erases the line of the log it is printed on.
    hostile = "src/refund\x1b[2K.py"
    serve_head_adding(github, tmp_path, hostile)
    monkeypatch.setenv("HONEST_CI_DISABLED_GATES", "regression")

    result = run_pr(github, monkeypatch, tmp_path, rules=PATH_RULES)

    assert result.exit_code == 1, result.output
    assert f"{hostile!r}  Refund code needs the payments team" in result.output
    [posted] = github.calls("POST", COMMENTS_PATH)
    assert f"· high · `{hostile!r}`" in posted.body["body"]
    assert hostile not in result.output
    assert hostile not in posted.body["body"]
