"""The benchmark's own GitHub client: GraphQL queries, pull request diffs and
rate limits.

It shares no code with the product's adapter (``honest_ci.adapters.github``)
on purpose. The adapter acts on one pull request with an installation's
permissions and should fail fast; this reads tens of thousands of public pull
requests with a personal token and has to survive running for hours. So it
waits out rate limits instead of failing, and retries what GitHub's search
backend drops.

The token goes into one place: the ``Authorization`` header of a request that
``open_validated_http_request`` keeps on ``https://api.github.com``, redirects
included. It is never logged, never part of an error message, and nothing
GitHub sends back contains it.

Caching is not done here but one level up, per search and per pull request
(``bench.source``), so that answers survive a change in how requests are
batched.
"""

from __future__ import annotations

import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from honest_ci.redaction import redact_text
from honest_ci.utils import open_validated_http_request

API_BASE = "https://api.github.com"
GRAPHQL_URL = f"{API_BASE}/graphql"
#: Answered without spending any budget, which makes it the check that tells a
#: diff GitHub fails on from GitHub failing.
RATE_LIMIT_URL = f"{API_BASE}/rate_limit"
API_VERSION = "2022-11-28"
USER_AGENT = "honest-ci-bench"
DIFF_MEDIA_TYPE = "application/vnd.github.diff"

#: Attempts at one request that fails in a way that usually passes: a dropped
#: connection, a 5xx, the HTML error page GitHub serves when a search times
#: out. Rate limits are not failures and do not count against this.
MAX_ATTEMPTS = 6
#: GitHub's guidance for a secondary rate limit without ``retry-after``: wait at
#: least a minute, and longer each time it happens again.
SECONDARY_LIMIT_WAIT = 60.0
MAX_SECONDARY_LIMITS = 8
#: The most of one answer that is read. GitHub renders a diff of up to 20,000
#: lines, a few megabytes of ordinary code; far past that, the size is long
#: lines of minified or generated files. One diff in the pilot was 495 MB.
MAX_RESPONSE_BYTES = 10 * 2**20
#: ``PullDiff.unavailable`` for a diff over ``MAX_RESPONSE_BYTES``.
OVERSIZED = f"over {MAX_RESPONSE_BYTES // 2**20} MiB"
#: Anything longer than this is worth a line on stderr, so a run that looks
#: stuck says what it is waiting for.
_ANNOUNCE_WAIT = 10.0

JSON = dict[str, Any]
Log = Callable[[str], None]

#: ``owner/name``, and neither half ``.`` or ``..``: the name goes into a URL path.
_REPO = re.compile(r"(?!\.\.?/)[A-Za-z0-9_.-]+/(?!\.\.?$)[A-Za-z0-9_.-]+")


class GitHubError(RuntimeError):
    """GitHub refused a request in a way that waiting will not fix."""


class ServerError(GitHubError):
    """Every attempt at one request reached GitHub, and every answer was a 5xx."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class TransportError(Exception):
    """The request never got an HTTP answer: DNS, a reset, a timeout."""


class ResponseTooLarge(GitHubError):
    """An answer over the size the client reads. Asking again will not shrink it."""


@dataclass(frozen=True)
class Response:
    status: int
    #: Header names lower-cased.
    headers: Mapping[str, str]
    body: bytes


Send = Callable[[urllib.request.Request], Response]


def urllib_send(
    request: urllib.request.Request,
    *,
    timeout_seconds: float = 90.0,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> Response:
    """Send one request with the product's outbound checks and no retries.

    An answer over ``max_bytes`` raises ``ResponseTooLarge``, without reading
    its body when its length is declared.
    """

    try:
        with open_validated_http_request(
            request, allowed_base=API_BASE, timeout_seconds=timeout_seconds
        ) as response:
            headers = _lower(response.headers.items())
            if (_int_or_none(headers.get("content-length")) or 0) > max_bytes:
                raise ResponseTooLarge(f"{request.full_url} answered with over {max_bytes} bytes")
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ResponseTooLarge(f"{request.full_url} answered with over {max_bytes} bytes")
            return Response(response.status, headers, body)
    except urllib.error.HTTPError as exc:
        body = exc.read(max_bytes) if exc.fp is not None else b""
        exc.close()
        return Response(exc.code, _lower(exc.headers.items() if exc.headers else ()), body)
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        # The URL is all these can carry; the token is in a header they never see.
        raise TransportError(redact_text(f"{type(exc).__name__}: {exc}")) from None


def _lower(items: Any) -> dict[str, str]:
    return {str(name).lower(): str(value) for name, value in items}


@dataclass(frozen=True)
class PullDiff:
    """A pull request's diff as GitHub renders it, or why it would not."""

    text: str | None
    #: Set when ``text`` is ``None``: GitHub's reason, e.g. ``too_large``.
    unavailable: str = ""


class GitHub:
    """Rate-limit-aware access to the two things the benchmark reads."""

    def __init__(
        self,
        token: str,
        *,
        send: Send = urllib_send,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        log: Log | None = None,
        min_interval: float = 0.25,
    ) -> None:
        self._token = token
        self._send = send
        self._sleep = sleep
        self._clock = clock
        self._log = log or _stderr
        self._min_interval = min_interval
        self._last_request = 0.0
        #: Per rate-limit resource ("graphql", "core"): when its window resets,
        #: once GitHub has said it is spent.
        self._resume_at: dict[str, float] = {}
        self.requests = 0

    def __repr__(self) -> str:
        # Spelled out so that no future default prints the token.
        return f"GitHub(requests={self.requests})"

    def graphql(
        self, query: str, variables: Mapping[str, Any], *, attempts: int = MAX_ATTEMPTS
    ) -> JSON:
        """The ``data`` of a GraphQL query.

        Partial answers are returned as they are -- a ``nodes`` lookup where
        one pull request has since been deleted still answers for the rest --
        with the missing entries as ``null``. ``attempts`` is for a query that
        has a cheaper way to be asked, such as several searches that can go
        one at a time: it gives up sooner.
        """

        body = json.dumps({"query": query, "variables": dict(variables)}).encode("utf-8")
        failures = 0
        while True:
            response = self._request(
                "graphql", GRAPHQL_URL, body, "application/vnd.github+json", attempts
            )
            if response.status != 200:
                raise GitHubError(_describe("GraphQL", response))
            payload = _json_or_none(response.body)
            if payload is None:
                # The HTML error page a search that timed out comes back as.
                failures = self._retry_later(
                    failures, "GraphQL answered with something not JSON", attempts
                )
                continue
            errors = payload.get("errors") or []
            if any(
                isinstance(error, dict) and error.get("type") == "RATE_LIMITED" for error in errors
            ):
                self._wait_for_reset("graphql", response.headers)
                continue
            data = payload.get("data")
            # A top-level field that came back null is one that failed: one
            # search of several in a query, typically, timing out on its own.
            # Entries inside a field may be null (``nodes`` for a pull request
            # deleted since), and are the caller's to handle.
            if (
                isinstance(data, dict)
                and data
                and all(value is not None for value in data.values())
            ):
                return data
            messages = "; ".join(
                str(error.get("message", "")) for error in errors if isinstance(error, dict)
            )
            failures = self._retry_later(
                failures, f"GraphQL returned no data: {redact_text(messages)[:300]}", attempts
            )

    def pull_diff(self, repo: str, number: int) -> PullDiff:
        """A pull request's diff as GitHub renders it, rename headers included."""

        if not _REPO.fullmatch(repo):
            raise ValueError(f"not a repository name: {repo!r}")
        url = f"{API_BASE}/repos/{repo}/pulls/{number}"
        try:
            response = self._request("core", url, None, DIFF_MEDIA_TYPE)
        except ResponseTooLarge:
            # Left out like a diff GitHub will not render: what makes it this
            # big is not code a person reviews line by line.
            return PullDiff(None, OVERSIZED)
        except ServerError as exc:
            # GitHub fails on some diffs every time: in the scale-up, a
            # six-file pull request got a 500 on every attempt, and again a
            # quarter of an hour later, while the diffs around it rendered.
            # That is a diff GitHub will not render, left out like a 406 --
            # once GitHub is seen answering. When it is failing everywhere,
            # the check stops the run instead, so that an outage is not
            # cached as a fact about this pull request.
            self._confirm_github_answers()
            return PullDiff(None, f"http_{exc.status}")
        if response.status == 200:
            return PullDiff(response.body.decode("utf-8", errors="replace"))
        if response.status in (403, 404, 406, 410, 422, 451):
            # 406 is GitHub declining to render a diff over its size limits
            # (errors[].code "too_large"); the rest are pull requests that are
            # gone or blocked. None of them changes by asking again.
            return PullDiff(None, _reason(response))
        raise GitHubError(_describe(f"diff of {repo}#{number}", response))

    def _confirm_github_answers(self) -> None:
        response = self._request("core", RATE_LIMIT_URL, None, "application/vnd.github+json")
        if response.status != 200:
            raise GitHubError(_describe("GitHub's rate limit status", response))

    # ── requests ───────────────────────────────────────────────────────────

    def _request(
        self,
        resource: str,
        url: str,
        body: bytes | None,
        accept: str,
        attempts: int = MAX_ATTEMPTS,
    ) -> Response:
        """One request, retried through whatever GitHub says will pass.

        Returns the first answer that is not a rate limit or a server error;
        what that answer means is the caller's business.
        """

        failures = 0
        # Of ``failures``, those that reached GitHub and got a 5xx back.
        server_errors = 0
        secondary = 0
        while True:
            self._pace(resource)
            request = urllib.request.Request(
                url,
                data=body,
                method="POST" if body is not None else "GET",
                headers=self._headers(accept, body is not None),
            )
            self.requests += 1
            try:
                response = self._send(request)
            except TransportError as exc:
                failures = self._retry_later(failures, str(exc), attempts)
                continue
            self._note_limits(resource, response.headers)
            if response.status in (403, 429):
                wait = self._rate_limit_wait(response, secondary)
                if wait is not None:
                    secondary += 1
                    if secondary > MAX_SECONDARY_LIMITS:
                        raise GitHubError(_describe("rate limit", response))
                    self._wait(wait, f"GitHub rate limit ({resource})")
                    continue
            if response.status >= 500:
                why = f"HTTP {response.status} from {resource}"
                server_errors += 1
                if server_errors >= attempts:
                    # Not one attempt was lost on the way; GitHub answered each
                    # and failed each. What that means is up to the caller.
                    raise ServerError(
                        f"giving up after {server_errors} attempts: {why}", response.status
                    )
                failures = self._retry_later(failures, why, attempts)
                continue
            return response

    def _headers(self, accept: str, has_body: bool) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "Authorization": f"Bearer {self._token}",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": API_VERSION,
        }
        if has_body:
            headers["Content-Type"] = "application/json"
        return headers

    # ── waiting ────────────────────────────────────────────────────────────

    def _pace(self, resource: str) -> None:
        now = self._clock()
        resume = max(self._resume_at.get(resource, 0.0), self._last_request + self._min_interval)
        if resume > now:
            self._wait(resume - now, f"GitHub {resource} quota")
        self._last_request = self._clock()

    def _note_limits(self, resource: str, headers: Mapping[str, str]) -> None:
        remaining = _int_or_none(headers.get("x-ratelimit-remaining"))
        reset = _int_or_none(headers.get("x-ratelimit-reset"))
        if remaining is not None and remaining <= 0 and reset is not None:
            self._resume_at[resource] = max(self._resume_at.get(resource, 0.0), reset + 1.0)

    def _rate_limit_wait(self, response: Response, earlier: int) -> float | None:
        """How long a 403/429 says to wait, or ``None`` when it is a plain refusal."""

        retry_after = _int_or_none(response.headers.get("retry-after"))
        if retry_after is not None:
            return float(max(retry_after, 1))
        if _int_or_none(response.headers.get("x-ratelimit-remaining")) == 0:
            reset = _int_or_none(response.headers.get("x-ratelimit-reset"))
            if reset is not None:
                return max(reset + 1.0 - self._clock(), 1.0)
        if b"rate limit" in response.body.lower():
            return SECONDARY_LIMIT_WAIT * 2.0 ** min(earlier, 4)
        return None

    def _wait_for_reset(self, resource: str, headers: Mapping[str, str]) -> None:
        reset = _int_or_none(headers.get("x-ratelimit-reset"))
        wait = reset + 1.0 - self._clock() if reset is not None else SECONDARY_LIMIT_WAIT
        self._wait(max(wait, 1.0), f"GitHub rate limit ({resource})")

    def _retry_later(self, failures: int, why: str, attempts: int) -> int:
        failures += 1
        if failures >= attempts:
            raise GitHubError(f"giving up after {failures} attempts: {why}")
        self._wait(5.0 * 2.0 ** (failures - 1), why)
        return failures

    def _wait(self, seconds: float, why: str) -> None:
        if seconds >= _ANNOUNCE_WAIT:
            until = datetime.fromtimestamp(self._clock() + seconds, UTC).strftime("%H:%M:%S")
            self._log(f"{why}: waiting {seconds:.0f}s (until {until} UTC)")
        self._sleep(seconds)


def _json_or_none(body: bytes) -> JSON | None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _reason(response: Response) -> str:
    payload = _json_or_none(response.body) or {}
    codes = [
        str(error.get("code"))
        for error in payload.get("errors") or []
        if isinstance(error, dict) and error.get("code")
    ]
    return redact_text(codes[0] if codes else f"http_{response.status}")


def _describe(what: str, response: Response) -> str:
    payload = _json_or_none(response.body) or {}
    message = str(payload.get("message") or response.body[:200].decode("utf-8", "replace"))
    return redact_text(f"{what}: HTTP {response.status}: {message}")


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
