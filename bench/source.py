"""What the benchmark reads from GitHub, one cached answer at a time.

Three kinds of question, each cached under its own key:

* one page of one search: ``(query, first, after)``;
* one pull request in full (body, commits, merge state): its node id;
* one pull request's diff: ``(repo, number)``.

The fields asked for are part of the key, so that changing a query reads
everything it touches again instead of mixing answers of two shapes.

Batching is how questions are asked, never how answers are kept. Several
searches go out as aliases of one GraphQL query and up to fifty pull requests
as one ``nodes`` lookup, but each answer is filed on its own, so a later run
that batches differently -- a larger sample, a different order -- still finds
it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar

from .cache import Cache, cache_key
from .github import JSON, MAX_ATTEMPTS, GitHubError, PullDiff

#: The most a search's pages can hold: GitHub stops paginating at 1000 results
#: whatever ``issueCount`` says, so a slice with more is split instead.
SEARCH_CAP = 1000
PAGE_SIZE = 100
SEARCHES_PER_QUERY = 10
PULLS_PER_QUERY = 50
#: Tries at a query that asks for several things before asking for fewer. A
#: batch that fails is usually too heavy for GitHub to answer in time, which
#: asking again does not change.
BATCH_ATTEMPTS = 2

_T = TypeVar("_T")

#: What a search result needs for the cheap decisions -- eligibility, and
#: ruling out bots and agent branches -- before anything bigger is fetched.
SEARCH_FIELDS = """id url number createdAt headRefName
        author { __typename login }
        repository { nameWithOwner stargazerCount isFork isPrivate }"""

SEARCH_QUERY = f"""query($q: String!, $first: Int!, $after: String) {{
  search(query: $q, type: ISSUE_ADVANCED, first: $first, after: $after) {{
    issueCount
    pageInfo {{ hasNextPage endCursor }}
    nodes {{ ... on PullRequest {{ {SEARCH_FIELDS} }} }}
  }}
}}"""

#: The last twenty commits are where an agent's trailer would be; a pull
#: request with more has its earlier ones unread (bench/README.md).
PULL_QUERY = """query($ids: [ID!]!) {
  nodes(ids: $ids) {
    ... on PullRequest {
      id url number createdAt mergedAt closedAt state isDraft
      headRefName title body additions deletions changedFiles
      author { __typename login }
      repository { nameWithOwner stargazerCount isFork isPrivate }
      commits(last: 20) { totalCount nodes { commit { message author { name email } } } }
    }
  }
}"""


class OfflineMiss(LookupError):
    """An offline run needed an answer the cache does not have."""


class Client(Protocol):
    """What ``Source`` asks GitHub through: ``bench.github.GitHub``, or a fake."""

    def graphql(
        self, query: str, variables: Mapping[str, Any], *, attempts: int = MAX_ATTEMPTS
    ) -> JSON: ...

    def pull_diff(self, repo: str, number: int) -> PullDiff: ...


class Reader(Protocol):
    """What sampling needs from GitHub. ``Source`` in a run, a fake in tests."""

    def search(self, query: str, first: int = PAGE_SIZE, after: str | None = None) -> JSON: ...

    def searches(self, queries: Sequence[str], first: int) -> list[JSON]: ...

    def prefetch(self, queries: Sequence[str], first: int) -> bool: ...

    def pull_requests(self, ids: Sequence[str]) -> dict[str, JSON | None]: ...

    def diff(self, repo: str, number: int) -> PullDiff: ...


class Source:
    """``Reader`` over the cache, asking ``github`` for what it lacks.

    ``github=None`` is an offline source: it answers from the cache alone and
    raises ``OfflineMiss`` for the rest, which is how a report is rebuilt
    without a token.
    """

    def __init__(self, cache: Cache, github: Client | None) -> None:
        self._cache = cache
        self._github = github

    def search(self, query: str, first: int = PAGE_SIZE, after: str | None = None) -> JSON:
        key = _search_key(query, first, after)
        cached = self._cache.get("search", key)
        if cached is not None:
            return dict(cached.value)
        data = self._online(f"search {query!r}").graphql(
            SEARCH_QUERY, {"q": query, "first": first, "after": after}
        )
        result: JSON = data["search"]
        self._cache.put("search", key, result)
        return result

    def searches(self, queries: Sequence[str], first: int) -> list[JSON]:
        """The first page of each search, asked ``SEARCHES_PER_QUERY`` at a time."""

        results: dict[int, JSON] = {}
        missing: list[int] = []
        for index, query in enumerate(queries):
            cached = self._cache.get("search", _search_key(query, first))
            if cached is not None:
                results[index] = dict(cached.value)
            else:
                missing.append(index)
        for chunk in _chunks(missing, SEARCHES_PER_QUERY):
            batch = [queries[index] for index in chunk]
            try:
                answers = self._search_batch(batch, first)
            except GitHubError:
                # One slow search fails the whole query; alone, the others
                # pass. ``search`` files each answer itself.
                for index, query in zip(chunk, batch, strict=True):
                    results[index] = self.search(query, first)
                continue
            for index, query, answer in zip(chunk, batch, answers, strict=True):
                self._cache.put("search", _search_key(query, first), answer)
                results[index] = answer
        return [results[index] for index in range(len(queries))]

    def prefetch(self, queries: Sequence[str], first: int) -> bool:
        """File the first page of each search, asked as one query: a head start.

        Returns whether GitHub answered. A batch it would not answer is left
        alone, to be asked search by search when each is needed, where a
        failure has somewhere to go (``bench.sample.candidates`` splits it).
        One search is not worth a batch, and offline there is nothing to ask.
        """

        missing = [
            query
            for query in dict.fromkeys(queries)
            if self._cache.get("search", _search_key(query, first)) is None
        ]
        if len(missing) < 2 or self._github is None:
            return True
        try:
            answers = self._search_batch(missing, first)
        except GitHubError:
            return False
        for query, answer in zip(missing, answers, strict=True):
            self._cache.put("search", _search_key(query, first), answer)
        return True

    def pull_requests(self, ids: Sequence[str]) -> dict[str, JSON | None]:
        """Each pull request in full, ``None`` for one that no longer resolves."""

        found: dict[str, JSON | None] = {}
        missing: list[str] = []
        for node_id in dict.fromkeys(ids):
            cached = self._cache.get("pull", _pull_key(node_id))
            if cached is not None:
                found[node_id] = cached.value
            else:
                missing.append(node_id)
        for chunk in _chunks(missing, PULLS_PER_QUERY):
            for node_id, node in self._pull_batch(chunk).items():
                self._cache.put("pull", _pull_key(node_id), node)
                found[node_id] = node
        return found

    def diff(self, repo: str, number: int) -> PullDiff:
        key = cache_key(repo, number)
        cached = self._cache.get("diff", key)
        if cached is not None:
            return PullDiff(cached.value.get("text"), cached.value.get("unavailable", ""))
        diff = self._online(f"the diff of {repo}#{number}").pull_diff(repo, number)
        self._cache.put("diff", key, {"text": diff.text, "unavailable": diff.unavailable})
        return diff

    def _search_batch(self, queries: Sequence[str], first: int) -> list[JSON]:
        variables: dict[str, Any] = {f"q{index}": query for index, query in enumerate(queries)}
        variables["first"] = first
        data = self._online(f"{len(queries)} searches").graphql(
            aliased_search_query(len(queries)), variables, attempts=BATCH_ATTEMPTS
        )
        return [data[f"s{index}"] for index in range(len(queries))]

    def _pull_batch(self, ids: Sequence[str]) -> dict[str, JSON | None]:
        try:
            data = self._online(f"{len(ids)} pull requests").graphql(
                PULL_QUERY,
                {"ids": list(ids)},
                attempts=BATCH_ATTEMPTS if len(ids) > 1 else MAX_ATTEMPTS,
            )
        except GitHubError:
            if len(ids) == 1:
                raise
            # A batch whose answer is too big to build in time; halves are not.
            half = len(ids) // 2
            return {**self._pull_batch(ids[:half]), **self._pull_batch(ids[half:])}
        nodes = data["nodes"]
        return {
            node_id: node if isinstance(node, dict) and node.get("id") == node_id else None
            for node_id, node in zip(ids, nodes, strict=True)
        }

    def _online(self, what: str) -> Client:
        if self._github is None:
            raise OfflineMiss(f"{what} is not in the cache; set GITHUB_TOKEN to fetch it")
        return self._github


def _search_key(query: str, first: int, after: str | None = None) -> str:
    return cache_key(SEARCH_FIELDS, query, first, after)


def _pull_key(node_id: str) -> str:
    return cache_key(PULL_QUERY, node_id)


def aliased_search_query(count: int) -> str:
    parameters = ", ".join(f"$q{index}: String!" for index in range(count))
    fields = "\n".join(
        f"  s{index}: search(query: $q{index}, type: ISSUE_ADVANCED, first: $first) {{"
        " issueCount pageInfo { hasNextPage endCursor }"
        f" nodes {{ ... on PullRequest {{ {SEARCH_FIELDS} }} }} }}"
        for index in range(count)
    )
    return f"query({parameters}, $first: Int!) {{\n{fields}\n}}"


def _chunks(items: Sequence[_T], size: int) -> list[Sequence[_T]]:
    return [items[start : start + size] for start in range(0, len(items), size)]
