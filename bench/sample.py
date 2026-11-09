"""Which pull requests the benchmark scans, and why those.

Agent groups
    The time window is cut into slices (ten minutes by default) and each agent
    visits them in an order shuffled from the seed. Every candidate in a
    visited slice is enumerated, and the eligible ones are judged in an order
    shuffled from the seed too, until the slice has given the group its share
    (``per_slice``, five by default) or the group is full. Enumerating whole
    slices rather than taking the top of a search is the point: GitHub ranks
    search results by relevance, and sorted by date its first thousand are a
    few days of one end of the window. Visiting slices at random gives every
    pull request in the window the same chance to be reached. It also makes a
    larger run a continuation of a smaller one: with the same seed, a
    1000-per-group run starts with the pilot's 50.

    The share is small because a group is only as many clusters as it has
    slices: ten minutes of Claude Code pull requests fill a group of fifty on
    their own, and fifty from ten minutes of one day say little about eight
    weeks. Five per slice spreads a group of fifty over at least ten slices
    and one of a thousand over two hundred. Slices are ten minutes long as a
    middle way: a longer one means more pages to enumerate for a busy agent,
    a shorter one more slices to visit for a quiet one, which rarely has five
    in a slice. The first pages of ten slices go out as one query.

    A candidate is taken when its repository is public, not a fork (unless
    asked for) and has the minimum number of stars; when the pull request in
    full identifies as this agent (``bench.agents``); when its repository has
    not yet reached the per-repository cap for the group, so that a few busy
    repositories cannot stand in for everyone; and when GitHub renders a
    non-empty diff for it. Every candidate not taken is counted under its
    reason, so a reader can see what the sample leaves out.

The human group
    One control per agent pull request: the pull request opened in the same
    repository, inside the window, closest in creation time, that identifies
    as human and is not already a control. Same repository means the same
    code, test suite, reviewers and merge habits, which is most of what
    differs between one pull request and another; closest in time keeps the
    two under the same rules and CI.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from honest_ci.diffs import file_patches, is_safe_relative_path
from honest_ci.integrity import IntegrityHit, detect, unread_deletions

from .agents import HUMAN, Agent, identify, plainly_not_human
from .github import JSON, GitHubError
from .source import (
    PAGE_SIZE,
    PULLS_PER_QUERY,
    SEARCH_CAP,
    SEARCHES_PER_QUERY,
    OfflineMiss,
    Reader,
)

Log = Callable[[str], None]

#: Search takes times to the second; below a minute, splitting a slice that
#: still has over a thousand results is not worth it.
MIN_SLICE = timedelta(minutes=1)
#: Pull requests read on each side of an agent pull request when looking for
#: its control. Busy repositories put many agent pull requests between two
#: human ones.
NEARBY = 20
#: Candidates per agent pull request fetched in full ahead of matching.
PREFETCH = 5


@dataclass(frozen=True)
class Config:
    since: datetime
    until: datetime
    per_group: int
    min_stars: int = 10
    include_forks: bool = False
    per_repo: int = 3
    seed: int = 1
    slice_length: timedelta = timedelta(minutes=10)
    per_slice: int = 5

    def describe(self) -> JSON:
        return {
            "since": iso(self.since),
            "until": iso(self.until),
            "per_group": self.per_group,
            "min_stars": self.min_stars,
            "include_forks": self.include_forks,
            "per_repo": self.per_repo,
            "seed": self.seed,
            "slice_minutes": self.slice_length.total_seconds() / 60,
            "per_slice": self.per_slice,
        }


@dataclass(frozen=True)
class PullRequest:
    id: str
    url: str
    repo: str
    number: int
    created_at: datetime
    merged: bool
    state: str
    draft: bool
    stars: int
    title: str
    additions: int
    deletions: int
    changed_files: int

    @classmethod
    def from_node(cls, node: Mapping[str, Any]) -> PullRequest:
        repository = node.get("repository") or {}
        return cls(
            id=str(node["id"]),
            url=str(node["url"]),
            repo=str(repository["nameWithOwner"]),
            number=int(node["number"]),
            created_at=parse_time(str(node["createdAt"])),
            merged=node.get("mergedAt") is not None,
            state=str(node.get("state") or ""),
            draft=bool(node.get("isDraft")),
            stars=int(repository.get("stargazerCount") or 0),
            title=str(node.get("title") or ""),
            additions=int(node.get("additions") or 0),
            deletions=int(node.get("deletions") or 0),
            changed_files=int(node.get("changedFiles") or 0),
        )


@dataclass(frozen=True)
class Scanned:
    group: str
    pr: PullRequest
    #: How the pull request was identified as its group's (``bench.agents``).
    via: tuple[str, ...]
    hits: tuple[IntegrityHit, ...]
    #: For a control: the agent pull request it stands beside.
    matched_to: PullRequest | None = None


@dataclass
class Group:
    name: str
    scanned: list[Scanned] = field(default_factory=list)
    #: Candidates looked at and not taken, by reason.
    left_out: Counter[str] = field(default_factory=Counter)
    #: Candidates looked at: search results for an agent, agent pull requests
    #: to match for the human group.
    seen: int = 0
    slices: int = 0


def sample_agent(reader: Reader, agent: Agent, config: Config, log: Log) -> Group:
    group = Group(agent.name)
    per_repo: Counter[str] = Counter()
    slices = time_slices(config.since, config.until, config.slice_length)
    random.Random(f"{config.seed}:{agent.name}").shuffle(slices)
    # Slices whose first page has been asked ahead, and how many to ask next:
    # ten at a time while GitHub answers that, fewer after a batch too heavy
    # to answer -- dense agents' slices are -- and back up as batches pass.
    asked, ahead = 0, SEARCHES_PER_QUERY
    for index, (start, end) in enumerate(slices):
        if len(group.scanned) >= config.per_group:
            break
        if index >= asked:
            batch = [frame_query(agent.search, *bounds) for bounds in slices[index : index + ahead]]
            answered = reader.prefetch(batch, PAGE_SIZE)
            asked = index + len(batch)
            ahead = min(2 * ahead, SEARCHES_PER_QUERY) if answered else max(ahead // 2, 2)
        group.slices += 1
        eligible: list[JSON] = []
        for node in candidates(reader, agent.search, start, end, group.left_out):
            group.seen += 1
            reason = ineligible(node, config, start, end)
            if reason is not None:
                group.left_out[reason] += 1
            else:
                eligible.append(node)
        # So that a slice's share is any few of it, not its first minute.
        random.Random(f"{config.seed}:{agent.name}:{iso(start)}").shuffle(eligible)
        _take(reader, agent, config, eligible, group, per_repo, log)
        if group.slices % 100 == 0:
            log(_progress(group, config))
    log(_progress(group, config, done=True))
    return group


def _take(
    reader: Reader,
    agent: Agent,
    config: Config,
    eligible: Sequence[JSON],
    group: Group,
    per_repo: Counter[str],
    log: Log,
) -> None:
    """Judge one slice's eligible candidates in turn, until the slice has given
    its share or the group is full.

    Candidates are read in full a batch at a time, as their turn comes; those
    not judged by the end are counted as left out, whether or not their batch
    was read.
    """

    taken = judged = 0
    while judged < len(eligible):
        needed = min(config.per_slice - taken, config.per_group - len(group.scanned))
        if needed <= 0:
            break
        chunk = eligible[judged : judged + batch_size(needed)]
        full = reader.pull_requests([node["id"] for node in chunk])
        for node in chunk:
            if taken >= config.per_slice or len(group.scanned) >= config.per_group:
                break
            judged += 1
            pull = full.get(node["id"])
            if pull is None:
                group.left_out["no longer resolves"] += 1
                continue
            identity = identify(pull)
            if identity.group != agent.name:
                group.left_out[f"identified as {identity.group}"] += 1
                continue
            repo = str(pull["repository"]["nameWithOwner"])
            if per_repo[repo] >= config.per_repo:
                group.left_out["over the per-repository cap"] += 1
                continue
            scanned = scan(reader, pull, agent.name, identity.via, group.left_out, log)
            if scanned is not None:
                per_repo[repo] += 1
                taken += 1
                group.scanned.append(scanned)
    if judged < len(eligible):
        full_group = len(group.scanned) >= config.per_group
        reason = "after the group was full" if full_group else "over the per-slice cap"
        group.left_out[reason] += len(eligible) - judged


def batch_size(needed: int) -> int:
    """How many candidates to read in full when ``needed`` more are wanted.

    Twice as many, since some turn out to be another's or to have no diff,
    and at least ten, since a lookup costs the same up to a hundred.
    """

    return min(PULLS_PER_QUERY, max(2 * needed, 10))


def match_controls(reader: Reader, samples: Sequence[Group], config: Config, log: Log) -> Group:
    group = Group(HUMAN)
    order = interleave([sample.scanned for sample in samples])
    # Agent pull requests can never be controls: none of them identifies as
    # human. Listed anyway, so that it does not rest on that alone.
    used = {scanned.pr.id for sample in samples for scanned in sample.scanned}
    queries = [query for scanned in order for query in nearby_queries(scanned.pr, config)]
    pages = reader.searches(queries, NEARBY)
    ranked = [
        rank_nearby([*pages[2 * index]["nodes"], *pages[2 * index + 1]["nodes"]], scanned.pr)
        for index, scanned in enumerate(order)
    ]
    reader.pull_requests(
        [
            node["id"]
            for candidates in ranked
            for node in candidates[:PREFETCH]
            if node["id"] not in used
        ]
    )
    for scanned, candidates in zip(order, ranked, strict=True):
        group.seen += 1
        control = _first_human(
            reader, scanned.pr, [node for node in candidates if node["id"] not in used], group, log
        )
        if control is None:
            group.left_out["agent pull requests with no human one nearby"] += 1
            continue
        used.add(control.pr.id)
        group.scanned.append(control)
        if group.seen % 100 == 0:
            log(f"[human] {len(group.scanned)} controls for {group.seen} agent pull requests")
    log(f"[human] done: {len(group.scanned)} controls for {group.seen} agent pull requests")
    return group


def _first_human(
    reader: Reader, agent_pr: PullRequest, candidates: Sequence[JSON], group: Group, log: Log
) -> Scanned | None:
    for start in range(0, len(candidates), PREFETCH):
        chunk = candidates[start : start + PREFETCH]
        full = reader.pull_requests([node["id"] for node in chunk])
        for node in chunk:
            pull = full.get(node["id"])
            if pull is None:
                continue
            identity = identify(pull)
            if identity.group != HUMAN:
                group.left_out[f"identified as {identity.group}"] += 1
                continue
            control = scan(reader, pull, HUMAN, (), group.left_out, log, matched_to=agent_pr)
            if control is not None:
                return control
    return None


def scan(
    reader: Reader,
    pull: Mapping[str, Any],
    group: str,
    via: tuple[str, ...],
    left_out: Counter[str],
    log: Log,
    matched_to: PullRequest | None = None,
) -> Scanned | None:
    """The detector's hits on one pull request, or ``None`` when it has no diff to judge."""

    pr = PullRequest.from_node(pull)
    diff = reader.diff(pr.repo, pr.number)
    if diff.text is None:
        left_out[f"diff unavailable ({diff.unavailable})"] += 1
        return None
    if not diff.text.strip():
        # An agent's placeholder that never got a commit, mostly: nothing a
        # test could have lost, and not a change anybody reviews.
        left_out["no changes"] += 1
        return None
    try:
        hits = integrity_hits(diff.text)
    except Exception as exc:
        # A detector bug must not end a run that has taken hours; it is counted
        # in the summary and named here, so it can be reproduced.
        left_out["detector error"] += 1
        log(f"detector error on {pr.url}: {type(exc).__name__}: {exc}")
        return None
    return Scanned(group, pr, via, tuple(hits), matched_to)


def integrity_hits(diff: str) -> list[IntegrityHit]:
    """What honest-ci's rule engine sees in ``diff`` for its integrity rules.

    ``detect`` reads the diff, and ``unread_deletions`` adds the test files
    deleted with none of their lines shown, which the rule engine takes from
    the forge's file list: the deleted files with no line counted. GitHub's
    rendered diff names every file it deletes by git's headers, a binary or
    empty one too, so here they are the deletions that come with no lines.
    """

    unread = [
        patch.old_path
        for patch in file_patches(diff)
        if patch.new_path is None
        and patch.old_path is not None
        and not patch.added
        and not patch.removed
        and is_safe_relative_path(patch.old_path)
    ]
    return [*unread_deletions(unread), *detect(diff)]


# ── searching ────────────────────────────────────────────────────────────────


def candidates(
    reader: Reader, terms: str, start: datetime, end: datetime, left_out: Counter[str]
) -> list[JSON]:
    """Every search result for ``terms`` created in ``[start, end)``, oldest first."""

    query = frame_query(terms, start, end)
    try:
        page = reader.search(query)
        if page["issueCount"] > SEARCH_CAP and end - start > MIN_SLICE:
            return _halves(reader, terms, start, end, left_out)
        nodes = list(page["nodes"])
        count = int(page["issueCount"])
        while page["pageInfo"]["hasNextPage"] and len(nodes) < SEARCH_CAP:
            page = reader.search(query, after=page["pageInfo"]["endCursor"])
            nodes += page["nodes"]
    except (GitHubError, OfflineMiss):
        # A search GitHub cannot finish in time is usually one over too many
        # results; half the time range is half the work. Offline, a slice
        # that was split when it was read has its halves in the cache and
        # not itself, so it is split again.
        if end - start <= MIN_SLICE:
            raise
        return _halves(reader, terms, start, end, left_out)
    if count > SEARCH_CAP:
        # Only a slice already down to MIN_SLICE gets here.
        left_out["beyond the search cap"] += count - len(nodes)
    return nodes


def _halves(
    reader: Reader, terms: str, start: datetime, end: datetime, left_out: Counter[str]
) -> list[JSON]:
    middle = start + timedelta(seconds=int((end - start).total_seconds()) // 2)
    return [
        *candidates(reader, terms, start, middle, left_out),
        *candidates(reader, terms, middle, end, left_out),
    ]


def frame_query(terms: str, start: datetime, end: datetime) -> str:
    # created-asc: the default order is relevance, which is not stable across
    # pages and is not what "every pull request in the slice" means.
    return f"is:pr is:public created:>={iso(start)} created:<{iso(end)} {terms} sort:created-asc"


def nearby_queries(pr: PullRequest, config: Config) -> tuple[str, str]:
    """The searches for the pull requests just before and just after ``pr`` in its repository."""

    created = iso(pr.created_at)
    return (
        f"repo:{pr.repo} is:pr created:>={iso(config.since)} created:<{created} sort:created-desc",
        f"repo:{pr.repo} is:pr created:>{created} created:<{iso(config.until)} sort:created-asc",
    )


def rank_nearby(nodes: Iterable[JSON | None], pr: PullRequest) -> list[JSON]:
    """Search results that could be a person's, nearest to ``pr`` first."""

    found: dict[str, JSON] = {}
    for node in nodes:
        if not node or not node.get("id") or node["id"] == pr.id or plainly_not_human(node):
            continue
        found.setdefault(str(node["id"]), node)

    def distance(node: JSON) -> tuple[float, datetime]:
        created = parse_time(str(node["createdAt"]))
        return abs((created - pr.created_at).total_seconds()), created

    return sorted(found.values(), key=distance)


def ineligible(node: JSON | None, config: Config, start: datetime, end: datetime) -> str | None:
    if not node or not node.get("id") or not node.get("repository"):
        return "not a pull request"
    repository = node["repository"]
    if repository.get("isPrivate"):
        return "private repository"
    if repository.get("isFork") and not config.include_forks:
        return "fork"
    if int(repository.get("stargazerCount") or 0) < config.min_stars:
        return f"under {config.min_stars} stars"
    if not start <= parse_time(str(node["createdAt"])) < end:
        # Search matched it loosely; the slice is the frame, not the search.
        return "outside the slice"
    return None


# ── time ─────────────────────────────────────────────────────────────────────


def time_slices(
    since: datetime, until: datetime, length: timedelta
) -> list[tuple[datetime, datetime]]:
    if length <= timedelta(0):
        raise ValueError("slice length must be positive")
    slices = []
    start = since
    while start < until:
        end = min(start + length, until)
        slices.append((start, end))
        start = end
    return slices


def interleave(groups: Sequence[Sequence[Scanned]]) -> list[Scanned]:
    """The first of every group, then the second of every group, and so on.

    The order controls are matched in. Taking them in turn keeps any one agent
    from using up the human pull requests of a repository two agents share,
    and keeps a larger run's first matches the same as a smaller run's.
    """

    longest = max((len(group) for group in groups), default=0)
    return [group[index] for index in range(longest) for group in groups if index < len(group)]


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _progress(group: Group, config: Config, *, done: bool = False) -> str:
    state = "done" if done else "sampling"
    exhausted = done and len(group.scanned) < config.per_group
    tail = " (the window ran out first)" if exhausted else ""
    return (
        f"[{group.name}] {state}: {len(group.scanned)}/{config.per_group} after {group.slices}"
        f" slices and {group.seen} candidates{tail}"
    )
