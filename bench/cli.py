"""``python -m bench``: sample, scan, and write the tables.

Run the same command again to resume: everything already read from GitHub is
in the cache, so a run picks up where the last one stopped, and a run with a
larger ``--per-group`` extends the smaller one it was started after.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .agents import AGENTS, Agent
from .cache import Cache
from .github import GitHub, GitHubError
from .report import render_markdown, summarize, write_results
from .sample import Config, Group, match_controls, sample_agent
from .source import OfflineMiss, Source

TOKEN_VARIABLE = "GITHUB_TOKEN"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = _config(arguments)
        agents = _agents(arguments.agents)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    github = None
    if not arguments.offline:
        # Read here and handed to the client, which puts it in a request header
        # and nowhere else. It is never printed, not even in part.
        token = os.environ.get(TOKEN_VARIABLE, "").strip()
        if not token:
            print(
                f"error: {TOKEN_VARIABLE} is not set; set it, or pass --offline to answer"
                " from the cache alone",
                file=sys.stderr,
            )
            return 2
        github = GitHub(token)
    source = Source(Cache(arguments.out / "cache"), github)
    started = time.monotonic()
    try:
        groups: list[Group] = [sample_agent(source, agent, config, _log) for agent in agents]
        groups.append(match_controls(source, groups, config, _log))
    except OfflineMiss as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except GitHubError as exc:
        print(
            f"error: {exc}\nEverything read so far is cached; run the same command to resume.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("interrupted; run the same command to resume", file=sys.stderr)
        return 130
    summary = summarize(groups, config, datetime.now(UTC), offline=github is None)
    directory = arguments.out / (arguments.name or run_name(config, agents))
    paths = write_results(directory, groups, summary)
    print(render_markdown(summary))
    requests = github.requests if github is not None else 0
    _log(f"{requests} requests to GitHub in {time.monotonic() - started:.0f}s")
    for path in paths:
        _log(f"wrote {path}")
    return 0


def run_name(config: Config, agents: Sequence[Agent]) -> str:
    name = (
        f"{config.since:%Y%m%d}-{config.until:%Y%m%d}_n{config.per_group}"
        f"_stars{config.min_stars}_cap{config.per_repo}_seed{config.seed}"
    )
    if config.include_forks:
        name += "_forks"
    if config.slice_length != Config.slice_length:
        name += f"_slice{config.slice_length.total_seconds() / 60:g}m"
    if config.per_slice != Config.per_slice:
        name += f"_perslice{config.per_slice}"
    if len(agents) != len(AGENTS):
        name += "_" + "-".join(agent.name for agent in agents)
    return name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description=(
            "Run honest-ci's test-integrity checks over pull requests that coding agents"
            " opened in public GitHub repositories, and over human pull requests in the"
            f" same repositories. Reads the token from {TOKEN_VARIABLE}."
        ),
    )
    parser.add_argument(
        "--since", required=True, type=_moment, help="window start, UTC (YYYY-MM-DD or ISO 8601)"
    )
    parser.add_argument("--until", required=True, type=_moment, help="window end, exclusive")
    parser.add_argument(
        "--per-group", type=int, default=50, help="pull requests per agent (default 50)"
    )
    parser.add_argument(
        "--agents",
        default=",".join(AGENTS),
        help=f"comma-separated, from {', '.join(AGENTS)} (default: all)",
    )
    parser.add_argument(
        "--min-stars", type=int, default=10, help="minimum repository stars (default 10)"
    )
    parser.add_argument("--include-forks", action="store_true", help="also sample forks")
    parser.add_argument(
        "--per-repo",
        type=int,
        default=3,
        help="most pull requests per repository per agent (default 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="shuffles the order time slices are visited in"
    )
    parser.add_argument(
        "--slice-minutes",
        type=float,
        default=Config.slice_length.total_seconds() / 60,
        help="length of a time slice (default %(default)g)",
    )
    parser.add_argument(
        "--per-slice",
        type=int,
        default=Config.per_slice,
        help="most pull requests one time slice gives an agent's group (default %(default)d)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("bench/out"), help="cache and results (default bench/out)"
    )
    parser.add_argument(
        "--name", help="results directory under --out (default: from the parameters)"
    )
    parser.add_argument(
        "--offline", action="store_true", help="answer from the cache only; no token needed"
    )
    return parser


def _config(arguments: argparse.Namespace) -> Config:
    if arguments.since >= arguments.until:
        raise ValueError("--since must be before --until")
    if arguments.per_group < 1:
        raise ValueError("--per-group must be at least 1")
    if arguments.per_repo < 1:
        raise ValueError("--per-repo must be at least 1")
    if arguments.slice_minutes < 1:
        raise ValueError("--slice-minutes must be at least 1")
    if arguments.per_slice < 1:
        raise ValueError("--per-slice must be at least 1")
    return Config(
        since=arguments.since,
        until=arguments.until,
        per_group=arguments.per_group,
        min_stars=arguments.min_stars,
        include_forks=arguments.include_forks,
        per_repo=arguments.per_repo,
        seed=arguments.seed,
        slice_length=timedelta(minutes=arguments.slice_minutes),
        per_slice=arguments.per_slice,
    )


def _agents(names: str) -> list[Agent]:
    chosen = [name.strip() for name in names.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in AGENTS]
    if unknown or not chosen:
        raise ValueError(
            f"unknown agents: {', '.join(unknown) or '(none given)'}; known: {', '.join(AGENTS)}"
        )
    return [AGENTS[name] for name in dict.fromkeys(chosen)]


def _moment(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a date or ISO 8601 time: {value!r}") from None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
