"""The numbers, and the files they are read from.

``summary.md`` / ``summary.json``
    Per group: how many pull requests were scanned; for each check, and for
    any check, how many of them have at least one hit; and of those, how many
    were merged. Every share carries a 95% Wilson score interval, which unlike
    the textbook ``p ± 1.96·se`` stays inside [0, 1] and does not collapse to
    zero width when nothing was hit -- the case this benchmark exists to tell
    apart from "rare".
``hits.csv``
    One row per hit, with a link to its line in the pull request's diff and an
    empty ``label`` column, for judging precision by hand.
``prs.csv``
    One row per scanned pull request, hit or not: the denominators.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import honest_ci
from honest_ci import diffs, integrity
from honest_ci.integrity import INTEGRITY_CHECKS, IntegrityHit

from .agents import AGENTS, HUMAN
from .github import JSON
from .sample import Config, Group, Scanned, iso

#: The two-sided 95% normal quantile.
Z95 = 1.959963984540054
#: What a spreadsheet would run as a formula when a cell starts with it.
_FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")

HITS_COLUMNS = (
    "group",
    "pr",
    "title",
    "check",
    "file",
    "line",
    "side",
    "link",
    "text",
    "replacement",
    "detail",
    "merged",
    "label",
)


def wilson(count: int, total: int) -> tuple[float, float] | None:
    """The 95% Wilson score interval for ``count`` of ``total``; ``None`` for no trials."""

    if total <= 0:
        return None
    if not 0 <= count <= total:
        raise ValueError(f"{count} of {total} is not a proportion")
    share = count / total
    z2 = Z95 * Z95
    denominator = 1 + z2 / total
    centre = (share + z2 / (2 * total)) / denominator
    half = Z95 * math.sqrt(share * (1 - share) / total + z2 / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def proportion(count: int, total: int) -> JSON:
    interval = wilson(count, total)
    return {
        "count": count,
        "of": total,
        "share": count / total if total else None,
        "low": interval[0] if interval else None,
        "high": interval[1] if interval else None,
    }


def title_of(group: str) -> str:
    return AGENTS[group].title if group in AGENTS else "Human (matched controls)"


def summarize(
    groups: Sequence[Group], config: Config, generated_at: datetime, *, offline: bool = False
) -> JSON:
    return {
        "generated_at": iso(generated_at),
        # Rebuilt from the cache alone, so nothing was read from GitHub at
        # ``generated_at``: the answers are older.
        "offline": offline,
        "config": config.describe(),
        "agents": [group.name for group in groups if group.name != HUMAN],
        "detector": detector_fingerprint(),
        "checks": list(INTEGRITY_CHECKS),
        "groups": {group.name: summarize_group(group) for group in groups},
    }


def summarize_group(group: Group) -> JSON:
    scanned = group.scanned
    total = len(scanned)
    hit = [item for item in scanned if item.hits]
    checks: JSON = {}
    for check in INTEGRITY_CHECKS:
        with_check = [item for item in scanned if any(one.check == check for one in item.hits)]
        checks[check] = {
            "prs": proportion(len(with_check), total),
            "hits": sum(one.check == check for item in scanned for one in item.hits),
            "merged": proportion(sum(item.pr.merged for item in with_check), len(with_check)),
        }
    summary: JSON = {
        "title": title_of(group.name),
        "scanned": total,
        "merged": proportion(sum(item.pr.merged for item in scanned), total),
        "any_check": {
            "prs": proportion(len(hit), total),
            "hits": sum(len(item.hits) for item in scanned),
            "merged": proportion(sum(item.pr.merged for item in hit), len(hit)),
        },
        "checks": checks,
        "candidates_seen": group.seen,
        "slices_visited": group.slices,
        "repositories": len({item.pr.repo for item in scanned}),
        "left_out": dict(group.left_out.most_common()),
    }
    if group.name == HUMAN:
        gaps = [gap_hours(item) for item in scanned]
        summary["median_gap_hours"] = statistics.median(gaps) if gaps else None
    else:
        summary["identified_via"] = dict(
            Counter("+".join(item.via) or "?" for item in scanned).most_common()
        )
    return summary


def gap_hours(item: Scanned) -> float:
    if item.matched_to is None:
        return 0.0
    return abs((item.pr.created_at - item.matched_to.created_at).total_seconds()) / 3600


def detector_fingerprint() -> JSON:
    """Which detector produced the numbers: its version and the digest of its source."""

    digest = hashlib.sha256()
    for module in (integrity, diffs):
        digest.update(Path(str(module.__file__)).read_bytes())
    return {"honest_ci": honest_ci.__version__, "sha256": digest.hexdigest()}


# ── markdown ─────────────────────────────────────────────────────────────────


def render_markdown(summary: Mapping[str, Any]) -> str:
    config = summary["config"]
    groups: Mapping[str, Any] = summary["groups"]
    checks: list[str] = summary["checks"]
    forks = "including forks" if config["include_forks"] else "no forks"
    read = "Rebuilt offline from the cache at" if summary["offline"] else "Read from GitHub up to"
    lines = [
        "# Test-integrity hits in agent and human pull requests",
        "",
        f"Pull requests created {config['since']} to {config['until']} (end exclusive) in public"
        f" repositories with at least {config['min_stars']} stars, {forks}; per agent, at most"
        f" {config['per_repo']} per repository and {config['per_slice']} per"
        f" {config['slice_minutes']:g}-minute slice; seed {config['seed']}. Detector"
        f" `honest_ci.integrity` from honest-ci {summary['detector']['honest_ci']}"
        f" (source sha256 `{summary['detector']['sha256'][:12]}`)."
        f" {read} {summary['generated_at']}; method in bench/README.md.",
        "",
        "## Pull requests with at least one hit",
        "",
        "Count, share and 95% Wilson interval.",
        "",
        _row(["Group", "PRs", "Any check", *checks]),
        _row(["---", "---:", *(["---"] * (len(checks) + 1))]),
    ]
    for data in groups.values():
        lines.append(
            _row(
                [
                    data["title"],
                    str(data["scanned"]),
                    _cell(data["any_check"]["prs"]),
                    *(_cell(data["checks"][check]["prs"]) for check in checks),
                ]
            )
        )
    lines += [
        "",
        "## Merged",
        "",
        "Of all scanned pull requests, and of those with a hit.",
        "",
        _row(["Group", "All PRs", "With any hit", *checks]),
        _row(["---", *(["---"] * (len(checks) + 2))]),
    ]
    for data in groups.values():
        lines.append(
            _row(
                [
                    data["title"],
                    _cell(data["merged"], fraction=True),
                    _cell(data["any_check"]["merged"], fraction=True),
                    *(_cell(data["checks"][check]["merged"], fraction=True) for check in checks),
                ]
            )
        )
    lines += [
        "",
        "## How the groups were drawn",
        "",
        _row(["Group", "Scanned", "Repositories", "Looked at", "Identified by", "Left out"]),
        _row(["---", "---:", "---:", "---:", "---", "---"]),
    ]
    for name, data in groups.items():
        if name == HUMAN:
            gap = data["median_gap_hours"]
            looked = f"{data['candidates_seen']} agent PRs to match"
            identified = "no agent signature" + (
                f"; median gap {gap:.1f} h" if gap is not None else ""
            )
        else:
            looked = f"{data['candidates_seen']} candidates in {data['slices_visited']} slices"
            identified = ", ".join(
                f"{via} {count}" for via, count in data["identified_via"].items()
            )
        left_out = ", ".join(f"{reason} {count}" for reason, count in data["left_out"].items())
        lines.append(
            _row(
                [
                    data["title"],
                    str(data["scanned"]),
                    str(data["repositories"]),
                    looked,
                    identified or "–",
                    left_out or "–",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _cell(share: Mapping[str, Any], *, fraction: bool = False) -> str:
    if not share["of"]:
        return "–"
    count = f"{share['count']}/{share['of']}" if fraction else str(share["count"])
    return (
        f"{count} · {100 * share['share']:.1f}%"
        f" ({100 * share['low']:.1f}–{100 * share['high']:.1f})"
    )


def _row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


# ── files ────────────────────────────────────────────────────────────────────


def write_results(directory: Path, groups: Sequence[Group], summary: JSON) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = {
        "summary.md": render_markdown(summary),
        "summary.json": json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    }
    paths = []
    for name, text in written.items():
        path = directory / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    scanned = [item for group in groups for item in group.scanned]
    paths.append(_write_csv(directory / "hits.csv", HITS_COLUMNS, hit_rows(scanned)))
    columns = (
        "group",
        "pr",
        "repo",
        "stars",
        "created_at",
        "state",
        "merged",
        "draft",
        "additions",
        "deletions",
        "changed_files",
        "identified_via",
        "matched_to",
        "gap_hours",
        "hits",
        *INTEGRITY_CHECKS,
    )
    paths.append(_write_csv(directory / "prs.csv", columns, pr_rows(scanned)))
    return paths


def hit_rows(scanned: Iterable[Scanned]) -> Iterable[Sequence[object]]:
    for item in scanned:
        for hit in item.hits:
            yield (
                item.group,
                item.pr.url,
                item.pr.title,
                hit.check,
                hit.file,
                "" if hit.line is None else hit.line,
                hit.side,
                line_link(item.pr.url, hit),
                hit.text,
                hit.replacement or "",
                hit.detail,
                item.pr.merged,
                "",
            )


def pr_rows(scanned: Iterable[Scanned]) -> Iterable[Sequence[object]]:
    for item in scanned:
        counts = Counter(hit.check for hit in item.hits)
        pr = item.pr
        yield (
            item.group,
            pr.url,
            pr.repo,
            pr.stars,
            iso(pr.created_at),
            pr.state,
            pr.merged,
            pr.draft,
            pr.additions,
            pr.deletions,
            pr.changed_files,
            "+".join(item.via),
            item.matched_to.url if item.matched_to else "",
            f"{gap_hours(item):.2f}" if item.matched_to else "",
            len(item.hits),
            *(counts[check] for check in INTEGRITY_CHECKS),
        )


def line_link(pr_url: str, hit: IntegrityHit) -> str:
    """The hit's line in the pull request's "Files changed" tab.

    GitHub anchors a file there by the SHA-256 of its path, and a line by the
    side it is on: ``L`` for the base, ``R`` for the head.
    """

    anchor = hashlib.sha256(hit.file.encode("utf-8")).hexdigest()
    line = "" if hit.line is None else f"{'L' if hit.side == 'base' else 'R'}{hit.line}"
    return f"{pr_url}/files#diff-{anchor}{line}"


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Sequence[object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([spreadsheet_safe(value) for value in row])
    return path


def spreadsheet_safe(value: object) -> object:
    """A cell a spreadsheet shows rather than runs.

    Hit lines are code from strangers' repositories, and ``=HYPERLINK(...)``
    is valid code in more than one language. A leading apostrophe is how
    spreadsheets spell "this is text".
    """

    if isinstance(value, str) and value.startswith(_FORMULA_LEAD):
        return "'" + value
    return value


def from_spreadsheet(cell: str) -> str:
    """What ``spreadsheet_safe`` was given, for showing a cell to a person.

    A value that began with an apostrophe of its own and then a formula's first
    character reads back one apostrophe short. Shown to a person, that costs
    less than every escaped line of code keeping one.
    """

    if cell.startswith("'") and cell[1:].startswith(_FORMULA_LEAD):
        return cell[1:]
    return cell
