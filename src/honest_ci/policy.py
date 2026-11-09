"""Structural guardrails evaluated before the gates spend any budget.

This is not the team rule library (that is ``rules.py``, and it fires inside the
review gate). It answers a narrower question: *can this change be judged
automatically at all, or does its shape alone demand a human?* A migration, an
auth path or a 6000-line refactor is escalated rather than waved through by an
automated verdict.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, get_args

from .contracts import ChangeRequest, PolicyDecision, Verdict
from .diffs import (
    count_changed_lines,
    extract_changed_files,
    is_safe_relative_path,
    markdown_path,
    normalize_diff_path,
)
from .utils import config_source, file_sha256, matches_any_glob, read_text

#: Paths where a wrong automated verdict is most expensive. Touching one does
#: not fail the change; it routes the change to a human.
DEFAULT_SENSITIVE_PATHS = [
    "**/migrations/**",
    "**/infra/**",
    "**/terraform/**",
    "**/auth/**",
    "**/security/**",
    "**/payment/**",
    "**/*secret*",
    "**/*password*",
    ".github/**",
    ".gitlab-ci.yml",
    "Jenkinsfile",
    # The files that configure these checks. A change that edits them may be
    # judged by the edit -- deleting a test and switching off the rule that
    # would report it look fine together -- so it is a person's call. A policy
    # that sets its own ``sensitive_paths`` replaces this list and has to name
    # them itself.
    "honest-ci-rules.toml",
    "honest-ci-policy.toml",
]


#: What an open review thread does to the verdict. Two words rather than a
#: boolean so the policy file reads as a decision someone made.
UnresolvedReviews = Literal["needs_human", "ignore"]
_UNRESOLVED_REVIEWS: tuple[str, ...] = get_args(UnresolvedReviews)


@dataclass(frozen=True)
class PolicyConfig:
    version: str
    sensitive_paths: list[str] = field(default_factory=lambda: list(DEFAULT_SENSITIVE_PATHS))
    #: Blast radius beyond which automated judgment is not trusted on its own.
    #: Both are read the same way wherever the change came from: a rename
    #: counts as two files, and a file moved without edits as no lines
    #: (``ChangeRequest.changed_lines``).
    max_changed_files: int = 60
    max_changed_lines: int = 4000
    required_checks: list[str] = field(default_factory=lambda: ["spec", "review", "regression"])
    #: What to do when the platform still carries an unanswered review thread.
    #: ``needs_human`` by default: somebody asked a question on this change and
    #: nobody answered it, and an automated pass there would be a machine
    #: overruling a person who is already on the record. ``ignore`` keeps the
    #: comments as evidence only -- a deployment whose teams leave threads open
    #: as a matter of habit can say so, in one auditable line of its own file.
    unresolved_reviews: UnresolvedReviews = "needs_human"
    #: Where the policy came from (relative to the repository, so an evidence
    #: pack names a configuration rather than a machine) and the SHA-256 of
    #: that file's bytes. Both ``None`` when no policy file exists and these
    #: defaults are what judged the change -- which the evidence pack then
    #: says, rather than naming a file nobody wrote.
    source: str | None = None
    sha256: str | None = None

    @classmethod
    def load(cls, repo_path: Path, policy_path: Path, default_version: str) -> PolicyConfig:
        candidate = policy_path if policy_path.is_absolute() else repo_path / policy_path
        if not candidate.exists():
            return cls(version=default_version)
        data = tomllib.loads(read_text(candidate))
        policy = data.get("policy", data)
        return cls(
            version=str(policy.get("version", default_version)),
            sensitive_paths=list(policy.get("sensitive_paths", DEFAULT_SENSITIVE_PATHS)),
            max_changed_files=int(policy.get("max_changed_files", 60)),
            max_changed_lines=int(policy.get("max_changed_lines", 4000)),
            required_checks=list(policy.get("required_checks", ["spec", "review", "regression"])),
            unresolved_reviews=_unresolved_reviews(policy.get("unresolved_reviews"), candidate),
            source=config_source(candidate, repo_path),
            sha256=file_sha256(candidate),
        )


def _unresolved_reviews(value: Any, source: Path) -> UnresolvedReviews:
    """Read the setting, or refuse the file.

    Defaulting an unrecognised word to ``needs_human`` would be the safe
    verdict reached the wrong way: the team wrote something they believed
    was configuration, and they have to be told it was not.
    """

    if value is None:
        return "needs_human"
    if not isinstance(value, str) or value not in _UNRESOLVED_REVIEWS:
        raise ValueError(
            f"Policy file {source}: unresolved_reviews must be "
            + " or ".join(f"{name!r}" for name in _UNRESOLVED_REVIEWS)
            + f", not {value!r}."
        )
    return cast(UnresolvedReviews, value)


class PolicyEngine:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def evaluate_change(self, change: ChangeRequest) -> PolicyDecision:
        # Both names of a rename: the old one is as much a part of the change,
        # and it is what local intake, which reads a move as a deletion and an
        # addition, counts and checks too.
        changed_files = change.touched_paths
        # A file moved without edits counts no lines, locally as on a forge,
        # where it has none to count: its size is in its two names above.
        changed_lines = change.changed_lines
        violations: list[str] = []

        if not changed_files:
            violations.append("change carries no files")
        if len(changed_files) > self.config.max_changed_files:
            violations.append(
                f"changed files {len(changed_files)} exceeds automated limit "
                f"{self.config.max_changed_files}"
            )
        if changed_lines > self.config.max_changed_lines:
            violations.append(
                f"changed lines {changed_lines} exceeds automated limit "
                f"{self.config.max_changed_lines}"
            )

        # Every name is written as a Markdown code span, escaped: these lines
        # reach the terminal, the approval summary and, as Markdown, the
        # evidence pack and the platform comment. A name refused for a newline
        # or an escape sequence must not get to break any of them apart, and a
        # sensitive one that holds "**" or "![x](...)" must not get to render.
        sensitive_hits: list[str] = []
        for file_name in changed_files:
            if not is_safe_relative_path(file_name):
                violations.append(f"unsafe path: {markdown_path(file_name)}")
                continue
            if matches_any_glob(file_name, self.config.sensitive_paths):
                sensitive_hits.append(markdown_path(file_name))
        if sensitive_hits:
            violations.append("sensitive paths require human review: " + ", ".join(sensitive_hits))

        risk: Literal["low", "medium", "high"] = "low"
        if sensitive_hits or any("unsafe" in item for item in violations):
            risk = "high"
        elif (
            changed_lines > self.config.max_changed_lines // 2
            or len(changed_files) > self.config.max_changed_files // 2
        ):
            risk = "medium"

        return PolicyDecision(
            allowed=not violations,
            policy_version=self.config.version,
            risk_level=risk,
            violations=violations,
            required_checks=self.config.required_checks,
            changed_files=changed_files,
            changed_lines=changed_lines,
        )


def unsatisfied_checks(
    decision: PolicyDecision | None,
    verdicts: Mapping[str, Verdict],
) -> list[str]:
    """Required gates that reached no conclusion on this change.

    Without this, deploying one gate would make every change it happens to pass
    look mergeable, even though the policy also demands spec and regression and
    neither exists yet. A deployment that only runs the review gate is only
    accountable for review — and saying so has to be an explicit line in
    ``honest-ci-policy.toml``, not a default this code chose on the team's
    behalf.

    Takes verdicts rather than a run: the same reading has to hold for a live
    run and for an evidence pack someone was handed months later.
    """

    if decision is None:
        return []
    return [
        name
        for name in decision.required_checks
        if verdicts.get(name) in {None, Verdict.SKIPPED}
    ]


__all__ = [
    "DEFAULT_SENSITIVE_PATHS",
    "PolicyConfig",
    "PolicyEngine",
    "UnresolvedReviews",
    # Re-exported so callers keep reading diff facts through one parser.
    "count_changed_lines",
    "extract_changed_files",
    "normalize_diff_path",
    "unsatisfied_checks",
]
