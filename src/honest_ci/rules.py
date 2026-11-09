"""The team rule library, in file form.

This is the "first review against your team's own rules" half of the review
gate, and it is deliberately a file rather than code: a team has to be able to
read every rule, argue with it, version it and take it with them. Each rule
carries an id, a description, how it is judged and how severe it is, which is
what lets a block be explained as "violates R-03" instead of "the tool
reported an error".

Five judgment kinds, four of them deterministic:

``path``
    Changing a file in scope is itself the finding (generated code, vendored
    trees, release manifests).
``regex``
    A pattern matched on a line the change **added**. Matching only added lines
    is the same rule the model trust boundary applies: a comment on code nobody
    touched is noise, not a review finding.
``integrity``
    One of the test-integrity checks in ``integrity.py`` -- a deleted test, a
    new skip, fewer assertions, a rewritten expected value, a loosened coverage
    threshold -- read off the diff alone. These read the lines a change
    **removed** as well, and that is the added-lines rule above applied to the
    other side of the diff rather than an exception to it: what that rule
    keeps out is code nobody touched, and a removed line is exactly what this
    change touched. Deleting the failing test leaves no added line to match.
    What a hit does is the rule's ``action``: ``needs_human`` (the default)
    sends the change to a person, ``block`` stops it, ``record`` only writes
    it into the evidence pack. Being rules, they live in this file under its
    ``version``, which is what the approval fingerprint binds.
``tool``
    A scanner's diagnostics, attributed to this rule. The tool list is fixed in
    code -- each registered toolchain declares its own, and
    ``adapters.scanners`` unions them -- rather than configurable, because a
    rule file that could append arbitrary arguments to a subprocess would be a
    remote-code-execution hole the moment the change under verification edits
    its own rule file.
``judgment``
    A criterion a model reads the change against — "an order's status only
    changes through the state machine" — for the rules a team really has but
    nobody can write as a path or a pattern. The rule's ``detail`` *is* the
    criterion, which is why it is mandatory here: a standard nobody wrote down
    cannot be judged, argued with or handed over. The model only proposes;
    ``grounding.py`` keeps the proposal only if it cites one of these rule ids
    and lands on a line the change added, and deterministic code applies the
    severity and the confidence threshold. Findings carry
    ``decided_by="model"`` and never block on their own — they route the change
    to a human, because a block is terminal and a probabilistic judgment must
    inform an approval rather than replace one.

A malformed rule file raises. A rule library nobody can parse is not a reason to
review nothing — it is a reason to stop.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_args

from .contracts import (
    BLOCKING_SEVERITIES,
    ChangeRequest,
    DiffSide,
    Finding,
    GateName,
    Severity,
)
from .diffs import added_lines, is_safe_relative_path, markdown_code
from .integrity import INTEGRITY_CHECKS, IntegrityCheck, IntegrityHit, detect, unread_deletions
from .redaction import redact_text
from .utils import config_source, file_sha256, matches_any_glob, read_text

RuleKind = Literal["path", "regex", "integrity", "tool", "judgment"]

#: What an ``integrity`` rule's hit does to the review verdict.
#:
#: ``block`` stops the change, ``needs_human`` sends it to a person, and
#: ``record`` only writes the hit into the evidence pack. The verdict still
#: reads nothing but ``Finding.blocking`` for a block -- high or critical,
#: decided by code -- so the loader holds each action to a severity that says
#: the same thing: ``block`` to high or above, the other two to medium or
#: below. An action that disagreed with its severity would leave two answers
#: to "does this stop the merge" in one rule, and the verdict would silently
#: pick one of them.
IntegrityAction = Literal["needs_human", "block", "record"]


_KINDS: tuple[str, ...] = get_args(RuleKind)
_SEVERITIES: tuple[str, ...] = get_args(Severity)
_ACTIONS: tuple[IntegrityAction, ...] = get_args(IntegrityAction)
_RULE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,31}$")

#: One rule reporting every hit in a 4000-line diff drowns the evidence pack.
#: The cap is recorded in the finding so the count is never silently wrong.
MAX_FINDINGS_PER_RULE = 20
#: Long generated lines (minified assets, embedded blobs) are skipped: they are
#: not readable evidence, and they are where a hand-written regex would hang.
MAX_SCANNED_LINE_CHARS = 2000
_MAX_EXCERPT_CHARS = 200


class RuleError(ValueError):
    """The rule library could not be read. Not retryable: someone must fix it."""


def supported_tools() -> tuple[str, ...]:
    """Scanners a ``tool`` rule may bind to.

    The union of what the registered toolchains declare (``adapters.scanners``)
    and nothing else: every one of them reaches a verdict with the network
    denied, with arguments fixed in code. ``pip-audit`` is absent for exactly
    that reason -- it needs an advisory feed, and the sandbox never grants a
    scanner network access. Imported lazily because the adapters are the
    heavier half of the package and a replay loads the rule engine first.
    """

    from .adapters.scanners import supported_tools as registered

    return registered()


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    kind: RuleKind
    severity: Severity = "medium"
    detail: str = ""
    #: Files the rule applies to. For ``path`` rules, being in scope is the
    #: finding; for the others it narrows where the rule looks.
    paths: tuple[str, ...] = ("**",)
    exclude_paths: tuple[str, ...] = ()
    pattern: re.Pattern[str] | None = None
    tool: str | None = None
    #: For ``integrity`` rules: which of ``integrity.INTEGRITY_CHECKS`` the
    #: rule reports, and what a hit does -- see ``IntegrityAction``.
    check: IntegrityCheck | None = None
    action: IntegrityAction | None = None
    #: For ``judgment`` rules: the confidence a model finding must reach before
    #: it is recorded as a finding at all. ``None`` takes the deployment-wide
    #: default, so a team can tune one noisy rule without moving the rest.
    min_confidence: float | None = None
    fix: str | None = None
    #: Which real changes this rule was mined from -- merge request refs,
    #: incident numbers. Rule mining goes from samples to a draft to a person's
    #: final version, and without this the last step throws away the only
    #: thing that answers "why does this rule exist" when someone challenges it
    #: a year later. Free-form on purpose: refusing an incident number would
    #: push provenance back into a comment.
    source_samples: tuple[str, ...] = ()

    def applies_to(self, path: str) -> bool:
        if matches_any_glob(path, self.exclude_paths):
            return False
        return matches_any_glob(path, self.paths)

    @property
    def blocking(self) -> bool:
        return self.severity in {"high", "critical"}


@dataclass(frozen=True)
class RuleSet:
    version: str
    rules: tuple[Rule, ...] = ()
    #: Where the rules came from, so the evidence pack can name the file that
    #: judged the change -- relative to the repository, because a pack that
    #: leaves the building should describe a configuration, not a machine.
    #: ``None`` means no rule library was configured.
    source: str | None = None
    #: SHA-256 of that file's bytes. The version string is what a team
    #: bumps; this is what says whether they bumped it after every edit -- two
    #: packs claiming ``v3`` with different digests here is a fact worth having.
    #: ``None`` when there is no file, never an empty string.
    sha256: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.rules

    def of_kind(self, kind: RuleKind) -> tuple[Rule, ...]:
        return tuple(rule for rule in self.rules if rule.kind == kind)

    @property
    def judgment_rules(self) -> tuple[Rule, ...]:
        """Rules whose verdict needs a model reading, in declaration order."""

        return self.of_kind("judgment")

    def integrity_rules(self, action: IntegrityAction) -> tuple[Rule, ...]:
        """The ``integrity`` rules whose hits do ``action``, in declaration order."""

        return tuple(rule for rule in self.of_kind("integrity") if rule.action == action)

    @property
    def required_tools(self) -> tuple[str, ...]:
        """Scanners this library binds rules to, in declaration order.

        The review gate runs these and nothing else: which scanners are
        mandatory is the team's configuration, not a hidden default.
        """

        tools: list[str] = []
        for rule in self.of_kind("tool"):
            if rule.tool is not None and rule.tool not in tools:
                tools.append(rule.tool)
        return tuple(tools)

    def tool_scopes(self, paths: Sequence[str]) -> dict[str, tuple[str, ...]]:
        """Which of ``paths`` each bound scanner is actually being asked about.

        A ``tool`` rule's ``paths`` is not decoration: it says which files that
        scanner is asked to judge. Handing the scanner every changed file
        regardless turns the rule's own scope into a comment -- which is how
        dogfooding got 23 mypy diagnostics out of a repository whose own mypy
        configuration only ever looks at ``src``.

        Several rules may bind the same tool, and then the scanner is asked
        about the union: each rule is a separate reason to run it, and running
        it on less than one rule asked for would leave that rule unreviewed.
        """

        scopes: dict[str, dict[str, None]] = {}
        for rule in self.of_kind("tool"):
            if rule.tool is None:
                continue
            covered = scopes.setdefault(rule.tool, {})
            for path in paths:
                if rule.applies_to(path):
                    covered.setdefault(path, None)
        return {tool: tuple(covered) for tool, covered in scopes.items()}

    @classmethod
    def load(cls, repo_path: Path, rules_path: Path, default_version: str) -> RuleSet:
        candidate = rules_path if rules_path.is_absolute() else repo_path / rules_path
        if not candidate.exists():
            return cls(version=default_version)
        try:
            data = tomllib.loads(read_text(candidate))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise RuleError(f"Rule file {candidate} could not be read: {exc}") from exc

        entries = data.get("rule", [])
        if not isinstance(entries, list):
            raise RuleError(f"Rule file {candidate}: 'rule' must be an array of tables.")
        rules: list[Rule] = []
        seen: set[str] = set()
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise RuleError(f"Rule file {candidate}: rule #{index} is not a table.")
            if not entry.get("enabled", True):
                # Kept in the file on purpose: a disabled rule is auditable
                # history, and deleting it loses why it was turned off.
                continue
            rule = _parse_rule(entry, index, candidate)
            if rule.id in seen:
                raise RuleError(f"Rule file {candidate}: duplicate rule id {rule.id!r}.")
            seen.add(rule.id)
            rules.append(rule)
        return cls(
            version=str(data.get("version", default_version)),
            rules=tuple(rules),
            source=config_source(candidate, repo_path),
            sha256=file_sha256(candidate),
        )


def evaluate_rules(
    rule_set: RuleSet,
    change: ChangeRequest,
    gate: GateName = "review",
) -> list[Finding]:
    """Judge a change against the deterministic rules, on its own diff.

    Only ``path``, ``regex`` and ``integrity`` rules are decided here. A
    ``tool`` rule needs a scanner run and a ``judgment`` rule needs a model
    call, and both of those are the gate's job: this function stays pure so
    the same rules can be replayed without a sandbox or a credential.
    """

    findings: list[Finding] = []
    added = added_lines(change.diff)
    # A rename out of a ruled path touches that path: whatever the rule
    # protects there is gone from it.
    changed_files = change.touched_paths

    for rule in rule_set.of_kind("path"):
        for path in changed_files:
            if not rule.applies_to(path):
                continue
            findings.append(
                _finding(
                    rule,
                    gate,
                    len(findings) + 1,
                    file=path,
                    detail=rule.detail or f"The change touches a path that rule {rule.id} constrains.",
                )
            )

    for rule in rule_set.of_kind("regex"):
        if rule.pattern is None:  # pragma: no cover - guaranteed by the loader
            continue
        hits = 0
        for path, lines in added.items():
            if not rule.applies_to(path):
                continue
            for line_number, text in lines:
                if len(text) > MAX_SCANNED_LINE_CHARS or not rule.pattern.search(text):
                    continue
                hits += 1
                if hits > MAX_FINDINGS_PER_RULE:
                    continue
                findings.append(
                    _finding(
                        rule,
                        gate,
                        len(findings) + 1,
                        file=path,
                        line=line_number,
                        detail=_regex_detail(rule, text),
                    )
                )
        if hits > MAX_FINDINGS_PER_RULE:
            findings[-1] = findings[-1].model_copy(
                update={
                    "detail": findings[-1].detail
                    + f"\n(This rule matched {hits} times; only the first "
                    f"{MAX_FINDINGS_PER_RULE} are listed.)"
                }
            )

    integrity_rules = rule_set.of_kind("integrity")
    # Read once for every integrity rule: the checks do not depend on the rule,
    # only which hits a rule keeps does.
    integrity_hits = _integrity_hits(change) if integrity_rules else []
    for rule in integrity_rules:
        matched = [
            hit for hit in integrity_hits if hit.check == rule.check and rule.applies_to(hit.file)
        ]
        for hit in matched[:MAX_FINDINGS_PER_RULE]:
            findings.append(
                _finding(
                    rule,
                    gate,
                    len(findings) + 1,
                    file=hit.file,
                    line=hit.line,
                    side=hit.side,
                    detail=_integrity_detail(rule, hit),
                )
            )
        if len(matched) > MAX_FINDINGS_PER_RULE:
            findings[-1] = findings[-1].model_copy(
                update={
                    "detail": findings[-1].detail
                    + f"\n(This rule matched {len(matched)} times; only the first "
                    f"{MAX_FINDINGS_PER_RULE} are listed.)"
                }
            )
    return findings


def _integrity_hits(change: ChangeRequest) -> list[IntegrityHit]:
    """What the diff shows happening to the tests, and what the diff leaves out.

    A forge lists a deleted file it sends no lines for -- too large, binary,
    cut short -- and the diff never names it, so ``detect`` cannot see it go.
    The file list does, as it does the pure renames. No line counted is how
    that file shows in the list: the forge adapters count from the diff they
    assembled, and local intake from git, which counts every line of a
    deleted text file. A name refused as a path has no lines because it was
    refused, and the guardrails already stop the change on it.
    """

    unread = [
        item.path
        for item in change.files_changed
        if item.change_type == "deleted"
        and not item.added_lines
        and not item.removed_lines
        and is_safe_relative_path(item.path)
    ]
    return [*unread_deletions(unread), *detect(change.diff, change.renames)]


def _finding(
    rule: Rule,
    gate: GateName,
    index: int,
    *,
    file: str | None = None,
    line: int | None = None,
    side: DiffSide = "head",
    detail: str = "",
) -> Finding:
    return Finding(
        id=f"{gate}.{rule.id}.{index}",
        gate=gate,
        rule_id=rule.id,
        severity=rule.severity,
        title=rule.title,
        detail=detail,
        file=file,
        line_start=line,
        line_end=line,
        side=side,
        decided_by="deterministic",
        suggested_fix=rule.fix,
    )


def _regex_detail(rule: Rule, text: str) -> str:
    detail = rule.detail or f"An added line matches the pattern of rule {rule.id}."
    # The line is the change author's text, and the detail is Markdown in the
    # evidence pack and in the comment the forge adapter posts on that line.
    return f"{detail}\nMatched: {markdown_code(_excerpt(text))}"


def _integrity_detail(rule: Rule, hit: IntegrityHit) -> str:
    """What happened, the line it happened on, then why the team cares.

    The line is what makes the finding checkable: "a test was removed" is a
    claim, and the removed ``def test_...`` is the evidence for it. Every
    line a hit quotes is the change author's text, and ``markdown_code`` is
    what keeps it from reading as Markdown in the pack and on the change.
    """

    lines = [hit.detail]
    if hit.replacement is not None:
        lines += [f"Was: {markdown_code(_excerpt(hit.text))}"]
        lines += [f"Now: {markdown_code(_excerpt(hit.replacement))}"]
    elif hit.text.strip():
        label = "Removed" if hit.side == "base" else "Added"
        lines.append(f"{label}: {markdown_code(_excerpt(hit.text))}")
    if rule.detail:
        lines.append(rule.detail)
    return "\n".join(lines)


def _excerpt(text: str) -> str:
    excerpt = redact_text(text.strip())
    if len(excerpt) > _MAX_EXCERPT_CHARS:
        excerpt = excerpt[:_MAX_EXCERPT_CHARS] + "…"
    return excerpt


def _parse_rule(entry: dict[str, Any], index: int, source: Path) -> Rule:
    where = f"Rule file {source}: rule #{index}"
    rule_id = str(entry.get("id", "")).strip()
    if not _RULE_ID_RE.match(rule_id):
        raise RuleError(
            f"{where} needs an id matching {_RULE_ID_RE.pattern} (e.g. 'R-03'); got {rule_id!r}."
        )
    where = f"Rule file {source}: rule {rule_id}"

    title = str(entry.get("title", "")).strip()
    if not title:
        raise RuleError(f"{where} needs a title; it is what a blocked developer reads.")

    kind = str(entry.get("kind", "")).strip()
    if kind not in _KINDS:
        raise RuleError(f"{where} has unknown kind {kind!r}. Expected one of: {', '.join(_KINDS)}.")

    check: IntegrityCheck | None = None
    action: IntegrityAction | None = None
    if kind == "integrity":
        check, action = _integrity_settings(entry, where)
    elif "check" in entry or "action" in entry:
        # Refused rather than ignored: ``action = "block"`` on a regex rule
        # reads as if it decided something, and only severity does there.
        raise RuleError(
            f"{where}: 'check' and 'action' only mean something for an integrity rule; "
            "the severity is what decides whether any other rule blocks."
        )

    # A block is what high and above already mean, so that is where a
    # blocking integrity rule starts; everything else starts where every other
    # rule does.
    severity = str(entry.get("severity", "high" if action == "block" else "medium")).strip()
    if severity not in _SEVERITIES:
        raise RuleError(
            f"{where} has unknown severity {severity!r}. "
            f"Expected one of: {', '.join(_SEVERITIES)}."
        )
    if action == "block" and severity not in BLOCKING_SEVERITIES:
        raise RuleError(
            f"{where} blocks on a hit, so its severity must be high or critical; got "
            f"{severity!r}. Only a finding that high stops a merge, so a lower one would "
            'leave action = "block" blocking nothing.'
        )
    if action in {"needs_human", "record"} and severity in BLOCKING_SEVERITIES:
        raise RuleError(
            f"{where} has action {action!r}, so its severity must be medium or lower; got "
            f'{severity!r}. A finding that high stops the merge, which is action = "block".'
        )

    pattern: re.Pattern[str] | None = None
    tool: str | None = None
    if kind == "regex":
        raw = entry.get("pattern")
        if not isinstance(raw, str) or not raw:
            raise RuleError(f"{where} is a regex rule and needs a non-empty 'pattern'.")
        flags = re.IGNORECASE if bool(entry.get("ignore_case", False)) else 0
        try:
            pattern = re.compile(raw, flags)
        except re.error as exc:
            raise RuleError(f"{where} has an invalid pattern: {exc}.") from exc
    elif kind == "tool":
        tool = str(entry.get("tool", "")).strip()
        known = supported_tools()
        if tool not in known:
            raise RuleError(
                f"{where} binds to unsupported tool {tool!r}. "
                f"Expected one of: {', '.join(known)} (each fixed in code by the toolchain "
                "that runs it). Scanners must be able to reach a verdict with the network denied."
            )
    elif kind == "judgment" and not str(entry.get("detail", "")).strip():
        raise RuleError(
            f"{where} is a judgment rule and needs a 'detail' saying what makes a change "
            "violate it. That text is the criterion the model reads the change against, "
            "and it is also what a blocked developer argues with, so there is no useful "
            "default for it."
        )

    min_confidence = _ratio(entry.get("min_confidence"), where)
    if min_confidence is not None and kind != "judgment":
        raise RuleError(
            f"{where}: 'min_confidence' only means something for a judgment rule; "
            "a deterministic rule either matched or it did not."
        )

    paths = _string_tuple(entry.get("paths"), where, "paths") or ("**",)
    if kind == "path" and "paths" not in entry:
        raise RuleError(f"{where} is a path rule and needs 'paths' to say which paths it flags.")

    return Rule(
        id=rule_id,
        title=title,
        kind=cast(RuleKind, kind),
        severity=cast(Severity, severity),
        detail=str(entry.get("detail", "")).strip(),
        paths=paths,
        exclude_paths=_string_tuple(entry.get("exclude_paths"), where, "exclude_paths"),
        pattern=pattern,
        tool=tool,
        check=check,
        action=action,
        min_confidence=min_confidence,
        fix=str(entry["fix"]).strip() if entry.get("fix") else None,
        source_samples=_string_tuple(entry.get("source_samples"), where, "source_samples"),
    )


def _integrity_settings(
    entry: dict[str, Any], where: str
) -> tuple[IntegrityCheck, IntegrityAction]:
    check = str(entry.get("check", "")).strip()
    if check not in INTEGRITY_CHECKS:
        raise RuleError(
            f"{where} is an integrity rule and needs a 'check', one of: "
            f"{', '.join(INTEGRITY_CHECKS)}; got {check!r}."
        )
    action = str(entry.get("action", "needs_human")).strip()
    if action not in _ACTIONS:
        raise RuleError(
            f"{where} has unknown action {action!r}. Expected one of: {', '.join(_ACTIONS)}."
        )
    return check, action


def _ratio(value: Any, where: str) -> float | None:
    # ``bool`` is an ``int`` in Python, and ``min_confidence = true`` reading as
    # 1.0 would silently switch a rule off rather than tell its author.
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuleError(f"{where}: 'min_confidence' must be a number between 0 and 1.")
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise RuleError(f"{where}: 'min_confidence' must be between 0 and 1; got {ratio}.")
    return ratio


def _string_tuple(value: Any, where: str, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuleError(f"{where}: '{field_name}' must be a list of strings.")
    return tuple(value)


__all__ = [
    "MAX_FINDINGS_PER_RULE",
    "IntegrityAction",
    "Rule",
    "RuleError",
    "RuleKind",
    "RuleSet",
    "evaluate_rules",
    "supported_tools",
]
