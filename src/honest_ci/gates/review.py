"""The review gate, v1: deterministic rules, static scanners, model judgment.

What the gate is for, in the product's terms: **a first review against your own
rules, able to say why it blocked**. Every finding it raises names the rule it
violated and points at a line the change added -- or, for test integrity,
removed -- so a block can be argued with instead of merely obeyed.

Three decisions shape this file:

* **The rule library decides what runs.** No rules, no findings — and no
  scanners either, because which scanners are mandatory is something a team
  configures and can audit, not a default this code chose for them.
* **An empty rule library is not a pass.** A gate that reviewed nothing has
  verified nothing, so it routes to a human and says why. That is also what
  pushes a new team to write the first draft of their rule file.
* **A scanner diagnostic only becomes a finding on a line this change added.**
  Pre-existing problems in a file someone happened to touch are reported in the
  evidence as context, and counted, but they do not block. When the change
  carries no diff to attribute against, the gate keeps every diagnostic instead
  of dropping them: unattributable is a reason to be stricter, not looser.

Test integrity -- a deleted test, a new skip, a rewritten expected value, a
lowered coverage bar -- is judged here as well, as ``integrity`` rules, rather
than as a policy violation beside the sensitive paths. A hit only helps anyone
if it names the file and the line, and this gate is where findings already
get that: a place in the evidence pack, the comment on the change, and
``annotate`` to mark one a false positive. A policy violation is a sentence.
The checks read the lines a change *removed*, and the added-lines decision
above does not stand in their way: it keeps out problems that were there
before the change, and a removed line is the change itself. What a hit does
is its rule's ``action``: ``block`` through ``Finding.blocking`` like any
other high finding, ``needs_human`` through ``_verdict``, and ``record``
nothing beyond the evidence pack.

The kind of rule a model reads — ``judgment``, the ones a team really has but
nobody can write as a path or a pattern — is what v1 adds, and it is bounded by
three further decisions:

* **A model finding routes to a human; it never blocks.** ``Finding.blocking``
  enforces that (see ``contracts.py``), and the reason is that a block here is
  terminal: only a new commit clears it, because an approval reaches runs that
  waited for a human and not blocked ones. A probabilistic judgment that could
  block would be the model replacing an approval instead of informing one —
  and while a team does not yet know how often the model is wrong about its
  rules, it would make every false positive cost a forced code change.
* **Not asking is not a pass.** No model enabled, no credential, judgment
  switched off, a failed call, output the trust boundary threw away: each
  leaves those rules unread, and unread rules send the change to a human with
  a line under "What this run could not verify" saying which one it was. That
  is the same "not run is not passed" the scanners obey — but ``needs_human``
  rather than a block, because unlike a scanner, the model was never allowed to
  decide anything in the first place.
* **Doubt below the bar is recorded, not escalated.** A finding under its
  rule's confidence threshold, and the model's open questions, go into the
  evidence rather than into the verdict. The threshold is exactly where a
  team says "below this, do not interrupt me"; routing to a human anyway
  would make it decorative, and a model asks questions on every change.

The change may also arrive carrying review comments somebody already wrote on
it at the platform. They are read two ways and neither of them is new authority
for a machine: deterministically, an **unresolved** thread routes the change to
a human, because a question a person asked and nobody answered is not something
an automated pass should walk past — and whether it does is one auditable line
in the team's policy file, not a default this code chose. As context, the
comments go to the model alongside the judgment rules; a finding still has to
cite a rule id and land on an added line, so nothing said in a comment thread
can become a citation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import (
    ChangeRequest,
    Finding,
    GateName,
    GateReport,
    PlatformReview,
    ToolDiagnostic,
    ToolReport,
    Verdict,
    WorkspaceKind,
    compute_fingerprint,
)
from ..diffs import changed_line_locations, markdown_code, markdown_code_block, markdown_path
from ..grounding import guarded_rule_judgment
from ..policy import UnresolvedReviews
from ..rules import IntegrityAction, Rule, RuleSet, evaluate_rules
from ..settings import MODEL_NOT_ENABLED
from ..state import RunState
from .base import Gate, unavailable_reasons

#: Authors named in the unresolved-threads finding before it is summarised.
_MAX_NAMED_REVIEWERS = 5


@dataclass(frozen=True)
class _Judgment:
    """What became of this run's judgment-class rules.

    ``judged`` is the field that carries the weight, and it is deliberately not
    inferable from the others: no findings and no judgment both come back as an
    empty ``findings``, and only one of them means the rules were read.
    """

    rules: tuple[Rule, ...] = ()
    judged: bool = False
    #: Why they went unread. Empty when they were read, or when there were none.
    reason: str = ""
    findings: tuple[Finding, ...] = ()
    below_threshold: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    summary: str = ""

    @property
    def unread(self) -> bool:
        return bool(self.rules) and not self.judged

    @property
    def needs_human(self) -> bool:
        """Whether these rules put the change in front of a person.

        A finding does, because a model's finding is a question for a reviewer
        rather than a verdict. Unread rules do, because nobody looked. Doubts
        below the threshold do not — see the module docstring.
        """

        return bool(self.findings) or self.unread


@dataclass(frozen=True)
class _PlatformReviews:
    """What the platform already carried, and what this policy makes of it.

    ``error`` is the same distinction ``_Judgment.judged`` draws: no comments
    and no *reading* of the comments both arrive as an empty ``reviews``, and
    only one of them is a statement about this change.
    """

    reviews: tuple[PlatformReview, ...] = ()
    policy: UnresolvedReviews = "needs_human"
    error: str | None = None

    @property
    def unresolved(self) -> tuple[PlatformReview, ...]:
        return tuple(review for review in self.reviews if review.unresolved)

    @property
    def needs_human(self) -> bool:
        """Whether an open thread is what puts this change in front of a person.

        Only when the policy says so. Deterministic either way -- this reads a
        flag the platform sets, not the prose of the comment -- and auditable
        either way, because the answer lives in the team's policy file.
        """

        return self.policy == "needs_human" and bool(self.unresolved)


def _platform_reviews(change: ChangeRequest, policy: UnresolvedReviews) -> _PlatformReviews:
    return _PlatformReviews(
        reviews=tuple(change.platform_reviews),
        policy=policy,
        error=change.platform_reviews_error,
    )


def _unresolved_finding(reviews: _PlatformReviews, index: int) -> Finding:
    """One finding for every open thread, not one per thread.

    The threads are on the merge request already, where they read better than
    any copy of them; what the gate adds is the fact that nobody closed them.
    ``medium`` because that fact is a reason to involve a person, not a defect
    in the code -- and because ``high`` here would block a change over somebody
    forgetting to click "resolve".
    """

    open_threads = reviews.unresolved
    authors = sorted({review.author for review in open_threads if review.author})
    named = ", ".join(authors[:_MAX_NAMED_REVIEWERS]) or "(unknown)"
    if len(authors) > _MAX_NAMED_REVIEWERS:
        named += f" ({len(authors)} people in all)"
    return Finding(
        id=f"review.platform-reviews.{index}",
        gate="review",
        severity="medium",
        title=f"Unresolved review threads on the change: {len(open_threads)}",
        detail=(
            f"{named} raised questions on this change, and the threads are still unresolved. "
            "Letting the change through automatically would overrule a person who has not "
            "been answered, so it goes to a person.\n"
            + "\n".join(_review_line(review) for review in open_threads[:_MAX_ROWS])
        ),
        decided_by="deterministic",
        suggested_fix=(
            "Answer and resolve these threads, then run again. If your team routinely "
            "leaves threads open, set unresolved_reviews = \"ignore\" in "
            "honest-ci-policy.toml."
        ),
    )


def _review_line(review: PlatformReview) -> str:
    # The path is the platform's, as the commenter's client sent it, and this
    # line is Markdown wherever the finding is shown.
    location = markdown_path(review.path) if review.path else "(the whole change)"
    if review.path and review.line is not None:
        location = f"{location}:{review.line}"
    who = review.author or "(unknown)"
    return f"- {who}{' (bot)' if review.is_bot else ''} · {location} · {_excerpt(review.body)}"


def _excerpt(body: str) -> str:
    text = " ".join(body.split())
    text = text[:_MAX_EXCERPT_CHARS] + "…" if len(text) > _MAX_EXCERPT_CHARS else text
    # Whoever can comment on the change wrote it. Flattened and cut short it is
    # no longer the Markdown they wrote, only Markdown that renders.
    return markdown_code(text)


#: Open threads quoted inside one finding, and how much of each is quoted. The
#: merge request itself holds the rest.
_MAX_ROWS = 10
_MAX_EXCERPT_CHARS = 200


class ReviewGate(Gate):
    name: GateName = "review"

    def evaluate(self, state: RunState) -> GateReport:
        rule_set = self.services.rule_set
        findings = evaluate_rules(rule_set, state.change, gate=self.name)
        tool_reports = self._run_scanners(rule_set, state)
        attribution = _line_attribution(state.change)
        tool_findings, kept, total = _findings_from_tools(
            rule_set, tool_reports, attribution, len(findings)
        )
        findings += tool_findings
        judgment = self._judge(rule_set, state)
        findings += list(judgment.findings)
        platform = _platform_reviews(
            state.change,
            self.services.policy_engine.config.unresolved_reviews,
        )
        if platform.needs_human:
            findings.append(_unresolved_finding(platform, len(findings) + 1))

        verdict = _verdict(findings, rule_set, judgment, platform)
        return GateReport(
            gate=self.name,
            verdict=verdict,
            summary=_summary(
                verdict, rule_set, findings, tool_reports, kept, total, judgment, platform
            ),
            findings=findings,
            tool_reports=tool_reports,
            metrics={
                "rules_total": float(len(rule_set.rules)),
                "rules_fired": float(len({f.rule_id for f in findings if f.rule_id})),
                "findings_total": float(len(findings)),
                "findings_blocking": float(len([f for f in findings if f.blocking])),
                "diagnostics_total": float(total),
                "diagnostics_on_added_lines": float(kept),
                "tools_run": float(len([t for t in tool_reports if t.available])),
                "tools_unavailable": float(len([t for t in tool_reports if not t.available])),
                "integrity_rules": float(len(rule_set.of_kind("integrity"))),
                "integrity_findings": float(
                    len(_integrity_findings(findings, rule_set.of_kind("integrity")))
                ),
                "judgment_rules": float(len(judgment.rules)),
                "judgment_findings": float(len(judgment.findings)),
                "judgment_rules_unread": float(len(judgment.rules) if not judgment.judged else 0),
                "platform_reviews": float(len(platform.reviews)),
                "platform_reviews_unresolved": float(len(platform.unresolved)),
            },
            verifier_report=_verifier_report(
                rule_set,
                tool_reports,
                attribution,
                self._workspace_kind(state),
                judgment,
                platform,
            ),
            rule_version=rule_set.version,
            policy_version=self.services.settings.policy_version,
        )

    def _judge(self, rule_set: RuleSet, state: RunState) -> _Judgment:
        """Put the judgment-class rules to a model, or say why they went unread.

        The ways this ends without a judgment are kept apart because "What
        this run could not verify" has to name the right one: "we configured no
        model" and "the model's answer was nonsense and was discarded" are
        different facts about the same unreviewed rule, and only one of them is
        something the team can fix by turning a model on.
        """

        rules = rule_set.judgment_rules
        if not rules:
            return _Judgment()
        settings = self.services.settings
        if not settings.model_judgment:
            return _Judgment(
                rules=rules,
                reason="model judgment is switched off in this deployment "
                "(HONEST_CI_MODEL_JUDGMENT=0)",
            )
        llm = self.services.llm
        if llm is None:
            if settings.model_provider is None:
                return _Judgment(rules=rules, reason=MODEL_NOT_ENABLED)
            return _Judgment(
                rules=rules, reason=f"no model credential is configured ({settings.credential_hint})"
            )

        judgment = guarded_rule_judgment(
            state.change,
            rules,
            # The same fingerprint an approval will be bound to, so a response
            # produced against another commit or another rule version cannot be
            # mistaken for one about this change.
            compute_fingerprint(
                state.change.head_sha,
                settings.policy_version,
                rule_set.version,
            ),
            llm,
            default_min_confidence=settings.model_min_confidence,
            gate=self.name,
        )
        if judgment is None:
            return _Judgment(
                rules=rules,
                reason="the model call failed, or its output failed the citation check or "
                "the line check and was discarded whole",
            )
        return _Judgment(
            rules=rules,
            judged=True,
            findings=judgment.findings,
            below_threshold=judgment.below_threshold,
            questions=judgment.questions,
            summary=judgment.summary,
        )

    def _run_scanners(self, rule_set: RuleSet, state: RunState) -> list[ToolReport]:
        tools = rule_set.required_tools
        if not tools:
            return []
        scanner = self.services.scanner
        if scanner is None:
            # A rule demands a scanner this deployment does not have. Reporting
            # it unavailable is what turns it into a block one layer up, rather
            # than a rule that silently never fires.
            return [
                ToolReport(
                    name=name,
                    command=f"{name} (no scanner adapter configured)",
                    available=False,
                    passed=False,
                    stderr="No ScannerAdapter is configured in this deployment.",
                    findings=["No ScannerAdapter is configured in this deployment."],
                )
                for name in tools
            ]
        return scanner.run_static_checks(
            self._workspace(state),
            state.change,
            tools=tools,
            # A tool rule's ``paths`` says which files that scanner is being
            # asked about. Without this the scope was a comment: dogfooding
            # handed mypy the change's test files in a repository whose own
            # mypy configuration only ever looks at ``src``, and got 23
            # diagnostics that no rule had asked for. These are the names
            # ``evaluate_rules`` reads, so one rule's ``paths`` cannot cover a
            # file for the deterministic pass and miss it for the scanner
            # pass; a name no longer on disk is the scanner's to drop.
            scopes=rule_set.tool_scopes(state.change.touched_paths),
        )


# ── attribution ──────────────────────────────────────────────────────────────


class _Attribution:
    """Which lines this change added, and whether that is knowable at all."""

    def __init__(self, change: ChangeRequest) -> None:
        self.known = bool(change.diff.strip())
        self.added: dict[str, set[int]] = changed_line_locations(change.diff)

    def covers(self, diagnostic: ToolDiagnostic) -> bool:
        if diagnostic.file is None or not self.known:
            return True
        added = self.added.get(_normalize(diagnostic.file))
        if added is None:
            return False
        if diagnostic.line is None:
            return True
        return diagnostic.line in added


def _line_attribution(change: ChangeRequest) -> _Attribution:
    return _Attribution(change)


def _normalize(path: str) -> str:
    return path[2:] if path.startswith("./") else path


# ── tool findings ────────────────────────────────────────────────────────────


def _findings_from_tools(
    rule_set: RuleSet,
    tool_reports: list[ToolReport],
    attribution: _Attribution,
    offset: int,
) -> tuple[list[Finding], int, int]:
    """Attribute scanner output to the rules that asked for those scanners."""

    by_name = {report.name: report for report in tool_reports}
    findings: list[Finding] = []
    kept = 0
    total = 0

    # Walk tools rather than rules: two rules may bind the same scanner, and its
    # diagnostics have to be counted once even when both of them fire on it.
    for tool in rule_set.required_tools:
        report = by_name.get(tool)
        if report is None or not report.available:
            # Unavailable tools are already in ``tool_reports``; the gate
            # template turns them into a block. Adding a finding here too would
            # report the same missing evidence twice.
            continue
        rules = [rule for rule in rule_set.of_kind("tool") if rule.tool == tool]
        total += len(report.diagnostics)
        for item in report.diagnostics:
            if not attribution.covers(item):
                continue
            matched = [rule for rule in rules if rule.applies_to(_normalize(item.file or ""))]
            if matched:
                kept += 1
            for rule in matched:
                findings.append(_tool_finding(rule, item, offset + len(findings) + 1))
        if not report.passed and not report.diagnostics:
            # It ran, it failed, and nothing parsed: the tool itself is broken
            # or misconfigured, which is missing evidence, not a clean file.
            for rule in rules:
                findings.append(
                    _opaque_failure_finding(rule, report, offset + len(findings) + 1)
                )
    return findings, kept, total


def _tool_finding(rule: Rule, diagnostic: ToolDiagnostic, index: int) -> Finding:
    code = f"[{diagnostic.code}] " if diagnostic.code else ""
    # A scanner quotes the code it flags -- a name, a string literal -- and the
    # detail is Markdown in the evidence pack and on the line comment.
    detail = f"{diagnostic.tool}: {code}{markdown_code(diagnostic.message)}"
    if rule.detail:
        detail = f"{rule.detail}\n{detail}"
    return Finding(
        id=f"review.{rule.id}.{index}",
        gate="review",
        rule_id=rule.id,
        severity=rule.severity,
        title=rule.title,
        detail=detail,
        file=_normalize(diagnostic.file) if diagnostic.file else None,
        line_start=diagnostic.line,
        line_end=diagnostic.line,
        decided_by="deterministic",
        suggested_fix=rule.fix,
    )


def _opaque_failure_finding(rule: Rule, report: ToolReport, index: int) -> Finding:
    detail = (
        f"{report.name} exited with code {report.exit_code} but left no diagnostics that "
        "could be relied on, so whether this change meets the rule cannot be told."
    )
    tail = (report.stderr or report.stdout or "")[-600:]
    if tail.strip():
        # A scanner that chokes echoes the source it choked on, and that is the
        # change author's text, many lines of it, into the evidence pack.
        detail += "\n" + markdown_code_block(tail)
    return Finding(
        id=f"review.{rule.id}.{index}",
        gate="review",
        rule_id=rule.id,
        severity=rule.severity,
        title=rule.title,
        detail=detail,
        decided_by="deterministic",
        suggested_fix=rule.fix,
    )


# ── verdict & reporting ──────────────────────────────────────────────────────


def _verdict(
    findings: list[Finding],
    rule_set: RuleSet,
    judgment: _Judgment,
    platform: _PlatformReviews,
) -> Verdict:
    # ``Finding.blocking`` is false for anything a model decided, so this line
    # cannot reach BLOCKED on model output however severe the rule. That is the
    # whole of "a model forms no verdict" as far as this gate is concerned;
    # everything a model contributes arrives below, as a reason to involve a
    # person.
    if any(finding.blocking for finding in findings):
        return Verdict.BLOCKED
    if rule_set.is_empty:
        return Verdict.NEEDS_HUMAN
    # Routed by the rule's action, not by a severity band: "high and above
    # blocks" stays the only way a finding stops a merge, and the loader keeps
    # these rules at medium or below so they cannot reach it.
    if _routed_to_human(findings, rule_set):
        return Verdict.NEEDS_HUMAN
    if judgment.needs_human:
        return Verdict.NEEDS_HUMAN
    # An open thread never blocks, for the same reason a model finding never
    # does: a block is terminal and only a new commit clears it, while what an
    # unanswered question needs is the person who asked it.
    if platform.needs_human:
        return Verdict.NEEDS_HUMAN
    return Verdict.PASS


def _integrity_findings(findings: list[Finding], rules: tuple[Rule, ...]) -> list[Finding]:
    ids = {rule.id for rule in rules}
    return [finding for finding in findings if finding.rule_id in ids]


def _routed_to_human(findings: list[Finding], rule_set: RuleSet) -> list[Finding]:
    """The findings of integrity rules whose action is to ask a person."""

    return _integrity_findings(findings, rule_set.integrity_rules("needs_human"))


#: How each action's share of the hits reads in the gate's summary line.
_INTEGRITY_SUMMARY: tuple[tuple[IntegrityAction, str], ...] = (
    ("block", "blocking"),
    ("needs_human", "for a human to review"),
    ("record", "recorded only"),
)


def _integrity_summary(findings: list[Finding], rule_set: RuleSet) -> list[str]:
    hits = _integrity_findings(findings, rule_set.of_kind("integrity"))
    if not hits:
        return []
    counts = [
        (len(_integrity_findings(hits, rule_set.integrity_rules(action))), label)
        for action, label in _INTEGRITY_SUMMARY
    ]
    shown = ", ".join(f"{count} {label}" for count, label in counts if count)
    return [f"test integrity: {len(hits)} hit{'s' if len(hits) != 1 else ''} ({shown})"]


def _summary(
    verdict: Verdict,
    rule_set: RuleSet,
    findings: list[Finding],
    tool_reports: list[ToolReport],
    kept: int,
    total: int,
    judgment: _Judgment,
    platform: _PlatformReviews,
) -> str:
    if rule_set.is_empty:
        also = _platform_summary(platform)
        return (
            "No team rule library is configured, so nothing was reviewed against rules; "
            "the verdict goes to a person."
            + (f" Also: {'; '.join(also)}." if also else "")
        )
    blocking = len([finding for finding in findings if finding.blocking])
    parts = [
        f"Reviewed against {_counted(len(rule_set.rules), 'rule')} ({rule_set.version})",
        f"{_counted(len(findings), 'finding')}, {blocking} blocking",
    ]
    if tool_reports:
        unavailable = unavailable_reasons(tool_reports)
        parts.append(f"{_counted(total, 'scanner diagnostic')}, {kept} on added lines")
        if unavailable:
            # With the first line of why: this summary is what the terminal,
            # the evidence pack's gate table and the pull request comment all
            # show, and the regression gate's already says why its suite did
            # not start. The whole reason is on the fail-closed finding.
            parts.append(
                "scanners that did not start: "
                + "; ".join(
                    ", ".join(names)
                    + (f" — {markdown_code(reason.splitlines()[0])}" if reason else "")
                    for names, reason in unavailable
                )
            )
    parts += _integrity_summary(findings, rule_set)
    parts += _judgment_summary(judgment)
    parts += _platform_summary(platform)
    if verdict is Verdict.PASS and total > kept:
        parts.append(
            "the other diagnostics are in code this change did not touch, "
            "so they are recorded and do not block"
        )
    return "; ".join(parts) + "."


def _judgment_summary(judgment: _Judgment) -> list[str]:
    if not judgment.rules:
        return []
    rules = _counted(len(judgment.rules), "judgment rule")
    if judgment.unread:
        return [f"{rules} not judged ({judgment.reason}), so the verdict goes to a person"]
    parts = [f"{rules} put to a model"]
    if judgment.findings:
        # Spelled out rather than folded into the hit count above: a reader of
        # one summary line has to be able to tell what code decided from what a
        # model proposed, without opening the evidence pack.
        parts.append(
            f"the model raised {_counted(len(judgment.findings), 'doubt')} "
            "(not blocking, for a person to judge)"
        )
    if judgment.below_threshold:
        parts.append(
            f"{len(judgment.below_threshold)} more below the confidence bar, recorded only"
        )
    return parts


def _platform_summary(platform: _PlatformReviews) -> list[str]:
    """One clause about the comments already on the change, when there is one."""

    if platform.error is not None:
        return [
            "the review comments already on the change could not be read, "
            "and this verdict does not take them into account"
        ]
    if not platform.reviews:
        return []
    parts = [f"{_counted(len(platform.reviews), 'review comment')} already on the change"]
    open_threads = len(platform.unresolved)
    if not open_threads:
        return parts
    if platform.policy == "needs_human":
        parts.append(f"{open_threads} unresolved, which the policy sends to a person")
    else:
        parts.append(
            f'{open_threads} unresolved, which the policy (unresolved_reviews = "ignore") '
            "only records"
        )
    return parts


def _counted(count: int, noun: str) -> str:
    """``count`` and ``noun``, the noun in the plural unless there is exactly one."""

    return f"{count} {noun}{'' if count == 1 else 's'}"


def _verifier_report(
    rule_set: RuleSet,
    tool_reports: list[ToolReport],
    attribution: _Attribution,
    workspace_kind: WorkspaceKind,
    judgment: _Judgment,
    platform: _PlatformReviews,
) -> dict[str, Any]:
    """The deterministic detail an evidence pack has to be able to show."""

    report: dict[str, Any] = {
        "rule_source": rule_set.source,
        "rule_version": rule_set.version,
        "rules_evaluated": [
            {
                "id": rule.id,
                "kind": rule.kind,
                "severity": rule.severity,
                "title": rule.title,
                "tool": rule.tool,
                # What an integrity rule checks and what its hits do: the
                # pack's reader needs both to tell a hit that stopped the
                # change from one that was only written down.
                "check": rule.check,
                "action": rule.action,
                # The samples the rule was mined from, carried into the pack so
                # "why does this rule exist" is answerable from the evidence
                # rather than from whoever ran the mining session.
                "source_samples": list(rule.source_samples),
            }
            for rule in rule_set.rules
        ],
        "tools": [
            {
                "name": report.name,
                "available": report.available,
                "passed": report.passed,
                "exit_code": report.exit_code,
                "diagnostics": len(report.diagnostics),
            }
            for report in tool_reports
        ],
        "line_attribution": "diff" if attribution.known else "unavailable",
        "workspace_kind": workspace_kind,
    }
    if judgment.rules:
        report["judgment"] = _judgment_report(judgment)
    if platform.reviews or platform.error is not None:
        report["platform_reviews"] = {
            "policy": platform.policy,
            "total": len(platform.reviews),
            "unresolved": len(platform.unresolved),
            "bots": len([review for review in platform.reviews if review.is_bot]),
            "not_read_reason": platform.error,
        }
    return report


def _judgment_report(judgment: _Judgment) -> dict[str, Any]:
    """What was asked of the model, and what survived the trust boundary."""

    return {
        "rules": [
            {
                "id": rule.id,
                "title": rule.title,
                "severity": rule.severity,
                "min_confidence": rule.min_confidence,
            }
            for rule in judgment.rules
        ],
        "judged": judgment.judged,
        "not_judged_reason": judgment.reason or None,
        "findings": len(judgment.findings),
        "below_threshold": list(judgment.below_threshold),
        "questions": list(judgment.questions),
        "summary": judgment.summary,
    }


__all__ = ["ReviewGate"]
