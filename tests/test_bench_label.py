"""Labelling the benchmark's hits (bench/label.py): flags, packets that keep
the group out of sight, the suggestions, the review, and the tally.

A run here is two pull requests with hits and one without, written as a real
run writes them: hits.csv, prs.csv, and their answers in the cache.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from bench import label
from bench.cache import Cache, cache_key
from bench.report import HITS_COLUMNS, spreadsheet_safe
from bench.source import PULL_QUERY

ROUNDING = "https://github.com/acme/shop/pull/1"
CART = "https://github.com/acme/shop/pull/2"
QUIET = "https://github.com/acme/shop/pull/3"
AT = "2026-10-13T00:00:00+00:00"

ROUNDING_DIFF = """\
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
diff --git a/tests/test_tax.py b/tests/test_tax.py
deleted file mode 100644
--- a/tests/test_tax.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def test_tax():
-    assert tax(10) == 1
diff --git a/tests/test_price.py b/tests/test_price.py
--- a/tests/test_price.py
+++ b/tests/test_price.py
@@ -1,2 +1,2 @@
 def test_price():
-    assert price(3) == 9
+    assert price(3) == 10
"""

CART_DIFF = """\
diff --git a/tests/test_cart.py b/tests/test_cart.py
--- a/tests/test_cart.py
+++ b/tests/test_cart.py
@@ -1,2 +1,3 @@
+@pytest.mark.skip
 def test_cart():
     assert cart() == []
"""


def _hit(
    group: str, pr: str, check: str, file: str, line: int, side: str, text: str, **more: Any
) -> dict[str, Any]:
    return {
        "group": group,
        "pr": pr,
        "title": "Round refunds down" if pr == ROUNDING else "Skip the cart test on Windows",
        "check": check,
        "file": file,
        "line": line,
        "side": side,
        "link": f"{pr}/files#diff-{file}",
        "text": text,
        "replacement": more.get("replacement", ""),
        "detail": more.get("detail", f"The {check} check fired."),
        "merged": pr == ROUNDING,
        "label": "",
    }


HITS = [
    _hit(
        "claude",
        ROUNDING,
        "deleted_test",
        "tests/test_refund.py",
        5,
        "base",
        "def test_rounding():",
    ),
    _hit("claude", ROUNDING, "deleted_test", "tests/test_tax.py", 1, "base", "def test_tax():"),
    _hit(
        "claude",
        ROUNDING,
        "changed_expectation",
        "tests/test_price.py",
        2,
        "base",
        "    assert price(3) == 9",
        replacement="    assert price(3) == 10",
    ),
    _hit("human", CART, "skip_or_focus", "tests/test_cart.py", 1, "head", "@pytest.mark.skip"),
]


def _pull(
    node_id: str, url: str, title: str, body: str, branch: str, messages: Sequence[str]
) -> dict[str, Any]:
    return {
        "id": node_id,
        "url": url,
        "title": title,
        "body": body,
        "headRefName": branch,
        "author": {"__typename": "User", "login": "rosalind"},
        "mergedAt": "2026-07-02T00:00:00Z" if url == ROUNDING else None,
        "commits": {
            "totalCount": len(messages),
            "nodes": [
                {
                    "commit": {
                        "message": message,
                        "author": {"name": "rosalind", "email": "r@example.com"},
                    }
                }
                for message in messages
            ],
        },
    }


@pytest.fixture
def run(tmp_path: Path) -> Path:
    cache = Cache(tmp_path / "out" / "cache")
    rounding = _pull(
        "PR_1",
        ROUNDING,
        "Round refunds down",
        "Refunds round down now.\r\n\r\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"
        "\r\n<!-- agent marker -->",
        "claude/round-refunds",
        [
            "Round refunds down\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
            "Merge branch 'main' into claude/round-refunds",
        ],
    )
    cache.put("pull", cache_key(PULL_QUERY, "PR_1"), rounding)
    # The same pull request as an older query asked it: another shape, not read.
    cache.put("pull", cache_key("an older query", "PR_1"), {**rounding, "title": "Stale title"})
    cache.put(
        "pull",
        cache_key(PULL_QUERY, "PR_2"),
        _pull(
            "PR_2",
            CART,
            "Skip the cart test on Windows",
            "Flaky on Windows, see #12.",
            "cart",
            ["Skip"],
        ),
    )
    cache.put("diff", cache_key("acme/shop", 1), {"text": ROUNDING_DIFF, "unavailable": ""})
    cache.put("diff", cache_key("acme/shop", 2), {"text": CART_DIFF, "unavailable": ""})
    directory = tmp_path / "out" / "run"
    directory.mkdir()
    _write_csv(
        directory / "hits.csv", HITS_COLUMNS, ([hit[name] for name in HITS_COLUMNS] for hit in HITS)
    )
    _write_csv(
        directory / "prs.csv",
        ("group", "pr", "merged"),
        [("claude", ROUNDING, True), ("human", CART, False), ("human", QUIET, True)],
    )
    return directory


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([spreadsheet_safe(value) for value in row])


def _sheet(run: Path, blind: int = 0) -> label.Sheet:
    return label.make_sheet(run, Cache(run.parent / "cache"), blind)


def _flag(sheet: label.Sheet, pr: str, check: str) -> label.Flag:
    return next(flag for flag in sheet.flags if flag.pr == pr and flag.check == check)


def _suggest(run: Path, *lines: Mapping[str, object] | str) -> None:
    with (run / "labels" / "codex.jsonl").open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write((line if isinstance(line, str) else json.dumps(line)) + "\n")


def _decided(
    verdict: str,
    *,
    files: Mapping[str, str] | None = None,
    suggested: str | None = None,
    blind: bool = False,
) -> label.Decision:
    return label.Decision(
        label.Verdict(verdict, dict(files or {})),
        "",
        label.Verdict(suggested) if suggested else None,
        blind,
        AT,
    )


class Script:
    """Answers for ``review``, in order; then the end of input."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)

    def __call__(self, prompt: str) -> str:
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def _review(run: Path, *answers: str, unit: str | None = None) -> list[str]:
    said: list[str] = []
    opened: list[str] = []
    label.review(
        run, unit=unit, ask=Script(*answers), say=said.append, page=print, open_url=opened.append
    )
    return said


# ── the sheet and the packets ────────────────────────────────────────────────


def test_a_flag_is_one_pull_requests_hits_of_one_check(run: Path) -> None:
    assert label.main(["sheet", str(run), "--blind", "0"]) == 0

    sheet = label.load_sheet(run)
    assert sorted((flag.pr, flag.check, flag.files) for flag in sheet.flags) == [
        (ROUNDING, "changed_expectation", ["tests/test_price.py"]),
        (ROUNDING, "deleted_test", ["tests/test_refund.py", "tests/test_tax.py"]),
        (CART, "skip_or_focus", ["tests/test_cart.py"]),
    ]
    with (run / "labels" / "units.csv").open(encoding="utf-8", newline="") as handle:
        units = list(csv.DictReader(handle))
    assert [row["unit"] for row in units] == [flag.unit for flag in sheet.flags]


def test_the_order_is_fixed_and_keeps_a_pull_requests_flags_together(run: Path) -> None:
    first = [flag.unit for flag in _sheet(run).flags]
    again = [flag.unit for flag in _sheet(run).flags]

    assert first == again
    rounding = [place for place, flag in enumerate(_sheet(run).flags) if flag.pr == ROUNDING]
    assert rounding[1] == rounding[0] + 1


def test_neither_the_sheet_nor_a_packet_says_who_opened_a_pull_request(run: Path) -> None:
    _sheet(run)
    workspace = run / "labels"
    texts = [
        (workspace / "units.csv").read_text(encoding="utf-8"),
        *(path.read_text(encoding="utf-8") for path in (workspace / "packets").glob("*.md")),
    ]

    for text in texts:
        for clue in (
            "claude",
            "Claude",
            "anthropic",
            "rosalind",
            "human",
            "Human",
            "True",
            "agent marker",
        ):
            assert clue not in text
    refund = (
        workspace / "packets" / f"{_flag(_sheet(run), ROUNDING, 'deleted_test').unit}.md"
    ).read_text(encoding="utf-8")
    assert "Refunds round down now." in refund
    assert "Round refunds down" in refund


def test_a_packet_marks_each_hit_in_the_diff_by_its_line_numbers(run: Path) -> None:
    sheet = _sheet(run)
    refund = (
        run / "labels" / "packets" / f"{_flag(sheet, ROUNDING, 'deleted_test').unit}.md"
    ).read_text(encoding="utf-8")
    cart = (
        run / "labels" / "packets" / f"{_flag(sheet, CART, 'skip_or_focus').unit}.md"
    ).read_text(encoding="utf-8")

    assert re.search(r"^>\s+5\s+-def test_rounding\(\):$", refund, re.MULTILINE)
    assert re.search(r"^>\s+1\s+-def test_tax\(\):$", refund, re.MULTILINE)
    # Around a hit, unmarked, with its numbers on both sides.
    assert re.search(r"^ \s+1\s+1  def test_whole_amounts\(\):$", refund, re.MULTILINE)
    assert "- D `tests/test_tax.py` +0 −2" in refund
    assert "Stale title" not in refund
    assert re.search(r"^>\s+1\s+\+@pytest\.mark\.skip$", cart, re.MULTILINE)
    # Shown as the code it is, not as the spreadsheet keeps it.
    assert "`@pytest.mark.skip`" in cart
    assert (run / "labels" / "diffs" / "acme__shop__2.diff").read_text(
        encoding="utf-8"
    ) == CART_DIFF


def test_making_the_sheet_again_keeps_the_suggestions_and_the_labels(run: Path) -> None:
    flag = _sheet(run).flags[0]
    _suggest(run, {"unit": flag.unit, "label": "wrong", "why": "Not a test."})
    label.write_labels(run / "labels", {flag.unit: _decided("wrong")})

    _sheet(run)

    suggestions, _ = label.read_suggestions(label.load_sheet(run))
    assert suggestions[flag.unit].verdict == label.Verdict("wrong")
    assert label.read_labels(run / "labels")[flag.unit].verdict == label.Verdict("wrong")


# ── suggestions ──────────────────────────────────────────────────────────────


def test_status_names_each_suggestion_it_cannot_use(run: Path) -> None:
    refund = _flag(_sheet(run), ROUNDING, "deleted_test")
    _suggest(
        run,
        "not JSON",
        {"unit": "00000000", "label": "real", "why": "No such flag."},
        {"unit": refund.unit, "label": "Real", "why": "Not a label."},
        {"unit": refund.unit, "label": "real", "why": " "},
        {
            "unit": refund.unit,
            "label": "real",
            "why": "x",
            "files": {"tests/test_cart.py": "wrong"},
        },
        {
            "unit": refund.unit,
            "label": "real",
            "why": "test_rounding goes while refund() stays.",
            "files": {"tests/test_tax.py": "justified"},
        },
    )
    said: list[str] = []

    assert label.status(run, said.append) == 1

    problems = [line for line in said if line.startswith("codex.jsonl")]
    assert [line.split(":")[0] for line in problems] == [
        f"codex.jsonl line {n}" for n in range(1, 6)
    ]
    assert said[0].startswith("3 flags · 1 with a suggestion · 0 labelled")
    suggestions, _ = label.read_suggestions(label.load_sheet(run))
    assert suggestions[refund.unit].verdict.per_file(refund) == {
        "tests/test_refund.py": "real",
        "tests/test_tax.py": "justified",
    }


def test_status_lists_the_blind_differences_once_every_blind_flag_is_labelled(
    run: Path,
) -> None:
    first, second, _ = _sheet(run, blind=2).flags
    _suggest(
        run,
        {"unit": first.unit, "label": "real", "why": "Agrees."},
        {"unit": second.unit, "label": "justified", "why": "THE-REASON."},
    )
    label.write_labels(run / "labels", {first.unit: _decided("real", suggested="real", blind=True)})
    before: list[str] = []
    label.status(run, before.append)
    label.write_labels(
        run / "labels",
        {
            first.unit: _decided("real", suggested="real", blind=True),
            second.unit: _decided("wrong", suggested="justified", blind=True),
        },
    )
    after: list[str] = []

    label.status(run, after.append)

    assert not any("differed" in line for line in before)
    assert after[after.index("Where the suggestion differed on the blind flags:") + 1 :] == [
        f"  {second.unit}: you wrong, suggestion justified. THE-REASON."
    ]


def test_check_names_the_unusable_lines_and_nothing_of_the_labels(run: Path) -> None:
    first, second, third = _sheet(run, blind=2).flags
    _suggest(
        run,
        "not JSON",
        {"unit": first.unit, "label": "real", "why": "Agrees."},
        {"unit": second.unit, "label": "justified", "why": "THE-REASON."},
    )
    label.write_labels(
        run / "labels",
        {
            first.unit: _decided("real", suggested="real", blind=True),
            second.unit: _decided("wrong", suggested="justified", blind=True),
        },
    )
    said: list[str] = []

    assert label.check(run, said.append) == 1

    assert said == [
        "3 flags · 2 with a suggestion",
        f"Next to suggest: {third.unit}, number 3 in units.csv.",
        "codex.jsonl line 1: not JSON",
    ]


def test_the_task_has_the_agent_run_check_not_status(run: Path) -> None:
    _sheet(run)

    task = (run / "labels" / "TASK.md").read_text(encoding="utf-8")

    assert " -m bench.label check " in task
    assert " -m bench.label status " not in task


def test_status_says_when_hits_csv_changed_after_the_sheet_was_made(run: Path) -> None:
    _sheet(run)
    hits = run / "hits.csv"
    hits.write_text(hits.read_text(encoding="utf-8").replace("check fired", "fired"), "utf-8")
    said: list[str] = []

    assert label.status(run, said.append) == 0

    assert "hits.csv has changed since the sheet was made: run `sheet` again." in said


def test_a_command_before_the_sheet_says_to_make_it(
    run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert label.main(["review", str(run)]) == 2

    assert "run `sheet` first" in capsys.readouterr().err


# ── the review ───────────────────────────────────────────────────────────────


def test_enter_takes_the_suggestion_and_a_label_typed_over_it_wins(run: Path) -> None:
    first, second, third = _sheet(run).flags
    for flag in (first, second, third):
        _suggest(run, {"unit": flag.unit, "label": "justified", "why": "Covered elsewhere."})

    _review(run, "", "r the code it covered is still there", "q")

    decisions = label.read_labels(run / "labels")
    assert decisions[first.unit].verdict == label.Verdict("justified")
    assert decisions[first.unit].suggested == label.Verdict("justified")
    assert (decisions[second.unit].verdict, decisions[second.unit].note) == (
        label.Verdict("real"),
        "the code it covered is still there",
    )
    assert third.unit not in decisions


def test_a_blind_flag_hides_the_suggestion_and_needs_a_label_of_its_own(run: Path) -> None:
    first, *rest = _sheet(run, blind=1).flags
    for flag in (first, *rest):
        _suggest(run, {"unit": flag.unit, "label": "wrong", "why": "THE-SUGGESTION'S-REASON"})

    said = _review(run, "", "j", "q")

    decision = label.read_labels(run / "labels")[first.unit]
    assert (decision.verdict, decision.blind, decision.suggested) == (
        label.Verdict("justified"),
        True,
        label.Verdict("wrong"),
    )
    # Hidden on the blind flag; shown on the next, which is not.
    assert "\n".join(said).count("THE-SUGGESTION'S-REASON") == 1


def test_a_flag_comes_up_once_it_has_a_suggestion(run: Path) -> None:
    last = _sheet(run).flags[-1]
    _suggest(run, {"unit": last.unit, "label": "wrong", "why": "Not a skip."})

    said = _review(run, "w")

    assert list(label.read_labels(run / "labels")) == [last.unit]
    assert said[-1] == "1 of 3 flags labelled; 2 still wait for a suggestion."


def test_a_flags_files_can_be_labelled_apart(run: Path) -> None:
    refund = _flag(_sheet(run), ROUNDING, "deleted_test")
    _suggest(run, {"unit": refund.unit, "label": "real", "why": "Both tests go."})

    _review(run, "f", "", "j", "tax went with its feature", unit=refund.unit)

    decision = label.read_labels(run / "labels")[refund.unit]
    assert decision.verdict.per_file(refund) == {
        "tests/test_refund.py": "real",
        "tests/test_tax.py": "justified",
    }
    assert decision.verdict.overall(refund) == "real"
    assert decision.note == "tax went with its feature"


# ── the report ───────────────────────────────────────────────────────────────


def _labelled(run: Path) -> label.Sheet:
    sheet = _sheet(run)
    label.write_labels(
        run / "labels",
        {
            _flag(sheet, ROUNDING, "deleted_test").unit: _decided(
                "real", files={"tests/test_tax.py": "justified"}
            ),
            _flag(sheet, ROUNDING, "changed_expectation").unit: _decided("justified"),
            _flag(sheet, CART, "skip_or_focus").unit: _decided("?"),
        },
    )
    return sheet


def test_the_report_puts_each_label_on_every_hit_of_its_flag(run: Path) -> None:
    _labelled(run)

    label.report(run)

    with (run / "labels" / "hits-labelled.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["file"], row["label"]) for row in rows] == [
        ("tests/test_refund.py", "real"),
        ("tests/test_tax.py", "justified"),
        ("tests/test_price.py", "justified"),
        ("tests/test_cart.py", "?"),
    ]
    # Every other cell as hits.csv has it, spreadsheet escape included.
    assert rows[3]["text"] == "'@pytest.mark.skip"


def test_the_report_counts_pull_requests_with_a_real_hit_against_all_scanned(run: Path) -> None:
    _labelled(run)

    text = label.report(run)

    assert "| Claude Code | 1 | 1 · 100.0% (20.7–100.0) | 1/1 · 100.0% (20.7–100.0) |" in text
    assert "| Human (matched controls) | 2 | 0 · 0.0% (0.0–65.8) | – |" in text
    # A ``?`` is labelled, but left out of every share.
    assert "| Human (matched controls) | 1 | 1 | 0 | 0 | 0 | 1 | – | – |" in text
    assert f"- Claude Code · deleted_test · {ROUNDING}" in text
    assert (run / "labels" / "labels.md").read_text(encoding="utf-8") == text


def test_the_report_measures_the_suggestions_against_the_blind_labels(run: Path) -> None:
    first, second, third = _sheet(run, blind=2).flags
    label.write_labels(
        run / "labels",
        {
            first.unit: _decided("real", suggested="real", blind=True),
            second.unit: _decided("wrong", suggested="justified", blind=True),
            third.unit: _decided("wrong", suggested="justified"),
        },
    )

    text = label.report(run)

    assert (
        "Labelled blind, the suggestion agreed on 1 of 2 flags."
        " Labelled with it shown, it was overridden on 1 of 1." in text
    )


# ── the published labels ─────────────────────────────────────────────────────


def _suggest_as_labelled(run: Path, sheet: label.Sheet, **otherwise: str) -> None:
    """A suggestion for each of ``_labelled``'s flags, with its labels unless ``otherwise``."""

    refund = _flag(sheet, ROUNDING, "deleted_test")
    price = _flag(sheet, ROUNDING, "changed_expectation")
    cart = _flag(sheet, CART, "skip_or_focus")
    _suggest(
        run,
        {
            "unit": refund.unit,
            "label": "real",
            "why": "test_rounding goes and refund() stays; the tax code goes too.",
            "files": {"tests/test_tax.py": "justified"},
        },
        {"unit": price.unit, "label": otherwise.get("price", "justified"), "why": "Prices rose."},
        {"unit": cart.unit, "label": "?", "why": "Issue #12 is not in the pull request."},
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_export_leaves_out_strangers_code_and_titles(run: Path, tmp_path: Path) -> None:
    _suggest_as_labelled(run, _labelled(run))
    out = tmp_path / "published"

    label.export(run, out)

    published = "".join(path.read_text(encoding="utf-8") for path in out.iterdir())
    for quoted in ("def test_rounding", "price(3)", "pytest.mark.skip", "Round refunds", "fired"):
        assert quoted not in published
    assert [(row["group"], row["pr"], row["merged"]) for row in _rows(out / "prs.csv")] == [
        ("claude", ROUNDING, "True"),
        ("human", CART, "False"),
        ("human", QUIET, "True"),
    ]
    assert [
        (row["check"], row["label"], row["reason"], row["reason_by"])
        for row in _rows(out / "flags.csv")
    ] == [
        (
            "deleted_test",
            "real",
            "test_rounding goes and refund() stays; the tax code goes too.",
            "suggestion",
        ),
        ("changed_expectation", "justified", "Prices rose.", "suggestion"),
        ("skip_or_focus", "?", "Issue #12 is not in the pull request.", "suggestion"),
    ]
    assert [(row["file"], row["line"], row["label"]) for row in _rows(out / "hits.csv")] == [
        ("tests/test_refund.py", "5", "real"),
        ("tests/test_tax.py", "1", "justified"),
        ("tests/test_price.py", "2", "justified"),
        ("tests/test_cart.py", "1", "?"),
    ]


def test_a_flag_labelled_otherwise_than_suggested_needs_a_reason_of_the_labellers_own(
    run: Path, tmp_path: Path
) -> None:
    sheet = _labelled(run)
    price = _flag(sheet, ROUNDING, "changed_expectation")
    _suggest_as_labelled(run, sheet, price="real")
    out = tmp_path / "published"

    with pytest.raises(label.SheetError, match="1 flag needs a reason of your own"):
        label.export(run, out)

    assert not out.exists()
    reasons = run / "labels" / "reasons.csv"
    rows = _rows(reasons)
    assert [(r["unit"], r["label"], r["suggested"], r["why"], r["reason"]) for r in rows] == [
        (price.unit, "justified", "real", "Prices rose.", "")
    ]
    rows[0]["reason"] = "The description says the price list changed."
    _write_csv(
        reasons, label.REASON_COLUMNS, [[row[c] for c in label.REASON_COLUMNS] for row in rows]
    )

    label.export(run, out)

    flag = next(row for row in _rows(out / "flags.csv") if row["flag"] == price.unit)
    assert (flag["reason"], flag["reason_by"]) == (
        "The description says the price list changed.",
        "labeller",
    )

    # Labelled again, the flag needs its reason written again.
    decisions = label.read_labels(run / "labels")
    label.write_labels(run / "labels", {**decisions, price.unit: _decided("wrong")})
    with pytest.raises(label.SheetError, match="1 flag needs"):
        label.export(run, out)
    assert [(r["label"], r["reason"]) for r in _rows(reasons)] == [("wrong", "")]
