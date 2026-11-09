from __future__ import annotations

from pathlib import Path

import pytest

from honest_ci.contracts import ChangedFile
from honest_ci.rules import MAX_FINDINGS_PER_RULE, RuleError, RuleSet, evaluate_rules

from .factories import WEAKENED_TESTS_DIFF, make_change, make_weakened_change

FLOAT_RULE = """
version = "v3"

[[rule]]
id = "R-01"
title = "金额计算不得使用 float"
detail = "浮点误差会在对账时累积。"
severity = "high"
kind = "regex"
pattern = 'float\\('
paths = ["**/*.py"]
fix = "改用 decimal.Decimal"
"""


def load(tmp_path: Path, body: str) -> RuleSet:
    (tmp_path / "honest-ci-rules.toml").write_text(body, encoding="utf-8")
    return RuleSet.load(tmp_path, Path("honest-ci-rules.toml"), "v1")


def test_a_missing_rule_file_is_an_empty_library_not_an_error(tmp_path: Path) -> None:
    rule_set = RuleSet.load(tmp_path, Path("honest-ci-rules.toml"), "v1")

    assert rule_set.is_empty
    assert rule_set.source is None


def test_the_file_owns_its_own_version(tmp_path: Path) -> None:
    # The version travels into the fingerprint, so editing a rule has to be
    # able to invalidate approvals that were made under the old one.
    assert load(tmp_path, FLOAT_RULE).version == "v3"


def test_a_regex_rule_fires_only_on_lines_the_change_added(tmp_path: Path) -> None:
    rule_set = load(tmp_path, FLOAT_RULE)

    findings = evaluate_rules(rule_set, make_change())

    assert [finding.rule_id for finding in findings] == ["R-01"]
    finding = findings[0]
    assert finding.file == "src/app/refund.py"
    assert finding.line_start == 11  # the added `share = float(amount) / ...`
    assert finding.severity == "high"
    assert finding.decided_by == "deterministic"
    assert finding.suggested_fix == "改用 decimal.Decimal"
    assert "float(amount)" in finding.detail


def test_untouched_code_is_never_a_review_finding(tmp_path: Path) -> None:
    rule_set = load(
        tmp_path,
        """
        [[rule]]
        id = "R-02"
        title = "不得返回裸 total"
        severity = "high"
        kind = "regex"
        pattern = 'return total'
        """,
    )

    # `return total` is a context line in the diff: the change did not add it.
    assert evaluate_rules(rule_set, make_change()) == []


def test_a_path_rule_flags_the_files_it_scopes(tmp_path: Path) -> None:
    rule_set = load(
        tmp_path,
        """
        [[rule]]
        id = "R-09"
        title = "生成的 SDK 不得手改"
        severity = "critical"
        kind = "path"
        paths = ["**/generated/**"]
        exclude_paths = ["**/generated/README.md"]
        """,
    )
    change = make_change(
        files_changed=[
            ChangedFile(path="src/generated/client.py"),
            ChangedFile(path="src/generated/README.md"),
            ChangedFile(path="src/app/refund.py"),
        ]
    )

    findings = evaluate_rules(rule_set, change)

    assert [finding.file for finding in findings] == ["src/generated/client.py"]


def test_a_path_rule_flags_a_file_moved_out_of_the_paths_it_scopes(tmp_path: Path) -> None:
    rule_set = load(
        tmp_path,
        """
        [[rule]]
        id = "R-09"
        title = "生成的 SDK 不得手改"
        severity = "critical"
        kind = "path"
        paths = ["**/generated/**"]
        """,
    )
    moved = make_change(
        files_changed=[
            ChangedFile(
                path="src/app/client.py",
                change_type="renamed",
                previous_path="src/generated/client.py",
            )
        ]
    )
    deleted = make_change(
        files_changed=[],
        diff="--- a/src/generated/client.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n",
    )

    # Taking the file out of the generated tree by hand is a hand edit of the
    # generated tree, whether the platform calls it a rename or the patch
    # deletes it.
    assert [finding.file for finding in evaluate_rules(rule_set, moved)] == [
        "src/generated/client.py"
    ]
    assert [finding.file for finding in evaluate_rules(rule_set, deleted)] == [
        "src/generated/client.py"
    ]


def test_a_noisy_rule_is_capped_and_says_how_many_it_found(tmp_path: Path) -> None:
    hits = MAX_FINDINGS_PER_RULE + 5
    diff = f"--- a/src/app/wide.py\n+++ b/src/app/wide.py\n@@ -1,0 +1,{hits} @@\n"
    diff += "".join(f"+value_{index} = float(1)\n" for index in range(hits))
    rule_set = load(tmp_path, FLOAT_RULE)

    findings = evaluate_rules(rule_set, make_change(diff=diff, files_changed=[]))

    assert len(findings) == MAX_FINDINGS_PER_RULE
    assert f"This rule matched {hits} times" in findings[-1].detail


def test_a_disabled_rule_stays_in_the_file_but_does_not_fire(tmp_path: Path) -> None:
    rule_set = load(tmp_path, FLOAT_RULE + '\nenabled = false\n')

    assert rule_set.is_empty
    assert evaluate_rules(rule_set, make_change()) == []


def test_only_tools_the_rules_ask_for_are_required(tmp_path: Path) -> None:
    rule_set = load(
        tmp_path,
        """
        [[rule]]
        id = "T-01"
        title = "静态检查必须通过"
        severity = "high"
        kind = "tool"
        tool = "ruff"

        [[rule]]
        id = "T-02"
        title = "类型必须自洽"
        severity = "medium"
        kind = "tool"
        tool = "ruff"
        """,
    )

    assert rule_set.required_tools == ("ruff",)


def test_a_tool_is_only_asked_about_the_paths_its_rules_cover(tmp_path: Path) -> None:
    rule_set = load(
        tmp_path,
        """
        [[rule]]
        id = "T-01"
        title = "类型标注是强制的"
        kind = "tool"
        tool = "mypy"
        paths = ["src/**"]

        [[rule]]
        id = "T-02"
        title = "静态检查必须通过"
        kind = "tool"
        tool = "ruff"
        """,
    )

    scopes = rule_set.tool_scopes(["src/app/refund.py", "tests/test_refund.py", "README.md"])

    # ``paths`` says which files a scanner is asked about. Without that, mypy
    # got the change's test files in a repository whose own mypy configuration
    # only ever looks at ``src``.
    assert scopes["mypy"] == ("src/app/refund.py",)
    # A rule that narrows nothing is still asked about everything.
    assert scopes["ruff"] == ("src/app/refund.py", "tests/test_refund.py", "README.md")


def test_two_rules_on_one_tool_ask_about_the_union_and_never_repeat_a_path(
    tmp_path: Path,
) -> None:
    rule_set = load(
        tmp_path,
        """
        [[rule]]
        id = "T-01"
        title = "a"
        kind = "tool"
        tool = "mypy"
        paths = ["src/**"]

        [[rule]]
        id = "T-02"
        title = "b"
        kind = "tool"
        tool = "mypy"
        paths = ["src/**", "tests/**"]
        """,
    )

    scopes = rule_set.tool_scopes(["src/app/refund.py", "tests/test_refund.py"])

    assert scopes["mypy"] == ("src/app/refund.py", "tests/test_refund.py")


def test_an_excluded_path_is_not_in_a_tools_scope(tmp_path: Path) -> None:
    rule_set = load(
        tmp_path,
        """
        [[rule]]
        id = "T-01"
        title = "a"
        kind = "tool"
        tool = "ruff"
        paths = ["src/**"]
        exclude_paths = ["src/generated/**"]
        """,
    )

    scopes = rule_set.tool_scopes(["src/app/refund.py", "src/generated/pb2.py"])

    assert scopes["ruff"] == ("src/app/refund.py",)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('[[rule]]\nid = ""\ntitle = "x"\nkind = "path"\npaths = ["a"]', "needs an id"),
        ('[[rule]]\nid = "R-1"\nkind = "path"\npaths = ["a"]', "needs a title"),
        ('[[rule]]\nid = "R-1"\ntitle = "x"\nkind = "ast"', "unknown kind"),
        ('[[rule]]\nid = "R-1"\ntitle = "x"\nkind = "path"\nseverity = "fatal"', "unknown severity"),
        ('[[rule]]\nid = "R-1"\ntitle = "x"\nkind = "path"', "needs 'paths'"),
        ('[[rule]]\nid = "R-1"\ntitle = "x"\nkind = "regex"', "needs a non-empty 'pattern'"),
        ('[[rule]]\nid = "R-1"\ntitle = "x"\nkind = "regex"\npattern = "(unclosed"', "invalid pattern"),
        ('[[rule]]\nid = "R-1"\ntitle = "x"\nkind = "tool"\ntool = "pip-audit"', "network denied"),
        ('[[rule]]\nid = "J-1"\ntitle = "x"\nkind = "judgment"', "needs a 'detail'"),
        (
            '[[rule]]\nid = "J-1"\ntitle = "x"\nkind = "judgment"\ndetail = "d"\n'
            "min_confidence = 1.5",
            "between 0 and 1",
        ),
        (
            '[[rule]]\nid = "J-1"\ntitle = "x"\nkind = "judgment"\ndetail = "d"\n'
            "min_confidence = true",
            "must be a number",
        ),
        (
            '[[rule]]\nid = "R-1"\ntitle = "x"\nkind = "path"\npaths = ["a"]\n'
            "min_confidence = 0.5",
            "only means something for a judgment rule",
        ),
        ('[[rule]]\nid = "I-1"\ntitle = "x"\nkind = "integrity"', "needs a 'check'"),
        (
            '[[rule]]\nid = "I-1"\ntitle = "x"\nkind = "integrity"\ncheck = "slow_test"',
            "needs a 'check'",
        ),
        (
            '[[rule]]\nid = "I-1"\ntitle = "x"\nkind = "integrity"\ncheck = "deleted_test"\n'
            'action = "warn"',
            "unknown action",
        ),
        (
            '[[rule]]\nid = "I-1"\ntitle = "x"\nkind = "integrity"\ncheck = "deleted_test"\n'
            'action = "block"\nseverity = "medium"',
            "must be high or critical",
        ),
        (
            '[[rule]]\nid = "I-1"\ntitle = "x"\nkind = "integrity"\ncheck = "deleted_test"\n'
            'severity = "high"',
            "must be medium or lower",
        ),
        (
            '[[rule]]\nid = "I-1"\ntitle = "x"\nkind = "integrity"\ncheck = "deleted_test"\n'
            'action = "record"\nseverity = "critical"',
            "must be medium or lower",
        ),
        (
            '[[rule]]\nid = "R-1"\ntitle = "x"\nkind = "regex"\npattern = "x"\naction = "block"',
            "only mean something for an integrity rule",
        ),
        ('rule = 3', "array of tables"),
        ('[[rule]]\nid = "R-1"\ntitle = "x"\nkind = "path"\npaths = "a"', "list of strings"),
    ],
)
def test_a_rule_file_nobody_can_parse_stops_the_run(
    tmp_path: Path, body: str, message: str
) -> None:
    # Reviewing nothing is not an acceptable fallback for a broken rule library.
    with pytest.raises(RuleError, match=message):
        load(tmp_path, body)


def test_duplicate_rule_ids_are_rejected(tmp_path: Path) -> None:
    body = '[[rule]]\nid = "R-1"\ntitle = "x"\nkind = "path"\npaths = ["a"]\n' * 2
    with pytest.raises(RuleError, match="duplicate rule id"):
        load(tmp_path, body)


JUDGMENT_RULE = """
[[rule]]
id = "J-01"
title = "订单状态变更必须走状态机"
detail = "直接给状态字段赋值会绕开状态机。"
severity = "high"
kind = "judgment"
paths = ["src/**/*.py"]
min_confidence = 0.8
"""


def test_a_judgment_rule_is_loaded_but_never_decided_by_the_rule_engine(
    tmp_path: Path,
) -> None:
    rule_set = load(tmp_path, JUDGMENT_RULE + FLOAT_RULE)

    assert [rule.id for rule in rule_set.judgment_rules] == ["J-01"]
    assert rule_set.judgment_rules[0].min_confidence == 0.8
    # ``evaluate_rules`` stays pure: only the regex rule can be decided without
    # a credential, which is what lets the same call be replayed offline.
    findings = evaluate_rules(rule_set, make_change())
    assert [finding.rule_id for finding in findings] == ["R-01"]


def test_a_judgment_rule_may_leave_the_threshold_to_the_deployment(tmp_path: Path) -> None:
    rule_set = load(tmp_path, JUDGMENT_RULE.replace("min_confidence = 0.8", ""))

    assert rule_set.judgment_rules[0].min_confidence is None


def test_a_rule_can_record_which_samples_it_was_mined_from(tmp_path: Path) -> None:
    """Rule mining goes from samples to a draft to a person's final version.

    Without this, the last step discards the only thing that answers "why
    does this rule exist" when someone argues with it a year later.
    """

    (tmp_path / "honest-ci-rules.toml").write_text(
        FLOAT_RULE + '\nsource_samples = ["S-014", "INC-EXAMPLE-1", "example/shop!1"]\n',
        encoding="utf-8",
    )

    rules = RuleSet.load(tmp_path, Path("honest-ci-rules.toml"), "v1")

    assert rules.rules[0].source_samples == ("S-014", "INC-EXAMPLE-1", "example/shop!1")


def test_a_rule_without_source_samples_still_loads(tmp_path: Path) -> None:
    # Optional on purpose: a library mined before this field existed keeps
    # working, and an empty tuple is what ``doctor`` counts as "not recorded".
    (tmp_path / "honest-ci-rules.toml").write_text(FLOAT_RULE, encoding="utf-8")

    rules = RuleSet.load(tmp_path, Path("honest-ci-rules.toml"), "v1")

    assert rules.rules[0].source_samples == ()


def test_source_samples_must_be_a_list_of_strings(tmp_path: Path) -> None:
    (tmp_path / "honest-ci-rules.toml").write_text(
        FLOAT_RULE + "\nsource_samples = 42\n", encoding="utf-8"
    )

    with pytest.raises(RuleError) as exc:
        RuleSet.load(tmp_path, Path("honest-ci-rules.toml"), "v1")

    assert "source_samples" in str(exc.value)


# ── integrity rules ──────────────────────────────────────────────────────────

INTEGRITY_RULES = """
[[rule]]
id = "I-01"
title = "Tests do not disappear"
kind = "integrity"
check = "deleted_test"
fix = "Keep the test, or say in the change why it no longer applies."

[[rule]]
id = "I-02"
title = "No new skipped tests"
kind = "integrity"
check = "skip_or_focus"

[[rule]]
id = "I-04"
title = "Expected values change for a reason"
kind = "integrity"
check = "changed_expectation"
detail = "Rewriting the expected value to the new output hides the regression it caught."
"""


def test_a_deleted_test_is_found_on_the_line_it_had_before_the_change(tmp_path: Path) -> None:
    rule_set = load(tmp_path, INTEGRITY_RULES)

    findings = evaluate_rules(rule_set, make_weakened_change())

    deleted = next(finding for finding in findings if finding.rule_id == "I-01")
    # The head has no line left to point at; the base does.
    assert (deleted.file, deleted.line_start, deleted.side) == ("tests/test_pricing.py", 3, "base")
    assert deleted.decided_by == "deterministic"
    assert "`test_total_doubles` is removed" in deleted.detail
    assert "Removed: `def test_total_doubles():`" in deleted.detail
    assert deleted.suggested_fix == "Keep the test, or say in the change why it no longer applies."


def test_each_integrity_rule_reports_its_own_check_and_nothing_else(tmp_path: Path) -> None:
    rule_set = load(tmp_path, INTEGRITY_RULES)

    findings = evaluate_rules(rule_set, make_weakened_change())

    assert [(finding.rule_id, finding.side) for finding in findings] == [
        ("I-01", "base"),
        ("I-02", "head"),
        ("I-04", "base"),
    ]


def test_a_new_skip_is_found_on_the_line_the_change_added(tmp_path: Path) -> None:
    rule_set = load(tmp_path, INTEGRITY_RULES)

    skip = next(
        finding
        for finding in evaluate_rules(rule_set, make_weakened_change())
        if finding.rule_id == "I-02"
    )

    assert (skip.line_start, skip.side) == (3, "head")
    assert 'Added: `@pytest.mark.skip(reason="flaky")`' in skip.detail


def test_a_long_line_an_integrity_hit_quotes_is_cut_short(tmp_path: Path) -> None:
    rule_set = load(tmp_path, INTEGRITY_RULES)
    long_reason = WEAKENED_TESTS_DIFF.replace('reason="flaky"', f'reason="{"x" * 400}"')

    skip = next(
        finding
        for finding in evaluate_rules(rule_set, make_weakened_change(diff=long_reason))
        if finding.rule_id == "I-02"
    )

    quoted = next(line for line in skip.detail.splitlines() if line.startswith("Added: "))
    assert quoted.endswith("…`")
    assert len(quoted) < 220


def test_a_rewritten_expectation_quotes_what_it_was_and_what_it_became(tmp_path: Path) -> None:
    rule_set = load(tmp_path, INTEGRITY_RULES)

    rewritten = next(
        finding
        for finding in evaluate_rules(rule_set, make_weakened_change())
        if finding.rule_id == "I-04"
    )

    assert rewritten.line_start == 8
    assert rewritten.detail.splitlines() == [
        "Only literal values change in this assertion (`6` → `9`), "
        "in a change that also edits non-test code.",
        "Was: `assert total(3) == 6`",
        "Now: `assert total(3) == 9`",
        # The team's reason comes after the fact it explains.
        "Rewriting the expected value to the new output hides the regression it caught.",
    ]


def test_an_integrity_rule_starts_at_medium_and_asks_a_person(tmp_path: Path) -> None:
    rule = load(tmp_path, INTEGRITY_RULES).rules[0]

    assert (rule.check, rule.action, rule.severity) == ("deleted_test", "needs_human", "medium")
    assert not rule.blocking


def test_a_blocking_integrity_rule_starts_at_high_so_the_block_is_real(tmp_path: Path) -> None:
    blocking = 'check = "deleted_test"\naction = "block"'
    rule_set = load(tmp_path, INTEGRITY_RULES.replace('check = "deleted_test"', blocking))

    rule = rule_set.rules[0]
    assert (rule.action, rule.severity) == ("block", "high")
    finding = next(
        finding
        for finding in evaluate_rules(rule_set, make_weakened_change())
        if finding.rule_id == "I-01"
    )
    # Blocking is still severity and nothing else: the action only chose it.
    assert finding.blocking


def test_an_integrity_rule_s_paths_narrow_which_files_it_reports(tmp_path: Path) -> None:
    rule_set = load(
        tmp_path,
        INTEGRITY_RULES.replace(
            'check = "deleted_test"', 'check = "deleted_test"\nexclude_paths = ["tests/test_*.py"]'
        ),
    )

    findings = evaluate_rules(rule_set, make_weakened_change())

    assert "I-01" not in {finding.rule_id for finding in findings}
    assert {"I-02", "I-04"} <= {finding.rule_id for finding in findings}


def test_a_test_file_renamed_out_of_collection_is_one_finding_for_the_whole_file(
    tmp_path: Path,
) -> None:
    rule_set = load(tmp_path, INTEGRITY_RULES)
    # A forge sends no lines for a pure rename; its two names are all there is.
    parked = make_change(
        diff="",
        files_changed=[
            ChangedFile(
                path="tests/disabled/pricing.py",
                previous_path="tests/test_pricing.py",
                change_type="renamed",
            )
        ],
    )

    (finding,) = evaluate_rules(rule_set, parked)

    assert (finding.file, finding.line_start, finding.side) == (
        "tests/test_pricing.py",
        None,
        "base",
    )
    assert "no longer run" in finding.detail
    assert "Removed:" not in finding.detail


def test_a_test_file_deleted_without_its_lines_is_one_finding_for_the_whole_file(
    tmp_path: Path,
) -> None:
    rule_set = load(tmp_path, INTEGRITY_RULES)
    # Listed as deleted, with no line in the diff: too large for the forge to show.
    withheld = make_change(
        diff="",
        files_changed=[ChangedFile(path="tests/test_pricing.py", change_type="deleted")],
    )

    (finding,) = evaluate_rules(rule_set, withheld)

    assert (finding.rule_id, finding.file, finding.line_start, finding.side) == (
        "I-01",
        "tests/test_pricing.py",
        None,
        "base",
    )
    assert "the diff shows none of its lines" in finding.detail


def test_a_deleted_test_file_whose_lines_are_shown_is_read_test_by_test(tmp_path: Path) -> None:
    rule_set = load(tmp_path, INTEGRITY_RULES)
    removed = "-def test_total_doubles():\n-    assert total(3) == 6\n"
    shown = make_change(
        diff=f"--- a/tests/test_pricing.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n{removed}",
        files_changed=[
            ChangedFile(path="tests/test_pricing.py", change_type="deleted", removed_lines=2)
        ],
    )

    (finding,) = evaluate_rules(rule_set, shown)

    assert (finding.file, finding.line_start) == ("tests/test_pricing.py", 1)
    assert "`test_total_doubles`" in finding.detail


def test_a_deleted_file_refused_as_a_path_is_left_to_the_guardrails(tmp_path: Path) -> None:
    refused = make_change(
        diff="",
        files_changed=[ChangedFile(path="../tests/test_pricing.py", change_type="deleted")],
    )

    assert evaluate_rules(load(tmp_path, INTEGRITY_RULES), refused) == []


def test_a_change_that_leaves_its_tests_alone_trips_no_integrity_rule(tmp_path: Path) -> None:
    assert evaluate_rules(load(tmp_path, INTEGRITY_RULES), make_change()) == []


def test_a_noisy_integrity_rule_is_capped_and_says_how_many_it_found(tmp_path: Path) -> None:
    hits = MAX_FINDINGS_PER_RULE + 3
    removed = "".join(
        f"-def test_case_{index}():\n-    assert run({index})\n" for index in range(hits)
    )
    diff = (
        "--- a/tests/test_wide.py\n+++ /dev/null\n"
        f"@@ -1,{2 * hits} +0,0 @@\n{removed}"
    )
    rule_set = load(tmp_path, INTEGRITY_RULES)

    findings = evaluate_rules(rule_set, make_change(diff=diff, files_changed=[]))

    assert len(findings) == MAX_FINDINGS_PER_RULE
    assert f"This rule matched {hits} times" in findings[-1].detail


def test_the_example_rule_file_loads(tmp_path: Path) -> None:
    # It is what a team copies; one that did not load would be the first
    # thing they hit.
    example = Path(__file__).resolve().parents[1] / "honest-ci-rules.example.toml"

    rule_set = RuleSet.load(example.parent, Path(example.name), "v1")

    assert {rule.check for rule in rule_set.of_kind("integrity")} == {
        "deleted_test",
        "skip_or_focus",
        "fewer_assertions",
        "changed_expectation",
        "loosened_threshold",
    }
