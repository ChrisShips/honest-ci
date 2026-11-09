from __future__ import annotations

import os
import subprocess
from pathlib import Path

from honest_ci.impact import (
    analyse,
    changed_line_coverage,
    changed_lines_for,
)

from .factories import make_change

# One added line inside ``split_refund`` and one inside a new method of the
# class below it, so a test can tell the innermost symbol from the file.
PYTHON_DIFF = """--- a/pkg/refund.py
+++ b/pkg/refund.py
@@ -1,7 +1,8 @@
 def split_refund(order, amount):
-    return 0
+    share = amount / len(order.items)
+    return share
 
 
 class Ledger:
     def post(self, entry):
         return entry
"""

PYTHON_SOURCE = """def split_refund(order, amount):
    share = amount / len(order.items)
    return share


class Ledger:
    def post(self, entry):
        return entry
"""

JAVA_DIFF = """--- a/src/main/java/com/acme/Refund.java
+++ b/src/main/java/com/acme/Refund.java
@@ -1,6 +1,7 @@
 package com.acme;
 
 public class Refund {
     public int split(int amount) {
+        return amount / 2;
     }
 }
"""

JAVA_SOURCE = """package com.acme;

public class Refund {
    public int split(int amount) {
        return amount / 2;
    }
}
"""


def write(workspace: Path, relative: str, content: str) -> None:
    target = workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def run(workspace: Path, *args: str) -> None:
    """A real repository, because "which files does git track" is the question."""

    subprocess.run(
        ["git", "-C", str(workspace), *args],
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"},
    )


# ── what changed ─────────────────────────────────────────────────────────────


def test_python_change_reports_the_innermost_symbol(tmp_path: Path) -> None:
    write(tmp_path, "pkg/refund.py", PYTHON_SOURCE)

    analysis = analyse(tmp_path, make_change(diff=PYTHON_DIFF), "python")

    assert analysis.analysed is True
    assert {(item.name, item.kind) for item in analysis.changed_symbols} == {
        ("split_refund", "function")
    }
    assert analysis.unanalysed_files == ()


def test_python_method_is_named_with_its_class(tmp_path: Path) -> None:
    diff = """--- a/pkg/refund.py
+++ b/pkg/refund.py
@@ -6,2 +6,3 @@ class Ledger:
     def post(self, entry):
+        entry.checked = True
         return entry
"""
    write(tmp_path, "pkg/refund.py", PYTHON_SOURCE)

    analysis = analyse(tmp_path, make_change(diff=diff), "python")

    assert [item.name for item in analysis.changed_symbols] == ["Ledger.post"]
    assert analysis.changed_symbols[0].kind == "method"


def test_line_outside_every_definition_is_attributed_to_the_module(tmp_path: Path) -> None:
    diff = """--- a/pkg/settings.py
+++ b/pkg/settings.py
@@ -1,1 +1,2 @@
 import os
+TIMEOUT = 30
"""
    write(tmp_path, "pkg/settings.py", "import os\nTIMEOUT = 30\n")

    analysis = analyse(tmp_path, make_change(diff=diff), "python")

    assert [(item.name, item.kind) for item in analysis.changed_symbols] == [
        ("pkg.settings", "module")
    ]


def test_unparseable_file_is_named_rather_than_dropped(tmp_path: Path) -> None:
    write(tmp_path, "pkg/refund.py", "def split_refund(:\n")

    analysis = analyse(tmp_path, make_change(diff=PYTHON_DIFF), "python")

    assert analysis.changed_symbols == ()
    assert analysis.unanalysed_files == ("pkg/refund.py",)


def test_file_missing_from_the_workspace_is_named_rather_than_dropped(tmp_path: Path) -> None:
    analysis = analyse(tmp_path, make_change(diff=PYTHON_DIFF), "python")

    assert analysis.unanalysed_files == ("pkg/refund.py",)


def test_language_without_a_reader_says_so_instead_of_reporting_nothing(tmp_path: Path) -> None:
    write(tmp_path, "pkg/refund.py", PYTHON_SOURCE)

    analysis = analyse(tmp_path, make_change(diff=PYTHON_DIFF), "rust")

    assert analysis.analysed is False
    assert analysis.changed_symbols == ()
    assert analysis.impacted_tests == ()


# ── which tests reach it ─────────────────────────────────────────────────────


def test_python_tests_are_selected_by_the_paths_pytest_takes(tmp_path: Path) -> None:
    write(tmp_path, "pkg/refund.py", PYTHON_SOURCE)
    write(tmp_path, "tests/test_refund.py", "from pkg.refund import split_refund\n")
    write(tmp_path, "tests/test_users.py", "def test_users():\n    pass\n")

    analysis = analyse(tmp_path, make_change(diff=PYTHON_DIFF), "python")

    assert analysis.impacted_tests == ("tests/test_refund.py",)


def test_a_test_file_the_change_edits_is_impacted_without_being_read(tmp_path: Path) -> None:
    diff = PYTHON_DIFF + """--- a/tests/test_users.py
+++ b/tests/test_users.py
@@ -1,1 +1,2 @@
 def test_users():
+    assert True
"""
    write(tmp_path, "pkg/refund.py", PYTHON_SOURCE)
    write(tmp_path, "tests/test_users.py", "def test_users():\n    assert True\n")

    analysis = analyse(tmp_path, make_change(diff=diff), "python")

    assert "tests/test_users.py" in analysis.impacted_tests


def test_a_dependency_venv_never_contributes_a_test_to_the_selection(tmp_path: Path) -> None:
    # The first bug dogfooding found: the workspace is not the repository. A
    # ``.venv`` ships its dependencies' own suites, and running mypy's tests
    # says nothing about this change.
    write(tmp_path, "pkg/refund.py", PYTHON_SOURCE)
    write(tmp_path, "tests/test_refund.py", "from pkg.refund import split_refund\n")
    write(tmp_path, ".venv/lib/mypy/test/test_refund.py", "def test_split_refund(): pass\n")
    write(tmp_path, "node_modules/pkg/test_refund.py", "def test_split_refund(): pass\n")

    analysis = analyse(tmp_path, make_change(diff=PYTHON_DIFF), "python")

    assert analysis.impacted_tests == ("tests/test_refund.py",)


def test_a_second_copy_of_the_repository_never_contributes_a_test(tmp_path: Path) -> None:
    # A git worktree under the repository holds these very test files again.
    # Selecting both copies is what made pytest refuse to collect either one
    # ("import file mismatch"), so a narrowed suite ran nothing at all.
    write(tmp_path, "pkg/refund.py", PYTHON_SOURCE)
    write(tmp_path, "tests/test_refund.py", "from pkg.refund import split_refund\n")
    run(tmp_path, "init")
    run(tmp_path, "add", "pkg", "tests")
    write(tmp_path, ".worktrees/other/tests/test_refund.py", "from pkg.refund import x\n")

    analysis = analyse(tmp_path, make_change(diff=PYTHON_DIFF), "python")

    assert analysis.impacted_tests == ("tests/test_refund.py",)


def test_a_workspace_git_cannot_read_still_selects_the_tests_it_can_see(tmp_path: Path) -> None:
    # A merge request materialised without its ``.git``: weaker, because an
    # untracked test is a real test here, but never empty.
    write(tmp_path, "pkg/refund.py", PYTHON_SOURCE)
    write(tmp_path, "tests/test_refund.py", "from pkg.refund import split_refund\n")

    analysis = analyse(tmp_path, make_change(diff=PYTHON_DIFF), "python")

    assert analysis.impacted_tests == ("tests/test_refund.py",)


def test_an_untracked_new_test_is_selected_once_git_tracks_it(tmp_path: Path) -> None:
    # The cost of asking git: a brand-new test file that nobody has staged is
    # not in the selection. It is recorded here because it is the trade, not an
    # accident -- and the gate's own rule puts the whole suite back when the
    # selection comes out empty.
    write(tmp_path, "pkg/refund.py", PYTHON_SOURCE)
    write(tmp_path, "tests/test_refund.py", "from pkg.refund import split_refund\n")
    run(tmp_path, "init")

    assert analyse(tmp_path, make_change(diff=PYTHON_DIFF), "python").impacted_tests == ()

    run(tmp_path, "add", "tests/test_refund.py")

    analysis = analyse(tmp_path, make_change(diff=PYTHON_DIFF), "python")
    assert analysis.impacted_tests == ("tests/test_refund.py",)


def test_java_tests_are_selected_by_the_class_names_maven_takes(tmp_path: Path) -> None:
    write(tmp_path, "src/main/java/com/acme/Refund.java", JAVA_SOURCE)
    write(
        tmp_path,
        "src/test/java/com/acme/RefundTest.java",
        "package com.acme;\nclass RefundTest { void t() { new Refund(); } }\n",
    )
    write(
        tmp_path,
        "src/test/java/com/acme/UserTest.java",
        "package com.acme;\nclass UserTest { void t() {} }\n",
    )

    analysis = analyse(tmp_path, make_change(diff=JAVA_DIFF), "java")

    assert analysis.impacted_tests == ("RefundTest",)
    assert [item.name for item in analysis.changed_symbols] == ["Refund.split"]


def test_evidence_shape_carries_symbols_tests_and_failures(tmp_path: Path) -> None:
    write(tmp_path, "pkg/refund.py", PYTHON_SOURCE)
    write(tmp_path, "tests/test_refund.py", "from pkg.refund import split_refund\n")

    payload = analyse(tmp_path, make_change(diff=PYTHON_DIFF), "python").as_dict()

    assert payload["language"] == "python"
    assert payload["analysed"] is True
    assert payload["changed_symbols"][0]["symbol"] == "split_refund"
    assert payload["impacted_tests"] == ["tests/test_refund.py"]
    assert payload["unanalysed_files"] == []


# ── the lines a coverage reader is asked about ───────────────────────────────


def test_changed_lines_are_the_added_ones_at_their_new_numbers() -> None:
    assert changed_lines_for(make_change(diff=PYTHON_DIFF)) == {"pkg/refund.py": {2, 3}}


def test_unsafe_paths_never_reach_the_coverage_reader() -> None:
    diff = """--- a/../../etc/passwd
+++ b/../../etc/passwd
@@ -0,0 +1,1 @@
+root
"""
    assert changed_lines_for(make_change(diff=diff)) == {}


# ── judging those lines ──────────────────────────────────────────────────────


def test_only_executable_lines_count_toward_the_bar() -> None:
    # Line 3 is in neither set: a comment cannot be covered, and demanding it
    # be covered would make the ratio say something it does not mean.
    result = changed_line_coverage(
        {"pkg/refund.py": ({1}, {2})},
        {"pkg/refund.py": {1, 2, 3}},
        (".py",),
    )

    assert (result.total, result.covered) == (2, 1)
    assert result.uncovered == (("pkg/refund.py", 2),)
    assert result.unmapped == ()


def test_a_changed_source_with_no_coverage_data_is_reported_not_assumed() -> None:
    result = changed_line_coverage({}, {"pkg/refund.py": {1, 2}}, (".py",))

    assert (result.total, result.covered) == (0, 0)
    assert result.unmapped == ("pkg/refund.py",)


def test_files_of_another_kind_are_not_judged_at_all() -> None:
    # A changed README has no line anybody could cover; counting it as
    # unmapped would block a documentation change on a coverage bar.
    result = changed_line_coverage({}, {"README.md": {1}}, (".py",))

    assert (result.total, result.covered, result.unmapped) == (0, 0, ())


def test_a_file_coverage_was_never_asked_to_measure_is_not_missing_data() -> None:
    # The third bug dogfooding found. With ``source = ["pkg"]`` a test file is
    # never in the report, so reading its absence as "nobody measured this"
    # blocked every change that touched a test.
    result = changed_line_coverage(
        {"pkg/refund.py": ({1}, {2})},
        {"pkg/refund.py": {1, 2}, "tests/test_refund.py": {1, 2, 3}},
        (".py",),
        ("pkg",),
    )

    assert (result.total, result.covered) == (2, 1)
    assert result.out_of_scope == ("tests/test_refund.py",)
    assert result.unmapped == ()  # nothing is missing; it was never in scope


def test_a_measured_source_with_no_data_still_blocks_even_with_a_scope() -> None:
    # The property the bar exists for survives the fix: inside the measured
    # scope, "the report never named it" is still missing evidence.
    result = changed_line_coverage({}, {"pkg/refund.py": {1, 2}}, (".py",), ("pkg",))

    assert result.unmapped == ("pkg/refund.py",)
    assert result.out_of_scope == ()


def test_an_unknown_scope_judges_every_source_file() -> None:
    # No coverage configuration to read: fail closed, exactly as before.
    result = changed_line_coverage({}, {"tests/test_refund.py": {1}}, (".py",), None)

    assert result.unmapped == ("tests/test_refund.py",)


def test_a_scope_matches_directories_not_name_prefixes() -> None:
    result = changed_line_coverage(
        {}, {"pkg_extra/refund.py": {1}, "pkg/refund.py": {1}}, (".py",), ("pkg",)
    )

    assert result.unmapped == ("pkg/refund.py",)
    assert result.out_of_scope == ("pkg_extra/refund.py",)
