from __future__ import annotations

import ast
import difflib
import textwrap
from collections.abc import Sequence
from pathlib import Path

import pytest

from honest_ci import integrity
from honest_ci.integrity import (
    INTEGRITY_CHECKS,
    MAX_SCANNED_LINE_CHARS,
    detect,
    unread_deletions,
)

# ── how the diffs are made ───────────────────────────────────────────────────


def _lines(text: str) -> list[str]:
    body = textwrap.dedent(text).lstrip("\n")
    if body and not body.endswith("\n"):
        body += "\n"
    return body.splitlines(keepends=True)


def _patch(old: str | None, new: str | None, before: str, after: str) -> str:
    body = difflib.unified_diff(
        _lines(before),
        _lines(after),
        fromfile=f"a/{old}" if old else "/dev/null",
        tofile=f"b/{new}" if new else "/dev/null",
    )
    return f"diff --git a/{old or new} b/{new or old}\n" + "".join(body)


def edit(path: str, before: str, after: str) -> str:
    return _patch(path, path, before, after)


def create(path: str, text: str) -> str:
    return _patch(None, path, "", text)


def delete(path: str, text: str) -> str:
    return _patch(path, None, text, "")


def move(old: str, new: str, text: str) -> str:
    """A move as a local diff shows it: ``--no-renames`` makes it two files."""

    return delete(old, text) + create(new, text)


def rename(old: str, new: str, before: str, after: str | None = None) -> str:
    """A move as a forge's diff shows it, with git's rename headers."""

    header = f"diff --git a/{old} b/{new}\n"
    if after is None:
        return header + f"similarity index 100%\nrename from {old}\nrename to {new}\n"
    body = difflib.unified_diff(
        _lines(before), _lines(after), fromfile=f"a/{old}", tofile=f"b/{new}"
    )
    return (
        header
        + f"similarity index 91%\nrename from {old}\nrename to {new}\n"
        + "index 1a2b3c4..5d6e7f8 100644\n"
        + "".join(body)
    )


def found(
    diff: str, renames: Sequence[tuple[str, str]] = ()
) -> list[tuple[str, str, int | None, str]]:
    return [(hit.check, hit.file, hit.line, hit.side) for hit in detect(diff, renames)]


def above(text: str, anchor: str, inserted: str) -> str:
    """``text`` with ``inserted`` put above the first line holding ``anchor``, at its indent."""

    rows = text.splitlines(keepends=True)
    at = next(index for index, row in enumerate(rows) if anchor in row)
    indent = rows[at][: len(rows[at]) - len(rows[at].lstrip())]
    lines = "".join(f"{indent}{line}\n" for line in inserted.splitlines())
    return "".join([*rows[:at], lines, *rows[at:]])


#: An edit to the code under test. A rewritten expectation is only worth a
#: hit next to one.
CODE = edit("src/shop/cart.py", "RATE = 1\n", "RATE = 2\n")

# ── one small suite per language ─────────────────────────────────────────────

PY_PATH = "tests/test_cart.py"
PY_HEAD = """\
from shop.cart import total


def test_empty_cart_costs_nothing():
    assert total([]) == 0
"""
PY_HALF_UP = """

def test_rounds_half_up():
    assert total([2.5]) == 3
    assert total([0.5]) == 1
"""
PY = PY_HEAD + PY_HALF_UP

JS_PATH = "src/cart.test.ts"
JS_HEAD = """\
import { total } from "./cart";

describe("total", () => {
  it("is zero for an empty cart", () => {
    expect(total([])).toBe(0);
  });
"""
JS_HALF_UP = """
  it("rounds half up", () => {
    expect(total([2.5])).toBe(3);
    expect(total([0.5])).toBe(1);
  });
"""
JS = JS_HEAD + JS_HALF_UP + "});\n"

JAVA_PATH = "src/test/java/shop/CartTest.java"
JAVA_HEAD = """\
package shop;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class CartTest {
    @Test
    void emptyCartCostsNothing() {
        assertEquals(0, Cart.total());
    }
"""
JAVA_HALF_UP = """
    @Test
    void roundsHalfUp() {
        assertEquals(3, Cart.total(2.5));
        assertEquals(1, Cart.total(0.5));
    }
"""
JAVA = JAVA_HEAD + JAVA_HALF_UP + "}\n"

GO_PATH = "cart/cart_test.go"
GO_HEAD = """\
package cart

import "testing"

func TestEmptyCartCostsNothing(t *testing.T) {
    if got := Total(nil); got != 0 {
        t.Errorf("Total(nil) = %v, want 0", got)
    }
}
"""
GO_HALF_UP = """
func TestRoundsHalfUp(t *testing.T) {
    want := 3.0
    if got := Total([]float64{2.5}); got != want {
        t.Errorf("Total(2.5) = %v, want %v", got, want)
    }
}
"""
GO = GO_HEAD + GO_HALF_UP


def test_the_checks_are_named_in_a_fixed_order() -> None:
    # The benchmark writes these names into its results and the rule engine
    # will key a rule off each one: renaming one is an interface change.
    assert INTEGRITY_CHECKS == (
        "deleted_test",
        "skip_or_focus",
        "fewer_assertions",
        "changed_expectation",
        "loosened_threshold",
    )


def test_the_detector_reads_nothing_but_the_diff_and_the_renames() -> None:
    # The benchmark imports detect() on its own, and the default GitHub setup
    # has no sandbox and no model: nothing beyond the diff parser and
    # redaction may creep in. The renames come in as plain name pairs rather
    # than as the contract's files, for the same reason.
    tree = ast.parse(Path(integrity.__file__).read_text(encoding="utf-8"))
    local = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }

    assert local == {"diffs", "redaction"}


def test_a_diff_that_touches_no_tests_has_no_hits() -> None:
    assert detect("") == []
    assert detect(CODE) == []
    assert detect(edit("README.md", "# Shop\n", "# Shop\n\nNo tests here.\n")) == []


# ── a. deleted_test ──────────────────────────────────────────────────────────

DELETIONS = [
    pytest.param(PY_PATH, PY, PY_HEAD, 8, "test_rounds_half_up", id="python"),
    pytest.param(JS_PATH, JS, JS_HEAD + "});\n", 8, "rounds half up", id="typescript"),
    pytest.param(JAVA_PATH, JAVA, JAVA_HEAD + "}\n", 14, "roundsHalfUp", id="java"),
    pytest.param(GO_PATH, GO, GO_HEAD, 11, "TestRoundsHalfUp", id="go"),
]


@pytest.mark.parametrize(("path", "before", "after", "line", "name"), DELETIONS)
def test_a_removed_test_is_reported_at_the_base_line_that_named_it(
    path: str, before: str, after: str, line: int, name: str
) -> None:
    hits = detect(edit(path, before, after) + CODE)

    # Its assertions went with it; they are this hit, not a second one.
    assert [(hit.check, hit.file, hit.line, hit.side) for hit in hits] == [
        ("deleted_test", path, line, "base")
    ]
    assert name in hits[0].text
    assert hits[0].detail == (
        f"Test `{name}` is removed, and no test by that name is added back in this change."
    )


RENAMES = [
    pytest.param(PY_PATH, PY, "test_rounds_half_up", "test_rounds_halves_up", id="python"),
    pytest.param(JS_PATH, JS, '"rounds half up"', '"rounds halves up"', id="typescript"),
    pytest.param(JAVA_PATH, JAVA, "roundsHalfUp", "roundsHalvesUp", id="java"),
    pytest.param(GO_PATH, GO, "TestRoundsHalfUp", "TestRoundsHalvesUp", id="go"),
]


@pytest.mark.parametrize(("path", "before", "old", "new"), RENAMES)
def test_a_test_renamed_where_it_stands_is_not_a_deletion(
    path: str, before: str, old: str, new: str
) -> None:
    assert detect(edit(path, before, before.replace(old, new)) + CODE) == []


#: The first test of each suite, and an unrelated one written where it stood.
REPLACEMENTS = [
    pytest.param(
        PY_PATH,
        PY,
        "def test_empty_cart_costs_nothing():\n    assert total([]) == 0\n",
        "def test_one_item_costs_its_price():\n    assert total([4]) == 4\n",
        4,
        "test_empty_cart_costs_nothing",
        id="python",
    ),
    pytest.param(
        JS_PATH,
        JS,
        '  it("is zero for an empty cart", () => {\n    expect(total([])).toBe(0);\n',
        '  it("costs its price for one item", () => {\n    expect(total([4])).toBe(4);\n',
        4,
        "is zero for an empty cart",
        id="typescript",
    ),
    pytest.param(
        JAVA_PATH,
        JAVA,
        "    void emptyCartCostsNothing() {\n        assertEquals(0, Cart.total());\n",
        "    void oneItemCostsItsPrice() {\n        assertEquals(4, Cart.total(4));\n",
        9,
        "emptyCartCostsNothing",
        id="java",
    ),
    pytest.param(
        GO_PATH,
        GO,
        "func TestEmptyCartCostsNothing(t *testing.T) {\n"
        "    if got := Total(nil); got != 0 {\n"
        '        t.Errorf("Total(nil) = %v, want 0", got)\n',
        "func TestOneItemCostsItsPrice(t *testing.T) {\n"
        "    if got := Total([]float64{4}); got != 4 {\n"
        '        t.Errorf("Total(4) = %v, want 4", got)\n',
        5,
        "TestEmptyCartCostsNothing",
        id="go",
    ),
]


@pytest.mark.parametrize(("path", "before", "old", "new", "line", "name"), REPLACEMENTS)
def test_a_different_test_written_where_a_removed_one_stood_is_a_deletion(
    path: str, before: str, old: str, new: str, line: int, name: str
) -> None:
    # One name out and one in, in the same block, is what a rename looks like
    # too. What tells them apart is whether the new test still says what the
    # old one did. The blank lines and closing brackets they share do not
    # count as saying anything.
    hits = detect(edit(path, before, before.replace(old, new)) + CODE)

    assert [(hit.check, hit.file, hit.line, hit.side) for hit in hits] == [
        ("deleted_test", path, line, "base")
    ]
    assert hits[0].detail == (
        f"Test `{name}` is removed, and no test by that name is added back in this change."
    )


def test_a_renamed_test_that_keeps_most_of_its_lines_is_still_a_rename() -> None:
    after = PY.replace("test_rounds_half_up", "test_rounds_halves_up") + (
        "    assert total([1.5]) == 2\n"
    )

    assert detect(edit(PY_PATH, PY, after) + CODE) == []


def test_a_renamed_test_whose_signature_now_wraps_is_still_a_rename() -> None:
    after = PY.replace(
        "def test_rounds_half_up():",
        "def test_rounds_halves_up(\n    capsys: pytest.CaptureFixture[str],\n) -> None:",
    )

    assert detect(edit(PY_PATH, PY, after) + CODE) == []


PY_WRAPPED = PY_HEAD + (
    "\n\ndef test_rounds_half_up(\n    items: list[float],\n) -> None:\n"
    "    assert total(items) == 3\n"
)


def test_a_wrapped_signature_does_not_count_as_what_a_test_says() -> None:
    # Only its one assertion changes, and that is all it said. The rows of
    # the signature it keeps are no more its own than a blank line is.
    after = PY_WRAPPED.replace("test_rounds_half_up", "test_rounds_halves_up").replace(
        "total(items) == 3", "total(items) > 2"
    )

    assert found(edit(PY_PATH, PY_WRAPPED, after) + CODE) == [("deleted_test", PY_PATH, 8, "base")]


def test_a_deleted_test_with_a_wrapped_signature_takes_its_assertions_with_it() -> None:
    assert found(edit(PY_PATH, PY_WRAPPED, PY_HEAD) + CODE) == [
        ("deleted_test", PY_PATH, 8, "base")
    ]


def test_a_renamed_test_that_keeps_only_half_of_its_lines_is_a_deletion() -> None:
    after = PY.replace("test_rounds_half_up", "test_rounds_halves_up").replace(
        "total([0.5]) == 1", "total([0.5, 0.5]) == 1"
    )

    # The assertions it held are this hit, not a second one.
    assert found(edit(PY_PATH, PY, after) + CODE) == [("deleted_test", PY_PATH, 8, "base")]


def test_a_test_deleted_beside_one_renamed_in_the_same_block_is_the_one_reported() -> None:
    empty_cart = "\n\ndef test_empty_cart_costs_nothing():\n    assert total([]) == 0\n"
    after = PY.replace(empty_cart, "").replace("test_rounds_half_up", "test_rounds_halves_up")

    assert found(edit(PY_PATH, PY, after) + CODE) == [("deleted_test", PY_PATH, 4, "base")]


MOVED_TESTS = [
    pytest.param(
        PY_PATH,
        PY,
        PY_HEAD,
        "tests/test_rounding.py",
        "from shop.cart import total\n" + PY_HALF_UP,
        id="python",
    ),
    pytest.param(
        JS_PATH,
        JS,
        JS_HEAD + "});\n",
        "src/rounding.test.ts",
        'import { total } from "./cart";\n' + JS_HALF_UP,
        id="typescript",
    ),
    pytest.param(
        JAVA_PATH,
        JAVA,
        JAVA_HEAD + "}\n",
        "src/test/java/shop/RoundingTest.java",
        "class RoundingTest {" + JAVA_HALF_UP + "}\n",
        id="java",
    ),
    pytest.param(
        GO_PATH,
        GO,
        GO_HEAD,
        "cart/rounding_test.go",
        'package cart\n\nimport "testing"\n' + GO_HALF_UP,
        id="go",
    ),
]


@pytest.mark.parametrize(("path", "before", "after", "destination", "moved"), MOVED_TESTS)
def test_a_test_moved_to_another_file_is_not_a_deletion(
    path: str, before: str, after: str, destination: str, moved: str
) -> None:
    assert detect(edit(path, before, after) + create(destination, moved) + CODE) == []


MOVED_FILES = [
    pytest.param(PY_PATH, "tests/shop/test_cart.py", PY, id="python"),
    pytest.param(JS_PATH, "src/__tests__/cart.test.ts", JS, id="typescript"),
    pytest.param(JAVA_PATH, "src/test/java/shop/checkout/CartTest.java", JAVA, id="java"),
    pytest.param(GO_PATH, "checkout/cart_test.go", GO, id="go"),
]


@pytest.mark.parametrize(("old", "new", "text"), MOVED_FILES)
def test_a_moved_test_file_is_not_a_deletion(old: str, new: str, text: str) -> None:
    # A local diff is taken with --no-renames and shows a deletion plus a
    # creation; a forge's diff shows a pure rename and no lines at all, and
    # the forge reports the rename beside it.
    assert detect(move(old, new, text) + CODE) == []
    assert detect(rename(old, new, text) + CODE, [(old, new)]) == []


#: Moves out of what each runner collects by default, into files that are
#: still test code by their path: a helper directory, a leading underscore, a
#: hidden directory, a name the runner does not match, Go's testdata.
UNCOLLECTED = [
    pytest.param(PY_PATH, "tests/disabled/cart.py", PY, id="python-helper-module"),
    pytest.param(PY_PATH, "tests/_cart.py", PY, id="python-underscore"),
    pytest.param(PY_PATH, "tests/.disabled/test_cart.py", PY, id="python-hidden-directory"),
    pytest.param(JS_PATH, "tests/cart.ts", JS, id="typescript"),
    pytest.param(JAVA_PATH, "src/test/java/shop/CartChecks.java", JAVA, id="java"),
    pytest.param(GO_PATH, "cart/_cart_test.go", GO, id="go-underscore"),
    pytest.param(GO_PATH, "cart/testdata/cart_test.go", GO, id="go-testdata"),
]


@pytest.mark.parametrize(("old", "new", "text"), UNCOLLECTED)
def test_a_test_file_moved_where_the_runner_does_not_collect_it_is_a_deletion(
    old: str, new: str, text: str
) -> None:
    hits = detect(move(old, new, text) + CODE)

    # Every name comes back in a file that is still test code, so matched by
    # name alone nothing was deleted -- while nothing runs these tests any
    # more. Each of the two is reported where it stood.
    assert [(hit.check, hit.file, hit.side) for hit in hits] == [("deleted_test", old, "base")] * 2
    assert all(
        hit.detail.endswith(
            f", and is added back only in `{new}`, which the test runner does not collect "
            "by default."
        )
        for hit in hits
    )


@pytest.mark.parametrize(("old", "new", "text"), UNCOLLECTED)
def test_a_test_file_renamed_where_the_runner_does_not_collect_it_is_one_deletion(
    old: str, new: str, text: str
) -> None:
    diff = rename(old, new, text) + CODE

    hits = detect(diff)

    # Git's rename headers name both sides, so a patch shows the move on its
    # own. A forge sends no patch for a pure rename and passes the pair beside
    # the diff instead. Given both, it is still one file's hit.
    assert detect(CODE, [(old, new)]) == hits
    assert detect(diff, [(old, new)]) == hits

    # Its names are enough. Which tests it held is not in the change, so the
    # hit is the file's and has no line of its own.
    assert [(hit.check, hit.file, hit.line, hit.side, hit.text) for hit in hits] == [
        ("deleted_test", old, None, "base", "")
    ]
    assert hits[0].detail == (
        f"Test file `{old}` is renamed to `{new}`, which the test runner does not collect by "
        "default: the tests in it no longer run."
    )


def test_a_rename_out_of_collection_that_edits_the_file_is_read_from_its_patch() -> None:
    # The patch names both sides, so no rename has to be passed in. The file's
    # hit covers the test the patch drops, rather than adding one for it.
    diff = rename(PY_PATH, "tests/disabled/cart.py", PY, PY_HEAD) + CODE

    assert found(diff) == [("deleted_test", PY_PATH, None, "base")]
    assert found(diff, [(PY_PATH, "tests/disabled/cart.py")]) == found(diff)


def test_a_test_moved_into_a_helper_module_is_a_deletion() -> None:
    helpers = "from shop.cart import total\n" + PY_HALF_UP

    hits = detect(edit(PY_PATH, PY, PY_HEAD) + create("tests/helpers.py", helpers) + CODE)

    assert [(hit.check, hit.file, hit.line) for hit in hits] == [("deleted_test", PY_PATH, 8)]
    assert hits[0].detail == (
        "Test `test_rounds_half_up` is removed, and is added back only in `tests/helpers.py`, "
        "which the test runner does not collect by default."
    )


def test_moving_code_between_files_the_runner_does_not_collect_is_not_a_deletion() -> None:
    # A helper named like a test, in a module pytest never collects. Whether
    # it ever ran as a test is not in the diff; that nothing left a file the
    # runner collects is.
    helpers = "def test_data_path():\n    return 'tests/data'\n"

    assert detect(move("tests/util.py", "tests/support/util.py", helpers)) == []
    assert detect("", [("tests/util.py", "tests/support/util.py")]) == []
    assert detect("", [("src/shop/cart.py", "src/shop/basket.py")]) == []


def test_a_test_file_deleted_with_none_of_its_lines_shown_is_one_deletion() -> None:
    hits = unread_deletions([PY_PATH, "tests/conftest.py", "src/shop/cart.py", PY_PATH])

    # Its name is all there is, so the hit is the file's and has no line. Only
    # a file the runner collects held a test that stopped running.
    assert [(hit.check, hit.file, hit.line, hit.side, hit.text) for hit in hits] == [
        ("deleted_test", PY_PATH, None, "base", "")
    ]
    assert hits[0].detail == (
        f"Test file `{PY_PATH}` is deleted, and the diff shows none of its lines: which tests "
        "it held, and whether the change adds them back elsewhere, cannot be read from it."
    )


def test_a_test_file_renamed_into_collection_is_not_a_hit() -> None:
    assert detect(move("tests/_cart.py", PY_PATH, PY) + CODE) == []
    assert detect(rename("tests/_cart.py", PY_PATH, PY) + CODE, [("tests/_cart.py", PY_PATH)]) == []


def test_a_rename_is_not_judged_for_a_runner_whose_defaults_are_unknown() -> None:
    # RSpec's defaults are not written down here, so nothing is claimed
    # about what it stops collecting.
    assert detect("", [("spec/cart_spec.rb", "spec/disabled/cart.rb")]) == []


REFORMATS = [
    pytest.param(
        PY_PATH,
        PY.replace("test_rounds_half_up():", "test_rounds_half_up( ):").replace(" == ", "=="),
        PY,
        id="python",
    ),
    pytest.param(JS_PATH, JS, JS.replace('"', "'").replace(";\n", "\n"), id="typescript"),
    pytest.param(JAVA_PATH, JAVA, JAVA.replace("@Test\n    void", "@Test void"), id="java"),
    pytest.param(GO_PATH, GO, GO.replace("(t *testing.T) {", "(\n    t *testing.T,\n) {"), id="go"),
]


@pytest.mark.parametrize(("path", "before", "after"), REFORMATS)
def test_a_formatting_change_is_not_a_hit(path: str, before: str, after: str) -> None:
    assert detect(edit(path, before, after) + CODE) == []


def test_removing_a_feature_with_its_tests_is_reported_in_neutral_words() -> None:
    feature = "def legacy_total(items):\n    return sum(items)\n"
    tests = """\
    from shop.legacy import legacy_total


    def test_legacy_total_adds_items():
        assert legacy_total([1, 2]) == 3
    """

    hits = detect(delete("src/shop/legacy.py", feature) + delete("tests/test_legacy.py", tests))

    assert [(hit.check, hit.file, hit.line, hit.side) for hit in hits] == [
        ("deleted_test", "tests/test_legacy.py", 4, "base")
    ]
    assert hits[0].detail == (
        "Test `test_legacy_total_adds_items` is removed along with its file, and no test by "
        "that name is added back in this change."
    )


def test_a_rename_that_drops_a_test_reports_it_under_the_old_path() -> None:
    diff = rename(PY_PATH, "tests/shop/test_cart.py", PY, PY_HEAD)

    assert found(diff) == [("deleted_test", PY_PATH, 8, "base")]


def test_a_java_test_whose_annotation_is_removed_no_longer_runs() -> None:
    after = JAVA.replace("    @Test\n    void roundsHalfUp", "    void roundsHalfUp")

    hits = detect(edit(JAVA_PATH, JAVA, after))

    assert [(hit.check, hit.line, hit.text.strip()) for hit in hits] == [
        ("deleted_test", 13, "@Test")
    ]
    assert hits[0].detail == (
        "`roundsHalfUp` no longer runs as a test: its `@Test` annotation is removed."
    )


def test_a_python_test_renamed_out_of_collection_is_a_deletion() -> None:
    after = PY.replace("def test_rounds_half_up", "def _test_rounds_half_up")

    assert found(edit(PY_PATH, PY, after)) == [("deleted_test", PY_PATH, 8, "base")]


def test_a_test_commented_out_is_a_deletion() -> None:
    after = PY_HEAD + textwrap.dedent(
        """

        # def test_rounds_half_up():
        #     assert total([2.5]) == 3
        #     assert total([0.5]) == 1
        """
    )

    assert found(edit(PY_PATH, PY, after)) == [("deleted_test", PY_PATH, 8, "base")]


def test_removing_a_fixture_named_like_a_test_is_not_a_deletion() -> None:
    before = PY + "\n\n@pytest.fixture\ndef test_data():\n    return [2.5]\n"

    assert detect(edit(PY_PATH, before, PY)) == []


def test_a_test_that_moves_between_languages_is_matched_only_within_its_own() -> None:
    # The same name in a Go file does not bring back a Python test.
    go = "package cart\n\nfunc test_rounds_half_up() {}\n"

    assert found(edit(PY_PATH, PY, PY_HEAD) + create("cart/names.go", go)) == [
        ("deleted_test", PY_PATH, 8, "base")
    ]


GO_SUBTESTS = """\
package cart

import "testing"

func TestTotal(t *testing.T) {
    t.Run("empty cart", func(t *testing.T) {
        if got := Total(nil); got != 0 {
            t.Errorf("got %v, want 0", got)
        }
    })
    t.Run("rounds half up", func(t *testing.T) {
        if got := Total([]float64{2.5}); got != 3 {
            t.Errorf("got %v, want 3", got)
        }
    })
}
"""


def test_a_removed_go_subtest_is_a_deleted_test() -> None:
    after = GO_SUBTESTS.replace(
        """\
    t.Run("rounds half up", func(t *testing.T) {
        if got := Total([]float64{2.5}); got != 3 {
            t.Errorf("got %v, want 3", got)
        }
    })
""",
        "",
    )

    hits = detect(edit(GO_PATH, GO_SUBTESTS, after))

    assert [(hit.check, hit.line) for hit in hits] == [("deleted_test", 11)]
    assert "`rounds half up`" in hits[0].detail


GO_TABLE = """\
package cart

import "testing"

func TestTotal(t *testing.T) {
    tests := []struct {
        name  string
        items []float64
        want  float64
    }{
        {name: "empty cart", items: nil, want: 0},
        {name: "one item", items: []float64{2}, want: 2},
        {name: "rounds half up", items: []float64{2.5}, want: 3},
    }
    for _, tc := range tests {
        t.Run(tc.name, func(t *testing.T) {
            if got := Total(tc.items); got != tc.want {
                t.Errorf("Total(%v) = %v, want %v", tc.items, got, tc.want)
            }
        })
    }
}
"""
GO_HALF_UP_CASE = '        {name: "rounds half up", items: []float64{2.5}, want: 3},\n'


def test_a_named_case_removed_from_a_go_table_is_a_deleted_test() -> None:
    hits = detect(edit(GO_PATH, GO_TABLE, GO_TABLE.replace(GO_HALF_UP_CASE, "")))

    assert [(hit.check, hit.line) for hit in hits] == [("deleted_test", 13)]
    assert hits[0].detail == (
        "Case `rounds half up` is removed from its test table, and no case by that name is "
        "added back in this change."
    )


def test_a_go_table_case_renamed_or_rewrapped_is_not_a_deletion() -> None:
    renamed = GO_TABLE.replace('"rounds half up"', '"rounds halves up"')
    rewrapped = GO_TABLE.replace(
        GO_HALF_UP_CASE,
        '        {\n            name:  "rounds half up",\n'
        "            items: []float64{2.5},\n            want:  3,\n        },\n",
    )

    assert detect(edit(GO_PATH, GO_TABLE, renamed) + CODE) == []
    assert detect(edit(GO_PATH, GO_TABLE, rewrapped) + CODE) == []


def test_a_different_case_written_where_a_removed_one_stood_is_a_deletion() -> None:
    # Same shape, other values: the rewritten values are this hit, not a
    # changed expectation on top of it.
    replaced = GO_TABLE.replace(
        GO_HALF_UP_CASE, '        {name: "two items", items: []float64{1, 2}, want: 3},\n'
    )

    hits = detect(edit(GO_PATH, GO_TABLE, replaced) + CODE)

    assert [(hit.check, hit.line) for hit in hits] == [("deleted_test", 13)]
    assert hits[0].detail == (
        "Case `rounds half up` is removed from its test table, and no case by that name is "
        "added back in this change."
    )


def test_a_go_case_over_several_lines_is_compared_by_all_of_them() -> None:
    def table(name: str, items: str) -> str:
        # The table's start has to be in the hunk for a name on a line of its
        # own to be read as a case at all.
        return (
            'package cart\n\nimport "testing"\n\nfunc TestTotal(t *testing.T) {\n'
            "    tests := []struct{ name string; items []float64; want float64 }{\n"
            f'        {{\n            name:  "{name}",\n'
            f"            items: []float64{{{items}}},\n            want:  3,\n        }},\n"
            "    }\n    _ = tests\n}\n"
        )

    before = table("rounds half up", "2.5")

    # Its name line alone is the same in any two cases.
    assert found(edit(GO_PATH, before, table("two items", "1, 2")) + CODE) == [
        ("deleted_test", GO_PATH, 8, "base")
    ]
    assert detect(edit(GO_PATH, before, table("rounds halves up", "2.5")) + CODE) == []


def test_a_different_parametrize_case_written_where_a_removed_one_stood_is_a_deletion() -> None:
    after = PY_PARAMETRIZED.replace(
        'pytest.param([2.5], 3, id="rounds half up")', 'pytest.param([1, 2], 3, id="two items")'
    )
    renamed = PY_PARAMETRIZED.replace('id="rounds half up"', 'id="rounds halves up"')

    assert found(edit(PY_PATH, PY_PARAMETRIZED, after) + CODE) == [
        ("deleted_test", PY_PATH, 10, "base")
    ]
    assert detect(edit(PY_PATH, PY_PARAMETRIZED, renamed) + CODE) == []


def test_a_go_fixture_slice_is_not_a_test_table() -> None:
    before = GO + '\nvar shoppers = []Shopper{\n    {name: "ann"},\n    {name: "bo"},\n}\n'
    after = before.replace('    {name: "bo"},\n', "")

    assert detect(edit(GO_PATH, before, after)) == []


def test_deleting_a_table_test_is_one_hit_not_one_per_case() -> None:
    assert found(edit(GO_PATH, GO_HEAD + "\n" + GO_TABLE.split("\n\n", 2)[2], GO_HEAD)) == [
        ("deleted_test", GO_PATH, 11, "base")
    ]


PY_PARAMETRIZED = """\
import pytest

from shop.cart import total


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        pytest.param([], 0, id="empty cart"),
        pytest.param([2.5], 3, id="rounds half up"),
    ],
)
def test_total(items, expected):
    assert total(items) == expected
"""


def test_a_named_parametrize_case_removed_is_a_deleted_test() -> None:
    after = PY_PARAMETRIZED.replace('        pytest.param([2.5], 3, id="rounds half up"),\n', "")

    assert found(edit(PY_PATH, PY_PARAMETRIZED, after)) == [("deleted_test", PY_PATH, 10, "base")]


def test_deleting_a_parametrized_test_is_one_hit_not_one_per_case() -> None:
    after = PY_PARAMETRIZED.split("\n\n\n")[0] + "\n"

    assert found(edit(PY_PATH, PY_PARAMETRIZED, after)) == [("deleted_test", PY_PATH, 13, "base")]


def test_a_one_line_each_test_is_a_test_too() -> None:
    each = '  it.each([[2.5, 3]])("rounds %d to %d", (value, rounded) => {\n' + (
        "    expect(total([value])).toBe(rounded);\n  });\n"
    )
    before = JS_HEAD + "\n" + each + "});\n"

    assert found(edit(JS_PATH, before, JS_HEAD + "});\n")) == [("deleted_test", JS_PATH, 8, "base")]


JUNIT3 = """\
package shop;

import junit.framework.TestCase;

public class CartTest extends TestCase {
    public void testEmptyCartCostsNothing() {
        assertEquals(0, Cart.total());
    }

    public void testRoundsHalfUp() {
        assertEquals(3, Cart.total(2.5));
    }
}
"""


def test_a_removed_junit3_test_is_a_deleted_test() -> None:
    after = JUNIT3.replace(
        "\n    public void testRoundsHalfUp() {\n        assertEquals(3, Cart.total(2.5));\n    }\n",
        "",
    )

    assert found(edit(JAVA_PATH, JUNIT3, after)) == [("deleted_test", JAVA_PATH, 10, "base")]


def test_removing_go_test_main_is_not_a_deleted_test() -> None:
    before = GO + "\nfunc TestMain(m *testing.M) {\n    os.Exit(m.Run())\n}\n"

    assert detect(edit(GO_PATH, before, GO)) == []


JS_ONE_LINER = """\
describe("total", () => {
  it("is zero for an empty cart", () => expect(total([])).toBe(0));
  it("rounds half up", () => {
    expect(total([2.5])).toBe(3);
    expect(total([0.5])).toBe(1);
  });
});
"""


def test_a_one_line_js_test_ends_on_its_own_line() -> None:
    after = JS_ONE_LINER.replace(
        '  it("is zero for an empty cart", () => expect(total([])).toBe(0));\n', ""
    ).replace("    expect(total([0.5])).toBe(1);\n", "")

    # The assertion lost from the next test is not part of the one-liner's hit.
    assert found(edit(JS_PATH, JS_ONE_LINER, after)) == [
        ("deleted_test", JS_PATH, 2, "base"),
        ("fewer_assertions", JS_PATH, 5, "base"),
    ]


# ── b. skip_or_focus ─────────────────────────────────────────────────────────

PY_TEST = "def test_rounds_half_up"
JAVA_TEST = "void roundsHalfUp"

MARKERS = [
    pytest.param(
        PY_PATH,
        PY,
        above(PY, PY_TEST, '@pytest.mark.skip(reason="flaky")'),
        8,
        "Adds `pytest.mark.skip`: the test it applies to is skipped.",
        id="python-skip",
    ),
    pytest.param(
        PY_PATH,
        PY,
        above(PY, PY_TEST, "@pytest.mark.xfail"),
        8,
        "Adds `pytest.mark.xfail`: a failure of the test it marks no longer fails the run.",
        id="python-xfail",
    ),
    pytest.param(
        PY_PATH,
        PY,
        above(PY, PY_TEST, '@pytest.mark.skipif(sys.platform == "win32", reason="rounding")'),
        8,
        "Adds `pytest.mark.skipif`: the test it applies to is skipped under some conditions.",
        id="python-skipif",
    ),
    pytest.param(
        PY_PATH,
        PY,
        above(PY, "assert total([2.5]) == 3", 'pytest.skip("not ready")'),
        9,
        "Adds `pytest.skip`: the test it applies to is skipped.",
        id="python-skip-call",
    ),
    pytest.param(
        PY_PATH,
        PY,
        above(PY, PY_TEST, '@unittest.skip("later")'),
        8,
        "Adds `unittest.skip`: the test it applies to is skipped.",
        id="python-unittest",
    ),
    pytest.param(
        JS_PATH,
        JS,
        JS.replace('it("rounds half up"', 'it.skip("rounds half up"'),
        8,
        "Adds `it.skip`: the test it applies to is skipped.",
        id="typescript-skip",
    ),
    pytest.param(
        JS_PATH,
        JS,
        JS.replace('it("rounds half up"', 'it.only("rounds half up"'),
        8,
        "Adds `it.only`: only focused tests run, and the others are skipped.",
        id="typescript-only",
    ),
    pytest.param(
        JS_PATH,
        JS,
        JS.replace('describe("total"', 'describe.only("total"'),
        3,
        "Adds `describe.only`: only focused tests run, and the others are skipped.",
        id="typescript-describe-only",
    ),
    pytest.param(
        JS_PATH,
        JS,
        JS.replace('  it("rounds half up"', '  xit("rounds half up"'),
        8,
        "Adds `xit`: the test it applies to is skipped.",
        id="typescript-xit",
    ),
    pytest.param(
        JS_PATH,
        JS,
        JS.replace('  it("rounds half up"', '  fit("rounds half up"'),
        8,
        "Adds `fit`: only focused tests run, and the others are skipped.",
        id="typescript-fit",
    ),
    pytest.param(
        JS_PATH,
        JS,
        JS_HEAD + JS_HALF_UP + '  test.todo("rounds half down");\n});\n',
        12,
        "Adds `test.todo`: a placeholder that does not run.",
        id="typescript-todo",
    ),
    pytest.param(
        JAVA_PATH,
        JAVA,
        above(JAVA, JAVA_TEST, '@Disabled("flaky")'),
        14,
        "Adds `@Disabled`: the test it applies to is skipped.",
        id="java-disabled",
    ),
    pytest.param(
        JAVA_PATH,
        JAVA,
        above(JAVA, "@Test", "@Ignore"),
        8,
        "Adds `@Ignore`: the test it applies to is skipped.",
        id="java-ignore",
    ),
    pytest.param(
        JAVA_PATH,
        JAVA,
        above(JAVA, "assertEquals(3, Cart.total(2.5))", "assumeTrue(Cart.supportsRounding());"),
        15,
        "Adds `assumeTrue`: the test it applies to is skipped under some conditions.",
        id="java-assume",
    ),
    pytest.param(
        GO_PATH,
        GO,
        above(GO, "want := 3.0", 't.Skip("flaky")'),
        12,
        "Adds `t.Skip`: the test it applies to is skipped.",
        id="go-skip",
    ),
]


@pytest.mark.parametrize(("path", "before", "after", "line", "detail"), MARKERS)
def test_a_new_skip_or_focus_marker_is_reported_on_the_head_side(
    path: str, before: str, after: str, line: int, detail: str
) -> None:
    hits = detect(edit(path, before, after))

    assert [(hit.check, hit.file, hit.line, hit.side) for hit in hits] == [
        ("skip_or_focus", path, line, "head")
    ]
    assert hits[0].detail == detail


NOT_MARKERS = [
    pytest.param(
        PY_PATH,
        PY,
        above(PY, PY_TEST, "# @pytest.mark.skip once rounding moves"),
        id="python-comment",
    ),
    pytest.param(
        PY_PATH,
        PY,
        above(PY, "assert total([0.5]) == 1", 'assert lint("@pytest.mark.skip") == ["skip"]'),
        id="python-string",
    ),
    pytest.param(
        PY_PATH,
        PY,
        above(
            PY, "assert total([2.5]) == 3", '"""\nNever calls pytest.skip(: halves round up.\n"""'
        ),
        id="python-docstring",
    ),
    pytest.param(
        JS_PATH,
        JS,
        above(JS, 'it("rounds half up"', '// it.only("rounds half up") while debugging'),
        id="typescript-comment",
    ),
    pytest.param(
        JS_PATH,
        JS,
        JS.replace('it("rounds half up"', 'it("skips empty rows and rounds half up"'),
        id="typescript-name",
    ),
    pytest.param(
        JAVA_PATH,
        JAVA,
        above(JAVA, JAVA_TEST, "// @Disabled until the fix lands"),
        id="java-comment",
    ),
    pytest.param(
        JAVA_PATH,
        JAVA,
        above(JAVA, JAVA_TEST, "/*\n * @Disabled was here before rounding was fixed.\n */"),
        id="java-block-comment",
    ),
    pytest.param(
        GO_PATH,
        GO,
        above(GO, "want := 3.0", '// t.Skip("flaky") used to be here'),
        id="go-comment",
    ),
    pytest.param(
        GO_PATH,
        GO,
        above(GO, "want := 3.0", '/*\nt.Skip("flaky")\n*/'),
        id="go-block-comment",
    ),
    pytest.param(
        JS_PATH,
        JS,
        above(
            JS,
            "expect(total([2.5]))",
            'const usage = `\n  it.only("x") runs alone, and \\`it.skip\\` not at all\n`;',
        ),
        id="typescript-template-literal",
    ),
    pytest.param(
        JAVA_PATH,
        JAVA,
        above(
            JAVA,
            "assertEquals(3, Cart.total(2.5))",
            'String usage = """\n    @Disabled tests are listed apart.\n    """;',
        ),
        id="java-text-block",
    ),
    pytest.param(
        GO_PATH,
        GO,
        above(GO, "want := 3.0", 't.Log(`\nt.Skip("flaky") is how a test bows out\n`)'),
        id="go-raw-string",
    ),
]


@pytest.mark.parametrize(("path", "before", "after"), NOT_MARKERS)
def test_a_marker_in_a_comment_string_or_name_is_not_a_skip(
    path: str, before: str, after: str
) -> None:
    assert detect(edit(path, before, after)) == []


def test_changing_the_reason_of_an_existing_skip_is_not_a_new_skip() -> None:
    before = above(PY, PY_TEST, '@pytest.mark.skip(reason="flaky")')
    after = before.replace('reason="flaky"', 'reason="flaky on CI"')

    assert detect(edit(PY_PATH, before, after)) == []


def test_turning_a_conditional_skip_into_a_plain_one_is_a_new_skip() -> None:
    before = above(PY, PY_TEST, '@pytest.mark.skipif(WINDOWS, reason="x")')
    after = before.replace('skipif(WINDOWS, reason="x")', 'skip(reason="x")')

    assert found(edit(PY_PATH, before, after)) == [("skip_or_focus", PY_PATH, 8, "head")]


def test_a_skip_that_moves_with_its_file_is_not_a_new_skip() -> None:
    skipped = JS.replace('it("rounds half up"', 'it.skip("rounds half up"')

    assert detect(move(JS_PATH, "src/__tests__/cart.test.ts", skipped)) == []


def test_removing_a_skip_is_not_a_hit() -> None:
    skipped = above(JAVA, JAVA_TEST, '@Disabled("flaky")')

    assert detect(edit(JAVA_PATH, skipped, JAVA)) == []


def test_a_skip_added_by_conftest_is_a_skip() -> None:
    before = "def pytest_collection_modifyitems(items):\n    pass\n"
    after = (
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n"
        '        item.add_marker(pytest.mark.skip(reason="slow"))\n'
    )

    assert found(edit("conftest.py", before, after)) == [
        ("skip_or_focus", "conftest.py", 3, "head")
    ]


def test_a_line_too_long_to_be_hand_written_is_not_read() -> None:
    bundle = 'it.only("x", () => {' + " " * MAX_SCANNED_LINE_CHARS + "});\n"

    assert detect(edit(JS_PATH, JS, JS + bundle)) == []


# ── c. fewer_assertions ──────────────────────────────────────────────────────

LOST_ASSERTIONS = [
    pytest.param(PY_PATH, PY, PY.replace("    assert total([0.5]) == 1\n", ""), 10, id="python"),
    pytest.param(
        JS_PATH, JS, JS.replace("    expect(total([0.5])).toBe(1);\n", ""), 10, id="typescript"
    ),
    pytest.param(
        JAVA_PATH,
        JAVA,
        JAVA.replace("        assertEquals(1, Cart.total(0.5));\n", ""),
        16,
        id="java",
    ),
    pytest.param(
        GO_PATH,
        GO,
        GO.replace(
            "    want := 3.0\n    if got := Total([]float64{2.5}); got != want {\n"
            '        t.Errorf("Total(2.5) = %v, want %v", got, want)\n    }\n',
            "    Total([]float64{2.5})\n",
        ),
        14,
        id="go",
    ),
]


@pytest.mark.parametrize(("path", "before", "after", "line"), LOST_ASSERTIONS)
def test_a_test_file_that_loses_assertions_is_reported_at_the_first_one_lost(
    path: str, before: str, after: str, line: int
) -> None:
    hits = detect(edit(path, before, after))

    assert [(hit.check, hit.file, hit.line, hit.side) for hit in hits] == [
        ("fewer_assertions", path, line, "base")
    ]
    assert hits[0].detail == "Removes 1 assertion and adds 0 in this file, a net loss of 1."


REWRITTEN_ASSERTIONS = [
    pytest.param(
        PY_PATH,
        PY,
        PY.replace("assert total([0.5]) == 1", "assert total([0.5]) == pytest.approx(1)"),
        id="python",
    ),
    pytest.param(JS_PATH, JS, JS.replace("toBe(1);", "toEqual(1);"), id="typescript"),
    pytest.param(
        JAVA_PATH,
        JAVA,
        JAVA.replace(
            "assertEquals(1, Cart.total(0.5));", "assertThat(Cart.total(0.5)).isEqualTo(1);"
        ),
        id="java",
    ),
    pytest.param(GO_PATH, GO, GO.replace('t.Errorf("Total(2.5)', 't.Fatalf("Total(2.5)'), id="go"),
]


@pytest.mark.parametrize(("path", "before", "after"), REWRITTEN_ASSERTIONS)
def test_an_assertion_rewritten_one_for_one_is_not_a_loss(
    path: str, before: str, after: str
) -> None:
    assert detect(edit(path, before, after)) == []


def test_an_assertion_moved_into_a_block_is_not_a_loss() -> None:
    after = PY.replace(
        "    assert total([2.5]) == 3\n",
        "    with localcontext():\n        assert total([2.5]) == 3\n",
    )

    assert detect(edit(PY_PATH, PY, after)) == []


def test_splitting_one_assertion_into_two_is_not_a_loss() -> None:
    after = JAVA.replace(
        "        assertEquals(1, Cart.total(0.5));\n",
        "        assertNotNull(Cart.total(0.5));\n        assertEquals(1, Cart.total(0.5));\n",
    )

    assert detect(edit(JAVA_PATH, JAVA, after)) == []


def test_the_same_assertion_removed_twice_and_restored_once_is_a_loss_of_one() -> None:
    before = PY.replace("    assert total([0.5]) == 1\n", "    assert total([0.5]) == 1\n" * 2)
    after = PY.replace("    assert total([0.5]) == 1\n", "")

    assert [hit.detail for hit in detect(edit(PY_PATH, before, after))] == [
        "Removes 2 assertions and adds 0 in this file, a net loss of 2."
    ]


def test_an_assertion_renamed_by_its_import_is_still_an_assertion() -> None:
    before = (
        'import { by, element, expect } from "detox";\n\n'
        'it("shows the total", async () => {\n'
        '  await expect(element(by.id("total"))).toHaveText("3");\n'
        "});\n"
    )
    renamed = before.replace("expect }", "expect as detoxExpect }").replace(
        "await expect(", "await detoxExpect("
    )
    dropped = renamed.replace('  await detoxExpect(element(by.id("total"))).toHaveText("3");\n', "")

    assert detect(edit("e2e/cart.spec.ts", before, renamed)) == []
    assert found(edit("e2e/cart.spec.ts", renamed, dropped)) == [
        ("fewer_assertions", "e2e/cart.spec.ts", 4, "base")
    ]


GO_SETUP = """\
package cart

func setup() error {
    Expect(Load("cart.json")).To(Succeed())
    return nil
}
"""


def test_a_go_check_that_returns_the_error_replaces_an_assertion() -> None:
    returned = GO_SETUP.replace(
        '    Expect(Load("cart.json")).To(Succeed())\n',
        '    if err := Load("cart.json"); err != nil {\n'
        '        return fmt.Errorf("load cart: %w", err)\n'
        "    }\n",
    )
    dropped = GO_SETUP.replace(
        '    Expect(Load("cart.json")).To(Succeed())\n', '    _ = Load("cart.json")\n'
    )

    assert detect(edit("cart/setup_test.go", GO_SETUP, returned)) == []
    assert found(edit("cart/setup_test.go", returned, dropped)) == [
        ("fewer_assertions", "cart/setup_test.go", 4, "base")
    ]


def test_copies_of_an_assertion_gathered_into_one_helper_are_moved() -> None:
    guard = (
        "\n\ndef _require_db():\n"
        "    if not db_up():\n"
        '        pytest.fail("the database is not running")\n'
    )
    cart, order = "tests/test_cart.py", "tests/test_order.py"
    helper = "tests/helpers/db.py"
    gathered = (
        edit(cart, PY + guard, PY)
        + edit(order, PY + guard, PY)
        + create(helper, "import pytest" + guard)
    )
    lost = edit(cart, PY + guard, PY) + edit(order, PY + guard, PY)

    assert detect(gathered) == []
    assert found(lost) == [
        ("fewer_assertions", cart, 15, "base"),
        ("fewer_assertions", order, 15, "base"),
    ]


# ── d. changed_expectation ───────────────────────────────────────────────────

REWRITTEN_EXPECTATIONS = [
    pytest.param(
        PY_PATH,
        PY,
        "    assert total([2.5]) == 3",
        "    assert total([2.5]) == 2",
        9,
        "assertion",
        id="python",
    ),
    pytest.param(
        JS_PATH,
        JS,
        "    expect(total([2.5])).toBe(3);",
        "    expect(total([2.5])).toBe(2);",
        9,
        "assertion",
        id="typescript",
    ),
    pytest.param(
        JAVA_PATH,
        JAVA,
        "        assertEquals(3, Cart.total(2.5));",
        "        assertEquals(2, Cart.total(2.5));",
        15,
        "assertion",
        id="java",
    ),
    pytest.param(GO_PATH, GO, "    want := 3.0", "    want := 2.0", 12, "expected value", id="go"),
]


@pytest.mark.parametrize(("path", "before", "old", "new", "line", "kind"), REWRITTEN_EXPECTATIONS)
def test_an_expectation_whose_literal_changes_with_the_code_is_reported(
    path: str, before: str, old: str, new: str, line: int, kind: str
) -> None:
    hits = detect(edit(path, before, before.replace(old, new)) + CODE)

    assert [(hit.check, hit.file, hit.line, hit.side) for hit in hits] == [
        ("changed_expectation", path, line, "base")
    ]
    assert hits[0].text == old
    assert hits[0].replacement == new
    was, now = ("3.0", "2.0") if path == GO_PATH else ("3", "2")
    assert hits[0].detail == (
        f"Only literal values change in this {kind} (`{was}` → `{now}`), in a change that "
        "also edits non-test code."
    )


@pytest.mark.parametrize(("path", "before", "old", "new", "line", "kind"), REWRITTEN_EXPECTATIONS)
def test_an_expectation_changed_without_touching_code_is_a_test_fix(
    path: str, before: str, old: str, new: str, line: int, kind: str
) -> None:
    assert detect(edit(path, before, before.replace(old, new))) == []


RESHAPED_EXPECTATIONS = [
    pytest.param(PY_PATH, PY, "total([2.5]) == 3", "total([2.5], rounding=UP) == 3", id="python"),
    pytest.param(JS_PATH, JS, "toBe(3);", "toBeCloseTo(3, 0);", id="typescript"),
    pytest.param(
        JAVA_PATH,
        JAVA,
        "assertEquals(3, Cart.total(2.5));",
        'assertEquals(3, Cart.total(2.5), "rounds half up");',
        id="java",
    ),
    pytest.param(GO_PATH, GO, "want := 3.0", "want := math.Round(2.5)", id="go"),
]


@pytest.mark.parametrize(("path", "before", "old", "new"), RESHAPED_EXPECTATIONS)
def test_an_assertion_that_changes_shape_is_not_a_changed_literal(
    path: str, before: str, old: str, new: str
) -> None:
    assert detect(edit(path, before, before.replace(old, new)) + CODE) == []


def test_a_quote_style_change_is_not_a_changed_literal() -> None:
    before = PY + "\n\ndef test_label():\n    assert label(3) == 'three'\n"
    after = before.replace("'three'", '"three"')

    assert detect(edit(PY_PATH, before, after) + CODE) == []


def test_a_negated_expected_value_is_a_changed_literal() -> None:
    before = JAVA.replace(
        "assertEquals(3, Cart.total(2.5));", "assertEquals(-3, Cart.total(-2.5));"
    )
    after = before.replace("assertEquals(-3,", "assertEquals(3,")

    hits = detect(edit(JAVA_PATH, before, after) + CODE)

    assert [(hit.check, hit.line) for hit in hits] == [("changed_expectation", 15)]
    assert "(`-3` → `3`)" in hits[0].detail


def test_a_value_inside_a_multi_line_assertion_is_an_expectation() -> None:
    before = (
        PY + '\n\ndef test_receipt():\n    assert receipt([2.5]) == {\n        "total": 3,\n    }\n'
    )
    after = before.replace('"total": 3,', '"total": 2,')

    assert found(edit(PY_PATH, before, after) + CODE) == [
        ("changed_expectation", PY_PATH, 15, "base")
    ]


def test_a_go_if_that_asserts_holds_its_expected_value() -> None:
    after = GO.replace("got != 0 {", "got != 1 {")

    assert found(edit(GO_PATH, GO, after) + CODE) == [("changed_expectation", GO_PATH, 6, "base")]


def test_a_changed_row_of_a_test_table_is_a_changed_expectation() -> None:
    after = GO_TABLE.replace("items: []float64{2.5}, want: 3}", "items: []float64{2.5}, want: 2}")

    hits = detect(edit(GO_PATH, GO_TABLE, after) + CODE)

    assert [(hit.check, hit.line) for hit in hits] == [("changed_expectation", 13)]
    assert "this test case (`3` → `2`)" in hits[0].detail


def test_a_changed_parametrize_case_is_a_changed_expectation() -> None:
    after = PY_PARAMETRIZED.replace(
        'pytest.param([2.5], 3, id="rounds half up")', 'pytest.param([2.5], 2, id="rounds half up")'
    )

    hits = detect(edit(PY_PATH, PY_PARAMETRIZED, after) + CODE)

    assert [(hit.check, hit.line) for hit in hits] == [("changed_expectation", 10)]
    assert "this test case (`3` → `2`)" in hits[0].detail


def test_a_number_is_compared_by_value_not_spelling() -> None:
    before = PY + "\n\ndef test_flags():\n    assert flags() == 0x10\n"

    assert detect(edit(PY_PATH, before, before.replace("0x10", "0X10")) + CODE) == []
    hits = detect(edit(PY_PATH, before, before.replace("0x10", "0x20")) + CODE)
    assert [(hit.check, hit.line) for hit in hits] == [("changed_expectation", 14)]
    assert "(`0x10` → `0x20`)" in hits[0].detail


#: Statements that hold a failure message beside their expected value, with
#: each: the message reworded, then the value changed.
MESSAGES = [
    pytest.param(
        PY_PATH,
        PY.replace(
            "    assert total([2.5]) == 3\n",
            '    assert total([2.5]) == 3, (\n        "rounds half up"\n    )\n',
        ),
        ('"rounds half up"', '"rounds 2.5 up to 3"'),
        ("== 3, (", "== 2, ("),
        id="python-assert",
    ),
    pytest.param(
        PY_PATH,
        PY.replace(
            "    assert total([2.5]) == 3\n",
            '    assert_allclose(total([2.5]), 3, err_msg="rounds half up")\n',
        ),
        ('"rounds half up"', '"rounds 2.5 up"'),
        ("]), 3,", "]), 2,"),
        id="python-err_msg",
    ),
    pytest.param(
        JS_PATH,
        JS.replace(
            "    expect(total([2.5])).toBe(3);\n",
            '    assert.match(label(2.5), /three/, "names the rounded total");\n',
        ),
        ('"names the rounded total"', '"spells the rounded total"'),
        ("/three/", "/two/"),
        id="node-assert",
    ),
    pytest.param(
        JS_PATH,
        JS.replace(
            "expect(total([2.5])).toBe(3);", 'expect(total([2.5]), "halves go up").toBe(3);'
        ),
        ('"halves go up"', '"2.5 goes up"'),
        ("toBe(3)", "toBe(2)"),
        id="expect-message",
    ),
    pytest.param(
        GO_PATH,
        GO.replace(
            "    want := 3.0\n    if got := Total([]float64{2.5}); got != want {\n"
            '        t.Errorf("Total(2.5) = %v, want %v", got, want)\n',
            "    if got := Total([]float64{2.5}); got != 3 {\n"
            '        t.Fatalf(\n            "Total(2.5) = %v, want 3",\n            got)\n',
        ),
        ('"Total(2.5) = %v, want 3"', '"Total(2.5) returned %v, want 3"'),
        ("got != 3", "got != 2"),
        id="go-fatalf",
    ),
    pytest.param(
        GO_PATH,
        GO.replace(
            "    want := 3.0\n    if got := Total([]float64{2.5}); got != want {\n"
            '        t.Errorf("Total(2.5) = %v, want %v", got, want)\n    }\n',
            '    Expect(Total([]float64{2.5})).To(Equal(3.0), "rounds half up")\n',
        ),
        ('"rounds half up"', '"rounds 2.5 up"'),
        ("Equal(3.0)", "Equal(2.0)"),
        id="gomega",
    ),
]


@pytest.mark.parametrize(("path", "before", "message", "value"), MESSAGES)
def test_a_reworded_failure_message_is_not_a_changed_expectation(
    path: str, before: str, message: tuple[str, str], value: tuple[str, str]
) -> None:
    line = next(n for n, row in enumerate(before.splitlines(), 1) if value[0] in row)

    assert detect(edit(path, before, before.replace(*message)) + CODE) == []
    assert found(edit(path, before, before.replace(*value)) + CODE) == [
        ("changed_expectation", path, line, "base")
    ]


def test_a_quoted_word_that_is_no_key_does_not_make_an_expected_value() -> None:
    before = JS_HEAD + "  const verdict = ok ? 'expected' : 'unexpected';\n});\n"
    keyed = JS_HEAD + "  const cases = { 'expected': 3 };\n});\n"

    assert (
        detect(edit(JS_PATH, before, before.replace("'unexpected'", "'surprising'")) + CODE) == []
    )
    assert found(edit(JS_PATH, keyed, keyed.replace("3 }", "2 }")) + CODE) == [
        ("changed_expectation", JS_PATH, 7, "base")
    ]


def test_the_names_of_a_parametrized_test_are_not_its_expectations() -> None:
    before = """\
import pytest


@pytest.mark.parametrize(
    ("items", "expected"),
    [([2.5], 3)],
    ids=["half up"],
)
def test_total(items, expected):
    assert total(items) == expected
"""
    retitled = before.replace('"half up"', '"rounds half up"')
    renamed = before.replace('"expected"', '"want"').replace("expected", "want")

    assert detect(edit(PY_PATH, before, retitled) + CODE) == []
    assert detect(edit(PY_PATH, before, renamed) + CODE) == []
    assert found(edit(PY_PATH, before, before.replace("3)]", "2)]")) + CODE) == [
        ("changed_expectation", PY_PATH, 6, "base")
    ]


def test_the_title_after_an_each_table_is_not_part_of_the_table() -> None:
    before = JS_HEAD + (
        "  it.each([[2.5, 3]])(\n"
        '    "rounds %d to %d",\n'
        "    (value, rounded) => {\n"
        '      const label = "rounding";\n'
        "      expect(total([value])).toBe(rounded);\n"
        "    },\n"
        "  );\n"
        "});\n"
    )
    retitled = before.replace('"rounds %d to %d"', '"rounds %d up to %d"').replace(
        '"rounding"', '"rounding up"'
    )

    assert detect(edit(JS_PATH, before, retitled) + CODE) == []
    assert found(edit(JS_PATH, before, before.replace("[[2.5, 3]]", "[[2.5, 2]]")) + CODE) == [
        ("changed_expectation", JS_PATH, 7, "base")
    ]


GO_REALIGNED = """\
package cart

import "testing"

func TestRound(t *testing.T) {
    tests := []struct {
        name string
        in   float64
        want float64
    }{
        {
            name: "rounds half up",
            in:   2.5,
            want: 3,
        },
    }
    for _, tc := range tests {
        if got := Round(tc.in); got != tc.want {
            t.Errorf("Round(%v) = %v, want %v", tc.in, got, tc.want)
        }
    }
}
"""


def test_a_line_realigned_in_its_own_case_is_not_rewritten_into_a_new_one() -> None:
    # A new field realigns every case, and the new case's lines take the
    # place of the old one's, which is still there below them.
    after = GO_REALIGNED.replace(
        "        name string\n        in   float64\n        want float64\n",
        "        name  string\n        in    float64\n        want  float64\n        exact bool\n",
    ).replace(
        '        {\n            name: "rounds half up",\n            in:   2.5,\n            want: 3,\n',
        '        {\n            name:  "rounds down",\n            in:    2.4,\n            want:  2,\n'
        "        },\n"
        '        {\n            name:  "rounds half up",\n            in:    2.5,\n            want:  3,\n'
        "            exact: true,\n",
    )

    assert detect(edit(GO_PATH, GO_REALIGNED, after) + CODE) == []
    changed = after.replace(
        "in:    2.5,\n            want:  3,", "in:    2.5,\n            want:  4,"
    )
    assert found(edit(GO_PATH, GO_REALIGNED, changed) + CODE) == [
        ("changed_expectation", GO_PATH, 14, "base")
    ]


def test_an_expectation_is_not_rewritten_into_the_next_test() -> None:
    split = JS.replace(
        "    expect(total([0.5])).toBe(1);\n  });\n",
        '  });\n\n  it("rounds a single half up", () => {\n    expect(total([1.5])).toBe(2);\n  });\n',
    )

    assert detect(edit(JS_PATH, JS, split) + CODE) == []
    assert found(edit(JS_PATH, JS, JS.replace("toBe(1);", "toBe(2);")) + CODE) == [
        ("changed_expectation", JS_PATH, 10, "base")
    ]


def test_a_regular_expression_is_one_literal() -> None:
    # Its slashes, quotes and brackets are not code: read as code, ``/*``
    # in it opened a comment that hid the lines below.
    before = JS.replace(
        "    expect(total([2.5])).toBe(3);\n",
        "    expect(receiptPath()).toMatch(/receipts\\/*\\.txt/);\n"
        "    expect(total([2.5])).toBe(3);\n",
    )

    assert found(edit(JS_PATH, before, before.replace("toBe(3)", "toBe(2)")) + CODE) == [
        ("changed_expectation", JS_PATH, 10, "base")
    ]
    hits = detect(edit(JS_PATH, before, before.replace(".txt/", ".pdf/")) + CODE)
    assert [(hit.check, hit.line) for hit in hits] == [("changed_expectation", 9)]
    assert "(`/receipts\\/*\\.txt/` → `/receipts\\/*\\.pdf/`)" in hits[0].detail


# ── e. loosened_threshold ────────────────────────────────────────────────────

PYPROJECT = """\
[tool.pytest.ini_options]
addopts = "--cov=shop --cov-fail-under=90"
testpaths = ["tests", "integration"]

[tool.coverage.report]
fail_under = 90
"""


def test_a_lowered_coverage_threshold_is_reported_with_what_it_became() -> None:
    after = PYPROJECT.replace("--cov-fail-under=90", "--cov-fail-under=80")

    hits = detect(edit("pyproject.toml", PYPROJECT, after))

    assert [(hit.check, hit.file, hit.line, hit.side) for hit in hits] == [
        ("loosened_threshold", "pyproject.toml", 2, "base")
    ]
    assert hits[0].detail == "Loosens the coverage threshold: `--cov-fail-under` 90 → 80."
    assert hits[0].replacement == 'addopts = "--cov=shop --cov-fail-under=80"'


def test_a_removed_coverage_threshold_is_a_lowered_one() -> None:
    after = PYPROJECT.replace("fail_under = 90\n", "")

    hits = detect(edit("pyproject.toml", PYPROJECT, after))

    assert [(hit.line, hit.detail) for hit in hits] == [
        (6, "Loosens the coverage threshold: `fail_under` 90 is removed.")
    ]


def test_a_threshold_raised_reformatted_or_moved_is_not_loosened() -> None:
    raised = PYPROJECT.replace("fail_under = 90", "fail_under = 95")
    reformatted = PYPROJECT.replace("fail_under = 90", "fail_under=90")
    moved = edit("pyproject.toml", PYPROJECT, PYPROJECT.replace("fail_under = 90\n", "")) + create(
        ".coveragerc", "[report]\nfail_under = 90\n"
    )

    assert detect(edit("pyproject.toml", PYPROJECT, raised)) == []
    assert detect(edit("pyproject.toml", PYPROJECT, reformatted)) == []
    assert detect(moved) == []


def test_a_narrower_testpaths_is_reported_and_a_reordered_one_is_not() -> None:
    narrowed = PYPROJECT.replace('["tests", "integration"]', '["tests"]')
    reordered = PYPROJECT.replace('["tests", "integration"]', '["integration", "tests"]')
    dropped = PYPROJECT.replace('testpaths = ["tests", "integration"]\n', "")

    hits = detect(edit("pyproject.toml", PYPROJECT, narrowed))

    assert [(hit.line, hit.side, hit.detail) for hit in hits] == [
        (3, "base", "Narrows `testpaths`: `integration` is no longer collected.")
    ]
    assert detect(edit("pyproject.toml", PYPROJECT, reordered)) == []
    # Without the setting pytest collects from the root: more tests, not fewer.
    assert detect(edit("pyproject.toml", PYPROJECT, dropped)) == []


def test_an_ini_testpaths_that_loses_a_line_is_narrower() -> None:
    before = "[pytest]\ntestpaths =\n    tests\n    integration\n"
    after = "[pytest]\ntestpaths =\n    tests\n"

    assert found(edit("pytest.ini", before, after)) == [
        ("loosened_threshold", "pytest.ini", 4, "base")
    ]


CI = """\
jobs:
  test:
    steps:
      - run: pip install -e .
      - run: pytest --cov=shop
"""

EXCLUSIONS = [
    pytest.param(
        "pytest --cov=shop --ignore=tests/slow",
        "Excludes `tests/slow` from test collection.",
        id="ignore",
    ),
    pytest.param(
        'pytest --cov=shop -k "not slow"',
        'Filters tests out by keyword (`-k "not slow"`).',
        id="keyword",
    ),
    pytest.param(
        'pytest --cov=shop -m "not integration"',
        'Filters tests out by marker (`-m "not integration"`).',
        id="marker",
    ),
    pytest.param(
        "pytest --cov=shop --deselect tests/test_cart.py::test_rounds_half_up",
        "Deselects `tests/test_cart.py::test_rounds_half_up` from the test run.",
        id="deselect",
    ),
    pytest.param(
        "pytest --cov=shop || true",
        "Test failures stop failing this command (`pytest ... || true`).",
        id="or-true",
    ),
]


@pytest.mark.parametrize(("command", "detail"), EXCLUSIONS)
def test_a_test_run_that_leaves_tests_out_is_reported_on_the_head_side(
    command: str, detail: str
) -> None:
    path = ".github/workflows/ci.yml"
    hits = detect(edit(path, CI, CI.replace("pytest --cov=shop", command)))

    assert [(hit.check, hit.file, hit.line, hit.side) for hit in hits] == [
        ("loosened_threshold", path, 5, "head")
    ]
    assert hits[0].detail == detail


def test_an_ignore_flag_for_a_linter_or_in_a_comment_is_not_an_exclusion() -> None:
    before = "lint:\n\truff check src tests\n"
    linter = before.replace("ruff check src tests", "ruff check --ignore E501 src tests")
    comment = CI.replace(
        "pytest --cov=shop", "pytest --cov=shop  # add --ignore=tests/slow if CI is slow"
    )

    assert detect(edit("Makefile", before, linter)) == []
    assert detect(edit(".github/workflows/ci.yml", CI, comment)) == []


def test_an_exclusion_that_only_moves_is_not_new() -> None:
    before = CI.replace("pytest --cov=shop", "pytest --cov=shop --ignore=tests/slow")
    moved = edit(".github/workflows/ci.yml", before, CI) + edit(
        "pyproject.toml",
        PYPROJECT,
        PYPROJECT.replace("--cov-fail-under=90", "--cov-fail-under=90 --ignore=tests/slow"),
    )

    assert detect(moved) == []


def test_a_conftest_that_stops_collecting_a_test_file_is_reported() -> None:
    before = 'collect_ignore = ["setup.py"]\n'
    after = 'collect_ignore = ["setup.py", "tests/test_legacy.py"]\n'

    hits = detect(edit("conftest.py", before, after))

    assert [(hit.check, hit.line, hit.side, hit.detail) for hit in hits] == [
        (
            "loosened_threshold",
            1,
            "head",
            "Adds `tests/test_legacy.py` to `collect_ignore`, so matching tests stop running.",
        )
    ]
    # A root conftest ignoring setup.py is housekeeping.
    assert detect(edit("conftest.py", "", before)) == []


JEST_CONFIG = """\
module.exports = {
  testPathIgnorePatterns: ["/node_modules/"],
  coverageThreshold: {
    global: { branches: 80, lines: 90 },
  },
};
"""


def test_a_lowered_jest_threshold_is_reported() -> None:
    after = JEST_CONFIG.replace("lines: 90", "lines: 70")

    hits = detect(edit("jest.config.js", JEST_CONFIG, after))

    assert [(hit.check, hit.line, hit.side) for hit in hits] == [("loosened_threshold", 4, "base")]
    assert hits[0].detail == "Loosens the coverage threshold: `lines` 90 → 70."


def test_a_new_jest_ignore_pattern_is_reported_and_node_modules_is_not() -> None:
    ignored = JEST_CONFIG.replace('["/node_modules/"]', '["/node_modules/", "/e2e/"]')
    default = JEST_CONFIG.replace('  testPathIgnorePatterns: ["/node_modules/"],\n', "")

    hits = detect(edit("jest.config.js", JEST_CONFIG, ignored))

    assert [(hit.line, hit.side, hit.detail) for hit in hits] == [
        (2, "head", "Adds `/e2e/` to `testPathIgnorePatterns`, so matching tests stop running.")
    ]
    assert detect(edit("jest.config.js", default, JEST_CONFIG)) == []


def test_a_jest_threshold_raised_or_commented_is_not_loosened() -> None:
    raised = JEST_CONFIG.replace("lines: 90", "lines: 95")
    commented = JEST_CONFIG.replace(
        "    global: { branches: 80, lines: 90 },\n",
        "    // global: { branches: 60, lines: 60 },\n    global: { branches: 80, lines: 90 },\n",
    )

    assert detect(edit("jest.config.js", JEST_CONFIG, raised)) == []
    assert detect(edit("jest.config.js", JEST_CONFIG, commented)) == []


POM = """\
<project>
  <build>
    <plugins>
      <plugin>
        <artifactId>jacoco-maven-plugin</artifactId>
        <configuration>
          <rules>
            <rule>
              <limits>
                <limit>
                  <minimum>0.80</minimum>
                </limit>
              </limits>
            </rule>
          </rules>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
"""


def test_a_lowered_jacoco_minimum_is_reported_and_an_equivalent_one_is_not() -> None:
    lowered = POM.replace("<minimum>0.80</minimum>", "<minimum>0.70</minimum>")
    percent = POM.replace("<minimum>0.80</minimum>", "<minimum>80%</minimum>")

    hits = detect(edit("pom.xml", POM, lowered))

    assert [(hit.check, hit.line, hit.side) for hit in hits] == [("loosened_threshold", 11, "base")]
    assert hits[0].detail == "Loosens the coverage threshold: `<minimum>` 0.80 → 0.70."
    assert detect(edit("pom.xml", POM, percent)) == []


def test_skipping_tests_in_maven_is_reported_and_a_commented_skip_is_not() -> None:
    skipped = POM.replace(
        "        <configuration>\n          <rules>",
        "        <configuration>\n          <skipTests>true</skipTests>\n          <rules>",
    )
    commented = POM.replace(
        "        <configuration>\n          <rules>",
        "        <configuration>\n          <!--\n          <skipTests>true</skipTests>\n"
        "          -->\n          <rules>",
    )

    assert found(edit("pom.xml", POM, skipped)) == [("loosened_threshold", "pom.xml", 7, "head")]
    assert detect(edit("pom.xml", POM, commented)) == []


def test_a_gradle_test_filter_or_skipped_test_task_is_reported() -> None:
    before = """\
    test {
        useJUnitPlatform()
    }
    """
    filtered = """\
    test {
        useJUnitPlatform()
        filter {
            excludeTestsMatching "*IntegrationTest"
        }
    }
    """
    ci_before = "      - run: ./gradlew build\n"
    ci_after = "      - run: ./gradlew build -x test\n"

    assert found(edit("build.gradle", before, filtered)) == [
        ("loosened_threshold", "build.gradle", 4, "head")
    ]
    assert found(edit(".gitlab-ci.yml", ci_before, ci_after)) == [
        ("loosened_threshold", ".gitlab-ci.yml", 1, "head")
    ]


def test_maven_skip_flags_on_a_command_line_are_reported() -> None:
    hits = detect(
        edit(".gitlab-ci.yml", "script: mvn verify\n", "script: mvn -DskipTests verify\n")
    )

    assert [(hit.check, hit.side, hit.detail) for hit in hits] == [
        (
            "loosened_threshold",
            "head",
            "Tests or their checks stop running or stop failing the build (`-DskipTests`).",
        )
    ]


def test_go_test_runs_that_skip_or_swallow_failures_are_reported() -> None:
    before = "test:\n\tgo test ./...\n"
    skip = before.replace("go test ./...", "go test ./... -skip TestSlow")
    swallowed = before.replace("go test ./...", "go test ./... || true")
    narrowed_by_run = before.replace("go test ./...", "go test -run TestTotal ./...")

    assert [hit.detail for hit in detect(edit("Makefile", before, skip))] == [
        "Skips tests matching `TestSlow` (`go test -skip`)."
    ]
    assert [hit.detail for hit in detect(edit("Makefile", before, swallowed))] == [
        "Test failures stop failing this command (`go test ... || true`)."
    ]
    assert detect(edit("Makefile", before, narrowed_by_run)) == []


def test_a_threshold_moved_to_another_file_and_lowered_is_loosened() -> None:
    diff = edit("pyproject.toml", PYPROJECT, PYPROJECT.replace("fail_under = 90\n", "")) + create(
        ".coveragerc", "[report]\nfail_under = 80\n"
    )

    assert [(hit.file, hit.line, hit.detail) for hit in detect(diff)] == [
        ("pyproject.toml", 6, "Loosens the coverage threshold: `fail_under` 90 → 80.")
    ]


def test_a_multi_line_testpaths_that_loses_an_entry_is_narrower() -> None:
    before = '[tool.pytest.ini_options]\ntestpaths = [\n    "tests",\n    "integration",\n]\n'
    after = before.replace('    "integration",\n', "")

    assert [(hit.line, hit.detail) for hit in detect(edit("pyproject.toml", before, after))] == [
        (4, "Narrows `testpaths`: `integration` is no longer collected.")
    ]


def test_a_commented_out_ini_option_is_not_a_setting() -> None:
    before = "[tool:pytest]\naddopts = --cov=shop\n"
    after = before + "; addopts = --cov=shop --ignore=tests/slow\n# testpaths = tests\n"

    assert detect(edit("setup.cfg", before, after)) == []


def test_a_conftest_among_the_tests_that_appends_to_collect_ignore_is_reported() -> None:
    before = "collect_ignore = []\n"
    after = before + 'collect_ignore.append("test_legacy.py")\n'

    assert found(edit("tests/conftest.py", before, after)) == [
        ("loosened_threshold", "tests/conftest.py", 2, "head")
    ]


def test_a_jest_setting_inside_a_block_comment_is_not_a_setting() -> None:
    after = JEST_CONFIG.replace(
        "  coverageThreshold: {\n",
        '  /*\n  testPathIgnorePatterns: ["/e2e/"],\n  */\n  coverageThreshold: {\n',
    )

    assert detect(edit("jest.config.js", JEST_CONFIG, after)) == []


def test_nyc_flags_on_a_test_script_are_thresholds() -> None:
    before = '{\n  "scripts": {\n    "test": "nyc --check-coverage --lines 90 mocha"\n  }\n}\n'
    after = before.replace("--lines 90", "--lines 80")

    assert [hit.detail for hit in detect(edit("package.json", before, after))] == [
        "Loosens the coverage threshold: `--lines` 90 → 80."
    ]


TEST_COMMAND_HOMES = [
    pytest.param(".gitlab-ci.yml", "test:\n  script: mvn verify\n", id="gitlab-ci"),
    pytest.param(
        ".github/workflows/ci.yml",
        "jobs:\n  test:\n    steps:\n      - run: mvn verify\n",
        id="github-actions",
    ),
    pytest.param("Jenkinsfile", "stage('test') {\n    sh 'mvn verify'\n}\n", id="jenkins"),
    pytest.param("Makefile", "test:\n\tmvn verify\n", id="make"),
    pytest.param("scripts/ci.sh", "set -e\nmvn verify\n", id="shell"),
]


@pytest.mark.parametrize(("path", "before"), TEST_COMMAND_HOMES)
def test_a_skipped_test_run_is_found_wherever_the_command_lives(path: str, before: str) -> None:
    after = before.replace("mvn verify", "mvn -DskipTests verify")

    assert [(hit.check, hit.file, hit.side) for hit in detect(edit(path, before, after))] == [
        ("loosened_threshold", path, "head")
    ]
    # The same words in documentation run nothing.
    assert detect(edit("docs/ci.md", before, after)) == []


# ── across checks ────────────────────────────────────────────────────────────


def test_hits_come_back_in_check_order_whatever_the_file_order() -> None:
    diff = (
        edit("pyproject.toml", PYPROJECT, PYPROJECT.replace("fail_under = 90", "fail_under = 50"))
        + edit(JS_PATH, JS, JS.replace('it("rounds half up"', 'it.skip("rounds half up"'))
        + edit(PY_PATH, PY, PY_HEAD)
    )

    assert [hit.check for hit in detect(diff)] == [
        "deleted_test",
        "skip_or_focus",
        "loosened_threshold",
    ]
    assert detect(diff) == detect(diff)


def test_a_hit_never_quotes_a_credential() -> None:
    token = "ghp_abcdefghijklmnopqrstuvwxyz012345"
    before = PY + f'\n\ndef test_client_uses_token():\n    assert client("{token}").ok\n'

    hits = detect(edit(PY_PATH, before, PY))

    assert [hit.check for hit in hits] == ["deleted_test"]
    assert token not in hits[0].text
    assert all(token not in hit.detail for hit in hits)
