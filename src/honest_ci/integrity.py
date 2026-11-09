"""Test-integrity checks: did this change weaken the tests that judge it?

The rule engine reads only the lines a change adds, and that is deliberate: a
review that reports what nobody touched stops being read. It leaves a blind
spot exactly where a change that needs a red build to turn green does its
work -- deleting the failing test, skipping it, dropping its assertions,
rewriting its expected value to whatever the code now returns, or lowering the
bar the suite has to clear. None of that is a suspicious *added* line. Here the
removed lines are the change, so reading them is the same "judge only what
changed" rule applied to the other side of the diff.

Everything is decided from the diff text, and the renames and withheld
deletions a forge reports beside it, alone: no workspace, no sandbox, no test
run, no model. The
regression gate needs Docker and a test command, and the default GitHub setup
leaves it off; these checks have to hold there too. It is also what lets
``detect`` stay a pure function that the benchmark imports on its own. The
renames are there because a forge's pure rename has no lines -- GitHub sends
no patch for it and GitLab an empty diff -- so the diff never names it, and
its two names are all there is to judge it by.

The five checks, in ``INTEGRITY_CHECKS`` order:

``deleted_test``
    A test stops existing -- its definition is removed (in Java, possibly just
    its ``@Test`` annotation), or a named case leaves a test table -- and
    nothing of that name is added back anywhere in the change. Names are
    matched across files, so moving a test or a whole file is not a deletion,
    and a definition renamed where it stands is a rename as long as more than
    half of its lines stay; a different test written where a removed one
    stood is a deletion. Renaming a test so the runner stops collecting it
    (``test_x`` -> ``_test_x``) is a deletion: nothing runs it any more. So
    is moving it, or its file, from where the runner collects tests by
    default to where it does not (``tests/test_x.py`` ->
    ``tests/disabled/x.py``): a test only counts as added back where it
    runs. A test file renamed that way is one hit for the whole file, since
    a forge's pure rename shows none of the tests in it. So is a test file a
    forge lists as deleted but sends none of the lines of -- too large,
    binary, cut short -- which the diff never names at all:
    ``unread_deletions`` reads those off the forge's file list.
``skip_or_focus``
    A newly added skip, expected-failure, todo or focus marker in a test file.
    Matched on code with comments and string contents removed, so a comment,
    or a test *named* "skips empty rows", is not a marker.
``fewer_assertions``
    A test file that stays loses more assertions than it gains. Assertions
    that went away with a deleted test are already that test's
    ``deleted_test`` hit and are not counted twice; assertions that moved
    verbatim, copies gathered from several files into one helper included,
    are not counted at all. An ``expect`` imported under another name is
    still an assertion, and so is a Go check that hands its error up
    (``if err != nil { return err }``).
``changed_expectation``
    An assertion, expected value or test-table row that keeps its exact shape
    and changes only a literal, in a change that also edits non-test code:
    what rewriting an expectation to match the code's new output looks like.
    One in a deleted test or case is already that ``deleted_test`` hit. A
    literal that only words a failure message or names a case
    (``t.Errorf("...")``, ``assert x, "..."``, a parametrize id) expects
    nothing. Lines are only paired within one test, and a line still there
    after the change, in the same test, was moved rather than rewritten.
``loosened_threshold``
    A coverage threshold lowered or removed, or tests excluded from the run
    (``--ignore``, ``-k "not ..."``, a narrower ``testpaths``,
    ``testPathIgnorePatterns``, ``-DskipTests``, ``|| true`` after a test
    command, ...), in build, CI and test-runner configuration.

A hit is a fact about the diff, worded neutrally. Removing a feature together
with its tests trips ``deleted_test`` exactly as removing a failing test does;
the hit says what went away and leaves the judgment to whoever reads it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import cache
from itertools import accumulate
from pathlib import PurePosixPath
from typing import Literal, get_args

from .diffs import FilePatch, file_patches, markdown_path
from .redaction import redact_text

IntegrityCheck = Literal[
    "deleted_test",
    "skip_or_focus",
    "fewer_assertions",
    "changed_expectation",
    "loosened_threshold",
]
INTEGRITY_CHECKS: tuple[IntegrityCheck, ...] = get_args(IntegrityCheck)

Side = Literal["base", "head"]

#: Minified bundles and embedded blobs are not test code anybody reads, and a
#: pathological line is where the scanning would cost the most.
MAX_SCANNED_LINE_CHARS = 2000
#: A run of changed lines this long is a rewrite, not an edit; pairing its
#: lines one by one would be quadratic work for no signal.
_MAX_PAIRED_BLOCK_LINES = 400
#: How far a multi-line assertion or expected value is followed. Brackets are
#: counted line by line, and the cap keeps one miscounted line from turning
#: the rest of a hunk into assertions.
_MAX_CONTINUATION_LINES = 50
#: The same for a test table, which legitimately runs much longer.
_MAX_TABLE_LINES = 500
_MAX_SHOWN_CHARS = 60


@dataclass(frozen=True)
class IntegrityHit:
    """One way a change weakened its tests, located on the side it happened.

    ``side`` says which file ``line`` counts in. A ``"base"`` hit is about
    something taken away or rewritten -- a removed test, a changed expected
    value -- so it points at the line as it stood before the change; the head
    no longer has it. A ``"head"`` hit is about something put in, such as a new
    ``@pytest.mark.skip``. ``file`` is the path on that same side, which only
    differs from the other side's for a rename.
    """

    check: IntegrityCheck
    file: str
    #: ``None`` for a hit about a whole file, which no line stands for: a test
    #: file renamed where the runner stops collecting it.
    line: int | None
    side: Side
    #: The line as it reads on ``side``, verbatim apart from secret redaction;
    #: empty when there is no line.
    text: str
    #: What happened, in neutral English: a hit is a fact, not an accusation.
    detail: str
    #: For a line the change rewrote rather than removed: what it became.
    replacement: str | None = None


def detect(diff: str, renames: Iterable[tuple[str, str]] = ()) -> list[IntegrityHit]:
    """Every integrity hit in a unified diff, in ``INTEGRITY_CHECKS`` order.

    ``renames`` are the files a forge reported as renamed, as ``(old, new)``
    names: ``ChangeRequest.renames``. A rename the diff has a patch for is read
    from the diff as well, so only the pure ones are missing without them.

    Pure: the diff and the renames are the only input, and nothing is read, run
    or sent.
    """

    patches = file_patches(diff)
    files = [_analyse(patch) for patch in patches]
    tests = [item for item in files if item.language is not None and item.test]
    configs = [item for item in files if item.config is not None]

    moved = [*renames]
    moved += [
        (patch.old_path, patch.new_path)
        for patch in patches
        if patch.old_path is not None
        and patch.new_path is not None
        and patch.old_path != patch.new_path
    ]
    renamed_out = _renamed_out_of_collection(moved)
    # A file's hit covers every test in it, the ones its patch shows leaving
    # included; a hit per test on top would count the same tests twice.
    covered = {hit.file for hit in renamed_out}
    deleted_hits, inside_deleted = _deleted_tests(tests)
    hits = [
        *renamed_out,
        *(hit for hit in deleted_hits if hit.file not in covered),
        *_skips(tests),
        *_fewer_assertions(tests, inside_deleted),
        *_loosened_thresholds(configs),
    ]
    # Rewriting an expected value only looks like fitting the test to the code
    # when the code changed too. A change that edits nothing but tests is
    # somebody fixing a test.
    if any(item.code for item in files):
        hits += _changed_expectations(tests, inside_deleted)
    return sorted(
        hits,
        # A whole file's hit comes before the lines in it.
        key=lambda hit: (INTEGRITY_CHECKS.index(hit.check), hit.file, hit.line or 0, hit.side),
    )


# ── reading test code ────────────────────────────────────────────────────────

_Language = Literal["python", "javascript", "java", "go"]

_NUMBER = (
    r"(?:0[xX][0-9a-fA-F_]+|0[oObB][0-7_]+"
    r"|(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d+)?)[a-zA-Z]*"
)
_QUOTED = r"\"(?:\\.|[^\"\\])*(?:\"|$)|'(?:\\.|[^'\\])*(?:'|$)"
_C_COMMENT = r"(?P<comment>//.*|/\*.*?(?:\*/|$))"
_C_TAIL = rf"|(?P<number>{_NUMBER})|(?P<word>[A-Za-z_$][\w$]*)|(?P<space>\s+)|(?P<other>.)"

#: One tokenizer per language. A string or comment still open at the end of a
#: line is carried to the next one by ``_lex``.
_TOKENS: dict[_Language, re.Pattern[str]] = {
    "python": re.compile(
        r"(?P<comment>#.*)"
        r"|(?P<string>[rRbBuUfF]{0,2}"
        r"(?:\"\"\"(?:\\.|.)*?(?:\"\"\"|$)|'''(?:\\.|.)*?(?:'''|$)|" + _QUOTED + "))"
        rf"|(?P<number>{_NUMBER})|(?P<word>[^\W\d]\w*)|(?P<space>\s+)|(?P<other>.)"
    ),
    "javascript": re.compile(
        _C_COMMENT + r"|(?P<string>`(?:\\.|[^`\\])*(?:`|$)|" + _QUOTED + ")" + _C_TAIL
    ),
    "java": re.compile(_C_COMMENT + r"|(?P<string>\"\"\".*$|" + _QUOTED + ")" + _C_TAIL),
    "go": re.compile(_C_COMMENT + r"|(?P<string>`[^`]*(?:`|$)|" + _QUOTED + ")" + _C_TAIL),
}
_KEYWORD_LITERALS: dict[_Language, frozenset[str]] = {
    "python": frozenset({"True", "False", "None"}),
    "javascript": frozenset({"true", "false", "null", "undefined"}),
    "java": frozenset({"true", "false", "null"}),
    "go": frozenset({"true", "false", "nil"}),
}
#: What may stand before a ``-`` for it to be a sign rather than a subtraction:
#: ``== -1`` changing to ``== 1`` is a changed literal, not a changed shape.
_SIGN_CONTEXT = frozenset("([{,=:<>!+-*/%&|^~?") | {"return", "in", "is", "not", "and", "or"}
_LITERAL = "\x00"
_DECIMAL = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_PREFIXED = re.compile(r"0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+")
#: A JavaScript regular expression literal, from its opening slash. It cannot
#: span lines, so one without its closing slash on the line is a division.
_JS_REGEX = re.compile(r"/(?![/*])(?:\\.|\[(?:\\.|[^\]\\])*\]|[^/\\\[])+/[A-Za-z]*")
#: What may stand before a ``/`` for it to open a regular expression rather
#: than divide. ``)``, ``]`` and ``}`` are left out, and so is ``<``, for the
#: closing tags of JSX: a regular expression after any of them is rare, and
#: one missed there is read as plain code, as every pattern used to be.
_JS_REGEX_AFTER = frozenset("(,=:[!&|?{;+-*%^~") | {
    "return",
    "typeof",
    "case",
    "do",
    "else",
    "in",
    "of",
    "new",
    "delete",
    "void",
    "throw",
    "instanceof",
    "yield",
    "await",
}


@dataclass(frozen=True)
class _Reading:
    """What a line of code says once comments and literal values are set aside."""

    #: Comments removed and string contents emptied: what patterns run on, so
    #: a comment or a string that mentions ``skip`` is not a skip.
    code: str
    #: The line with every literal replaced by one placeholder and whitespace
    #: removed. Two lines with the same shape differ at most in literals.
    shape: str
    #: Each literal as written, and a value to compare it by, so that ``'a'``
    #: and ``"a"`` or ``1.0`` and ``1.00`` count as the same literal.
    literals: tuple[tuple[str, tuple[str, object]], ...] = ()
    #: What ``shape`` is made of, one token each: the brackets, commas and
    #: names a statement is followed through, the literals as placeholders.
    tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Open:
    """A string or comment that a line leaves open for the next one."""

    closer: str
    comment: bool
    #: Whether a backslash inside it escapes the character after it.
    escapes: bool = False


def _lex(rows: Iterable[_Row], language: _Language) -> None:
    """Read one side of a hunk top to bottom, carrying open strings and comments.

    A hunk starts wherever the diff says, so a string already open above its
    first line cannot be seen; rows are read as code until something visibly
    opens. That is the safe direction: a docstring line mistaken for code is
    at worst one more line to match, never a hidden one.
    """

    still_open: _Open | None = None
    for row in rows:
        row.reading, still_open = _read(row.text, language, still_open)


def _read(
    text: str, language: _Language, still_open: _Open | None
) -> tuple[_Reading, _Open | None]:
    if len(text) > MAX_SCANNED_LINE_CHARS:
        return _Reading(code="", shape=text), None
    code: list[str] = []
    shape: list[str] = []
    literals: list[tuple[str, tuple[str, object]]] = []
    start = 0
    if still_open is not None:
        end = _closing(text, still_open)
        if not still_open.comment:
            inside = text if end is None else text[: end - len(still_open.closer)]
            shape.append(_LITERAL)
            literals.append((inside, ("string", inside)))
        if end is None:
            return _Reading("", "".join(shape), tuple(literals), tuple(shape)), still_open
        start, still_open = end, None

    keywords = _KEYWORD_LITERALS[language]
    tokens = _TOKENS[language]
    position = start
    while position < len(text):
        regex = (
            _JS_REGEX.match(text, position)
            if language == "javascript"
            and text[position] == "/"
            and (not shape or shape[-1] in _JS_REGEX_AFTER or shape[-2:] == ["=", ">"])
            else None
        )
        if regex is not None:
            # Read as one literal, so that the brackets and quotes in a
            # pattern are neither counted nor taken for strings, and a
            # changed pattern is a changed value, as a changed string is.
            value = regex.group()
            code.append("//" + value[value.rindex("/") + 1 :])
            shape.append(_LITERAL)
            literals.append((value, ("regex", value)))
            position = regex.end()
            continue
        match = tokens.match(text, position)
        assert match is not None  # ``other`` matches any one character
        position = match.end()
        kind, value = match.lastgroup, match.group()
        if kind == "space":
            code.append(value)
        elif kind == "comment":
            still_open = _left_open(value, language)
        elif kind == "string":
            code.append(_emptied(value))
            shape.append(_LITERAL)
            literals.append((value, ("string", _string_value(value))))
            still_open = _left_open(value, language)
        elif kind == "number":
            code.append(value)
            number = _number_value(value)
            if shape[-1:] == ["-"] and (len(shape) == 1 or shape[-2] in _SIGN_CONTEXT):
                shape.pop()
                value = f"-{value}"
                number = -number if isinstance(number, int | float) else value
            shape.append(_LITERAL)
            literals.append((value, ("number", number)))
        elif kind == "word" and value in keywords:
            code.append(value)
            shape.append(_LITERAL)
            literals.append((value, ("keyword", value)))
        else:
            code.append(value)
            shape.append(value)
    return _Reading("".join(code), "".join(shape), tuple(literals), tuple(shape)), still_open


def _left_open(token: str, language: _Language) -> _Open | None:
    """What a token that ran to the end of its line leaves open, if anything."""

    if token.startswith("/*"):
        return None if len(token) >= 4 and token.endswith("*/") else _Open("*/", comment=True)
    body = token.lstrip("rRbBuUfF")
    if language == "python" and body[:3] in ('"""', "'''"):
        if len(body) >= 6 and body.endswith(body[:3]):
            return None
        raw = "r" in token[: len(token) - len(body)].lower()
        return _Open(body[:3], comment=False, escapes=not raw)
    if language == "java" and body.startswith('"""'):
        # A text block's opening line cannot hold its closing delimiter.
        return _Open('"""', comment=False, escapes=True)
    if language in ("javascript", "go") and body.startswith("`"):
        if len(body) >= 2 and body.endswith("`"):
            return None
        return _Open("`", comment=False, escapes=language == "javascript")
    return None


def _closing(text: str, still_open: _Open) -> int | None:
    """Just past where an open string or comment ends on this line, if it does."""

    index = 0
    while index < len(text):
        if still_open.escapes and text[index] == "\\":
            index += 2
        elif text.startswith(still_open.closer, index):
            return index + len(still_open.closer)
        else:
            index += 1
    return None


def _emptied(string: str) -> str:
    body = string.lstrip("rRbBuUfF")
    quote = body[:3] if body[:3] in ('"""', "'''") else body[:1]
    return string[: len(string) - len(body)] + quote + quote


def _string_value(string: str) -> str:
    body = string.lstrip("rRbBuUfF")
    quote = body[:3] if body[:3] in ('"""', "'''") else body[:1]
    content = body[len(quote) :]
    if len(content) >= len(quote) and content.endswith(quote):
        content = content[: -len(quote)]
    return content


def _number_value(raw: str) -> object:
    body = raw.replace("_", "")
    try:
        prefixed = _PREFIXED.match(body)
        if prefixed is not None:
            return int(prefixed.group(), 0)
        decimal = _DECIMAL.match(body)
        if decimal is None:
            return raw
        text = decimal.group()
        return float(text) if any(char in text for char in ".eE") else int(text)
    except ValueError:
        return raw


def _balance(code: str) -> int:
    return sum(code.count(char) for char in "([{") - sum(code.count(char) for char in ")]}")


@dataclass
class _Frame:
    """An open bracket of a statement, as far as the statement has been read."""

    #: The dotted name the bracket calls (``assert.equal``, ``.toBe`` for a
    #: method of whatever came before), ``f()`` for a call on what the call
    #: to ``f`` returned, and empty for a bracket that calls nothing.
    callee: str
    #: Which argument the reading is in, counting from 0.
    argument: int = 0
    #: The name of an argument passed as ``name=value``.
    keyword: str = ""
    #: Whether nothing of the current argument has been read yet.
    fresh: bool = True


class _Walk:
    """Follows one statement across its rows, bracket by bracket.

    It is what tells the literals of an assertion apart: the expected value,
    and the message printed when it fails, are both literals of the same
    statement, and only where each one stands says which is which.
    """

    def __init__(self, language: _Language, aliases: dict[str, str] | None = None) -> None:
        self.language = language
        self.aliases = aliases or {}
        #: The statement itself, then each bracket still open in it.
        self.frames = [_Frame("")]
        self.started = False
        #: The callee of the bracket closed last, while nothing has followed.
        self.closed: str | None = None
        #: Whether a ``.each(...)`` table's own brackets have closed.
        self.each_closed = False

    @property
    def depth(self) -> int:
        return len(self.frames) - 1

    def row(self, row: _Row) -> frozenset[int]:
        """Read one more row; the positions of its literals that expect nothing."""

        tokens = row.reading.tokens if row.reading is not None else ()
        unstated: set[int] = set()
        literal = 0
        for index, token in enumerate(tokens):
            if not self.started:
                self.started = True
                if self.language == "python" and token == "assert":
                    self.frames[0].callee = "assert"
                    continue
            closed, self.closed = self.closed, None
            frame = self.frames[-1]
            if token == _LITERAL:
                if _states_nothing(self.frames, self.language):
                    unstated.add(literal)
                literal += 1
                frame.fresh = False
            elif token in ("(", "[", "{"):
                callee = ""
                if token == "(":
                    callee = self._callee(tokens, index)
                    if not callee and closed is not None and index and tokens[index - 1] == ")":
                        callee = f"{closed}()"
                frame.fresh = False
                self.frames.append(_Frame(callee))
            elif token in (")", "]", "}"):
                if self.depth:
                    done = self.frames.pop()
                    self.closed = done.callee
                    self.each_closed = self.each_closed or done.callee.endswith(".each")
            elif token == ",":
                frame.argument += 1
                frame.keyword = ""
                frame.fresh = True
            elif (
                frame.fresh
                and _is_word(token)
                and tokens[index + 1 : index + 2] == ("=",)
                and tokens[index + 2 : index + 3] != ("=",)
            ):
                frame.keyword = token
                frame.fresh = False
            else:
                frame.fresh = False
        return frozenset(unstated)

    def _callee(self, tokens: Sequence[str], index: int) -> str:
        parts: list[str] = []
        at = index - 1
        while at >= 0:
            token = tokens[at]
            if token == "." and (not parts or parts[-1] != "."):
                parts.append(token)
            elif _is_word(token) and (not parts or parts[-1] == "."):
                parts.append(token)
            else:
                break
            at -= 1
        callee = "".join(reversed(parts))
        head, dot, rest = callee.partition(".")
        return self.aliases.get(head, head) + dot + rest


def _is_word(token: str) -> bool:
    return token[:1].isalpha() or token[:1] in ("_", "$")


# Where a literal of an assertion is a message or a name rather than a value
# the test expects. Only what each framework's signature says for certain:
# JUnit puts the message first in version 4 and last in version 5, and
# testify's messages trail a different number of values per function, so
# neither is guessed at, and a changed literal there is still reported.

#: Calls that only report a failure: everything passed to them is a message.
#: In Go the comparison is the ``if`` above ``t.Errorf``, and is read there.
_REPORTS: dict[_Language, re.Pattern[str]] = {
    "python": re.compile(r"(?:pytest|self|cls)\.fail"),
    "javascript": re.compile(r"(?:assert\.)?fail"),
    "java": re.compile(r"(?:Assertions\.|Assert\.)?fail"),
    "go": re.compile(r"(?:t|b|tb|f)?\.(?:Errorf?|Fatalf?|Logf?)"),
}
#: From which argument on a call's arguments are its failure message.
_MESSAGE_FROM: dict[_Language, dict[str, int]] = {
    "python": {
        # unittest's: self.assertEqual(first, second, msg).
        **dict.fromkeys(("assertTrue", "assertFalse", "assertIsNone", "assertIsNotNone"), 1),
        **dict.fromkeys(
            (
                *("assertEqual", "assertNotEqual", "assertIs", "assertIsNot", "assertIn"),
                *("assertNotIn", "assertIsInstance", "assertNotIsInstance", "assertGreater"),
                *("assertGreaterEqual", "assertLess", "assertLessEqual", "assertRegex"),
                *("assertNotRegex", "assertCountEqual", "assertMultiLineEqual"),
                *("assertSequenceEqual", "assertListEqual", "assertTupleEqual"),
                *("assertSetEqual", "assertDictEqual"),
            ),
            2,
        ),
        # The third is ``places``, which is a tolerance, not a message.
        **dict.fromkeys(("assertAlmostEqual", "assertNotAlmostEqual"), 3),
    },
    "javascript": {
        # Node's assert, whose signatures chai's assert shares.
        "assert": 1,
        "assert.ok": 1,
        **{
            f"assert.{name}": 2
            for name in (
                *("equal", "notEqual", "strictEqual", "notStrictEqual", "deepEqual"),
                *("notDeepEqual", "deepStrictEqual", "notDeepStrictEqual", "match"),
                "doesNotMatch",
            )
        },
        # expect(value, message), in Vitest, Playwright and chai.
        "expect": 1,
    },
    "java": {},
    # Gomega: Expect(x).To(Equal(3), "description").
    "go": {f".{name}": 1 for name in ("To", "ToNot", "NotTo", "Should", "ShouldNot")},
}
_MESSAGE_KEYWORDS = frozenset({"msg", "err_msg"})


def _states_nothing(frames: Sequence[_Frame], language: _Language) -> bool:
    """Whether a literal read inside ``frames`` is a message or a name.

    A test that rewords the message it fails with, or renames a case, has not
    changed what it expects.
    """

    for depth, frame in enumerate(frames):
        callee = frame.callee
        if depth == 0:
            # ``assert total == 3, "message"``
            if callee == "assert" and frame.argument > 0:
                return True
            continue
        if _REPORTS[language].fullmatch(callee) or frame.keyword in _MESSAGE_KEYWORDS:
            return True
        method = callee.rpartition(".")[2] if callee.startswith(("self.", "cls.")) else callee
        start = _MESSAGE_FROM[language].get(method)
        if start is not None and frame.argument >= start:
            return True
        if _names(frame, language):
            return True
    return False


def _names(frame: _Frame, language: _Language) -> bool:
    """Whether ``frame``'s current argument names the cases of a test table."""

    if language == "python":
        if frame.callee == "pytest.mark.parametrize":
            # The argument names, and the ids the runner prints.
            return frame.argument == 0 or frame.keyword == "ids"
        return (frame.callee == "pytest.fixture" and frame.keyword == "ids") or (
            frame.callee == "pytest.param" and frame.keyword == "id"
        )
    # it.each([...])("title %s", ...): the title.
    return language == "javascript" and frame.callee.endswith(".each()") and frame.argument == 0


# ── which files are which ────────────────────────────────────────────────────

_JS_SUFFIXES = frozenset({".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"})
#: Source files whose change counts as "the code under test changed" for
#: ``changed_expectation``. Broad on purpose: the test may be Python and the
#: code it pins down a Rust extension.
_CODE_SUFFIXES = (
    frozenset({".py", ".pyi", ".pyx", ".java", ".kt", ".kts", ".scala", ".groovy", ".go"})
    | {".rs", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".cs", ".fs", ".rb", ".php", ".swift"}
    | {".m", ".mm", ".dart", ".vue", ".svelte", ".ex", ".exs", ".erl", ".clj", ".lua", ".r"}
    | {".jl", ".sql"}
    | _JS_SUFFIXES
)
_TEST_DIRS = frozenset({"test", "tests", "__tests__", "spec", "specs", "testing", "e2e"})


def _language(path: str) -> _Language | None:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in _JS_SUFFIXES:
        return "javascript"
    if suffix == ".java":
        return "java"
    if suffix == ".go":
        return "go"
    return None


def _is_test_path(path: str) -> bool:
    """Where each toolchain looks for tests by default, and nothing more."""

    pure = PurePosixPath(path)
    name, directories = pure.name, pure.parts[:-1]
    language = _language(path)
    if language == "python":
        # conftest.py configures collection, and a skip it adds is a skip like
        # any other; it is read as both.
        return (
            name.startswith("test_")
            or name.endswith("_test.py")
            or name in {"test.py", "tests.py", "conftest.py"}
            or any(part in {"test", "tests"} for part in directories)
        )
    if language == "javascript":
        return any(part in {"test", "spec", "cy"} for part in name.split(".")[1:-1]) or any(
            part in {"__tests__", "test", "tests", "e2e"} for part in directories
        )
    if language == "java":
        in_test_tree = any(
            directories[index : index + 2] == ("src", "test") for index in range(len(directories))
        )
        named = re.fullmatch(r"Test\w*|\w*(?:Tests?|IT|TestCase)", pure.stem) is not None
        return in_test_tree or named
    if language == "go":
        return name.endswith("_test.go")
    # A test in another language is still not "the code under test changed".
    stem = pure.stem
    return (
        any(part.lower() in _TEST_DIRS for part in directories)
        or re.search(r"(?:^|[._-])(?:tests?|specs?)(?:$|[._-])", stem.lower()) is not None
        or re.search(r"(?:Tests?|Spec)$", stem) is not None
    )


#: Maven Surefire's and Failsafe's default class names.
_JAVA_COLLECTED = re.compile(r"(?:Test|IT)\w*|\w*(?:Tests?|TestCase|IT|ITCase)")


def _collected(path: str) -> bool:
    """Whether the test runner picks up tests in ``path`` without being told to.

    Narrower than ``_is_test_path``, which also takes in the helpers, fixtures
    and conftest files that sit beside the tests: a test moved into one of
    those is still test code, and nothing runs it. Each toolchain's defaults
    and nothing more, and a hit built on this says "by default", so a project
    that configured its runner to collect more can read it as the question it
    is.
    """

    if not _is_test_path(path):
        return False
    pure = PurePosixPath(path)
    name, directories = pure.name, pure.parts[:-1]
    language = _language(path)
    if language == "python":
        # pytest's file patterns, and the module names unittest and Django
        # discover; pytest does not look inside hidden directories.
        return (
            name.startswith("test_") or name.endswith("_test.py") or name in {"test.py", "tests.py"}
        ) and not any(part.startswith(".") for part in directories)
    if language == "javascript":
        # Jest, Vitest, Playwright and Cypress go by the name; Jest also runs
        # everything under __tests__, and Mocha everything directly in test/.
        return "node_modules" not in directories and (
            any(part in {"test", "spec", "cy"} for part in name.split(".")[1:-1])
            or "__tests__" in directories
            or directories[-1:] == ("test",)
        )
    if language == "java":
        # Surefire and Failsafe go by the class name alone. Gradle finds every
        # test class in the test source set, these names included, but Maven is
        # the build that runs here, and a Gradle project keeps these names too.
        return _JAVA_COLLECTED.fullmatch(pure.stem) is not None
    if language == "go":
        # The go tool skips files and directories whose names start with "."
        # or "_", and directories named testdata.
        return not any(part.startswith((".", "_")) or part == "testdata" for part in pure.parts)
    # A runner whose defaults are not written down here: nothing is claimed
    # about what it collects, so nothing is read as leaving it either.
    return False


@dataclass(frozen=True)
class _Config:
    """How to read one build, CI or test-runner configuration file."""

    #: Comment syntax: ``#``, ``ini`` (``#`` and ``;``), ``//``, ``xml`` or none.
    comments: str
    #: Whose settings it can carry, beyond command lines.
    tools: frozenset[str] = frozenset()
    #: An INI value continues on the indented lines below its key.
    ini: bool = False


_JS_CONFIG_PREFIXES = (
    "jest.config.",
    "vitest.config.",
    "vite.config.",
    "vitest.workspace.",
    ".nycrc",
    ".c8rc",
    "karma.conf.",
)
_CI_DIRECTORIES = (".github/workflows/", ".circleci/", ".buildkite/", ".gitlab/")
_CI_FILES = frozenset(
    {".gitlab-ci.yml", ".travis.yml", "azure-pipelines.yml", "bitbucket-pipelines.yml"}
    | {"taskfile.yml", "taskfile.yaml"}
)


def _config(path: str) -> _Config | None:
    """Files that decide which tests run and what coverage has to be reached."""

    pure = PurePosixPath(path)
    name = pure.name.lower()
    suffix = pure.suffix.lower()
    if name == "pyproject.toml":
        return _Config("#", frozenset({"pytest", "coverage"}))
    if name in {"setup.cfg", "tox.ini", "pytest.ini"}:
        return _Config("ini", frozenset({"pytest", "coverage"}), ini=True)
    if name == ".coveragerc":
        return _Config("ini", frozenset({"coverage"}), ini=True)
    if name == "conftest.py":
        return _Config("#", frozenset({"conftest"}))
    if name == "noxfile.py":
        return _Config("#")
    if name == "package.json" or name.startswith(_JS_CONFIG_PREFIXES):
        comments = "//" if suffix in _JS_SUFFIXES else "#" if suffix in {".yml", ".yaml"} else ""
        return _Config(comments, frozenset({"jest"}))
    if name == "pom.xml":
        return _Config("xml", frozenset({"maven"}))
    if name.endswith((".gradle", ".gradle.kts")):
        return _Config("//", frozenset({"gradle"}))
    if name == "jenkinsfile":
        return _Config("//")
    if name in {"makefile", "gnumakefile", "justfile"} or suffix in {".mk", ".sh", ".bash"}:
        return _Config("#")
    if name in _CI_FILES or (
        suffix in {".yml", ".yaml"} and pure.as_posix().startswith(_CI_DIRECTORIES)
    ):
        return _Config("#")
    return None


def _uncommented(text: str, comments: str, inside: bool) -> tuple[str, bool]:
    """A configuration line with its comments removed and its strings intact.

    Also returns whether a block comment is still open after the line.
    """

    if len(text) > MAX_SCANNED_LINE_CHARS:
        return "", False
    if comments == "xml":
        kept: list[str] = []
        rest = text
        while rest:
            if inside:
                end = rest.find("-->")
                if end < 0:
                    break
                rest, inside = rest[end + 3 :], False
            else:
                start = rest.find("<!--")
                if start < 0:
                    kept.append(rest)
                    break
                kept.append(rest[:start])
                rest, inside = rest[start + 4 :], True
        return "".join(kept), inside
    if comments == "ini" and text.lstrip().startswith((";", "#")):
        return "", False

    kept = []
    quote: str | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if inside:
            if text.startswith("*/", index):
                inside = False
                index += 1
        elif quote is not None:
            kept.append(char)
            if char == "\\" and index + 1 < len(text):
                index += 1
                kept.append(text[index])
            elif char == quote:
                quote = None
        elif comments == "//" and text.startswith("/*", index):
            inside = True
            index += 1
        elif comments == "//" and text.startswith("//", index):
            break
        elif (
            comments in {"#", "ini"}
            and char == "#"
            # ``https://x#frag`` and ``${#list}`` are not comments.
            and (index == 0 or text[index - 1].isspace())
        ):
            break
        else:
            if char in "\"'" or (comments == "//" and char == "`"):
                quote = char
            kept.append(char)
        index += 1
    return "".join(kept), inside


# ── one file, as both sides read it ──────────────────────────────────────────


@dataclass(eq=False)
class _Row:
    """One line of a hunk as one side of the change reads it.

    Compared by identity: a removed row, the block it sits in and the test it
    belongs to are all looked up by the row itself.
    """

    kind: str
    number: int
    text: str
    reading: _Reading | None = None
    #: The line without its comments, for configuration files.
    setting: str = ""
    #: Inside a test table: ``@pytest.mark.parametrize``, a Go ``[]struct``...
    in_table: bool = False
    #: The positions of the literals that name a table's cases or parameters
    #: rather than hold its values.
    names: frozenset[int] = frozenset()


@dataclass(frozen=True)
class _Block:
    """A run of changed lines with no context line between them."""

    removed: tuple[_Row, ...]
    added: tuple[_Row, ...]


@dataclass
class _File:
    old_path: str | None
    new_path: str | None
    language: _Language | None = None
    test: bool = False
    config: _Config | None = None
    #: Non-test, non-configuration source: "the code under test changed".
    code: bool = False
    #: Names the file imports JavaScript's ``expect`` or ``assert`` under, as
    #: ``{alias: name}``: ``import { expect as detoxExpect } from "detox"``.
    aliases: dict[str, str] = field(default_factory=dict)
    #: Per hunk, the rows the base file has (context and removed) ...
    base: list[list[_Row]] = field(default_factory=list)
    #: ... and the rows the head file has (context and added), in order.
    head: list[list[_Row]] = field(default_factory=list)
    blocks: list[_Block] = field(default_factory=list)

    @property
    def base_path(self) -> str:
        return self.old_path or self.new_path or ""

    @property
    def head_path(self) -> str:
        return self.new_path or self.old_path or ""


def _analyse(patch: FilePatch) -> _File:
    item = _File(patch.old_path, patch.new_path)
    paths = [path for path in (patch.new_path, patch.old_path) if path is not None]
    if not paths:
        return item
    item.language = _language(paths[0])
    item.config = _config(paths[0])
    item.test = any(_is_test_path(path) for path in paths)
    item.code = (
        item.config is None
        and not item.test
        and PurePosixPath(paths[0]).suffix.lower() in _CODE_SUFFIXES
    )
    language = item.language if item.test else None
    if language is None and item.config is None:
        return item

    for hunk in patch.hunks:
        base: list[_Row] = []
        head: list[_Row] = []
        removed: list[_Row] = []
        added: list[_Row] = []
        for line in hunk:
            if line.kind == " " and line.old is not None and line.new is not None:
                if removed or added:
                    item.blocks.append(_Block(tuple(removed), tuple(added)))
                    removed, added = [], []
                base.append(_Row(" ", line.old, line.text))
                head.append(_Row(" ", line.new, line.text))
            elif line.kind == "-" and line.old is not None:
                row = _Row("-", line.old, line.text)
                base.append(row)
                removed.append(row)
            elif line.kind == "+" and line.new is not None:
                row = _Row("+", line.new, line.text)
                head.append(row)
                added.append(row)
        if removed or added:
            item.blocks.append(_Block(tuple(removed), tuple(added)))
        for view in (base, head):
            if language is not None:
                _lex(view, language)
                _mark_tables(view, language)
            if item.config is not None:
                inside = False
                for row in view:
                    row.setting, inside = _uncommented(row.text, item.config.comments, inside)
        item.base.append(base)
        item.head.append(head)
    if language == "javascript":
        item.aliases = {
            alias: name
            for rows in (*item.base, *item.head)
            for row in rows
            for pattern in _JS_ALIASES
            for name, alias in pattern.findall(_code(row))
        }
    return item


#: ``import { expect as detoxExpect } from "detox"`` and
#: ``const { assert: check } = require("chai")``.
_JS_ALIASES = (
    re.compile(r"\b(expect|assert)\s+as\s+([A-Za-z_$][\w$]*)"),
    re.compile(r"\b(expect|assert)\s*:\s*([A-Za-z_$][\w$]*)(?=[^}]*\}\s*=\s*require\b)"),
)


_TABLE_START: dict[_Language, re.Pattern[str]] = {
    "python": re.compile(r"^\s*@pytest\.mark\.parametrize\s*\("),
    "javascript": re.compile(r"^\s*(?:it|test|describe)(?:\.\w+)*?\.each\s*\("),
    "java": re.compile(r"^\s*@(?:CsvSource|ValueSource)\s*\("),
    "go": re.compile(r"(?:\[\]|map\[[^\]]*\])struct\s*\{"),
}


def _mark_tables(view: Sequence[_Row], language: _Language) -> None:
    """Flag the rows of tables that feed one test many cases.

    A ``.each([...])`` table ends where its own call closes, not where the
    statement does: the test body passed to the call that follows is not part
    of it, and only its title is read as one of the table's names.
    """

    walk: _Walk | None = None
    span = 0
    for row in view:
        if walk is None:
            if not _TABLE_START[language].search(_code(row)):
                continue
            walk, span = _Walk(language), 0
        row.in_table = True
        row.names = walk.row(row)
        span += 1
        if walk.depth == 0 or walk.each_closed or span >= _MAX_TABLE_LINES:
            walk = None


def _code(row: _Row) -> str:
    return row.reading.code if row.reading is not None else ""


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _shown(text: str) -> str:
    """Text quoted inside a sentence: one line, bounded, safe in backticks."""

    text = " ".join(text.split()).replace("`", "'")
    return text if len(text) <= _MAX_SHOWN_CHARS else text[: _MAX_SHOWN_CHARS - 1] + "…"


def _hit(
    check: IntegrityCheck,
    file: str,
    row: _Row,
    side: Side,
    detail: str,
    replacement: str | None = None,
) -> IntegrityHit:
    # Redacted here, not by each consumer: a hit quotes test code verbatim, and
    # test code is where fixtures keep things that look like credentials.
    return IntegrityHit(
        check=check,
        file=file,
        line=row.number,
        side=side,
        text=redact_text(row.text),
        detail=redact_text(detail),
        replacement=redact_text(replacement) if replacement is not None else None,
    )


# ── a. deleted_test ──────────────────────────────────────────────────────────


@dataclass(eq=False)
class _TestDef:
    """A test, or a named case of a test table, as one side of a hunk shows it."""

    name: str
    #: The row that names it.
    decl: _Row
    #: Every row that makes it a test: the declaration, and for Java the
    #: annotation that marks it.
    rows: tuple[_Row, ...]
    #: The side of the hunk it was read from.
    view: Sequence[_Row]
    case: bool = False
    #: For Java, the annotation as written (``@Test``).
    annotation: str | None = None


_PY_DEF = re.compile(r"^\s*(?:async\s+)?def\s+(?P<name>test\w*)\s*\(")
_PY_DECORATOR = re.compile(r"^\s*@")
_PY_FIXTURE = re.compile(r"^\s*@(?:pytest\.)?fixture\b")
_JS_TEST_CALL = re.compile(r"^\s*(?:it|test|specify)\b")
_JS_TEST = re.compile(
    r"^\s*(?:it|test|specify)"
    r"(?:\.(?:only|skip|todo|concurrent|failing|fails|fixme|fail|sequential)"
    r"|\.(?:skipIf|runIf)\([^)]*\)|\.each\s*(?:\(.*?\)|`[^`]*`))*"
    r"\s*\(\s*(?P<quote>['\"`])(?P<name>(?:\\.|(?!(?P=quote)).)*)(?P=quote)"
)
_JAVA_TEST_ANNOTATION = re.compile(
    r"@(?:org\.junit\.(?:jupiter\.api\.)?)?"
    r"(?P<annotation>Test|ParameterizedTest|RepeatedTest|TestFactory|TestTemplate)\b"
)
_JAVA_ANNOTATION = re.compile(r"^\s*@")
_JAVA_DECLARATION = re.compile(
    r"^\s*(?:@[\w.]+(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|protected|private|static|final|synchronized|abstract|default)\s+)*"
    r"(?:<[^>]*>\s+)?(?P<type>[\w$.]+(?:<[^()]*>)?(?:\[\])*)\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*\("
)
_JAVA_JUNIT3 = re.compile(r"^\s*public\s+void\s+(?P<name>test\w*)\s*\(")
_JAVA_STATEMENT_WORDS = frozenset({"return", "new", "throw", "else", "case", "yield"})
#: Annotations and blank lines a JUnit annotation may sit above its method by.
_JAVA_ANNOTATION_REACH = 8
_GO_TEST = re.compile(
    r"^func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s+(?P<method>Test\w*)\s*\(\s*\)"
    r"|(?P<name>(?:Test|Fuzz)(?:[A-Z0-9_]\w*)?)\s*\(\s*(?:\w+\s+\*testing\.[TF]\s*\)|$))"
)
_GO_SUBTEST_CALL = re.compile(r"^\s*\w+\.Run\(")
_GO_SUBTEST = re.compile(r"^\s*\w+\.Run\(\s*(?P<quote>[\"`])(?P<name>[^\"`]*)(?P=quote)\s*,")
_CASE_FIELDS = r"(?:name|desc|description|title|scenario|testName)"
_GO_CASE_CODE = re.compile(rf"^\s*\{{?\s*{_CASE_FIELDS}\s*:\s*\"")
_GO_CASE = re.compile(rf"^\s*\{{?\s*{_CASE_FIELDS}\s*:\s*\"(?P<name>(?:\\.|[^\"\\])*)\"")
_GO_KEYED_CASE_CODE = re.compile(r"^\s*\"\"\s*:\s*\{")
_GO_KEYED_CASE = re.compile(r"^\s*\"(?P<name>(?:\\.|[^\"\\])*)\"\s*:\s*\{")
_PY_CASE_CODE = re.compile(r"\bpytest\.param\s*\(")
_PY_CASE = re.compile(
    r"\bpytest\.param\s*\(.*?\bid\s*=\s*"
    r"(?P<quote>['\"])(?P<name>(?:\\.|(?!(?P=quote)).)*)(?P=quote)"
)


def _test_defs(view: Sequence[_Row], language: _Language) -> Iterator[_TestDef]:
    if language == "python":
        yield from _python_defs(view)
    elif language == "java":
        yield from _java_defs(view)
    else:
        for row in view:
            code = _code(row)
            if language == "javascript":
                match = _JS_TEST.match(row.text) if _JS_TEST_CALL.match(code) else None
            else:
                test = _GO_TEST.match(code)
                if test is not None:
                    name = test["method"] or test["name"]
                    if name != "TestMain":
                        yield _TestDef(name, row, (row,), view)
                    continue
                match = _GO_SUBTEST.match(row.text) if _GO_SUBTEST_CALL.match(code) else None
            if match is not None:
                yield _TestDef(match["name"], row, (row,), view)
    for row in view:
        name = _case_name(row, language)
        if name is not None:
            yield _TestDef(name, row, (row,), view, case=True)


def _python_defs(view: Sequence[_Row]) -> Iterator[_TestDef]:
    for index, row in enumerate(view):
        match = _PY_DEF.match(_code(row))
        if match is None:
            continue
        # A fixture called ``test_data`` is not a test, and removing one is
        # not removing a test.
        above = index - 1
        fixture = False
        while above >= 0 and _PY_DECORATOR.match(_code(view[above])):
            fixture = fixture or _PY_FIXTURE.match(_code(view[above])) is not None
            above -= 1
        if not fixture:
            yield _TestDef(match["name"], row, (row,), view)


def _java_defs(view: Sequence[_Row]) -> Iterator[_TestDef]:
    claimed: set[_Row] = set()
    for index, row in enumerate(view):
        annotation = _JAVA_TEST_ANNOTATION.search(_code(row))
        if annotation is None:
            continue
        for candidate in view[index : index + _JAVA_ANNOTATION_REACH]:
            code = _code(candidate)
            declaration = _JAVA_DECLARATION.match(code)
            if declaration is not None and declaration["type"] not in _JAVA_STATEMENT_WORDS:
                claimed.add(candidate)
                rows = (row,) if candidate is row else (row, candidate)
                marker = "@" + annotation["annotation"]
                yield _TestDef(declaration["name"], candidate, rows, view, annotation=marker)
                break
            if candidate is not row and code.strip() and not _JAVA_ANNOTATION.match(code):
                break
    # JUnit 3: no annotation, the name is what makes it a test.
    for row in view:
        match = _JAVA_JUNIT3.match(_code(row))
        if match is not None and row not in claimed:
            yield _TestDef(match["name"], row, (row,), view)


def _case_name(row: _Row, language: _Language) -> str | None:
    """The name of the test-table case a row starts, where it has one.

    A Go case names itself with a ``name`` field or a map key. That alone is
    also how a slice of fixtures looks, so the row has to sit in a table whose
    start is in view, or say on the same line what it wants. A
    ``pytest.param`` names itself with ``id=``. Bare values have no name.
    """

    code = _code(row)
    match: re.Match[str] | None = None
    if language == "go":
        pattern = (
            _GO_CASE
            if _GO_CASE_CODE.match(code)
            else _GO_KEYED_CASE
            if _GO_KEYED_CASE_CODE.match(code)
            else None
        )
        if pattern is not None and (row.in_table or _EXPECTATION_NAME.search(code)):
            match = pattern.match(row.text)
    elif language == "python" and _PY_CASE_CODE.search(code):
        match = _PY_CASE.search(row.text)
    return match["name"] if match is not None else None


def _deleted_tests(tests: Sequence[_File]) -> tuple[list[IntegrityHit], set[_Row]]:
    """Hits for tests that stop existing, and the base rows those hits already cover.

    Those rows are the tests' and the cases' own: a lost assertion or a
    rewritten value on one of them is part of its test's hit, not another.
    """

    removed: list[tuple[_File, _TestDef]] = []
    added: list[tuple[_File, _TestDef]] = []
    for item in tests:
        assert item.language is not None
        for view in item.base:
            removed += [
                (item, definition)
                for definition in _test_defs(view, item.language)
                if any(row.kind == "-" for row in definition.rows)
            ]
        for view in item.head:
            added += [
                (item, definition)
                for definition in _test_defs(view, item.language)
                if any(row.kind == "+" for row in definition.rows)
            ]

    # Names are matched per language across the whole change: that is what
    # makes a moved test, or a whole moved file, not a deletion. Where a name
    # comes back matters as much as whether it does. Added back only where the
    # runner does not look, a test no longer runs -- the same as ``test_x``
    # renamed to ``_test_x`` -- and a move from tests/test_x.py into
    # tests/disabled/ took every test in the file out of the run with no line
    # of test code gone.
    collected: set[_Name] = set()
    #: Names added back only outside collection, and the first file each one
    #: went to, for the hit to say where.
    parked: dict[_Name, str] = {}

    def add_back(item: _File, name: _Name) -> None:
        if _collected(item.head_path):
            collected.add(name)
        else:
            parked.setdefault(name, item.head_path)

    for item, definition in added:
        add_back(item, _key(item, definition))
    # A name written as a string on an added line has not gone anywhere, even
    # where the line it now sits on is not one this module recognises as a
    # definition: a case rewrapped over several lines, a table whose start is
    # out of the hunk's view.
    for item in tests:
        for view in item.head:
            for row in view:
                if row.kind != "+" or row.reading is None:
                    continue
                for _, (kind, value) in row.reading.literals:
                    if kind == "string" and isinstance(value, str):
                        add_back(item, (item.language, True, value))
                        if item.language in ("javascript", "go"):
                            add_back(item, (item.language, False, value))
    # A test in a file the runner does not collect by default may never have
    # run, so it only has to come back somewhere among the tests: moving
    # tests/util.py to tests/support/util.py takes nothing out of the run.
    lost = {
        definition
        for item, definition in removed
        if _key(item, definition) not in collected
        and (_collected(item.base_path) or _key(item, definition) not in parked)
    }
    taken = {_key(item, definition) for item, definition in removed}
    renamed = _renamed_in_place(tests, removed, added, lost, taken)
    gone = [
        (item, definition)
        for item, definition in removed
        if definition in lost and definition not in renamed
    ]

    deleted = {definition for _, definition in gone if not definition.case}
    inside_deleted: set[_Row] = set()
    for item in tests:
        assert item.language is not None
        for view in item.base:
            inside_deleted |= _owned_rows(view, item.language, deleted)

    covered = set(inside_deleted)
    hits: list[IntegrityHit] = []
    for item, definition in gone:
        assert item.language is not None
        name = _shown(definition.name)
        where = parked.get(_key(item, definition))
        back = (
            f"and is added back only in {markdown_path(where)}, which the test runner does not "
            "collect by default."
            if where is not None
            else f"and no {'case' if definition.case else 'test'} by that name is added back "
            "in this change."
        )
        if definition.case:
            # A case inside a test that went away is part of that test's hit.
            if definition.decl in inside_deleted:
                continue
            row = definition.decl
            covered.update(_case_rows(definition.view, definition.view.index(row), item.language))
            detail = f"Case `{name}` is removed from its test table, {back}"
        elif definition.decl.kind == "-":
            row = definition.decl
            detail = (
                f"Test `{name}` is removed along with its file, {back}"
                if item.new_path is None
                else f"Test `{name}` is removed, {back}"
            )
        else:
            row = next(row for row in definition.rows if row.kind == "-")
            detail = (
                f"`{name}` no longer runs as a test: its `{definition.annotation}` annotation "
                "is removed."
            )
        hits.append(_hit("deleted_test", item.base_path, row, "base", detail))
    return hits, covered


#: A test or case as ``deleted_test`` matches it across the change: language,
#: whether it is a case, name.
_Name = tuple[str | None, bool, str]


def _key(item: _File, definition: _TestDef) -> _Name:
    return item.language, definition.case, definition.name


def _renamed_in_place(
    tests: Sequence[_File],
    removed: Sequence[tuple[_File, _TestDef]],
    added: Sequence[tuple[_File, _TestDef]],
    lost: set[_TestDef],
    taken: set[_Name],
) -> set[_TestDef]:
    """Definitions renamed where they stand: one out, one in, same block, same test.

    Only names that do not survive on the other side take part -- ``lost``
    ones out, names not ``taken`` from anywhere in -- so a block that edits
    one test and renames another pairs the rename, not the edit.

    One out and one in is also what deleting a failing test and writing
    another where it stood looks like, so the one in has to say what the one
    out said: more than half of the old test's lines are still there, with a
    new name above them. A test that keeps most of itself was renamed and
    edited, and the edit is judged like any other. One that keeps half or
    less is a different test in the old one's place, and the old one is gone.
    """

    removed_at = {definition.decl: (item, definition) for item, definition in removed}
    added_at = {definition.decl: (item, definition) for item, definition in added}
    renamed: set[_TestDef] = set()
    for item in tests:
        assert item.language is not None
        language = item.language
        for block in item.blocks:
            for case in (False, True):
                gone = [
                    removed_at[row][1]
                    for row in block.removed
                    if row in removed_at
                    and removed_at[row][1].case is case
                    and removed_at[row][1] in lost
                ]
                unpaired = {
                    added_at[row][1]: Counter(_said(added_at[row][1], language))
                    for row in block.added
                    if row in added_at
                    and added_at[row][1].case is case
                    and _key(*added_at[row]) not in taken
                }
                for definition in gone:
                    before = Counter(_said(definition, language))
                    kept = {new: (before & after).total() for new, after in unpaired.items()}
                    # The closest one, so that deleting one test and renaming
                    # its neighbour in the same block pairs the rename.
                    best = max(kept, key=kept.__getitem__, default=None)
                    if best is not None and 2 * kept[best] > before.total():
                        renamed.add(definition)
                        del unpaired[best]
    return renamed


def _said(definition: _TestDef, language: _Language) -> list[tuple[str, tuple[object, ...]]]:
    """What a test or case says, line by line, for telling a rename from a replacement.

    Its lines other than the ones that declare it, as far as its side of the
    hunk shows them; lines the hunk does not show are the same on both sides.
    Each line is compared by its shape and literal values, so re-quoting or
    re-commenting it changes nothing. Blank lines, comments, lone brackets and
    the rest of a signature written over several lines are left out: every
    test has them, and they would make two unrelated tests look alike. A
    one-liner has no other line, so its own line stands in, with the name
    taken out.
    """

    view = definition.view
    index = view.index(definition.decl)
    if definition.case:
        rows = _case_rows(view, index, language)
        declared = {definition.decl}
    else:
        rows = list(_owned_rows(view, language, {definition}))
        declared = {*definition.rows, *_signature(view, index, language)}
    said = [
        _saying(row)
        for row in rows
        if row not in declared and any(char.isalnum() or char == _LITERAL for char in _shape(row))
    ]
    if said:
        return said
    shape, values = _saying(definition.decl)
    named = ("string", definition.name)
    if named in values:
        rest = list(values)
        rest.remove(named)
        return [(shape, tuple(rest))]
    return [(shape.replace(definition.name, "", 1), values)]


def _saying(row: _Row) -> tuple[str, tuple[object, ...]]:
    assert row.reading is not None
    return row.reading.shape, tuple(value for _, value in row.reading.literals)


def _case_rows(view: Sequence[_Row], index: int, language: _Language) -> list[_Row]:
    """A case's own line, and for a Go case written over several lines the lines after it.

    A case whose ``{`` opens on its own line ends where its brackets close;
    one whose ``{`` is on the line above ends at the ``}`` that closes it.
    Fields written above the name are not followed; by convention the name
    comes first.
    """

    decl = view[index]
    if language != "go":
        return [decl]
    code = _code(decl)
    depth = _balance(code)
    end = 0 if code.lstrip().startswith("{") or _GO_KEYED_CASE_CODE.match(code) else -1
    rows = [decl]
    for row in view[index + 1 : index + _MAX_TABLE_LINES]:
        if depth <= end:
            break
        rows.append(row)
        depth += _balance(_code(row))
    return rows


def _signature(view: Sequence[_Row], index: int, language: _Language) -> list[_Row]:
    """The rows a test's declaration spans, from its name to where its body opens.

    One row unless the signature wraps: Python's ends where its brackets
    close, a brace language's at the ``{`` that opens the body. A JavaScript
    test or a Go subtest is a call whose brackets close only after the body,
    and its name and ``{`` share a row.
    """

    rows: list[_Row] = []
    depth = 0
    for row in view[index : index + _MAX_CONTINUATION_LINES]:
        rows.append(row)
        code = _code(row)
        depth += _balance(code)
        if depth <= 0 or (language != "python" and "{" in code):
            break
    return rows


def _renamed_out_of_collection(renames: Iterable[tuple[str, str]]) -> list[IntegrityHit]:
    """One hit per test file renamed from where the runner collects to where it does not.

    Per file rather than per test: a forge sends no lines for a pure rename, so
    which tests the file held is not in the change. Its two names are, and
    they are enough -- whatever tests it held, the runner no longer runs them.
    Locally the same move is a deletion and an addition, and the tests in it
    are matched one by one instead.
    """

    hits: list[IntegrityHit] = []
    for old, new in sorted(set(renames)):
        if not _collected(old) or _collected(new):
            continue
        detail = (
            f"Test file {markdown_path(old)} is renamed to {markdown_path(new)}, which the test "
            "runner does not collect by default: the tests in it no longer run."
        )
        hits.append(IntegrityHit("deleted_test", old, None, "base", "", redact_text(detail)))
    return hits


def unread_deletions(paths: Iterable[str]) -> list[IntegrityHit]:
    """One ``deleted_test`` hit per test file deleted with none of its lines shown.

    ``paths`` are the files a change deletes that its diff has no line of: a
    forge lists such a file but sends no lines when it is too large, binary or
    cut short. The diff never names it, so ``detect`` cannot see it go, and
    nothing else in a run looks at a file that is gone. Per file, like a
    rename out of collection: which tests it held is not in the change. A file
    the runner does not collect by default held no test that stopped running.

    Pure, like ``detect``: the paths are the only input.
    """

    hits: list[IntegrityHit] = []
    for path in sorted(set(paths)):
        if not _collected(path):
            continue
        detail = (
            f"Test file {markdown_path(path)} is deleted, and the diff shows none of its "
            "lines: which tests it held, and whether the change adds them back elsewhere, "
            "cannot be read from it."
        )
        hits.append(IntegrityHit("deleted_test", path, None, "base", "", redact_text(detail)))
    return hits


def _owned_rows(view: Sequence[_Row], language: _Language, tests: set[_TestDef]) -> set[_Row]:
    """The rows of ``view`` that sit inside one of ``tests``, where visible.

    Python by indentation, the brace languages by counting braces from the
    declaration. A hunk that starts inside a test does not show where that
    test began, and its rows are simply not claimed.
    """

    starts = {definition.decl for definition in tests}
    owned: set[_Row] = set()
    if language == "python":
        #: (indentation of a def, whether it is one of ``tests``)
        indents: list[tuple[int, bool]] = []
        # A line inside brackets left open above it continues that line, at
        # whatever indentation: the ``) -> None:`` that closes a signature
        # written over several lines does not end the def it belongs to.
        unclosed = span = 0
        for index, row in enumerate(view):
            code = _code(row)
            if code.strip() and not unclosed:
                indent = _indent(row)
                while indents and indent <= indents[-1][0]:
                    indents.pop()
                if row in starts or _PY_DEF.match(code):
                    indents.append((indent, row in starts))
                    if row in starts:
                        owned.update(_decorations(view, index))
            if indents and indents[-1][1]:
                owned.add(row)
            unclosed = max(0, unclosed + _balance(code))
            # One miscounted bracket -- a hunk that starts inside a string --
            # must not turn everything below it into one continued line.
            span = span + 1 if unclosed else 0
            if span > _MAX_CONTINUATION_LINES:
                unclosed = span = 0
        return owned

    depth = 0
    #: [depth at the declaration, whether its body has opened]
    open_tests: list[list[int]] = []
    for row in view:
        code = _code(row)
        if row in starts:
            open_tests.append([depth, 0])
        if open_tests:
            owned.add(row)
        for char in code:
            if char == "{":
                depth += 1
                if open_tests and depth > open_tests[-1][0]:
                    open_tests[-1][1] = 1
            elif char == "}":
                depth -= 1
                while open_tests and open_tests[-1][1] and depth <= open_tests[-1][0]:
                    open_tests.pop()
        # ``it("x", () => expect(1).toBe(1));`` has no body of its own to close.
        if (
            language == "javascript"
            and row in starts
            and open_tests
            and not open_tests[-1][1]
            and _balance(code) <= 0
        ):
            open_tests.pop()
    return owned


def _indent(row: _Row) -> int:
    return len(row.text) - len(row.text.lstrip())


def _decorations(view: Sequence[_Row], index: int) -> list[_Row]:
    """The decorators stacked above the def at ``view[index]``, parametrize tables included.

    They sit above the def rather than inside it, and belong to it all the
    same: a deleted test's ``pytest.param`` rows are part of its deletion.
    """

    indent = _indent(view[index])
    claimed: list[_Row] = []
    pending: list[_Row] = []
    #: Brackets closed below and not yet opened: a decorator's arguments wrap
    #: onto lines at any indentation, its closing ``)`` included.
    unopened = 0
    for position in range(index - 1, -1, -1):
        row = view[position]
        code = _code(row)
        unopened -= _balance(code)
        if unopened > 0 or not code.strip() or _indent(row) > indent:
            pending.append(row)
        elif unopened == 0 and _indent(row) == indent and code.lstrip().startswith("@"):
            claimed += [*pending, row]
            pending = []
        else:
            break
    return claimed


# ── b. skip_or_focus ─────────────────────────────────────────────────────────

_MARKERS: dict[_Language, tuple[re.Pattern[str], ...]] = {
    "python": (
        re.compile(r"(?P<marker>\bpytest\.mark\.(?P<kind>skipif|skip|xfail))\b"),
        re.compile(r"(?P<marker>\bpytest\.(?P<kind>skip|xfail))\s*\("),
        re.compile(
            r"^\s*@(?P<marker>(?:unittest\.)?(?P<kind>skipIf|skipUnless|skip|expectedFailure))\b"
        ),
        re.compile(r"(?P<marker>\bself\.(?P<kind>skipTest))\s*\("),
        re.compile(r"\braise\s+(?P<marker>(?:unittest\.)?(?P<kind>SkipTest))\b"),
    ),
    "javascript": (
        re.compile(
            r"^\s*(?:await\s+)?(?P<marker>(?:it|test|describe|context|suite|specify)(?:\.\w+)*?"
            r"\.(?P<kind>skipIf|skip|only|todo|fixme|failing|fails|fail|runIf))\b"
        ),
        re.compile(
            r"^\s*(?P<marker>(?P<kind>x(?:it|describe|test|context|specify)|f(?:it|describe)))"
            r"\s*\("
        ),
        re.compile(r"(?P<marker>\bthis\.(?P<kind>skip))\s*\("),
    ),
    "java": (
        re.compile(
            r"(?P<marker>@(?:org\.junit\.(?:jupiter\.api\.(?:condition\.)?)?)?"
            r"(?P<kind>Disabled\w*|Ignore|Enabled(?:If|On)\w*))\b"
        ),
        re.compile(
            r"(?P<marker>\b(?:Assumptions\.|Assume\.)?(?P<kind>assume(?:True|False|That)))\s*\("
        ),
    ),
    "go": (re.compile(r"(?P<marker>\b(?:t|b|tb|f)\.(?P<kind>Skip|Skipf|SkipNow))\s*\("),),
}
_FOCUS = frozenset({"only", "fit", "fdescribe"})
_EXPECTED_FAILURE = frozenset({"xfail", "expectedfailure", "failing", "fails", "fail"})
_CONDITIONAL = re.compile(
    r"(?i)^(?:skipif|skipunless|runif|assume\w+|(?:disabled|enabled)(?:if|on)\w*)$"
)


def _markers(row: _Row, language: _Language) -> list[tuple[str, str]]:
    code = _code(row)
    found: list[tuple[str, str]] = []
    for pattern in _MARKERS[language]:
        found += [(match["marker"], match["kind"]) for match in pattern.finditer(code)]
    return found


def _marker_detail(marker: str, kind: str) -> str:
    lowered = kind.lower()
    if lowered in _FOCUS:
        return f"Adds `{marker}`: only focused tests run, and the others are skipped."
    if lowered == "todo":
        return f"Adds `{marker}`: a placeholder that does not run."
    if lowered in _EXPECTED_FAILURE:
        return f"Adds `{marker}`: a failure of the test it marks no longer fails the run."
    if _CONDITIONAL.match(kind):
        return f"Adds `{marker}`: the test it applies to is skipped under some conditions."
    return f"Adds `{marker}`: the test it applies to is skipped."


def _skips(tests: Sequence[_File]) -> list[IntegrityHit]:
    # A marker line removed somewhere and added somewhere else moved with its
    # test -- a moved file, or tests reordered.
    moved = Counter(
        _normalized(row.text)
        for item in tests
        for view in item.base
        for row in view
        if row.kind == "-"
    )
    hits: list[IntegrityHit] = []
    for item in tests:
        assert item.language is not None
        language = item.language
        block_of = {row: block for block in item.blocks for row in block.added}
        for view in item.head:
            for row in view:
                found = _markers(row, language) if row.kind == "+" else []
                if not found:
                    continue
                key = _normalized(row.text)
                if moved[key] > 0:
                    moved[key] -= 1
                    continue
                # The same marker on a line this block rewrites was there
                # before: a changed reason or condition, not a new skip.
                block = block_of.get(row)
                existing = {
                    kind.lower()
                    for removed in (block.removed if block is not None else ())
                    for _, kind in _markers(removed, language)
                }
                fresh = [(marker, kind) for marker, kind in found if kind.lower() not in existing]
                if fresh:
                    detail = _marker_detail(*fresh[0])
                    hits.append(_hit("skip_or_focus", item.head_path, row, "head", detail))
    return hits


# ── c. fewer_assertions ──────────────────────────────────────────────────────

_ASSERTIONS: dict[_Language, re.Pattern[str]] = {
    "python": re.compile(
        r"^\s*(?:assert\b|(?:self|cls)\.(?:assert|fail)\w*\s*\(|pytest\.fail\s*\("
        r"|(?:await\s+)?(?:[\w.]+\.)?assert_\w+\s*\(|with\s+(?:pytest\.)?(?:raises|warns)\s*\()"
    ),
    "javascript": re.compile(
        r"^\s*(?:(?:await|return|void)\s+)?(?:expect(?:\.\w+)*\s*\(|assert(?:\.\w+)*\s*\("
        r"|t\.(?:is|not|deepEqual|notDeepEqual|true|false|truthy|falsy|throws|throwsAsync"
        r"|notThrows|notThrowsAsync|equal|notEqual|strictEqual|notStrictEqual|same|notSame|ok"
        r"|notOk|match|notMatch|like|regex|rejects|resolves)\s*\()"
        r"|\.should(?:\.|\()"
    ),
    "java": re.compile(
        r"^\s*(?:(?:[\w$]+\.)*assert[A-Z]\w*\s*\(|(?:Assertions\.|Assert\.)?fail\s*\("
        r"|assert\s|(?:Mockito\.)?verify\s*\()"
    ),
    "go": re.compile(
        r"^\s*(?:(?:t|b|tb|f)\.(?:Error|Errorf|Fatal|Fatalf|Fail|FailNow)\s*\("
        r"|(?:assert|require)\.[A-Z]\w*\s*\(|[\w.]+\.(?:Require|Assert)\(\)\.[A-Z]\w*\s*\("
        r"|(?:g\.|gomega\.)?(?:Expect|Ω)\s*\()"
    ),
}


def _is_assertion(row: _Row, language: _Language, aliases: dict[str, str] | None = None) -> bool:
    code = _code(row)
    if _ASSERTIONS[language].search(code) is not None:
        return True
    if not aliases:
        return False
    return _alias_calls(frozenset(aliases)).search(code) is not None


@cache
def _alias_calls(aliases: frozenset[str]) -> re.Pattern[str]:
    names = "|".join(re.escape(alias) for alias in sorted(aliases))
    return re.compile(rf"^\s*(?:(?:await|return|void)\s+)?(?:{names})(?:\.\w+)*\s*\(")


def _unmatched(
    rows: Sequence[tuple[_File, _Row]], others: Sequence[tuple[_File, _Row]]
) -> list[tuple[_File, _Row]]:
    """``rows`` less those whose text reappears, whitespace aside, in ``others``."""

    available = Counter(_normalized(row.text) for _, row in others)
    left: list[tuple[_File, _Row]] = []
    for item, row in rows:
        text = _normalized(row.text)
        if available[text] > 0:
            available[text] -= 1
        else:
            left.append((item, row))
    return left


#: The ``return err`` under ``if err != nil {``: a Go test, or the helper it
#: calls, that checks and hands the error up instead of asserting on it.
_GO_ERROR_RETURN = re.compile(
    r"^\s*(?:return\b.*(?:\berr\w*\b|\bErrorf\s*\(|\berrors\.\w+\s*\()|panic\s*\()"
)


def _assertions(item: _File, view: Sequence[_Row], kind: str) -> Iterator[_Row]:
    """The rows of ``kind`` on one side of a hunk that check something."""

    assert item.language is not None
    for index, row in enumerate(view):
        if row.kind != kind:
            continue
        if _is_assertion(row, item.language, item.aliases):
            yield row
        elif (
            item.language == "go"
            and _IF.match(_code(row))
            and _code(row).rstrip().endswith("{")
            and index + 1 < len(view)
            and _GO_ERROR_RETURN.match(_code(view[index + 1]))
        ):
            yield row


def _fewer_assertions(tests: Sequence[_File], inside_deleted: set[_Row]) -> list[IntegrityHit]:
    removed = [
        (item, row) for item in tests for view in item.base for row in _assertions(item, view, "-")
    ]
    added = [
        (item, row) for item in tests for view in item.head for row in _assertions(item, view, "+")
    ]
    # Across the whole change: an assertion that reappears verbatim was moved,
    # re-indented or reformatted, not lost -- and one that went away with a
    # deleted test is that test's hit.
    lost = [(item, row) for item, row in _unmatched(removed, added) if row not in inside_deleted]
    gained = _unmatched(added, removed)
    # Copies of one assertion gathered from several files into a shared
    # helper are moved too, though fewer of them arrive than left.
    landed: dict[str, list[_File]] = {}
    for item, row in added:
        landed.setdefault(_normalized(row.text), []).append(item)
    lost = [
        (item, row)
        for item, row in lost
        if all(other is item for other in landed.get(_normalized(row.text), ()))
    ]

    hits: list[IntegrityHit] = []
    for item in tests:
        if item.old_path is None or item.new_path is None:
            continue
        rows = sorted((row for owner, row in lost if owner is item), key=lambda row: row.number)
        count = sum(1 for owner, _ in gained if owner is item)
        if len(rows) > count:
            plural = "s" if len(rows) != 1 else ""
            detail = (
                f"Removes {len(rows)} assertion{plural} and adds {count} in this file, "
                f"a net loss of {len(rows) - count}."
            )
            hits.append(_hit("fewer_assertions", item.base_path, rows[0], "base", detail))
    return hits


# ── d. changed_expectation ───────────────────────────────────────────────────

#: ``expected = 3``, ``want := 3``, ``want: 3`` in a Go table, and the same
#: keys spelled as strings (``"expected": 3``).
_EXPECTATION_NAME = re.compile(
    r"(?i)(?<![\w$])(?:want|expect(?:ed|ation)?)\w*\s*(?::=|=(?![=>])|:(?![:=]))"
)
_EXPECTATION_KEY = re.compile(r"(?i)(?:^|[{,])\s*([\"'])(?:want|expect(?:ed|ation)?)\w*\1\s*:")
#: The tail of an ``expect(...)`` chain on a line of its own.
_JS_MATCHER = re.compile(r"^\s*\)*\s*\.(?:not\.|resolves\.|rejects\.)*to[A-Z]\w*\s*\(")
#: The condition half of ``if got != 3 { t.Errorf(...) }``.
_IF = re.compile(r"^\s*(?:\}\s*else\s+)?if\b")


#: Rows that state an expectation: what kind of statement each is, and the
#: positions of its literals that expect nothing.
_Stated = dict[_Row, tuple[str, frozenset[int]]]


def _expectations(
    view: Sequence[_Row], language: _Language, aliases: dict[str, str] | None = None
) -> _Stated:
    """Rows that state what a test expects, and what kind of statement each is.

    Each comes with the positions of its literals that expect nothing: a
    failure message, a case's name. A multi-line assertion or expected value
    is followed bracket by bracket, so the line that changed inside
    ``assert result == {...}`` counts too.
    """

    flagged: _Stated = {}
    walk: _Walk | None = None
    kind = ""
    span = 0
    for index, row in enumerate(view):
        code = _code(row)
        # A named Go case is a case even when its table starts above the hunk.
        case = (language == "python" and _PY_CASE_CODE.search(code)) or (
            language == "go" and _case_name(row, language) is not None
        )
        if row.in_table or case:
            flagged[row] = ("test case", row.names)
            continue
        if walk is not None:
            flagged[row] = (kind, walk.row(row))
            span += 1
            if walk.depth == 0 or span >= _MAX_CONTINUATION_LINES:
                walk = None
            continue
        if _is_assertion(row, language, aliases) or (
            language == "javascript" and _JS_MATCHER.match(code)
        ):
            kind = "assertion"
            # Go asserts with an ``if`` and a ``t.Errorf`` under it; the
            # expected value is in the ``if``.
            previous = view[index - 1] if index else None
            if previous is not None and previous not in flagged and _IF.match(_code(previous)):
                flagged[previous] = (kind, frozenset())
        elif _EXPECTATION_NAME.search(code) or (code.strip() and _EXPECTATION_KEY.search(row.text)):
            kind = "expected value"
        else:
            continue
        walk, span = _Walk(language, aliases), 0
        flagged[row] = (kind, walk.row(row))
        if walk.depth == 0:
            walk = None
    return flagged


def _changed_expectations(tests: Sequence[_File], inside_deleted: set[_Row]) -> list[IntegrityHit]:
    hits: list[IntegrityHit] = []
    for item in tests:
        if item.old_path is None or item.new_path is None:
            continue
        assert item.language is not None
        language = item.language
        before: _Stated = {}
        after: _Stated = {}
        for view in item.base:
            before.update(_expectations(view, language, item.aliases))
        for view in item.head:
            after.update(_expectations(view, language, item.aliases))
        owners = _owners(item.base, language) | _owners(item.head, language)
        starts = {owner.decl for owner in owners.values()}
        pairs = [*_same_shape_pairs(item.blocks, starts)]
        kept = _kept(item.blocks, pairs, owners)
        for old, new in pairs:
            if old not in before or new not in after or old in inside_deleted:
                continue
            assert old.reading is not None and new.reading is not None
            kind, unstated = before[old]
            unstated |= after[new][1]
            # A case renamed in its table is a rename, not a new expectation.
            renamed = (
                ("string", _case_name(old, language)),
                ("string", _case_name(new, language)),
            )
            changed = [
                (was, now)
                for position, ((was, was_value), (now, now_value)) in enumerate(
                    zip(old.reading.literals, new.reading.literals, strict=True)
                )
                if was_value != now_value
                and (was_value, now_value) != renamed
                and position not in unstated
            ]
            if not changed:
                continue
            owner = owners.get(old)
            if owner is not None and kept[owner.name, _normalized(old.text)] > 0:
                kept[owner.name, _normalized(old.text)] -= 1
                continue
            was, now = changed[0]
            detail = (
                f"Only literal values change in this {kind} "
                f"(`{_shown(was)}` → `{_shown(now)}`), in a change that also edits "
                "non-test code."
            )
            hits.append(_hit("changed_expectation", item.base_path, old, "base", detail, new.text))
    return hits


def _owners(views: Iterable[Sequence[_Row]], language: _Language) -> dict[_Row, _TestDef]:
    """The test or case each row of a side belongs to, where it starts in view."""

    owners: dict[_Row, _TestDef] = {}
    for view in views:
        starts = {definition.decl: definition for definition in _test_defs(view, language)}
        owner: _TestDef | None = None
        for row in view:
            owner = starts.get(row, owner)
            if owner is not None:
                owners[row] = owner
    return owners


def _kept(
    blocks: Iterable[_Block], pairs: Sequence[tuple[_Row, _Row]], owners: dict[_Row, _TestDef]
) -> Counter[tuple[str, str]]:
    """Removed lines that are still there after the change, in the same test.

    Such a line was moved or re-indented, not rewritten, and the pairing only
    put it next to another line of the same shape. Keyed by test name and
    text; a line paired with its own copy has the first claim on it.
    """

    def owned(rows: Iterable[_Row]) -> Counter[tuple[str, str]]:
        return Counter((owners[row].name, _normalized(row.text)) for row in rows if row in owners)

    blocks = list(blocks)
    kept = owned(row for block in blocks for row in block.removed) & owned(
        row for block in blocks for row in block.added
    )
    kept.subtract(
        owned(
            old
            for old, new in pairs
            if old in owners
            and new in owners
            and owners[old].name == owners[new].name
            and _normalized(old.text) == _normalized(new.text)
        )
    )
    return kept


def _same_shape_pairs(blocks: Iterable[_Block], starts: set[_Row]) -> Iterator[tuple[_Row, _Row]]:
    """Removed and added rows that line up with the same shape.

    Aligned by ``SequenceMatcher`` rather than position, so an edit that also
    adds a line still pairs the lines it rewrote. Two rows only pair when as
    many tests start above each of them in the block: one test's line next
    to another test's is two tests, not a rewrite.
    """

    for block in blocks:
        if not block.removed or not block.added:
            continue
        if max(len(block.removed), len(block.added)) > _MAX_PAIRED_BLOCK_LINES:
            continue
        before = [_shape(row) for row in block.removed]
        after = [_shape(row) for row in block.added]
        tests_before = list(accumulate((row in starts for row in block.removed), initial=0))
        tests_after = list(accumulate((row in starts for row in block.added), initial=0))
        matcher = SequenceMatcher(None, before, after, autojunk=False)
        for tag, old_start, old_end, new_start, _ in matcher.get_opcodes():
            if tag != "equal":
                continue
            for offset in range(old_end - old_start):
                old, new = old_start + offset, new_start + offset
                if tests_before[old] == tests_after[new]:
                    yield block.removed[old], block.added[new]


def _shape(row: _Row) -> str:
    return row.reading.shape if row.reading is not None else row.text


# ── e. loosened_threshold ────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Threshold:
    key: str
    value: float
    shown: str
    #: A limit on what may stay uncovered: raising it is the loosening.
    ceiling: bool = False


_COVERAGE_FLAG = re.compile(
    r"(?<![\w-])(?P<key>--cov-fail-under|--fail-under)(?:=|\s+)[\"']?(?P<value>\d+(?:\.\d+)?)"
)
_FAIL_UNDER = re.compile(r"^\s*(?P<key>fail_under)\s*[=:]\s*[\"']?(?P<value>\d+(?:\.\d+)?)")
_JS_COVERAGE_METRIC = re.compile(
    r"[\"']?\b(?P<key>branches|functions|lines|statements)\b[\"']?\s*:\s*"
    r"(?P<value>-?\d+(?:\.\d+)?)"
)
#: nyc and c8 take the same metrics as command-line flags.
_JS_COVERAGE_TOOL = re.compile(r"(?<![\w-])(?:nyc|c8)\b|--check-coverage\b")
_JS_COVERAGE_FLAG = re.compile(
    r"(?<![\w-])--(?P<key>branches|functions|lines|statements)(?:=|\s+)(?P<value>\d+(?:\.\d+)?)"
)
_JACOCO_LIMIT = re.compile(
    r"<(?P<key>minimum|maximum)>\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<percent>%?)\s*</(?P=key)>"
)
_GRADLE_LIMIT = re.compile(
    r"\b(?P<key>minimum|maximum)\s*=\s*(?:BigDecimal(?:\.valueOf)?\(\s*)?[\"']?"
    r"(?P<value>\d+(?:\.\d+)?)"
)


def _thresholds(row: _Row, config: _Config) -> list[_Threshold]:
    setting = row.setting
    found = [
        _Threshold(match["key"], float(match["value"]), match["value"])
        for match in _COVERAGE_FLAG.finditer(setting)
    ]
    if "coverage" in config.tools:
        found += [
            _Threshold(match["key"], float(match["value"]), match["value"])
            for match in _FAIL_UNDER.finditer(setting)
        ]
    if "jest" in config.tools:
        # A negative Jest threshold is the number of uncovered entities
        # allowed, so lowering the number loosens it either way.
        found += [
            _Threshold(match["key"], float(match["value"]), match["value"])
            for match in _JS_COVERAGE_METRIC.finditer(setting)
        ]
    if _JS_COVERAGE_TOOL.search(setting):
        found += [
            _Threshold(f"--{match['key']}", float(match["value"]), match["value"])
            for match in _JS_COVERAGE_FLAG.finditer(setting)
        ]
    if "maven" in config.tools:
        found += [
            _Threshold(
                f"<{match['key']}>",
                float(match["value"]) / (100 if match["percent"] else 1),
                match["value"] + match["percent"],
                ceiling=match["key"] == "maximum",
            )
            for match in _JACOCO_LIMIT.finditer(setting)
        ]
    if "gradle" in config.tools:
        found += [
            _Threshold(
                match["key"], float(match["value"]), match["value"], match["key"] == "maximum"
            )
            for match in _GRADLE_LIMIT.finditer(setting)
        ]
    return found


def _looser(old: _Threshold, new: _Threshold) -> bool:
    return new.value > old.value if old.ceiling else new.value < old.value


_TEST_COMMAND = (
    r"(?<![\w/.-])(?P<value>py\.?test|python3?\s+-m\s+pytest|jest|vitest"
    r"|(?:npm|yarn|pnpm)(?:\s+run)?\s+test|go\s+test|(?:\./)?mvnw?|(?:\./)?gradlew?)(?![\w-])"
)
#: (key, pattern, what the change does). ``value`` is what a hit names; the
#: same key and value removed elsewhere in the change means it only moved.
_EXCLUSIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "--ignore",
        re.compile(r"(?<![\w-])--(?:ignore|ignore-glob)(?:=|\s+)[\"']?(?P<value>[^\s\"',\]]+)"),
        "Excludes `{value}` from test collection.",
    ),
    (
        "--deselect",
        re.compile(r"(?<![\w-])--deselect(?:=|\s+)[\"']?(?P<value>[^\s\"',\]]+)"),
        "Deselects `{value}` from the test run.",
    ),
    (
        "-k",
        re.compile(
            r"(?<![\w-])-k(?:\s*=\s*|\s+)?(?P<quote>[\"'])(?P<value>[^\"']*\bnot\b[^\"']*)"
            r"(?P=quote)"
        ),
        'Filters tests out by keyword (`-k "{value}"`).',
    ),
    (
        "-m",
        re.compile(
            r"(?<![\w-])-m(?:\s*=\s*|\s+)?(?P<quote>[\"'])(?P<value>[^\"']*\bnot\b[^\"']*)"
            r"(?P=quote)"
        ),
        'Filters tests out by marker (`-m "{value}"`).',
    ),
    (
        "--no-cov",
        re.compile(r"(?<![\w-])(?P<value>--no-cov)\b"),
        "Turns coverage measurement off (`--no-cov`), so its threshold is no longer checked.",
    ),
    (
        "--testPathIgnorePatterns",
        re.compile(r"(?<![\w-])--testPathIgnorePatterns(?:=|\s+)[\"']?(?P<value>[^\s\"']+)"),
        "Excludes tests matching `{value}` from the run.",
    ),
    (
        "maven",
        re.compile(
            r"(?<![\w-])-D(?P<value>skipTests|skipITs|maven\.test\.skip"
            r"|maven\.test\.failure\.ignore|jacoco\.skip)(?:=(?:true)?)?(?![\w.=])"
        ),
        "Tests or their checks stop running or stop failing the build (`-D{value}`).",
    ),
    (
        "maven",
        re.compile(
            r"<(?P<value>skipTests|skipITs|maven\.test\.skip|testFailureIgnore"
            r"|maven\.test\.failure\.ignore|jacoco\.skip)>\s*true\s*</"
        ),
        "Tests or their checks stop running or stop failing the build (`<{value}>true`).",
    ),
    (
        "maven",
        re.compile(r"<(?P<value>haltOnFailure)>\s*false\s*</"),
        "A failed coverage check stops failing the build (`<{value}>false`).",
    ),
    (
        "<exclude>",
        re.compile(r"<exclude>\s*(?P<value>[^<]*(?:Tests?|IT)\b[^<]*?)\s*</exclude>"),
        "Excludes `{value}` from the test run.",
    ),
    (
        "excludeTestsMatching",
        re.compile(r"\bexcludeTestsMatching\s*\(?\s*[\"'](?P<value>[^\"']+)[\"']"),
        "Excludes tests matching `{value}` from the run.",
    ),
    (
        "-x",
        re.compile(r"(?<![\w-])-x\s+(?P<value>(?::?[\w-]+:)*test)\b"),
        "Skips the Gradle `{value}` task.",
    ),
    (
        "ignoreFailures",
        re.compile(r"\b(?P<value>ignoreFailures)\s*=\s*true\b"),
        "Test failures stop failing the build (`ignoreFailures = true`).",
    ),
    (
        "enabled",
        re.compile(r"\b(?P<value>test)\s*(?:\.|\{)\s*enabled\s*=\s*false\b"),
        "Disables the Gradle `{value}` task.",
    ),
    (
        "-skip",
        re.compile(
            r"\bgo\s+test\b[^|;&\n]*?(?<![\w-])-(?:test\.)?skip(?:=|\s+)[\"']?"
            r"(?P<value>[^\s\"']+)"
        ),
        "Skips tests matching `{value}` (`go test -skip`).",
    ),
    (
        "|| true",
        re.compile(_TEST_COMMAND + r"[^|;&\n]*\|\|\s*(?:true\b|:(?!\S)|exit\s+0\b)"),
        "Test failures stop failing this command (`{value} ... || true`).",
    ),
)
#: ``--ignore`` is also flake8's and ruff's flag. It only counts on a pytest
#: command line or setting, or when what it ignores is plainly a test.
_PYTEST_CONTEXT = re.compile(r"\bpy\.?test\b|\baddopts\b")
_TESTISH = re.compile(r"(?i)test|spec")


def _exclusions(row: _Row) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for key, pattern, template in _EXCLUSIONS:
        for match in pattern.finditer(row.setting):
            value = match["value"]
            if key == "--ignore" and not (
                _PYTEST_CONTEXT.search(row.setting) or _TESTISH.search(value)
            ):
                continue
            found.append((key, value, template.format(value=_shown(value))))
    return found


@dataclass(frozen=True)
class _ListEntry:
    """Items one row contributes to a list-valued setting."""

    key: str
    items: tuple[str, ...]
    #: The row names the setting, rather than continuing its value.
    names_key: bool


_LIST_SETTING = re.compile(
    r"^\s*[\"']?(?P<key>testpaths|testPathIgnorePatterns|collect_ignore_glob|collect_ignore)"
    r"[\"']?\s*(?:\+=|=|:)\s*(?P<rest>.*)$"
)
_LIST_CALL = re.compile(
    r"\b(?P<key>collect_ignore_glob|collect_ignore)\.(?:append|extend)\s*\((?P<rest>.*)$"
)
_STRING = re.compile(r"\"((?:\\.|[^\"\\])*)\"|'((?:\\.|[^'\\])*)'")
_LIST_TOOLS = {
    "testpaths": "pytest",
    "testPathIgnorePatterns": "jest",
    "collect_ignore": "conftest",
    "collect_ignore_glob": "conftest",
}


def _strings(text: str) -> tuple[str, ...]:
    return tuple(double or single for double, single in _STRING.findall(text))


def _list_entries(view: Sequence[_Row], config: _Config) -> dict[_Row, _ListEntry]:
    """Follow list-valued settings across the lines they span.

    TOML and JavaScript arrays run to their closing bracket; an INI value runs
    over the indented lines below its key. The key line has to be in the hunk
    for its items to be recognised at all.
    """

    entries: dict[_Row, _ListEntry] = {}
    open_key: str | None = None
    depth = 0
    indented_key: str | None = None
    indent = 0
    for row in view:
        setting = row.setting
        if open_key is not None:
            entries[row] = _ListEntry(open_key, _strings(setting), False)
            depth += setting.count("[") - setting.count("]")
            if depth <= 0:
                open_key = None
            continue
        if indented_key is not None:
            if setting.strip() and len(setting) - len(setting.lstrip()) > indent:
                entries[row] = _ListEntry(indented_key, tuple(setting.split()), False)
                continue
            indented_key = None
        match = _LIST_SETTING.match(setting) or _LIST_CALL.search(setting)
        if match is None or _LIST_TOOLS[match["key"]] not in config.tools:
            continue
        key, rest = match["key"], match["rest"]
        if "[" in rest:
            depth = rest.count("[") - rest.count("]")
            open_key = key if depth > 0 else None
            entries[row] = _ListEntry(key, _strings(rest), True)
        elif config.ini:
            indented_key, indent = key, len(setting) - len(setting.lstrip())
            items = tuple(item.strip("\"'") for item in rest.split())
            entries[row] = _ListEntry(key, items, True)
        else:
            entries[row] = _ListEntry(key, _strings(rest), True)
    return entries


def _loosened_thresholds(configs: Sequence[_File]) -> list[IntegrityHit]:
    return [*_lowered_thresholds(configs), *_added_exclusions(configs), *_list_changes(configs)]


def _lowered_thresholds(configs: Sequence[_File]) -> list[IntegrityHit]:
    # Anywhere in the change, so a threshold that moved from .coveragerc to
    # pyproject.toml is compared with where it went.
    added_anywhere: dict[str, list[_Threshold]] = {}
    for item in configs:
        assert item.config is not None
        for view in item.head:
            for row in view:
                if row.kind == "+":
                    for threshold in _thresholds(row, item.config):
                        added_anywhere.setdefault(threshold.key, []).append(threshold)

    hits: list[IntegrityHit] = []
    for item in configs:
        assert item.config is not None
        for block in item.blocks:
            replacements = [
                (row, threshold)
                for row in block.added
                for threshold in _thresholds(row, item.config)
            ]
            for row in block.removed:
                changes: list[str] = []
                replacement: str | None = None
                for old in _thresholds(row, item.config):
                    paired = next(
                        (entry for entry in replacements if entry[1].key == old.key), None
                    )
                    if paired is not None:
                        replacements.remove(paired)
                        new_row, new = paired
                        if _looser(old, new):
                            changes.append(f"`{old.key}` {old.shown} → {new.shown}")
                            replacement = replacement or new_row.text
                        continue
                    elsewhere = added_anywhere.get(old.key, [])
                    if any(not _looser(old, new) for new in elsewhere):
                        continue
                    if elsewhere:
                        loosest = max(
                            elsewhere, key=lambda new: new.value if old.ceiling else -new.value
                        )
                        changes.append(f"`{old.key}` {old.shown} → {loosest.shown}")
                    else:
                        changes.append(f"`{old.key}` {old.shown} is removed")
                if changes:
                    detail = "Loosens the coverage threshold: " + "; ".join(changes) + "."
                    hits.append(
                        _hit("loosened_threshold", item.base_path, row, "base", detail, replacement)
                    )
    return hits


def _added_exclusions(configs: Sequence[_File]) -> list[IntegrityHit]:
    existing = Counter(
        (key, value)
        for item in configs
        for view in item.base
        for row in view
        if row.kind == "-"
        for key, value, _ in _exclusions(row)
    )
    hits: list[IntegrityHit] = []
    for item in configs:
        for view in item.head:
            for row in view:
                if row.kind != "+":
                    continue
                fresh: list[str] = []
                for key, value, detail in _exclusions(row):
                    if existing[(key, value)] > 0:
                        existing[(key, value)] -= 1
                    else:
                        fresh.append(detail)
                if fresh:
                    hits.append(_hit("loosened_threshold", item.head_path, row, "head", fresh[0]))
    return hits


def _list_changes(configs: Sequence[_File]) -> list[IntegrityHit]:
    base: list[tuple[_File, _Row, _ListEntry]] = []
    head: list[tuple[_File, _Row, _ListEntry]] = []
    for item in configs:
        assert item.config is not None
        for view in item.base:
            base += [(item, row, entry) for row, entry in _list_entries(view, item.config).items()]
        for view in item.head:
            head += [(item, row, entry) for row, entry in _list_entries(view, item.config).items()]

    hits: list[IntegrityHit] = []
    # testpaths is an allow-list: an entry that disappears is a directory
    # nobody collects any more -- as long as the setting itself remains.
    # Deleting the whole setting makes pytest collect from the root instead.
    collected = {path for _, _, entry in head if entry.key == "testpaths" for path in entry.items}
    still_set = any(entry.key == "testpaths" and entry.names_key for _, _, entry in head)
    for item, row, entry in base:
        if entry.key != "testpaths" or row.kind != "-" or not still_set:
            continue
        dropped = [path for path in entry.items if path not in collected]
        if dropped:
            more = f", nor {len(dropped) - 1} more" if len(dropped) > 1 else ""
            detail = f"Narrows `testpaths`: `{_shown(dropped[0])}` is no longer collected{more}."
            hits.append(_hit("loosened_threshold", item.base_path, row, "base", detail))

    # The ignore lists are deny-lists: an entry that appears is a test that
    # stops running. Jest's own default, node_modules, is not a test.
    listed = {(entry.key, path) for _, _, entry in base for path in entry.items}
    for item, row, entry in head:
        if entry.key == "testpaths" or row.kind != "+":
            continue
        fresh = [
            path
            for path in entry.items
            if (entry.key, path) not in listed and "node_modules" not in path
        ]
        if entry.key.startswith("collect_ignore") and not _in_test_directory(item.head_path):
            # A root conftest.py ignoring setup.py is housekeeping; ignoring
            # something that looks like a test is not.
            fresh = [path for path in fresh if _TESTISH.search(path)]
        if fresh:
            detail = f"Adds `{_shown(fresh[0])}` to `{entry.key}`, so matching tests stop running."
            hits.append(_hit("loosened_threshold", item.head_path, row, "head", detail))
    return hits


def _in_test_directory(path: str) -> bool:
    return any(part.lower() in _TEST_DIRS for part in PurePosixPath(path).parts[:-1])


__all__ = [
    "INTEGRITY_CHECKS",
    "IntegrityCheck",
    "IntegrityHit",
    "Side",
    "detect",
    "unread_deletions",
]
