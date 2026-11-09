"""Unified-diff parsing, in one place.

Three layers ask the same questions of a diff: the structural guardrails (which
files, how many lines), the rule engine (which text was added, and on which
line), and the model trust boundary (which lines a finding may point at). They
have to agree. A rule that fired on a line the grounding check would reject
would produce a block nobody could explain from the evidence pack, so there is
exactly one parser and everyone reads the same answer out of it.

The test-integrity checks ask the fourth question -- which lines were taken
away, and where they sat in the base file -- and read it out of the same
per-file patches ``added_lines`` is built on.

The diff is read the way git reads it, because its content is written by
whoever wrote the change and must not be able to pass for its structure: a line
ends at ``\\n`` and nowhere else (``split_lines``), and a hunk is exactly as long
as the counts in its ``@@`` header (``file_patches``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

from .redaction import redact_text

_HUNK_RE = re.compile(
    r"^@@ -(?P<old>\d+)(?:,(?P<old_count>\d+))? \+(?P<new>\d+)(?:,(?P<new_count>\d+))? @@"
)

#: A path in git's C-style quotes, as quote.c's ``unquote_c_style`` accepts
#: one: any character but ``"`` and ``\``, a one-letter escape, or three octal
#: digits for one byte (a first digit over 3 would not fit in it).
_QUOTED_PATH_RE = re.compile(r'"((?:[^"\\]|\\[abtnvfr"\\]|\\[0-3][0-7]{2})*)"')
_ESCAPE_RE = re.compile(rb'\\([abtnvfr"\\]|[0-3][0-7]{2})')
_ESCAPES = {
    b"a": b"\a",
    b"b": b"\b",
    b"t": b"\t",
    b"n": b"\n",
    b"v": b"\v",
    b"f": b"\f",
    b"r": b"\r",
    b'"': b'"',
    b"\\": b"\\",
}


class AddedLine(NamedTuple):
    """One line the change introduced, at its position in the new file."""

    line: int
    text: str


class RemovedLine(NamedTuple):
    """One line the change took out, at its position in the base file."""

    line: int
    text: str


class PatchLine(NamedTuple):
    """One line of a hunk, numbered on every side it exists on.

    ``old`` is ``None`` for an added line and ``new`` is ``None`` for a
    removed one; a context line carries both.
    """

    kind: Literal[" ", "-", "+"]
    old: int | None
    new: int | None
    text: str


@dataclass(frozen=True)
class FilePatch:
    """One file's share of a diff, with both of its names.

    A rename is one patch with two paths, not a deletion plus an addition:
    anything that asks "did this change take a test away" has to be able to
    tell the two apart.
    """

    #: ``None`` when the change creates the file.
    old_path: str | None
    #: ``None`` when the change deletes the file.
    new_path: str | None
    hunks: tuple[tuple[PatchLine, ...], ...] = ()

    @property
    def added(self) -> list[AddedLine]:
        return [
            AddedLine(line.new, line.text)
            for hunk in self.hunks
            for line in hunk
            if line.kind == "+" and line.new is not None
        ]

    @property
    def removed(self) -> list[RemovedLine]:
        return [
            RemovedLine(line.old, line.text)
            for hunk in self.hunks
            for line in hunk
            if line.kind == "-" and line.old is not None
        ]


def normalize_diff_path(path: str) -> str | None:
    """Normalize a git diff header path, excluding deleted-file markers.

    Git writes a path it cannot put down as it is -- one with a byte past
    ASCII (under the default ``core.quotePath``), a ``"``, a ``\\`` or a control
    character -- in C-style quotes, prefix and all:
    ``"b/config/\\345\\257\\206\\351\\222\\245.env"`` is ``config/密钥.env``.
    Taken as written, the file would go by a name no tree holds and no glob or
    rule matches, and not by the one git's own file list gives it, which is
    what the guardrails read. Unquoted, a path can hold a control character
    again, which ``is_safe_relative_path`` refuses as it refuses any other.

    Nothing else is trimmed. A label ends at a tab -- git puts one after a
    label that holds a space (``diff_label``), GNU diff one before its
    timestamp -- and a name git has not quoted never holds a tab itself, so
    everything before the first one is the name, whitespace and all.
    ``src/a.py `` and ``src/a.py<NBSP>`` are names git writes unquoted -- the
    first always, the second with ``core.quotePath`` off -- and trimmed, each
    would be read as ``src/a.py``, lending its lines to the file every rule,
    the tree check and the grounding check look at under that name. The line
    ending is not here to trim: ``split_lines`` has taken it.
    """

    path = path.split("\t", maxsplit=1)[0]
    if path.startswith('"'):
        path = _unquote(path)
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def diff_label(label: str) -> str:
    """``label`` as git writes it after ``--- `` or ``+++ ``.

    For the diffs written here rather than by git: the forge adapters rebuild
    the headers a platform leaves out, and local intake writes a hunk for each
    untracked file. They land in the same evidence as git's own output and go
    through the same ``normalize_diff_path``, so they follow git's rule: a
    label that holds a space ends in a tab, which GNU patch reads as the end of
    the name. The tab is for an ASCII space only, as git's is.

    Nothing is quoted. A label written here starts with ``a/`` or ``b/``, or is
    ``/dev/null``, so it is never read as quoted, and the names that would need
    escaping -- a tab, a line break -- are refused by ``is_safe_relative_path``
    before any caller gets this far.
    """

    return f"{label}\t" if " " in label else label


def _unquote(quoted: str) -> str:
    """Read a quoted path back the way git does, up to its closing quote.

    An escape stands for a byte, not a character -- ``密`` is three of them --
    so the bytes are gathered first and decoded together, anything that is not
    UTF-8 replaced, as intake decodes the paths in git's own file list.
    Quoting git itself could not unquote is kept as written: the file keeps a
    name, if not a real one, rather than dropping out of the diff.
    """

    match = _QUOTED_PATH_RE.match(quoted)
    if match is None:
        return quoted
    # A str decoded from JSON can hold a lone surrogate, which strict UTF-8
    # will not encode; reading a header must not raise over one.
    raw = match[1].encode("utf-8", errors="surrogatepass")
    return _ESCAPE_RE.sub(_unescape, raw).decode("utf-8", errors="replace")


def _unescape(escape: re.Match[bytes]) -> bytes:
    code = escape[1]
    return bytes([int(code, 8)]) if len(code) == 3 else _ESCAPES[code]


def split_lines(text: str) -> list[str]:
    """Split ``text`` into lines where git does: at ``\\n``, and only there.

    ``str.splitlines`` also breaks at a lone ``\\r``, form feeds, ``\\x1c`` to
    ``\\x1e``, NEL and the Unicode line and paragraph separators. Git does not,
    so a line holding one of them is one line of a diff, under one number.
    Broken anywhere else, the rest of an added line loses its ``+`` and drops
    out of the added lines -- text no rule reads, though Python runs code after
    a lone ``\\r`` -- and a file stops agreeing with its own diff about which
    line is which. The ``\\r`` of a ``\\r\\n`` ending is dropped, as
    ``splitlines`` drops it, so a CRLF file reads like any other.
    """

    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def extract_changed_files(diff_text: str) -> list[str]:
    """Every path the diff names on its new side, sorted."""

    return sorted({patch.new_path for patch in file_patches(diff_text) if patch.new_path})


def count_changed_lines(diff_text: str) -> int:
    """Lines added plus lines removed, including hunks that name no file."""

    return _read(diff_text)[1]


def file_patches(diff: str) -> list[FilePatch]:
    """Split a unified diff into per-file patches, numbering both sides.

    A hunk runs for the number of lines its ``@@`` header counts on each side,
    which is how git reads one, and until they are used up every line is
    content, whatever it looks like: a removed ``-- comment`` in SQL or Lua
    renders as ``--- ...`` and an added ``++ ...`` line as ``+++ ...``. Taken
    for a file header, such a pair hands the rest of the hunk to a file that
    does not exist, and hides its lines from every reader of this module. Inside
    a hunk an empty line is an empty context line, as git reads it, and ``\\ No
    newline at end of file`` counts for neither side.

    Headers are read between hunks only. There every ``+++`` line opens a file
    and a ``---`` line right before it names the base side; a ``+++`` without
    one (hand-written diffs) is the same file on both sides.

    A ``diff --git`` line opens a file too, and its ``+++`` line, when one
    follows, names it. Git writes none for a change with no lines to show: a
    pure rename or copy, an empty file created or deleted, a mode change, a
    binary file. Read by ``+++`` lines alone, those files were not in the diff,
    and a patch handed over with ``--diff`` carries no file list besides it:
    moving ``.github/workflows/deploy.yml`` out of the tree's sensitive paths
    or a test file out of the runner's reach was a change no guardrail saw.
    Such a file is named by git's extended header lines (``_name_by_header``)
    and has no hunks.


    A hunk the diff cuts short, as a platform does when it truncates a diff, ends
    at the next ``diff`` or ``@@`` line, which no content line can start with. A
    line that cannot be content at all -- no ``+``, ``-`` or space in front -- is
    skipped rather than taken for the end of the hunk, which would drop every
    line after it.
    """

    return _read(diff)[0]


def _read(diff: str) -> tuple[list[FilePatch], int]:
    """The one pass over a diff: its per-file patches, and how many lines it changes.

    The count takes in hunks that come before any file header. They belong to
    no file, but they are still lines the change adds or removes.
    """

    files: list[_Entry] = []
    #: The file a ``diff --git`` line opened, until a ``+++`` line names it or
    #: a hunk starts: meanwhile git's extended header lines name its sides.
    opened: _Entry | None = None
    hunk: list[PatchLine] = []
    old_number = new_number = 0
    #: What the open hunk still owes on each side; both zero between hunks.
    old_left = new_left = 0
    #: The line before, when it was a ``--- `` line: the base side of a
    #: ``+++`` header that follows it at once.
    base: str | None = None
    changed = 0

    for line in split_lines(diff):
        if old_left or new_left:
            if line.startswith("diff ") or _HUNK_RE.match(line):
                old_left = new_left = 0  # cut short; the line is read as a header below
            else:
                kind = line[:1] or " "
                if kind == "+":
                    hunk.append(PatchLine("+", None, new_number, line[1:]))
                    new_number += 1
                    new_left = max(new_left - 1, 0)
                    changed += 1
                elif kind == "-":
                    hunk.append(PatchLine("-", old_number, None, line[1:]))
                    old_number += 1
                    old_left = max(old_left - 1, 0)
                    changed += 1
                elif kind == " ":
                    hunk.append(PatchLine(" ", old_number, new_number, line[1:]))
                    old_number += 1
                    new_number += 1
                    old_left, new_left = max(old_left - 1, 0), max(new_left - 1, 0)
                continue

        match = _HUNK_RE.match(line)
        if match:
            old_number, new_number = int(match["old"]), int(match["new"])
            old_left = _hunk_count(match["old_count"])
            new_left = _hunk_count(match["new_count"])
            # A hunk before any file header belongs to nothing and is dropped,
            # as it always was -- but it is still read by its counts, so its
            # lines cannot pass for headers either.
            hunk = []
            if files:
                files[-1].hunks.append(hunk)
            opened = None
        elif line.startswith("+++ "):
            new_path = normalize_diff_path(line[4:])
            old_path = normalize_diff_path(base) if base is not None else new_path
            if opened is None:
                files.append(_Entry(old_path, new_path))
            else:
                opened.old_path, opened.new_path = old_path, new_path
            opened = None
        elif line.startswith(_GIT_HEADER):
            name = _git_header_name(line[len(_GIT_HEADER) :])
            opened = _Entry(name, name)
            files.append(opened)
        elif opened is not None:
            _name_by_header(opened, line)
        base = line[4:] if line.startswith("--- ") else None

    patches = [
        FilePatch(entry.old_path, entry.new_path, tuple(tuple(hunk) for hunk in entry.hunks))
        for entry in files
    ]
    return patches, changed


_GIT_HEADER = "diff --git "


@dataclass
class _Entry:
    """A file ``_read`` is still gathering, before it becomes a ``FilePatch``."""

    old_path: str | None
    new_path: str | None
    hunks: list[list[PatchLine]] = field(default_factory=list)


def _git_header_name(names: str) -> str | None:
    """The one name a ``diff --git`` line gives both sides, if it gives one.

    Its two names are not delimited: git puts neither a tab nor quotes around a
    name with a space, so ``a/my dir/x.py b/my dir/x.py`` could split at any
    of its spaces. Git's own reader takes the one split whose halves name the
    same file, and so does this one. The halves are then the same length, so
    the split is the middle -- quoted or not, since a name is quoted the same
    way on both sides. A line whose two names differ is a rename or a copy, and
    its ``rename``/``copy`` lines name the sides; without them, as for git, no
    name is read from it.
    """

    middle = len(names) // 2
    if len(names) % 2 == 0 or names[middle] != " ":
        return None
    old, new = normalize_diff_path(names[:middle]), normalize_diff_path(names[middle + 1 :])
    return old if old == new else None


def _name_by_header(entry: _Entry, line: str) -> None:
    """Name ``entry``'s sides from one of git's extended header lines, if it is one.

    A copy's source is left out. It is the same file after the change as
    before -- a platform does not list it either -- and paired with the copy
    as a patch's two sides, it would read as a file moved away, which is what
    the test-integrity checks look for.
    """

    for prefix, side in _HEADER_SIDES:
        if not line.startswith(prefix):
            continue
        path = _header_path(line[len(prefix) :]) if side in {"old", "new"} else None
        if side in {"old", "created"}:
            entry.old_path = path
        else:
            entry.new_path = path
        return


#: Git's extended header lines that name a side of a file, and the side each
#: names. ``created`` and ``deleted`` name the side the file does not have.
_HEADER_SIDES: tuple[tuple[str, Literal["old", "new", "created", "deleted"]], ...] = (
    ("rename from ", "old"),
    ("rename to ", "new"),
    ("copy from ", "created"),
    ("copy to ", "new"),
    ("new file mode ", "created"),
    ("deleted file mode ", "deleted"),
)


def _header_path(name: str) -> str:
    """A name from a ``rename``/``copy`` line, which git writes bare.

    No ``a/``/``b/`` prefix and no tab, so nothing is stripped or cut: a file
    under a directory named ``a`` is ``a/x.py`` there. It is quoted when it
    has to be, as in any other header.
    """

    return _unquote(name) if name.startswith('"') else name


def _hunk_count(value: str | None) -> int:
    # "@@ -3 +3 @@": a count left out is one line.
    return 1 if value is None else int(value)


def added_lines(diff: str) -> dict[str, list[AddedLine]]:
    """Return the lines a diff adds, with their new-file line numbers.

    A file that only loses lines still appears, with an empty list: callers use
    the keys to answer "did this change touch that file at all".
    """

    added: dict[str, list[AddedLine]] = {}
    for patch in file_patches(diff):
        if patch.new_path is not None:
            added.setdefault(patch.new_path, []).extend(patch.added)
    return added


def removed_lines(diff: str) -> dict[str, list[RemovedLine]]:
    """Return the lines a diff removes, with their base-file line numbers.

    The mirror of ``added_lines``, keyed by the path on the base side: a file
    that only gains lines still appears, with an empty list, and a file the
    change creates does not appear at all.
    """

    removed: dict[str, list[RemovedLine]] = {}
    for patch in file_patches(diff):
        if patch.old_path is not None:
            removed.setdefault(patch.old_path, []).extend(patch.removed)
    return removed


def file_line_counts(diff: str) -> dict[str, tuple[int, int]]:
    """Added and removed line counts per file.

    A forge hands over one diff per file with no counts attached, and the blast
    radius the structural guardrails compare against has to be read out of the
    same text the rule engine reads its added lines from. Counting it here keeps
    that single reading.
    """

    counts: dict[str, tuple[int, int]] = {}
    for patch in file_patches(diff):
        # A deletion's ``+++`` header is /dev/null, so it is counted under its
        # base-side path: a deleted file that vanishes from the inventory is a
        # sensitive path the guardrails would stop seeing.
        path = patch.new_path or patch.old_path
        if path is None:
            continue
        added, removed = counts.get(path, (0, 0))
        counts[path] = (added + len(patch.added), removed + len(patch.removed))
    return counts


class Move(NamedTuple):
    """A file a diff deletes whole and creates again, line for line, under another name."""

    old_path: str
    new_path: str
    #: Lines in the file: the diff removes this many and adds them all back.
    lines: int


def moved_files(diff: str) -> list[Move]:
    """Files a diff deletes and creates again elsewhere with the same lines.

    That is how git without rename detection -- which local intake runs --
    shows a moved file. A deletion pairs with a creation only when every line
    matches, in order: when the move also edited the file, telling which new
    file carries on which deleted one takes a similarity guess, and only an
    exact match makes "nothing but the name changed" a fact of the diff.
    Deletions pair with creations in the order the diff lists them, each at
    most once.
    """

    patches = file_patches(diff)
    deleted: dict[tuple[str, ...], list[str]] = {}
    for patch in patches:
        if patch.old_path is not None and patch.new_path is None and patch.removed:
            lines = tuple(line.text for line in patch.removed)
            deleted.setdefault(lines, []).append(patch.old_path)
    moves: list[Move] = []
    for patch in patches:
        if patch.old_path is None and patch.new_path is not None and patch.added:
            lines = tuple(line.text for line in patch.added)
            sources = deleted.get(lines)
            if sources:
                moves.append(Move(sources.pop(0), patch.new_path, len(lines)))
    return moves


def changed_line_locations(diff: str) -> dict[str, set[int]]:
    """Return newly-added line numbers per file from a git-style unified diff."""

    return {path: {item.line for item in lines} for path, lines in added_lines(diff).items()}


#: Unicode categories no path here may hold. None of them has a glyph of its
#: own; each one controls the terminal, changes how the characters around it
#: are shown, or hides between them, so a name holding one displays as a name
#: it is not:
#:
#: * ``Cc``, the controls: C0, DEL, and C1, where U+009B is read as ``ESC [``
#:   by some terminals and U+0085 breaks a line for Python and for many editors.
#: * ``Cf``, the format characters. The bidi controls (U+202A-U+202E,
#:   U+2066-U+2069, U+200E, U+200F, U+061C) reorder the name, so
#:   ``src/<U+202E>yp.nimda`` reads as ``src/admin.py``. The zero-width ones
#:   (U+200B-U+200D, U+2060, U+FEFF) make two different names look alike, so
#:   ``src/pay<U+200B>ment/`` reads as a ``**/payment/**`` path no glob matches.
#:   The whole category rather than a list of those, because each of its
#:   characters is invisible or acts on its neighbours, and a list would miss
#:   the next one. What that costs: a zero-width joiner inside an emoji, or a
#:   zero-width non-joiner in a Persian or Indic name, sends the change to a
#:   person rather than past an automated verdict.
#: * ``Zl`` and ``Zp``, U+2028 and U+2029, which break a line in JavaScript, in
#:   ``str.splitlines`` and in many editors.
#: * ``Cs``, a lone surrogate: JSON can carry one, and no file name can.
#:
#: Left out: ``Co``, private use, which a font draws as one character like any
#: other; and ``Cn``, unassigned, since which code points those are depends on
#: the Unicode version of the interpreter, and a verdict must not change with a
#: Python upgrade. ``printable_path`` still shows both escaped.
_UNSAFE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})


def is_safe_relative_path(path: str) -> bool:
    """Reject anything that could point outside the tree under verification.

    Paths here come from a diff someone else wrote, and they end up as tool
    arguments and report locations. Absolute paths, traversal, drive letters,
    leading dashes (which a tool would read as a flag) and characters that
    cannot be shown as they are (``_UNSAFE_CATEGORIES``) are all rejected
    rather than sanitised.
    """

    if not path or path.startswith("-"):
        return False
    normalized = Path(path)
    if normalized.is_absolute() or ":" in path or ".." in normalized.parts:
        return False
    return not any(unicodedata.category(char) in _UNSAFE_CATEGORIES for char in path)


def printable_path(path: str) -> str:
    """``path`` as a report, a comment or a terminal line can show it.

    A refused name still has to be named wherever the change is reported. Shown
    raw, a newline in it starts a new line of Markdown and an escape sequence
    rewrites the terminal the log is read in, so a name with anything
    unprintable is shown quoted and escaped, the way ``repr`` writes it.
    """

    return path if path.isprintable() else repr(path)


_BACKTICKS_RE = re.compile(r"`+")


def markdown_code(text: str) -> str:
    """``text`` as an inline code span in Markdown, whatever it holds.

    For text whoever wrote the change chose -- a file name, a line it added --
    quoted in a report or a comment this product writes. A single-backtick span
    ends at the first backtick inside it: ``a` **Approved** ![x](…) `b.py``
    would close it, and the rest would render as bold text and an image in the
    platform comment. CommonMark closes a span only on a run of exactly as many
    backticks as opened it, so the fence is one longer than the longest run in
    the text. A span also drops one space at each end when it has both, so text
    that starts or ends with a backtick or a space is padded to keep all of it.
    Inside a span nothing else is Markdown -- no emphasis, link, image, HTML or
    mention -- and what cannot be printed is escaped first, as
    ``printable_path`` escapes a name: a line break would end the span's line,
    and a lone ``\\r`` is one to CommonMark.

    What can still open the span is the redaction every renderer runs over its
    finished text. A ``token=`` value runs to the next quote or space, so at the
    end of the text it would take the closing fence with it, and whatever
    followed a backtick left open would read as Markdown again. Padding cannot
    stop it on its own -- after ``token=`` a space is skipped over too -- so the
    value is redacted here, where it ends with the text, and the pad keeps the
    fence out of reach of the next pass.
    """

    shown = printable_path(text)
    if not shown:
        # Two backticks with nothing between them are a literal run, not a
        # span; a lone space is the nearest span CommonMark keeps whole.
        return "` `"
    # A backtick after the text stands for the fence, and for whatever follows it.
    redacted = redact_text(f"{shown}`")
    exposed = not redacted.endswith("`")
    if exposed:
        shown = redacted
    runs = _BACKTICKS_RE.findall(shown)
    fence = "`" * (max((len(run) for run in runs), default=0) + 1)
    # A span of nothing but spaces is the one kind CommonMark keeps whole.
    pad = " " if shown.strip(" ") and (exposed or shown[0] in "` " or shown[-1] in "` ") else ""
    return f"{fence}{pad}{shown}{pad}{fence}"


def markdown_code_block(text: str) -> str:
    """``text`` as a fenced code block in Markdown, whatever it holds.

    ``markdown_code`` for output many lines long, which a span cannot hold: a
    scanner's or a test run's, which the change under verification controls
    -- a failing test prints what it likes, and a scanner echoes the source it
    choked on. CommonMark closes a fenced block only at a line holding a run of
    the fence's own character at least as long as the fence, so the fence is
    one backtick longer than the longest run in the text, and three at least.
    A tilde cannot close a backtick fence, so tildes are not counted. Inside,
    nothing is Markdown. What cannot be printed is escaped, one character at a
    time so each line keeps its shape: an escape sequence would rewrite the
    terminal the pack is read in, and a lone ``\\r`` is a line break to
    CommonMark. A tab is left as it is, since tools line their output up
    with it. The redaction every renderer runs over its finished text stops
    at the end of a line, so a line ending in ``token=`` cannot take the
    closing fence, or the ``>`` that keeps the block inside a quote, as its
    value.
    """

    lines = [_printable_line(line) for line in split_lines(text)]
    runs = _BACKTICKS_RE.findall("\n".join(lines))
    fence = "`" * max(3, max((len(run) for run in runs), default=0) + 1)
    return "\n".join((fence, *lines, fence))


def _printable_line(line: str) -> str:
    return "".join(
        char if char.isprintable() or char == "\t" else repr(char)[1:-1] for char in line
    )


#: A fenced code block's opening line: a run of three or more backticks or
#: tildes, and the info string after it.
_FENCE_RE = re.compile(r"( *)(`{3,}|~{3,})(.*)")
#: Blank lines, which end a paragraph and every code span open in it.
_PARAGRAPH_BREAK_RE = re.compile(r"(\n(?:[ \t]*\n)+)")
#: What a platform links or notifies from wherever it stands in text: a URL by
#: its scheme or its ``www.``, an address or a mention by its ``@``. Taken from
#: the start of the run it stands in, to the end of the printable ASCII after
#: it, so no part of what could be linked is left in text.
_REACHING_RE = re.compile(
    r"(?i)(?<![a-z0-9+.%_\-])[a-z0-9+.%_\-]*"
    r"(?:://|www\.|(?:mailto|xmpp):|@(?=[a-z0-9_]))[!-~]*"
)
#: What opens a link, an image, raw HTML, an entity or a code span, what
#: escapes it, and a run of tildes long enough to open a code block.
_PROSE_SYNTAX_RE = re.compile(r"[\\`\[\]<&]|~{3,}")


def markdown_prose(text: str) -> str:
    """Model-written ``text`` as Markdown that can format but cannot reach out.

    A model's prose is meant to be Markdown, so a code span around it would
    wreck it; but the model wrote it after reading the diff, and a change
    author can steer it into an image that fetches a tracking pixel, a link,
    raw HTML or a mention, posted under this product's name. Emphasis, lists
    and code are kept. What could reach out is not rendered:

    - ``[`` and ``]`` are escaped, which leaves no link, image or reference
      definition; ``<`` too, which leaves no raw HTML and no ``<…>`` autolink;
      ``&``, since the platforms find links and mentions in text after an
      entity has become the character it names; and ``\\``, so that a
      backslash of the model's cannot undo one of these escapes.
    - A bare URL, a ``www.`` address, an e-mail address and an ``@`` mention
      are quoted as code. An escape would not do: GitHub and GitLab find them
      in the rendered HTML, where the escape is gone, and leave code alone.
    - A backtick outside a code span is escaped, and a run of three tildes, so
      nothing the model writes opens a code block of its own that would run on
      over whatever the report says next.
    - The model's code spans and fenced blocks are written anew with
      ``markdown_code`` and ``markdown_code_block``, which nothing in them can
      close. A block the model never closed ends where its text ends, and the
      info string is dropped: ``mermaid`` or ``plantuml`` there is a diagram
      the platform renders, links and all. Nor does a block open the text: a
      report writes it after a label on the same line (``The model's reasoning:``), where
      a fence is not one, and the code under it would be read as Markdown.

    Where this reading of the model's code differs from a renderer's -- an
    escaped backtick, a code span inside a link -- the difference is how the
    text looks, never what it can do, since every piece is written out here
    in a form whose reading does not depend on its neighbours. The text is
    redacted first, as the final pass over the whole report will redact it,
    so that pass has nothing left to take from the fences.
    """

    lines = [_printable_line(line) for line in split_lines(redact_text(text))]
    written: list[str] = []
    prose: list[str] = []
    index = 0
    while index < len(lines):
        opening = _FENCE_RE.fullmatch(lines[index])
        # CommonMark does not open a backtick fence whose info string holds a
        # backtick; it is a line of text, with code spans in it.
        if opening is None or (opening[2][0] == "`" and "`" in opening[3]):
            prose.append(lines[index])
            index += 1
            continue
        indent, fence = opening[1], opening[2]
        closing = re.compile(rf" *{re.escape(fence[0])}{{{len(fence)},}}[ \t]*")
        end = next(
            (at for at in range(index + 1, len(lines)) if closing.fullmatch(lines[at])),
            len(lines),
        )
        if prose:
            written.append(_inert_prose("\n".join(prose)))
            prose = []
        elif not written:
            written.append("")
        body = [line.removeprefix(indent) for line in lines[index + 1 : end]]
        written += [f"{indent}{line}" for line in markdown_code_block("\n".join(body)).split("\n")]
        index = end + 1
    if prose:
        written.append(_inert_prose("\n".join(prose)))
    return "\n".join(written)


def _inert_prose(text: str) -> str:
    # A code span ends with the paragraph it opened in.
    return "".join(
        part if _PARAGRAPH_BREAK_RE.fullmatch(part) else _inert_paragraph(part)
        for part in _PARAGRAPH_BREAK_RE.split(text)
    )


def _inert_paragraph(text: str) -> str:
    """One paragraph: its code spans written anew, the text around them escaped."""

    pieces: list[tuple[bool, str]] = []
    runs = list(_BACKTICKS_RE.finditer(text))
    position = 0
    index = 0
    while index < len(runs):
        opener = runs[index]
        closer = next(
            (at for at in range(index + 1, len(runs)) if runs[at][0] == opener[0]),
            None,
        )
        if closer is None:
            # Unmatched, a run is literal backticks, and the next run may open.
            index += 1
            continue
        pieces += _inert_text(text[position : opener.start()])
        content = text[opener.end() : runs[closer].start()].replace("\n", " ")
        if content.startswith(" ") and content.endswith(" ") and content.strip(" "):
            content = content[1:-1]
        pieces.append((True, markdown_code(content)))
        position = runs[closer].end()
        index = closer + 1
    pieces += _inert_text(text[position:])
    pieces = [(code, piece) for code, piece in pieces if piece]
    written = ""
    for at, (code, piece) in enumerate(pieces):
        # Two spans side by side are one run of backticks where they meet, and
        # a renderer would pair their fences otherwise than they were written.
        if code and at and pieces[at - 1][0]:
            written += " "
        written += piece
    return written


def _inert_text(text: str) -> list[tuple[bool, str]]:
    """Text outside code: what could reach out quoted as code, the rest escaped."""

    pieces: list[tuple[bool, str]] = []
    position = 0
    for reaching in _REACHING_RE.finditer(text):
        # Trailing punctuation reads as the sentence's, and a renderer leaves it
        # out of the link too; the part that could be linked stays quoted.
        quoted = reaching[0].rstrip(".,:;!?'\")]>")
        pieces += [
            (False, _escaped(text[position : reaching.start()])),
            (True, markdown_code(quoted)),
        ]
        position = reaching.start() + len(quoted)
    return [*pieces, (False, _escaped(text[position:]))]


def _escaped(text: str) -> str:
    return _PROSE_SYNTAX_RE.sub(lambda syntax: "".join(f"\\{char}" for char in syntax[0]), text)


def markdown_path(path: str) -> str:
    """A file name in Markdown: ``markdown_code``, since a name is the change author's text."""

    return markdown_code(path)


__all__ = [
    "AddedLine",
    "FilePatch",
    "Move",
    "PatchLine",
    "RemovedLine",
    "added_lines",
    "changed_line_locations",
    "count_changed_lines",
    "diff_label",
    "extract_changed_files",
    "file_line_counts",
    "file_patches",
    "is_safe_relative_path",
    "markdown_code",
    "markdown_code_block",
    "markdown_path",
    "markdown_prose",
    "moved_files",
    "normalize_diff_path",
    "printable_path",
    "removed_lines",
    "split_lines",
]
