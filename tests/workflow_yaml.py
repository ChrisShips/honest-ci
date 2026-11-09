"""Enough YAML to read the two workflow files this repository ships.

``action.yml`` and the workflow template carry the properties that make the
Action safe to install -- which events start it, what its token may do, which
commit each action it uses is pinned to -- and the tests hold those on the
parsed files rather than on their text: a comment that says "never
pull_request_target" is not a trigger. A YAML library would be a new
dependency for two files this repository writes itself, so this reads the
subset they are written in and refuses the rest. A file that grows past the
subset fails the tests loudly instead of being half read.

The subset: block mappings and block sequences indented with spaces, unquoted
keys, plain and quoted scalars, ``|`` and ``>`` block scalars (clipped, or
stripped with ``-``), and ``#`` comments. Every scalar stays the string it is
written as: ``fetch-depth: 0`` reads as ``"0"``.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: ``key:`` or ``key: value``. Every key in the two files is a bare word; a
#: quoted or a complex key is outside the subset.
_ENTRY = re.compile(r"(?P<key>[A-Za-z0-9_-]+):(?: +(?P<value>.*))?")

#: A plain scalar cannot start with these: each opens something the subset
#: does not read (a flow collection, an anchor, an alias, a tag, a directive).
_UNSUPPORTED_START = frozenset("[]{}&*!%@`,?")


class UnsupportedYaml(ValueError):
    """The text uses YAML this reader does not read."""


def load(text: str) -> dict[str, Any]:
    """The document as nested dicts, lists and strings."""

    reader = _Reader(text.splitlines())
    peeked = reader.peek()
    if peeked is None or peeked[0] != 0:
        raise UnsupportedYaml("the document does not start with a key at column 0")
    document = reader.node(0)
    if not isinstance(document, dict):
        raise UnsupportedYaml("the document is not a mapping")
    return document


class _Reader:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.index = 0

    def error(self, message: str) -> UnsupportedYaml:
        return UnsupportedYaml(f"line {self.index + 1}: {message}")

    def peek(self) -> tuple[int, str] | None:
        """Indentation and content of the next line that holds any, past comments."""

        while self.index < len(self.lines):
            line = self.lines[self.index]
            content = line.lstrip(" ")
            if content.startswith("\t"):
                raise self.error("indented with a tab")
            if content.strip() and not content.startswith("#"):
                return len(line) - len(content), content.rstrip()
            self.index += 1
        return None

    def node(self, indent: int) -> dict[str, Any] | list[Any]:
        peeked = self.peek()
        if peeked is not None and _is_item(peeked[1]):
            return self.sequence(indent)
        return self.mapping(indent)

    def mapping(self, indent: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while (peeked := self.peek()) is not None and peeked[0] == indent:
            if _is_item(peeked[1]):
                # Also what a sequence written level with its key looks like;
                # the files indent theirs.
                raise self.error("a sequence item where a mapping key belongs")
            match = _ENTRY.fullmatch(peeked[1])
            if match is None:
                raise self.error(f"not a `key: value` line: {peeked[1]!r}")
            key = match["key"]
            if key in result:
                raise self.error(f"duplicate key {key!r}")
            self.index += 1
            result[key] = self.value(indent, match["value"])
        self.expect_dedent(indent)
        return result

    def sequence(self, indent: int) -> list[Any]:
        items: list[Any] = []
        while (peeked := self.peek()) is not None and peeked[0] == indent:
            if not _is_item(peeked[1]):
                raise self.error("a mapping key where a sequence item belongs")
            rest = peeked[1][1:].lstrip(" ")
            if not rest or rest.startswith("#"):
                raise self.error("a sequence item with nothing on its own line")
            if _ENTRY.fullmatch(rest):
                # "- key: value" opens a mapping whose keys line up with the
                # first one. Read the dash as the space it stands in for.
                inner = indent + len(peeked[1]) - len(rest)
                self.lines[self.index] = " " * inner + rest
                items.append(self.mapping(inner))
            else:
                self.index += 1
                items.append(self.scalar(rest))
        self.expect_dedent(indent)
        return items

    def value(self, indent: int, raw: str | None) -> Any:
        if raw is None or raw.startswith("#"):
            peeked = self.peek()
            if peeked is None or peeked[0] <= indent:
                return None
            return self.node(peeked[0])
        if raw in {"|", "|-", ">", ">-"}:
            return self.block_scalar(indent, folded=raw[0] == ">", strip=raw.endswith("-"))
        return self.scalar(raw)

    def block_scalar(self, indent: int, *, folded: bool, strip: bool) -> str:
        lines: list[str] = []
        block: int | None = None
        while self.index < len(self.lines):
            line = self.lines[self.index]
            if not line.strip():
                lines.append("")
                self.index += 1
                continue
            current = len(line) - len(line.lstrip(" "))
            if current <= indent:
                break
            if block is None:
                block = current
            elif current < block or (folded and current > block):
                raise self.error("a block scalar line out of line with the first one")
            lines.append(line[block:])
            self.index += 1
        while lines and not lines[-1]:
            lines.pop()
        if folded:
            paragraphs = "\n".join(lines).split("\n\n")
            text = "\n".join(" ".join(part.split("\n")) for part in paragraphs)
        else:
            text = "\n".join(lines)
        return text if strip or not text else f"{text}\n"

    def scalar(self, raw: str) -> str:
        if raw[0] == "'":
            value, end = _single_quoted(raw)
        elif raw[0] == '"':
            try:
                decoded, end = json.JSONDecoder().raw_decode(raw)
            except ValueError as exc:
                raise self.error(f"a double-quoted value this reader cannot read: {raw!r}") from exc
            value = str(decoded)
        else:
            if raw[0] in _UNSUPPORTED_START or raw[0] in "|>" or _is_item(raw):
                raise self.error(f"a value outside the subset: {raw!r}")
            plain = re.split(r"\s#", raw, maxsplit=1)[0].rstrip()
            if ": " in plain or plain.endswith(":"):
                raise self.error(f"a plain value that reads as a mapping: {raw!r}")
            return plain
        rest = raw[end:].strip()
        if rest and not rest.startswith("#"):
            raise self.error(f"text after a quoted value: {raw!r}")
        return value

    def expect_dedent(self, indent: int) -> None:
        peeked = self.peek()
        if peeked is not None and peeked[0] > indent:
            raise self.error("indented deeper than anything above it allows")


def _is_item(content: str) -> bool:
    return content == "-" or content.startswith("- ")


def _single_quoted(raw: str) -> tuple[str, int]:
    """A single-quoted scalar on one line, where ``''`` is a quote."""

    value: list[str] = []
    index = 1
    while index < len(raw):
        if raw[index] == "'":
            if raw[index + 1 : index + 2] == "'":
                value.append("'")
                index += 2
                continue
            return "".join(value), index + 1
        value.append(raw[index])
        index += 1
    raise UnsupportedYaml(f"a single-quoted value that does not close on its line: {raw!r}")
