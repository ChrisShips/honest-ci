"""Filesystem and outbound-HTTP helpers shared by the whole verification layer.

The URL helpers are the credential boundary for every hosted model call: a
request that carries an API key must stay inside its configured origin and base
path, on the initial request and on every redirect.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import ipaddress
import shlex
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

RUNTIME_PATH_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".honest-ci",
    "__pycache__",
    "node_modules",
    "target",
}


def is_runtime_path(path: Path) -> bool:
    """Whether a path belongs to tooling state rather than to the code itself."""

    return any(part in RUNTIME_PATH_PARTS for part in path.parts)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*RUNTIME_PATH_PARTS))


def file_sha256(path: Path) -> str | None:
    """Digest of a file's bytes, or ``None`` when there is no file.

    Over bytes rather than decoded text so anyone handed the file can check
    it with ``sha256sum`` and get the same string. ``None`` rather than the
    digest of an empty input, because "there was no rule library" and "the rule
    library was empty" explain a quiet review gate differently.
    """

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def path_inside(path: Path, root: Path) -> str | None:
    """``path`` relative to ``root``, or ``None`` when it lies outside.

    Everything an evidence pack names on disk is named this way, so that a pack
    handed to someone else describes the change rather than the machine that
    produced it. What a file from *outside* the root deserves differs by what
    the file is, so this answers only the question and leaves that to callers.
    """

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def config_source(path: Path, root: Path) -> str:
    """How a configuration file is named in evidence: relative to the repo it configures.

    An evidence pack is handed to other people, and an absolute path inside one
    describes the machine that produced it rather than the configuration that
    judged the change. A file outside the repository keeps its full path --
    shortening that would name a different file, which is worse than a long
    name.
    """

    return path_inside(path, root) or str(path)


def hash_tree(root: Path) -> str:
    """Content fingerprint of a working tree, ignoring runtime state."""

    digest = hashlib.sha256()
    for file_path in sorted(
        p for p in root.rglob("*") if p.is_file() and not is_runtime_path(p.relative_to(root))
    ):
        relative = file_path.relative_to(root)
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def unified_diff(before: str, after: str, fromfile: str, tofile: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def iter_source_files(root: Path, suffixes: tuple[str, ...] | None = None) -> list[Path]:
    """List candidate source files, skipping caches, venvs and vendored trees."""

    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and not is_runtime_path(p.relative_to(root))
        and (suffixes is None or p.suffix in suffixes)
    )


def ensure_relative_path(path: str | Path) -> str:
    return str(Path(path).as_posix())


def recorded_command(command: list[str]) -> str:
    """A command as an evidence pack names it: what ran, not where it lives.

    ``argv[0]`` is an absolute path whenever the deployment resolved the
    interpreter for itself (``sys.executable``) or an operator configured one
    that way, and such a path describes the machine rather than the run: an
    auditor reading the pack cannot check ``/home/ci/.venv/bin/python3.12``,
    and the directory is not what they came for. The basename keeps every fact
    the path was carrying about what ran, the interpreter version included.

    Two things are deliberately left alone. A relative ``argv[0]`` points into
    the repository, which is exactly where a reader should be able to follow
    it. And arguments are never rewritten: they are the operator's own string,
    where a path may be the thing that makes the command mean what it means.

    Takes a command that has an executable in it; an empty one is a caller's
    bug, and raising beats recording an empty string as what ran.
    """

    head, *rest = command
    if Path(head).is_absolute():
        head = Path(head).name or head
    return shlex.join([head, *rest])


def matches_any_glob(path: str, patterns: list[str] | tuple[str, ...]) -> bool:
    """Match a repo-relative path against configured glob patterns.

    Both the structural guardrails and the team rule library filter paths this
    way. They share one implementation because a pattern that means one thing in
    ``honest-ci-policy.toml`` and another in ``honest-ci-rules.toml`` is a configuration
    people cannot audit.
    """

    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def validate_outbound_http_url(
    url: str,
    *,
    allowed_base: str | None = None,
    allow_loopback_http: bool = False,
) -> str:
    """Reject non-HTTP schemes and keep a derived URL under its configured origin/path."""

    parsed = _split_outbound_http_url(url, allow_loopback_http=allow_loopback_http)
    if allowed_base is None:
        return url

    base = _split_outbound_http_url(
        allowed_base,
        allow_loopback_http=allow_loopback_http,
        is_base=True,
    )
    if _url_origin(parsed) != _url_origin(base):
        raise ValueError("Outbound URL does not match its configured origin.")
    path = _decoded_outbound_path(parsed.path)
    base_path = _decoded_outbound_path(base.path).rstrip("/")
    if base_path and path != base_path and not path.startswith(f"{base_path}/"):
        raise ValueError("Outbound URL escapes its configured base path.")
    return url


def open_validated_http_request(
    request: urllib.request.Request,
    *,
    allowed_base: str,
    timeout_seconds: float,
    allow_loopback_http: bool = False,
) -> Any:
    """Open one request while constraining its initial URL and every redirect."""

    validate_outbound_http_url(
        request.full_url,
        allowed_base=allowed_base,
        allow_loopback_http=allow_loopback_http,
    )
    opener = urllib.request.build_opener(
        _ValidatedRedirectHandler(
            allowed_base=allowed_base,
            allow_loopback_http=allow_loopback_http,
        )
    )
    return opener.open(request, timeout=timeout_seconds)


class _ValidatedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow redirects only inside the credential-bearing request's configured base."""

    def __init__(self, *, allowed_base: str, allow_loopback_http: bool) -> None:
        super().__init__()
        self.allowed_base = validate_outbound_http_url(
            allowed_base,
            allow_loopback_http=allow_loopback_http,
        )
        self.allow_loopback_http = allow_loopback_http

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        validate_outbound_http_url(
            newurl,
            allowed_base=self.allowed_base,
            allow_loopback_http=self.allow_loopback_http,
        )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _split_outbound_http_url(
    url: str,
    *,
    allow_loopback_http: bool,
    is_base: bool = False,
) -> urllib.parse.SplitResult:
    if not url or url != url.strip():
        raise ValueError("Outbound URL must be a non-empty value without surrounding whitespace.")
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError("Outbound URL must not contain control characters.")
    try:
        parsed = urllib.parse.urlsplit(url)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Outbound URL is malformed.") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Outbound URL must not contain embedded credentials.")
    if not parsed.hostname:
        raise ValueError("Outbound URL must contain a hostname.")
    if parsed.fragment:
        raise ValueError("Outbound URL must not contain a fragment.")
    if is_base and parsed.query:
        raise ValueError("Outbound base URL must not contain a query string.")
    _decoded_outbound_path(parsed.path)
    if parsed.scheme == "https":
        return parsed
    if parsed.scheme == "http" and allow_loopback_http and _is_loopback_host(parsed.hostname):
        return parsed
    raise ValueError("Outbound URL must use HTTPS (HTTP is allowed only for loopback tests).")


def _decoded_outbound_path(path: str) -> str:
    """Decode path escapes and reject ambiguous traversal or separator forms."""

    decoded = path
    for _ in range(4):
        try:
            next_value = urllib.parse.unquote(decoded, errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("Outbound URL path contains invalid encoding.") from exc
        if next_value == decoded:
            break
        decoded = next_value
    if "%" in decoded:
        raise ValueError("Outbound URL path contains ambiguous percent encoding.")
    if any(ord(char) < 32 or ord(char) == 127 for char in decoded):
        raise ValueError("Outbound URL path must not contain control characters.")
    if "\\" in decoded:
        raise ValueError("Outbound URL path must not contain backslashes.")
    if any(segment in {".", ".."} for segment in decoded.split("/")):
        raise ValueError("Outbound URL path must not contain dot segments.")
    return decoded


def _url_origin(parsed: urllib.parse.SplitResult) -> tuple[str, str, int]:
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, parsed.hostname or "", parsed.port or default_port


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False
