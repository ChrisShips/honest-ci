"""Secret redaction for everything that leaves the process.

Every artifact, log line and model payload passes through here first. The
patterns cover the credentials this product actually handles: forge tokens
(GitLab/Gitee/GitHub), ``sk-`` model keys (OpenAI, Anthropic, DeepSeek,
Bailian/Qwen), Z.ai/GLM keys, honest-ci's own license keys, and generic
``key: value`` shapes found in config dumps.
"""

from __future__ import annotations

import re
from typing import Any

# A pattern may keep a readable prefix by naming it ``keep``; everything the
# pattern matches outside that group is replaced. Token-shaped patterns have no
# ``keep`` group precisely because the whole match IS the secret -- capturing it
# would re-emit the credential next to the ``[REDACTED]`` marker.
#
# No pattern reaches past the end of a line. A line that ends in ``token=`` has
# no value, and taking the next line's first word as one replaced whatever
# stood there: a diff's ``+``/``-`` marker, which corrupts the hunk, or the
# fence that closes a code block in a report.
#
# Nor does a value reach past a comma. A comma separates the fields of a CSV
# row, the arguments of a call and the pairs of a log line, and a value that
# ran across it took the rest with it: ``a,token:,b`` became
# ``a,token:[REDACTED]``, one column short. No token shape below holds a
# comma; a password that does is masked up to it.
#
# A token starts where a word does. Without that, ``task-management-system``
# holds an ``sk-`` key and ``test_github_adapter_rejects_forged_paths`` a
# ``github_`` one, and the repository, path or test name the evidence is about
# is masked. A ``-`` in front still starts one: a diff shows a removed line's
# first character that way, and a key alone on its line must be caught there.
_TOKEN_START = r"(?<![a-z0-9_])"

SECRET_PATTERNS = [
    re.compile(r"(?i)(?P<keep>bearer[ \t]+)[a-z0-9._\-]{16,}"),
    re.compile(rf"(?i){_TOKEN_START}(?:github|ghp|gho|ghu|ghs|ghr)_[a-z0-9_]{{16,}}"),
    re.compile(rf"(?i){_TOKEN_START}glpat-[a-z0-9_\-]{{16,}}"),  # GitLab personal access token
    re.compile(rf"(?i){_TOKEN_START}gldt-[a-z0-9_\-]{{16,}}"),  # GitLab deploy token
    # sk-proj-… (OpenAI), sk-ant-… (Anthropic), DeepSeek, Bailian/DashScope
    re.compile(rf"(?i){_TOKEN_START}sk-[a-z0-9_\-]{{16,}}"),
    re.compile(rf"(?i){_TOKEN_START}zai_[a-z0-9_\-]{{16,}}"),
    # honest-ci's license key, hci1.<payload>.<signature>. Neither part alone
    # makes a key, but the whole of it is masked, with whatever more parts
    # follow: a key cut short or run together with something else is masked as
    # far as it reaches, where requiring the exact shape would let it through.
    # Its variable's name can't be the cue, as ``token`` is for a forge token:
    # HONEST_CI_LICENSE_KEY names no credential word.
    re.compile(rf"(?i){_TOKEN_START}hci1\.[a-z0-9_\-]{{16,}}(?:\.[a-z0-9_\-]+)*"),
    re.compile(
        r"(?i)(?P<keep>(?:api[_-]?key|token|secret|password)[ \t]*[:=][ \t]*[\"']?)[^\"'\s,]+"
    ),
]

#: Key names that carry a credential word but hold a count. ``token`` is the one
#: credential word that doubles as a unit of measurement, which is the only
#: reason this list exists -- and why it stays closed: a name that merely looks
#: like a counter is not worth the chance of shipping a credential past the key
#: check. These are the names the contracts write (``RunMetrics.token_usage``,
#: ``LLMCallRecord.input_tokens``/``output_tokens``/``total_tokens``) plus the
#: OpenAI-compatible usage fields the structured adapter reads. Redacting them
#: is not a harmless over-reach: the contracts type them as integers, so a
#: ``[REDACTED]`` string is what made ``evidence.json`` -- the machine-readable
#: half of an evidence pack -- fail ``EvidencePack.model_validate``.
COUNTER_KEYS = frozenset(
    {
        "token_usage",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    }
)


def redact_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        if "keep" in pattern.groupindex:
            redacted = pattern.sub(lambda match: f"{match.group('keep')}[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_obj(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_obj(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_obj(item) for item in value)
    if isinstance(value, dict):
        # A key is text too, and ``ArtifactStore.write_json`` no longer passes
        # the serialized JSON through ``redact_text``, which was the only thing
        # that reached it. Two keys that differ only in a credential become one,
        # which loses an entry but never shows the credential.
        return {
            (redact_text(key) if isinstance(key, str) else key): _redact_entry(str(key), item)
            for key, item in value.items()
        }
    return value


def _redact_entry(key: str, value: Any) -> Any:
    if _looks_sensitive(key) and not _is_counter(key, value):
        return "[REDACTED]"
    # A spared counter is spared at the key-name layer only: it keeps walking,
    # so every string under it still meets the pattern layer on the way out.
    return redact_obj(value)


def _looks_sensitive(key: str) -> bool:
    return bool(re.search(r"(?i)(token|secret|password|api[_-]?key|authorization)", key))


def _is_counter(key: str, value: Any) -> bool:
    """Whether this entry counts tokens rather than holding a credential.

    Both halves have to hold. The name alone is not enough -- a count is a
    number, so a string parked under a counter name is not one and goes back to
    being redacted wholesale -- and the shape alone is not enough either,
    because an integer beside an unrecognised credential word proves nothing.
    """

    if key.strip().lower() not in COUNTER_KEYS:
        return False
    if isinstance(value, bool):
        # ``bool`` is an ``int`` in Python, and a flag is not a count.
        return False
    # ``None`` is a count this run does not have: the contracts type per-call
    # counters as ``int | None`` for providers that report no usage. A mapping
    # is one too (``token_usage``), because recursion judges every entry inside
    # it by its own name rather than inheriting this exemption.
    return value is None or isinstance(value, int | dict)
