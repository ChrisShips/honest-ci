from __future__ import annotations

import csv
import io

from honest_ci.license_check import KEY_VARIABLE
from honest_ci.redaction import redact_obj, redact_text

from .test_licensing import issue


def test_forge_and_model_tokens_are_removed() -> None:
    text = (
        "gitlab=glpat-abcdefghijklmnopqrst "
        "github=ghp_abcdefghijklmnopqrstuvwxyz012345 "
        "deepseek=sk-abcdefghijklmnopqrstuvwx "
        "zai=zai_abcdefghijklmnopqrstuvwx"
    )
    redacted = redact_text(text)
    assert "glpat-" not in redacted
    assert "ghp_" not in redacted
    assert "sk-abcdefghijklmnopqrstuvwx" not in redacted
    assert "zai_abcdefghijklmnopqrstuvwx" not in redacted
    assert redacted.count("[REDACTED]") == 4


def test_openai_and_anthropic_key_shapes_are_removed_whole() -> None:
    # Built from the allowlisted placeholder instead of written out, so this
    # adds no new secret-shaped literal for the scanner to judge.
    body = "abcdefghijklmnopqrstuvwx"
    for key in (f"sk-proj-{body}", f"sk-ant-api03-{body}"):
        redacted = redact_text(f"key={key} rest")

        assert body not in redacted
        assert key.split("-")[1] not in redacted  # no vendor prefix left dangling
        assert redacted.endswith(" rest")


def test_bearer_prefix_is_kept_but_credential_is_not() -> None:
    redacted = redact_text("Authorization: Bearer abcdefghijklmnopqrstuvwxyz")
    assert "Bearer [REDACTED]" in redacted
    assert "abcdefghijklmnop" not in redacted


def test_sensitive_keys_are_replaced_wholesale() -> None:
    payload = {
        "api_key": "whatever-this-is",
        "nested": {"Authorization": "Bearer abcdefghijklmnopqrstuvwx", "safe": "keep me"},
        "list": ["token=abcdefghijklmnop", "plain"],
    }
    redacted = redact_obj(payload)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["Authorization"] == "[REDACTED]"
    assert redacted["nested"]["safe"] == "keep me"
    assert redacted["list"][0].startswith("token=[REDACTED]")
    assert redacted["list"][1] == "plain"


def test_token_counters_survive_while_credential_keys_beside_them_do_not() -> None:
    payload = {
        "token_usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        "input_tokens": 100,
        "total_tokens": None,
        "api_token": "glpat-abcdefghijklmnopqrst",
    }
    redacted = redact_obj(payload)
    assert redacted["token_usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert redacted["input_tokens"] == 100
    assert redacted["total_tokens"] is None
    assert redacted["api_token"] == "[REDACTED]"


def test_a_counter_name_holding_something_other_than_a_count_is_still_redacted() -> None:
    # The exemption is for counts, not for the word "token": a credential does
    # not become publishable by being filed under a counter's name.
    redacted = redact_obj(
        {
            "total_tokens": "glpat-abcdefghijklmnopqrst",
            "input_tokens": True,  # ``bool`` is an ``int``, but a flag is not a count
            "token_usage": {"api_key": "whatever-this-is", "total_tokens": 120},
        }
    )
    assert redacted["total_tokens"] == "[REDACTED]"
    assert redacted["input_tokens"] == "[REDACTED]"
    assert redacted["token_usage"]["api_key"] == "[REDACTED]"
    assert redacted["token_usage"]["total_tokens"] == 120


def test_token_shapes_are_caught_wherever_a_credential_stands() -> None:
    body = "abcdefghijklmnopqrstuvwx"
    tokens = [
        f"{prefix}{body}"
        for prefix in ("sk-", "sk-proj-", "ghp_", "github_pat_", "glpat-", "gldt-", "zai_")
    ]
    for token in tokens:
        for text in (
            f"key={token}",
            f"key: {token}",
            f'"{token}"',
            f"'{token}'",
            f"Authorization: Bearer {token}",
            token,
            f"first line\n{token}",
            # Removed and added lines of a diff.
            f"-{token}",
            f"+{token}",
            f"https://{token}@example.invalid/repo.git",
        ):
            assert body not in redact_text(text), text


def test_a_license_key_is_removed_whole_wherever_it_stands() -> None:
    """Signed here rather than written out: the test adds no key for the scanner to judge."""

    key = issue()
    _, payload, signature = key.split(".")
    for text in (
        f"{KEY_VARIABLE}={key}",
        f"license-key: {key}",
        f'"{key}"',
        key,
        f"first line\n{key}\nlast line",
        # Removed and added lines of a diff.
        f"-{key}",
        f"+{key}",
        # Cut short at its second dot, and run on into another part.
        f"hci1.{payload}",
        f"{key}.{payload}",
    ):
        redacted = redact_text(text)

        assert payload not in redacted, text
        assert signature not in redacted, text
    # The shape of an environment dump: the name is no credential word.
    assert redact_obj({KEY_VARIABLE: key}) == {KEY_VARIABLE: "[REDACTED]"}
    assert redact_text(f"key {key}.") == "key [REDACTED]."


def test_names_that_merely_contain_a_token_prefix_are_left_alone() -> None:
    for text in (
        "https://github.com/acme/task-management-system/pull/3",
        "src/desk-organizer-application/test_app.py",
        "def test_github_adapter_rejects_forged_paths_in_review():",
        "feature/bonzai_tree_rendering_pipeline",
        "drivers/usb/host/xhci1.controller_quirks_table.c",
        "config/hci1.yml",
    ):
        assert redact_text(text) == text


def test_redacting_a_diff_keeps_every_line_and_its_marker() -> None:
    diff = (
        "@@ -1,4 +1,4 @@\n"
        "-  password:\n"
        "+  password: hunter2\n"
        "   token=\n"
        "-Authorization: Bearer\n"
        "+abcdefghijklmnopqrstuvwxyz\n"
    )
    redacted = redact_text(diff)

    assert "hunter2" not in redacted
    before, after = diff.splitlines(), redacted.splitlines()
    assert len(after) == len(before)
    assert [line[0] for line in after] == [line[0] for line in before]
    # Only the line that has a value on it changes.
    assert [a == b for a, b in zip(before, after, strict=True)] == [
        True,
        True,
        False,
        True,
        True,
        True,
    ]


def test_a_value_ends_at_a_comma_so_a_csv_row_keeps_its_columns() -> None:
    rows = [
        ["claude", "  password:", "deleted_test"],
        ["codex", "token=abcdefghijklmnop", "fewer_assertions"],
    ]
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)

    redacted = redact_text(buffer.getvalue())

    assert "abcdefghijklmnop" not in redacted
    assert list(csv.reader(io.StringIO(redacted))) == [
        rows[0],
        ["codex", "token=[REDACTED]", "fewer_assertions"],
    ]
    assert redact_text("a,token:,b") == "a,token:,b"
    assert redact_text("f(token=abcdefgh, retries=3)") == "f(token=[REDACTED], retries=3)"


def test_keys_are_redacted_like_values() -> None:
    body = "abcdefghijklmnopqrstuvwx"
    redacted = redact_obj({f"sk-{body}": 1, "task-management-system": {f"ghp_{body}": "x"}})

    assert body not in repr(redacted)
    assert redacted == {"[REDACTED]": 1, "task-management-system": {"[REDACTED]": "x"}}
