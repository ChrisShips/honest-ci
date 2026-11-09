from __future__ import annotations

import pytest

from honest_ci.utils import validate_outbound_http_url

BASE = "https://api.deepseek.com/v1"


def test_url_under_base_is_accepted() -> None:
    assert validate_outbound_http_url(f"{BASE}/chat/completions", allowed_base=BASE)


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.com/v1/chat/completions",  # different origin
        "https://api.deepseek.com/other/chat",  # escapes the base path
        "http://api.deepseek.com/v1/chat",  # not https
        "https://user:pw@api.deepseek.com/v1/chat",  # embedded credentials
        "https://api.deepseek.com/v1/../admin",  # dot segment
        "https://api.deepseek.com/v1/%2e%2e/admin",  # encoded dot segment
        "ftp://api.deepseek.com/v1",  # non-http scheme
    ],
)
def test_unsafe_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        validate_outbound_http_url(url, allowed_base=BASE)


def test_loopback_http_is_allowed_only_when_requested() -> None:
    url = "http://127.0.0.1:8080/v1/chat"
    assert validate_outbound_http_url(
        url, allowed_base="http://127.0.0.1:8080/v1", allow_loopback_http=True
    )
    with pytest.raises(ValueError):
        validate_outbound_http_url(url, allowed_base="http://127.0.0.1:8080/v1")
