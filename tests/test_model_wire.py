"""What each model provider actually puts on the wire.

The adapter unit tests swap the HTTP layer for a fake transport, so they never
see a request body. These run the real ``urllib`` path against a server on
loopback -- the one plain-HTTP host ``_request`` accepts -- and read back what
arrived: path, headers, body.

The server also holds the line the real services hold on the fields this file
is about. OpenAI's reasoning models refuse ``max_tokens`` and ``temperature``,
and current Claude models refuse ``temperature``, each with a 400. A body the
real service would refuse is refused here too.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from honest_ci.adapters import build_llm_adapter
from honest_ci.adapters.openai import OpenAIError
from honest_ci.adapters.structured import StructuredModelAdapter
from honest_ci.contracts import BlockBrief
from honest_ci.settings import Settings

# The placeholder the gitleaks config already allows by exact value.
KEY = "sk-test-key-value-1234"

BRIEF = {
    "gate": "review",
    "attempt": 1,
    "source_fingerprint": "src-1",
    "summary": "Refund amounts are split in float.",
    "root_causes": ["float instead of Decimal"],
    "recommended_actions": ["Use decimal.Decimal and assert on the remainder."],
    "cited_evidence_ids": ["review.R-01"],
    "confidence": 0.7,
}


# ── a model service that answers on loopback ─────────────────────────────────


@dataclass
class Received:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


Answer = Callable[[Received], tuple[int, Any]]


def vendor_contract(received: Received) -> tuple[int, Any]:
    """Refuse what the real service refuses; otherwise answer with a brief."""

    body = received.body
    if received.path.endswith("/chat/completions"):
        if body["model"].startswith("gpt-") and {"max_tokens", "temperature"} & body.keys():
            return 400, {"error": {"type": "invalid_request_error", "message": "unsupported"}}
        return 200, {
            "choices": [{"message": {"role": "assistant", "content": json.dumps(BRIEF)}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        }
    if received.path.endswith("/v1/messages"):
        if body["model"].startswith("claude-") and "temperature" in body:
            return 400, {"type": "error", "error": {"type": "invalid_request_error"}}
        return 200, {
            # Current Claude models think by default; the thinking comes back
            # as its own block, ahead of the text that holds the answer.
            "content": [
                {"type": "thinking", "thinking": "", "signature": "placeholder"},
                {"type": "text", "text": json.dumps(BRIEF)},
            ],
            "usage": {"input_tokens": 120, "output_tokens": 30},
        }
    return 404, {"error": "no such route"}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        server = self.server
        assert isinstance(server, _Server)
        length = int(self.headers.get("Content-Length") or 0)
        received = Received(
            path=self.path,
            headers={key.lower(): value for key, value in self.headers.items()},
            body=json.loads(self.rfile.read(length)),
        )
        server.received.append(received)
        status, payload = server.answer(received)
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: Any) -> None:
        """Keep the test output readable."""


class _Server(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.received: list[Received] = []
        self.answer: Answer = vendor_contract

    @property
    def origin(self) -> str:
        host, port = self.server_address[:2]
        assert isinstance(host, str)
        return f"http://{host}:{port}"


@pytest.fixture
def server() -> Iterator[_Server]:
    server = _Server()
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def adapter_for(settings: Settings) -> StructuredModelAdapter:
    adapter = build_llm_adapter(settings)
    assert adapter is not None
    return adapter


def explain(adapter: StructuredModelAdapter) -> BlockBrief:
    return adapter.explain_block(
        gate="review",
        attempt=1,
        source_fingerprint="src-1",
        summary="Refund amounts are split in float.",
        deterministic_feedback="GATE BLOCKED: review",
        evidence_items=[{"id": "review.R-01", "detail": "float used for money"}],
    )


def without_messages(body: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in body.items() if key != "messages"}


# ── openai ───────────────────────────────────────────────────────────────────


def openai_settings(server: _Server, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "model_provider": "openai",
        "openai_api_key": KEY,
        "openai_base_url": f"{server.origin}/v1",
        "openai_max_retries": 0,
    }
    return Settings(**{**base, **overrides})


def test_openai_defaults_send_only_what_a_reasoning_model_accepts(server: _Server) -> None:
    adapter = adapter_for(openai_settings(server))

    brief = explain(adapter)

    (request,) = server.received
    assert request.path == "/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert without_messages(request.body) == {
        "model": "gpt-6-astra",
        "max_completion_tokens": 16000,
    }
    assert request.body["messages"][0]["role"] == "user"
    assert brief.summary == BRIEF["summary"]
    (record,) = adapter.call_records
    assert (record.provider, record.protocol, record.total_tokens) == ("openai", "openai", 150)


def test_a_server_that_reads_max_tokens_gets_it_under_that_name(server: _Server) -> None:
    # The Ollama / DeepSeek shape: the older field name, and a temperature.
    settings = openai_settings(
        server,
        openai_api_key="ollama",
        openai_model="qwen3:8b",
        openai_max_tokens=4096,
        openai_max_tokens_param="max_tokens",
        openai_temperature=0.2,
    )

    explain(adapter_for(settings))

    (request,) = server.received
    assert without_messages(request.body) == {
        "model": "qwen3:8b",
        "max_tokens": 4096,
        "temperature": 0.2,
    }


def test_a_body_the_model_refuses_is_a_recorded_failure_not_a_crash(server: _Server) -> None:
    # The body every provider used to send, pointed at a reasoning model.
    adapter = adapter_for(
        openai_settings(server, openai_max_tokens_param="max_tokens", openai_temperature=0.1)
    )

    with pytest.raises(OpenAIError, match="openai API failed: 400"):
        explain(adapter)

    (record,) = adapter.call_records
    assert record.schema_valid is False
    assert record.provider == "openai"


# ── anthropic ────────────────────────────────────────────────────────────────


def test_anthropic_sends_no_temperature_and_reads_past_the_thinking(server: _Server) -> None:
    adapter = adapter_for(
        Settings(
            model_provider="anthropic",
            anthropic_api_key=KEY,
            anthropic_base_url=server.origin,
            anthropic_max_retries=0,
        )
    )

    brief = explain(adapter)

    (request,) = server.received
    assert request.path == "/v1/messages"
    assert request.headers["x-api-key"] == KEY
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in request.headers
    assert without_messages(request.body) == {"model": "claude-opus-5", "max_tokens": 16000}
    assert brief.summary == BRIEF["summary"]
    (record,) = adapter.call_records
    assert (record.provider, record.protocol, record.total_tokens) == (
        "anthropic",
        "anthropic",
        150,
    )


# ── the providers that were already here ─────────────────────────────────────


def test_the_existing_providers_send_what_they_always_sent(server: _Server) -> None:
    deepseek = Settings(
        model_provider="deepseek",
        deepseek_api_key=KEY,
        deepseek_base_url=f"{server.origin}/v1",
        deepseek_max_retries=0,
    )
    qwen = Settings(
        model_provider="qwen",
        qwen_api_key=KEY,
        qwen_base_url=f"{server.origin}/compatible-mode/v1",
        qwen_max_retries=0,
    )
    glm = Settings(
        model_provider="glm",
        glm_api_key=KEY,
        glm_base_url=f"{server.origin}/api/anthropic",
        glm_max_retries=0,
    )
    for settings in (deepseek, qwen, glm):
        explain(adapter_for(settings))

    deepseek_call, qwen_call, glm_call = server.received
    assert deepseek_call.path == "/v1/chat/completions"
    assert without_messages(deepseek_call.body) == {
        "model": "deepseek-chat",
        "max_tokens": 8192,
        "temperature": 0.1,
    }
    assert qwen_call.path == "/compatible-mode/v1/chat/completions"
    assert without_messages(qwen_call.body) == {
        "model": "qwen3.8-max",
        "max_tokens": 8192,
        "temperature": 0.1,
        "enable_thinking": False,
    }
    assert glm_call.path == "/api/anthropic/v1/messages"
    assert glm_call.headers["x-api-key"] == KEY
    assert without_messages(glm_call.body) == {
        "model": "glm-5.2",
        "max_tokens": 4096,
        "temperature": 0.1,
    }
