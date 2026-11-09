from __future__ import annotations

import json
from typing import Any

import pytest

from honest_ci.adapters import build_llm_adapter
from honest_ci.adapters.anthropic import AnthropicAdapter, AnthropicError
from honest_ci.adapters.deepseek import DeepSeekAdapter, DeepSeekError
from honest_ci.adapters.openai import OpenAIAdapter, OpenAIError
from honest_ci.adapters.qwen import QwenAdapter
from honest_ci.adapters.structured import (
    StructuredModelAdapter,
    StructuredModelError,
    extract_json,
)
from honest_ci.contracts import EvidencePack
from honest_ci.settings import Settings

from .factories import make_change

BRIEF_JSON = {
    "gate": "review",
    "attempt": 1,
    "source_fingerprint": "src-1",
    "summary": "金额计算使用 float",
    "root_causes": ["未使用 Decimal"],
    "recommended_actions": ["改用 Decimal 并补一条断言"],
    "cited_evidence_ids": ["review.C-12"],
    "confidence": 0.7,
}


class Transport:
    """Stands in for a provider: records prompts, replays scripted responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def __call__(self, prompt: str, settings: Settings) -> tuple[str, dict[str, Any]]:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[index], {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
        }


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "model_provider": "deepseek",
        "deepseek_api_key": "sk-test-key-value-1234",
        "deepseek_max_retries": 1,
    }
    base.update(overrides)
    return Settings(**base)


def explain(adapter: StructuredModelAdapter) -> Any:
    return adapter.explain_block(
        gate="review",
        attempt=1,
        source_fingerprint="src-1",
        summary="金额计算使用 float",
        deterministic_feedback="GATE BLOCKED: review",
        evidence_items=[{"id": "review.C-12", "detail": "float 用于金额"}],
    )


def test_structured_call_validates_and_records_telemetry() -> None:
    transport = Transport([f"```json\n{json.dumps(BRIEF_JSON)}\n```"])
    adapter = DeepSeekAdapter(settings(), transport)

    brief = explain(adapter)

    assert brief.gate == "review"
    assert brief.cited_evidence_ids == ["review.C-12"]
    record = adapter.call_records[-1]
    assert record.provider == "deepseek"
    assert record.schema_valid
    assert record.total_tokens == 150
    assert record.attempts == 1


def test_malformed_output_is_retried_then_reported() -> None:
    transport = Transport(["not json at all"])
    adapter = DeepSeekAdapter(settings(deepseek_max_retries=1), transport)

    with pytest.raises(StructuredModelError):
        explain(adapter)

    # One initial attempt plus one retry, and the failure is recorded rather
    # than silently dropped.
    assert len(transport.prompts) == 2
    assert adapter.call_records[-1].schema_valid is False
    assert adapter.call_records[-1].error


def test_schema_violation_is_not_accepted() -> None:
    bad = {**BRIEF_JSON, "confidence": 4.2}
    adapter = DeepSeekAdapter(settings(deepseek_max_retries=0), Transport([json.dumps(bad)]))
    with pytest.raises(DeepSeekError):
        explain(adapter)


def test_payload_is_redacted_before_it_reaches_the_provider() -> None:
    transport = Transport([json.dumps(BRIEF_JSON)])
    adapter = DeepSeekAdapter(settings(), transport)

    adapter.explain_block(
        gate="review",
        attempt=1,
        source_fingerprint="src-1",
        summary="CI log leaked glpat-abcdefghijklmnopqrst",
        deterministic_feedback="GATE BLOCKED: review",
        evidence_items=[{"id": "review.C-12", "detail": "token=glpat-abcdefghijklmnopqrst"}],
    )

    prompt = transport.prompts[0]
    assert "glpat-abcdefghijklmnopqrst" not in prompt
    assert "[REDACTED]" in prompt


def test_review_prompt_carries_the_diff_and_the_fingerprint() -> None:
    transport = Transport(
        [
            json.dumps(
                {
                    "evidence_fingerprint": "fp-1",
                    "summary": "s",
                    "change_summary": "c",
                    "findings": [],
                    "review_questions": [],
                    "confidence": 0.5,
                }
            )
        ]
    )
    adapter = DeepSeekAdapter(settings(), transport)
    report = adapter.review_change(
        EvidencePack(run_id="r1", change=make_change(), fingerprint="fp-1")
    )

    assert report.evidence_fingerprint == "fp-1"
    assert "src/app/refund.py" in transport.prompts[0]


def test_missing_credential_is_an_explicit_error() -> None:
    adapter = DeepSeekAdapter(Settings(), Transport([json.dumps(BRIEF_JSON)]))
    with pytest.raises(DeepSeekError, match="DEEPSEEK_API_KEY"):
        explain(adapter)


def test_provider_endpoints_match_their_vendor_contract() -> None:
    deepseek = DeepSeekAdapter(settings()).endpoint()
    assert deepseek.protocol == "openai"
    assert deepseek.base_url == "https://api.deepseek.com/v1"

    qwen = QwenAdapter(Settings(qwen_api_key="sk-q", qwen_model="qwen3.8-max")).endpoint()
    assert qwen.base_url.endswith("/compatible-mode/v1")
    # Bailian's default thinking budget would stall a bounded JSON call.
    assert qwen.extra_body == {"enable_thinking": False}

    opt_in = QwenAdapter(
        Settings(qwen_api_key="sk-q", qwen_model="qwen3.8-max", qwen_enable_thinking=True)
    ).endpoint()
    assert opt_in.extra_body == {"enable_thinking": True}


def test_openai_and_anthropic_default_to_what_their_current_models_accept() -> None:
    openai = OpenAIAdapter(Settings(openai_api_key="sk-o")).endpoint()
    assert (openai.protocol, openai.base_url) == ("openai", "https://api.openai.com/v1")
    assert openai.model == "gpt-6-astra"
    # Reasoning models take the cap only under this name, and no temperature.
    assert (openai.max_tokens_param, openai.max_tokens) == ("max_completion_tokens", 16000)
    assert openai.temperature is None

    anthropic = AnthropicAdapter(Settings(anthropic_api_key="sk-a")).endpoint()
    assert (anthropic.protocol, anthropic.base_url) == ("anthropic", "https://api.anthropic.com")
    assert anthropic.model == "claude-opus-5"
    # Thinking is on by default and shares this cap with the answer.
    assert anthropic.max_tokens == 16000
    assert anthropic.temperature is None


def test_the_openai_adapter_goes_wherever_the_settings_point_it() -> None:
    ollama = OpenAIAdapter(
        Settings(
            openai_api_key="ollama",
            openai_base_url="http://localhost:11434/v1/",
            openai_model="qwen3:8b",
            openai_max_tokens_param="max_tokens",
            openai_temperature=0.2,
        )
    ).endpoint()

    assert ollama.base_url == "http://localhost:11434/v1"  # trailing slash dropped
    assert (ollama.model, ollama.max_tokens_param, ollama.temperature) == (
        "qwen3:8b",
        "max_tokens",
        0.2,
    )


@pytest.mark.parametrize(
    ("adapter_cls", "overrides"),
    [
        (OpenAIAdapter, {"openai_api_key": "sk-o"}),
        (AnthropicAdapter, {"anthropic_api_key": "sk-a"}),
    ],
)
def test_the_new_providers_validate_and_record_like_the_others(
    adapter_cls: type[OpenAIAdapter | AnthropicAdapter], overrides: dict[str, Any]
) -> None:
    transport = Transport(["not json", f"```json\n{json.dumps(BRIEF_JSON)}\n```"])
    adapter = adapter_cls(Settings(**overrides), transport)

    brief = explain(adapter)

    assert brief.cited_evidence_ids == ["review.C-12"]
    (record,) = adapter.call_records
    assert record.provider == adapter.endpoint().provider
    assert record.protocol == adapter.endpoint().protocol
    assert (record.attempts, record.schema_valid, record.total_tokens) == (2, True, 150)


def test_the_new_providers_name_their_own_missing_key() -> None:
    brief = Transport([json.dumps(BRIEF_JSON)])
    with pytest.raises(OpenAIError, match="OPENAI_API_KEY"):
        explain(OpenAIAdapter(Settings(), brief))
    with pytest.raises(AnthropicError, match="ANTHROPIC_API_KEY"):
        explain(AnthropicAdapter(Settings(), brief))


def test_adapter_factory_follows_the_configured_provider() -> None:
    assert build_llm_adapter(Settings()) is None  # nothing enabled, no adapter
    # A key alone is not a provider: nobody asked for this model.
    assert build_llm_adapter(Settings(deepseek_api_key="sk-test-key-value-1234")) is None
    # A provider alone is not a credential either.
    assert build_llm_adapter(Settings(model_provider="deepseek")) is None
    adapter = build_llm_adapter(settings())
    assert isinstance(adapter, DeepSeekAdapter)
    glm = build_llm_adapter(Settings(model_provider="glm", glm_api_key="zai_x"))
    assert glm is not None
    assert glm.endpoint().provider == "glm"
    openai = build_llm_adapter(Settings(model_provider="openai", openai_api_key="sk-o"))
    assert isinstance(openai, OpenAIAdapter)
    anthropic = build_llm_adapter(Settings(model_provider="anthropic", anthropic_api_key="sk-a"))
    assert isinstance(anthropic, AnthropicAdapter)
    # The new keys are no more of a switch than the old ones.
    assert build_llm_adapter(Settings(openai_api_key="sk-o", anthropic_api_key="sk-a")) is None


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('prose before {"a": 1} prose after') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    with pytest.raises(StructuredModelError):
        extract_json("no object here")


JUDGMENT_JSON = {
    "evidence_fingerprint": "fp-1",
    "summary": "按 J-01 读过这次变更。",
    "change_summary": "",
    "findings": [
        {
            "rule_id": "J-01",
            "title": "绕开了状态机",
            "detail": "直接给 status 赋值。",
            "file": "src/app/refund.py",
            "line_start": 13,
            "line_end": 13,
            "confidence": 0.82,
        }
    ],
    "review_questions": ["退款失败时状态回滚到哪一步？"],
}


def test_a_rule_judgment_carries_the_rules_and_the_citable_lines() -> None:
    transport = Transport([json.dumps(JUDGMENT_JSON)])
    adapter = DeepSeekAdapter(settings(), transport)

    report = adapter.judge_rules(
        change=make_change(),
        fingerprint="fp-1",
        rules=[{"rule_id": "J-01", "title": "走状态机", "criterion": "状态只能经 transition"}],
        locations=[{"file": "src/app/refund.py", "line": "13", "text": "item.refund = share"}],
    )

    assert report.findings[0].rule_id == "J-01"
    prompt = transport.prompts[0]
    # The model is told which rule ids exist and which lines it may cite -- both
    # are what grounding will check it against, so withholding them would just
    # buy discarded responses.
    assert "J-01" in prompt
    assert "状态只能经 transition" in prompt
    assert '"line": "13"' in prompt
    # And it is told the two things it does not get to decide.
    assert "Severity comes from the rule file" in prompt
    assert "Never approve or reject" in prompt
    assert adapter.call_records[0].schema_name == "ModelReviewReport"


def test_a_judgment_that_is_not_valid_json_raises_after_the_retry_budget() -> None:
    transport = Transport(["not json at all"])
    adapter = DeepSeekAdapter(settings(deepseek_max_retries=0), transport)

    with pytest.raises(DeepSeekError):
        adapter.judge_rules(change=make_change(), fingerprint="fp-1", rules=[], locations=[])

    # The failed call is on the record: tokens spent on output nobody kept are
    # still tokens spent.
    assert adapter.call_records[0].schema_valid is False
