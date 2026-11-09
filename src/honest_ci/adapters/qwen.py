"""Alibaba Bailian (Model Studio) / DashScope provider binding — Qwen.

Wire details: Bailian's OpenAI-compatible endpoint
(``https://dashscope.aliyuncs.com/compatible-mode/v1``); the DashScope-native
protocol is intentionally not implemented. Override the base URL with
``HONEST_CI_QWEN_BASE_URL`` for a workspace-scoped or non-Beijing region host.

Qwen3.8-Max enables deep thinking by default upstream. Gate calls are bounded
and non-streaming, so an unset ``HONEST_CI_QWEN_ENABLE_THINKING`` is normalized to
``false`` for that model; set it to ``1`` only to opt in deliberately. Other
model ids keep the tri-state behavior and omit the field when unset.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..settings import Settings
from .structured import ModelEndpoint, StructuredModelAdapter, StructuredModelError


class QwenError(StructuredModelError):
    pass


class QwenAdapter(StructuredModelAdapter):
    error_cls: ClassVar[type[StructuredModelError]] = QwenError
    missing_credential_message: ClassVar[str] = (
        "DASHSCOPE_API_KEY is required for the Qwen (Alibaba Bailian) adapter."
    )

    def __init__(self, settings: Settings, transport: Any | None = None) -> None:
        super().__init__(settings, transport)

    def endpoint(self) -> ModelEndpoint:
        extra_body: dict[str, Any] = {}
        if self.settings.qwen_enable_thinking is not None:
            extra_body["enable_thinking"] = self.settings.qwen_enable_thinking
        elif self.settings.qwen_model.startswith("qwen3.8-max"):
            # Bailian defaults Qwen3.8-Max to a very large thinking budget,
            # which can make a non-streaming JSON call look hung until the
            # socket timeout and retry budgets are exhausted.
            extra_body["enable_thinking"] = False
        return ModelEndpoint(
            provider="qwen",
            protocol="openai",  # Bailian exposes Qwen through OpenAI-compatible mode only.
            base_url=self.settings.effective_qwen_base_url,
            model=self.settings.qwen_model,
            api_key=self.settings.qwen_api_key,
            max_tokens=self.settings.qwen_max_tokens,
            temperature=self.settings.qwen_temperature,
            timeout_seconds=self.settings.qwen_timeout_seconds,
            max_retries=self.settings.qwen_max_retries,
            extra_body=extra_body,
        )
