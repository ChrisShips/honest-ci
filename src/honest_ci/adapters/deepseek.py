"""DeepSeek provider binding (OpenAI-compatible).

Prompts, redaction, schema validation, bounded retries and call telemetry live
in ``structured.py``; this module only describes the endpoint.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..settings import Settings
from .structured import ModelEndpoint, StructuredModelAdapter, StructuredModelError


class DeepSeekError(StructuredModelError):
    pass


class DeepSeekAdapter(StructuredModelAdapter):
    error_cls: ClassVar[type[StructuredModelError]] = DeepSeekError
    missing_credential_message: ClassVar[str] = (
        "DEEPSEEK_API_KEY is required for the DeepSeek adapter."
    )

    def __init__(self, settings: Settings, transport: Any | None = None) -> None:
        super().__init__(settings, transport)

    def endpoint(self) -> ModelEndpoint:
        return ModelEndpoint(
            provider="deepseek",
            protocol="openai",
            base_url=self.settings.effective_deepseek_base_url,
            model=self.settings.deepseek_model,
            api_key=self.settings.deepseek_api_key,
            max_tokens=self.settings.deepseek_max_tokens,
            temperature=self.settings.deepseek_temperature,
            timeout_seconds=self.settings.deepseek_timeout_seconds,
            max_retries=self.settings.deepseek_max_retries,
        )
