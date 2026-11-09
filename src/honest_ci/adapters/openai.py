"""OpenAI-compatible provider binding.

OpenAI by default; through ``HONEST_CI_OPENAI_BASE_URL``, anything else that
speaks the chat-completions wire -- DeepSeek's compatible endpoint, a local
Ollama, a company gateway. The key in ``OPENAI_API_KEY`` goes to whichever of
them that names.

Two defaults follow OpenAI's reasoning models, which reject what older chat
models took: the output cap travels as ``max_completion_tokens`` and
``temperature`` is not sent. A server that reads only ``max_tokens`` gets
``HONEST_CI_OPENAI_MAX_TOKENS_PARAM=max_tokens``; one that wants a temperature
gets ``HONEST_CI_OPENAI_TEMPERATURE``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..settings import Settings
from .structured import ModelEndpoint, StructuredModelAdapter, StructuredModelError


class OpenAIError(StructuredModelError):
    pass


class OpenAIAdapter(StructuredModelAdapter):
    error_cls: ClassVar[type[StructuredModelError]] = OpenAIError
    missing_credential_message: ClassVar[str] = (
        "OPENAI_API_KEY is required for the OpenAI-compatible adapter."
    )

    def __init__(self, settings: Settings, transport: Any | None = None) -> None:
        super().__init__(settings, transport)

    def endpoint(self) -> ModelEndpoint:
        return ModelEndpoint(
            provider="openai",
            protocol="openai",
            base_url=self.settings.effective_openai_base_url,
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key,
            max_tokens=self.settings.openai_max_tokens,
            max_tokens_param=self.settings.openai_max_tokens_param,
            temperature=self.settings.openai_temperature,
            timeout_seconds=self.settings.openai_timeout_seconds,
            max_retries=self.settings.openai_max_retries,
        )
