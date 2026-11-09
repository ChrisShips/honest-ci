"""Anthropic provider binding (Messages API).

The request carries ``model``, ``max_tokens`` and the one user message, and
nothing else by default. Current Claude models reject ``temperature``, so it is
sent only when ``HONEST_CI_ANTHROPIC_TEMPERATURE`` asks for it. Thinking is left
at the model's default -- on, for current models -- which is why
``max_tokens`` defaults high: it caps the thinking and the JSON together, and a
call that runs out mid-object is a failed call.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..settings import Settings
from .structured import ModelEndpoint, StructuredModelAdapter, StructuredModelError


class AnthropicError(StructuredModelError):
    pass


class AnthropicAdapter(StructuredModelAdapter):
    error_cls: ClassVar[type[StructuredModelError]] = AnthropicError
    missing_credential_message: ClassVar[str] = (
        "ANTHROPIC_API_KEY is required for the Anthropic adapter."
    )

    def __init__(self, settings: Settings, transport: Any | None = None) -> None:
        super().__init__(settings, transport)

    def endpoint(self) -> ModelEndpoint:
        return ModelEndpoint(
            provider="anthropic",
            protocol="anthropic",
            base_url=self.settings.effective_anthropic_base_url,
            model=self.settings.anthropic_model,
            api_key=self.settings.anthropic_api_key,
            max_tokens=self.settings.anthropic_max_tokens,
            temperature=self.settings.anthropic_temperature,
            timeout_seconds=self.settings.anthropic_timeout_seconds,
            max_retries=self.settings.anthropic_max_retries,
        )
