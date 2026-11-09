"""Z.ai / GLM provider binding.

Z.ai exposes both an Anthropic-shaped and an OpenAI-shaped API; the protocol is
selected with ``HONEST_CI_GLM_PROTOCOL``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..settings import Settings
from .structured import ModelEndpoint, StructuredModelAdapter, StructuredModelError


class GLMError(StructuredModelError):
    pass


class GLMAdapter(StructuredModelAdapter):
    error_cls: ClassVar[type[StructuredModelError]] = GLMError
    missing_credential_message: ClassVar[str] = (
        "ZAI_API_KEY or ANTHROPIC_AUTH_TOKEN is required for the GLM adapter."
    )

    def __init__(self, settings: Settings, transport: Any | None = None) -> None:
        super().__init__(settings, transport)

    def endpoint(self) -> ModelEndpoint:
        return ModelEndpoint(
            provider="glm",
            protocol="openai" if self.settings.glm_protocol == "openai" else "anthropic",
            base_url=self.settings.effective_glm_base_url,
            model=self.settings.glm_model,
            api_key=self.settings.glm_api_key,
            max_tokens=self.settings.glm_max_tokens,
            temperature=self.settings.glm_temperature,
            timeout_seconds=self.settings.glm_timeout_seconds,
            max_retries=self.settings.glm_max_retries,
        )
