"""Adapter registry.

Both builders return ``None`` rather than raising when a deployment is not
configured for them, because both absences are supported states:

* no model enabled, or no credential for it — deterministic rules keep
  running and the model-assisted part of a gate is recorded as skipped;
* no forge, or a forge nobody has implemented yet — the local entry point
  (``honest-ci verify``) does not need one, so only the commands that really write
  back to a platform have to complain, and they can say exactly what is
  missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..settings import Settings
from .anthropic import AnthropicAdapter, AnthropicError
from .base import (
    AdapterSet,
    ForgeAdapter,
    ForgeError,
    LedgerStore,
    LLMAdapter,
    ScannerAdapter,
)
from .deepseek import DeepSeekAdapter, DeepSeekError
from .github import GitHubError, GitHubForgeAdapter
from .glm import GLMAdapter, GLMError
from .openai import OpenAIAdapter, OpenAIError
from .qwen import QwenAdapter, QwenError
from .scanners import (
    LANGUAGE_AUTO,
    SCANNER_ADAPTERS,
    LanguageRouter,
    ScannerConfigError,
    ToolchainChoice,
    build_scanner_adapter,
    detect_languages,
    supported_tools,
    tool_languages,
)
from .structured import ModelEndpoint, StructuredModelAdapter, StructuredModelError

LLM_ADAPTERS: dict[str, type[StructuredModelAdapter]] = {
    "deepseek": DeepSeekAdapter,
    "qwen": QwenAdapter,
    "glm": GLMAdapter,
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
}

#: Forge name → how to build one. ``local`` is deliberately absent: a local diff
#: has no platform to fetch from or write back to. A factory rather than a class
#: keeps construction out of the port, which describes behaviour only.
FORGE_ADAPTERS: dict[str, Callable[[Settings], ForgeAdapter]] = {
    "github": GitHubForgeAdapter,
}


def build_forge_adapter(settings: Settings) -> ForgeAdapter | None:
    """The configured platform adapter, or ``None`` when there is not one.

    ``None`` covers three different situations on purpose — ``local``, a forge
    with no implementation yet, and a missing token — because none of them is a
    failure for a run that never touches a platform. The command that does
    (``honest-ci run-pr``) distinguishes them itself, so an operator is told which
    one it is.
    """

    adapter_cls = FORGE_ADAPTERS.get(settings.forge)
    if adapter_cls is None or not settings.forge_token():
        return None
    return adapter_cls(settings)


def build_llm_adapter(
    settings: Settings,
    transport: Any | None = None,
) -> StructuredModelAdapter | None:
    provider = settings.active_provider
    if provider is None:
        return None
    adapter_cls = LLM_ADAPTERS.get(provider)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown model provider {provider!r}. Expected one of: {', '.join(LLM_ADAPTERS)}."
        )
    return adapter_cls(settings, transport)


__all__ = [
    "FORGE_ADAPTERS",
    "LANGUAGE_AUTO",
    "LLM_ADAPTERS",
    "SCANNER_ADAPTERS",
    "AdapterSet",
    "AnthropicAdapter",
    "AnthropicError",
    "DeepSeekAdapter",
    "DeepSeekError",
    "ForgeAdapter",
    "ForgeError",
    "GitHubError",
    "GitHubForgeAdapter",
    "GLMAdapter",
    "GLMError",
    "LLMAdapter",
    "LanguageRouter",
    "LedgerStore",
    "ModelEndpoint",
    "OpenAIAdapter",
    "OpenAIError",
    "QwenAdapter",
    "QwenError",
    "ScannerAdapter",
    "ScannerConfigError",
    "StructuredModelAdapter",
    "StructuredModelError",
    "ToolchainChoice",
    "build_forge_adapter",
    "build_llm_adapter",
    "build_scanner_adapter",
    "detect_languages",
    "supported_tools",
    "tool_languages",
]
