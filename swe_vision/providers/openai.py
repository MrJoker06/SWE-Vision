"""
OpenAI chat-completions provider adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from swe_vision.providers.base import (
    ModelCapabilities,
    ModelEndpoint,
    ReasoningConfig,
)


class OpenAIAdapter:
    provider = "openai"

    def resolve_endpoint(
        self,
        model: str,
        base_url: Optional[str] = None,
    ) -> ModelEndpoint:
        capabilities = ModelCapabilities(
            supports_tools=True,
            supports_vision=None,
            supports_reasoning=is_openai_reasoning_model(model),
            reasoning_style="openai_chat_reasoning_effort",
            supported_reasoning_efforts=("minimal", "low", "medium", "high", "max"),
            default_reasoning_effort="medium",
            preserves_reasoning_for_tools=True,
        )
        return ModelEndpoint(
            provider=self.provider,
            protocol="openai_chat",
            model=model,
            model_vendor="openai",
            base_url=base_url,
            capabilities=capabilities,
        )

    def apply_reasoning(
        self,
        kwargs: Dict[str, Any],
        endpoint: ModelEndpoint,
        config: ReasoningConfig,
    ) -> List[str]:
        if config.effort == "off":
            if endpoint.capabilities.supports_reasoning:
                if "gpt-5.1" in endpoint.model.lower():
                    kwargs["reasoning_effort"] = "none"
                    return []
                kwargs["reasoning_effort"] = "minimal"
                return [
                    "This OpenAI reasoning model does not support disabling "
                    "reasoning entirely; sent minimal reasoning effort instead."
                ]
            return []
        if not endpoint.capabilities.supports_reasoning:
            if config.effort != "auto" or config.max_tokens is not None:
                return [
                    "Reasoning was requested, but this OpenAI model is not "
                    "recognized as a reasoning model; no reasoning parameter was sent."
                ]
            return []

        effort = _map_openai_effort(endpoint.model, config.effort)
        if effort:
            kwargs["reasoning_effort"] = effort
        return []


def is_openai_reasoning_model(model: str) -> bool:
    normalized = model.lower().split(":", 1)[0]
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    return (
        normalized.startswith("gpt-5")
        or normalized.startswith("o1")
        or normalized.startswith("o3")
        or normalized.startswith("o4")
    )


def _map_openai_effort(model: str, effort: str) -> Optional[str]:
    normalized = model.lower()
    if "gpt-5-pro" in normalized:
        return "high"
    if "gpt-5.1" in normalized:
        if effort == "auto":
            return "medium"
        if effort in {"minimal", "max"}:
            return "low" if effort == "minimal" else "high"
        return effort
    if effort == "auto":
        return "medium"
    if effort == "max":
        return "xhigh"
    if effort == "minimal":
        return "minimal"
    return effort
