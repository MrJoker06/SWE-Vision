"""
OpenRouter provider adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from swe_vision.providers.base import (
    ModelCapabilities,
    ModelEndpoint,
    ReasoningConfig,
    merge_extra_body,
)


class OpenRouterAdapter:
    provider = "openrouter"

    def resolve_endpoint(
        self,
        model: str,
        base_url: Optional[str] = None,
    ) -> ModelEndpoint:
        return ModelEndpoint(
            provider=self.provider,
            protocol="openai_chat",
            model=model,
            model_vendor=_infer_model_vendor(model),
            base_url=base_url,
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_vision=None,
                supports_reasoning=True,
                reasoning_style="openrouter_reasoning",
                supported_reasoning_efforts=(
                    "minimal",
                    "low",
                    "medium",
                    "high",
                    "max",
                ),
                default_reasoning_effort="medium",
                preserves_reasoning_for_tools=True,
            ),
        )

    def apply_reasoning(
        self,
        kwargs: Dict[str, Any],
        endpoint: ModelEndpoint,
        config: ReasoningConfig,
    ) -> List[str]:
        reasoning: Dict[str, Any] = {}
        if config.effort == "off":
            reasoning["effort"] = "none"
        elif config.max_tokens is not None:
            reasoning["max_tokens"] = config.max_tokens
        elif config.effort == "auto":
            reasoning["enabled"] = True
        else:
            reasoning["effort"] = "xhigh" if config.effort == "max" else config.effort

        if config.exclude:
            reasoning["exclude"] = True
        merge_extra_body(kwargs, {"reasoning": reasoning})
        return []


def _infer_model_vendor(model: str) -> str:
    normalized = model.lower()
    if "/" not in normalized:
        return "unknown"
    vendor = normalized.split("/", 1)[0]
    if vendor in {"openai", "anthropic", "google", "qwen", "deepseek", "minimax"}:
        return vendor
    if vendor in {"alibaba"}:
        return "qwen"
    return vendor
