"""
DeepSeek OpenAI-compatible provider adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from swe_vision.providers.base import (
    ModelCapabilities,
    ModelEndpoint,
    ReasoningConfig,
    merge_extra_body,
)


class DeepSeekAdapter:
    provider = "deepseek"

    def resolve_endpoint(
        self,
        model: str,
        base_url: Optional[str] = None,
    ) -> ModelEndpoint:
        return ModelEndpoint(
            provider=self.provider,
            protocol="openai_chat",
            model=model,
            model_vendor="deepseek",
            base_url=base_url,
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_vision=False,
                supports_reasoning=True,
                reasoning_style="deepseek_thinking",
                supported_reasoning_efforts=("high", "max"),
                default_reasoning_effort="high",
                preserves_reasoning_for_tools=True,
            ),
        )

    def apply_reasoning(
        self,
        kwargs: Dict[str, Any],
        endpoint: ModelEndpoint,
        config: ReasoningConfig,
    ) -> List[str]:
        if config.effort == "off":
            merge_extra_body(kwargs, {"thinking": {"type": "disabled"}})
            return []

        effort = _map_deepseek_effort(config.effort)
        kwargs["reasoning_effort"] = effort
        merge_extra_body(kwargs, {"thinking": {"type": "enabled"}})
        return []


def _map_deepseek_effort(effort: str) -> str:
    if effort in {"auto", "low", "medium"}:
        return "high"
    if effort == "max":
        return "max"
    if effort == "minimal":
        return "high"
    return effort
