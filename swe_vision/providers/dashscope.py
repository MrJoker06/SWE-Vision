"""
Alibaba DashScope / Qwen OpenAI-compatible provider adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from swe_vision.providers.base import (
    ModelCapabilities,
    ModelEndpoint,
    ReasoningConfig,
    merge_extra_body,
)


class DashScopeAdapter:
    provider = "dashscope"

    def resolve_endpoint(
        self,
        model: str,
        base_url: Optional[str] = None,
    ) -> ModelEndpoint:
        return ModelEndpoint(
            provider=self.provider,
            protocol="openai_chat",
            model=model,
            model_vendor="qwen",
            base_url=base_url,
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_vision=None,
                supports_reasoning=True,
                reasoning_style="dashscope_thinking",
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
        if config.effort == "off":
            merge_extra_body(kwargs, {"enable_thinking": False})
            return []

        extra: Dict[str, Any] = {"enable_thinking": True}
        budget = config.max_tokens
        if budget is None and config.effort != "auto":
            budget = _effort_to_budget(config.effort)
        if budget is not None:
            extra["thinking_budget"] = budget
        merge_extra_body(kwargs, extra)
        return []


def _effort_to_budget(effort: str) -> Optional[int]:
    budgets = {
        "minimal": 512,
        "low": 1024,
        "medium": 4096,
        "high": 8192,
        "max": 16384,
    }
    return budgets.get(effort)
