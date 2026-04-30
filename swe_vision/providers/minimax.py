"""
MiniMax OpenAI-compatible provider adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from swe_vision.providers.base import (
    ModelCapabilities,
    ModelEndpoint,
    ReasoningConfig,
    merge_extra_body,
)


class MiniMaxAdapter:
    provider = "minimax"

    def resolve_endpoint(
        self,
        model: str,
        base_url: Optional[str] = None,
    ) -> ModelEndpoint:
        return ModelEndpoint(
            provider=self.provider,
            protocol="openai_chat",
            model=model,
            model_vendor="minimax",
            base_url=base_url,
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_vision=False,
                supports_reasoning=True,
                reasoning_style="minimax_reasoning_split",
                supported_reasoning_efforts=("auto", "off"),
                default_reasoning_effort="auto",
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
            return []
        merge_extra_body(kwargs, {"reasoning_split": True})
        if config.effort not in {"auto", "off"} or config.max_tokens is not None:
            return [
                "MiniMax OpenAI-compatible API exposes reasoning_split, but no "
                "confirmed reasoning strength parameter; requested reasoning "
                "strength was not sent."
            ]
        return []
