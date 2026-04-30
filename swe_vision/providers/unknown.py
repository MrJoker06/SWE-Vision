"""
Fallback provider adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from swe_vision.providers.base import (
    ModelCapabilities,
    ModelEndpoint,
    ReasoningConfig,
)


class UnknownAdapter:
    provider = "unknown"

    def resolve_endpoint(
        self,
        model: str,
        base_url: Optional[str] = None,
    ) -> ModelEndpoint:
        return ModelEndpoint(
            provider=self.provider,
            protocol="openai_chat",
            model=model,
            model_vendor="unknown",
            base_url=base_url,
            capabilities=ModelCapabilities(),
        )

    def apply_reasoning(
        self,
        kwargs: Dict[str, Any],
        endpoint: ModelEndpoint,
        config: ReasoningConfig,
    ) -> List[str]:
        if config.effort != "auto" or config.max_tokens is not None:
            return [
                "Reasoning was requested, but the provider is unknown; no "
                "provider-specific reasoning parameter was sent."
            ]
        return []
