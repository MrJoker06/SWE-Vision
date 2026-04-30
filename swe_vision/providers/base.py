"""
Provider abstractions for model endpoint resolution and request adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple


REASONING_EFFORTS = ("auto", "off", "minimal", "low", "medium", "high", "max")


@dataclass(frozen=True)
class ReasoningConfig:
    effort: str = "auto"
    max_tokens: Optional[int] = None
    exclude: bool = False

    @classmethod
    def from_legacy(
        cls,
        reasoning: bool = True,
        effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        exclude: bool = False,
    ) -> "ReasoningConfig":
        resolved_effort = (effort or "auto").strip().lower()
        if not reasoning:
            resolved_effort = "off"
        if resolved_effort not in REASONING_EFFORTS:
            raise ValueError(
                "reasoning effort must be one of: "
                + ", ".join(REASONING_EFFORTS)
            )
        if resolved_effort == "off":
            max_tokens = None
        return cls(
            effort=resolved_effort,
            max_tokens=max_tokens,
            exclude=exclude,
        )


@dataclass(frozen=True)
class ModelCapabilities:
    supports_tools: bool = True
    supports_vision: Optional[bool] = None
    supports_reasoning: bool = False
    reasoning_style: str = "none"
    supported_reasoning_efforts: Tuple[str, ...] = ()
    default_reasoning_effort: Optional[str] = None
    preserves_reasoning_for_tools: bool = False


@dataclass(frozen=True)
class ModelEndpoint:
    provider: str
    protocol: str
    model: str
    model_vendor: str
    base_url: Optional[str] = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)


class ProviderAdapter(Protocol):
    provider: str

    def resolve_endpoint(
        self,
        model: str,
        base_url: Optional[str] = None,
    ) -> ModelEndpoint:
        ...

    def apply_reasoning(
        self,
        kwargs: Dict[str, Any],
        endpoint: ModelEndpoint,
        config: ReasoningConfig,
    ) -> List[str]:
        ...


def merge_extra_body(kwargs: Dict[str, Any], values: Dict[str, Any]) -> None:
    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body.update(values)
    kwargs["extra_body"] = extra_body
