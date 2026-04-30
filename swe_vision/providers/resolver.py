"""
Endpoint resolver for OpenAI-compatible model providers.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from swe_vision.providers.base import ModelEndpoint, ProviderAdapter
from swe_vision.providers.dashscope import DashScopeAdapter
from swe_vision.providers.deepseek import DeepSeekAdapter
from swe_vision.providers.minimax import MiniMaxAdapter
from swe_vision.providers.openai import OpenAIAdapter
from swe_vision.providers.openrouter import OpenRouterAdapter
from swe_vision.providers.unknown import UnknownAdapter


ADAPTERS = {
    "openai": OpenAIAdapter(),
    "openrouter": OpenRouterAdapter(),
    "deepseek": DeepSeekAdapter(),
    "dashscope": DashScopeAdapter(),
    "qwen": DashScopeAdapter(),
    "minimax": MiniMaxAdapter(),
    "unknown": UnknownAdapter(),
}


def resolve_endpoint(
    model: str,
    base_url: Optional[str] = None,
    provider: str = "auto",
) -> ModelEndpoint:
    provider_name = _normalize_provider(provider)
    if provider_name == "auto":
        provider_name = infer_provider(model, base_url)
    adapter = get_adapter(provider_name)
    return adapter.resolve_endpoint(model, base_url)


def get_adapter(provider: str) -> ProviderAdapter:
    return ADAPTERS.get(_normalize_provider(provider), ADAPTERS["unknown"])


def infer_provider(model: str, base_url: Optional[str] = None) -> str:
    host = _host(base_url)
    normalized_model = model.lower()

    if "openrouter.ai" in host:
        return "openrouter"
    if "deepseek.com" in host:
        return "deepseek"
    if "dashscope" in host or "aliyuncs.com" in host:
        return "dashscope"
    if "minimax" in host or "minimaxi" in host:
        return "minimax"
    if "api.openai.com" in host or not host:
        return "openai"

    if normalized_model.startswith(("deepseek/", "deepseek-")):
        return "deepseek"
    if normalized_model.startswith(("qwen/", "alibaba/", "qwen-")):
        return "dashscope"
    if normalized_model.startswith(("minimax/", "minimax-")):
        return "minimax"
    if normalized_model.startswith("openai/"):
        return "openai"

    return "unknown"


def _normalize_provider(provider: Optional[str]) -> str:
    normalized = (provider or "auto").strip().lower()
    aliases = {
        "qwen_dashscope": "dashscope",
        "aliyun": "dashscope",
        "alibaba": "dashscope",
        "auto": "auto",
    }
    return aliases.get(normalized, normalized)


def _host(base_url: Optional[str]) -> str:
    if not base_url:
        return ""
    parsed = urlparse(base_url)
    if parsed.netloc:
        return parsed.netloc.lower()
    return base_url.lower()
