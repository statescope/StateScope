"""OpenRouter backend (OpenAI-compatible client pointed at openrouter.ai).

Model strings are fully config-driven, e.g. ``openai/gpt-4o``,
``anthropic/claude-3.5-sonnet``, ``google/gemini-pro``, ``qwen/qwen-2.5-72b-instruct``,
``deepseek/deepseek-r1``. Uses the ``OPENROUTER_API_KEY`` environment variable.
"""

from __future__ import annotations

from typing import Any

from driftmath.models.openai_compat import OpenAICompatModel
from driftmath.models.registry import register_model


@register_model
class OpenRouterModel(OpenAICompatModel):
    type = "openrouter"

    def __init__(
        self,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        http_referer: str | None = None,
        x_title: str | None = None,
        extra_headers: dict | None = None,
        **kw: Any,
    ) -> None:
        headers = dict(extra_headers or {})
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if x_title:
            headers["X-Title"] = x_title
        super().__init__(base_url=base_url, api_key_env=api_key_env, extra_headers=headers or None, **kw)
