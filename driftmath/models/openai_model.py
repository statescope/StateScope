"""OpenAI API backend (uses the OpenAI-compatible core with the OpenAI base URL)."""

from __future__ import annotations

from typing import Any

from driftmath.models.openai_compat import OpenAICompatModel
from driftmath.models.registry import register_model


@register_model
class OpenAIModel(OpenAICompatModel):
    type = "openai"

    def __init__(self, *, api_key_env: str = "OPENAI_API_KEY", base_url: str | None = None, **kw: Any) -> None:
        # base_url=None lets the openai SDK use api.openai.com; no key read at construction.
        super().__init__(api_key_env=api_key_env, base_url=base_url, **kw)
