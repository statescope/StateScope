"""Anthropic (Claude) backend. Lazy ``anthropic`` import; ANTHROPIC_API_KEY."""

from __future__ import annotations

import os
from typing import Any

from driftmath.models.base import MissingCredentialError, Model, ModelResponse
from driftmath.models.registry import register_model


@register_model
class AnthropicModel(Model):
    type = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        api_key: str | None = None,
        timeout: float = 120,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        supports_tools: bool = False,
        name: str | None = None,
        **_ignored: Any,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self._api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._supports_tools = bool(supports_tools)
        self.name = name or model
        self._client_obj = None

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    def _resolve_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise MissingCredentialError(f"AnthropicModel: environment variable {self.api_key_env} is not set.")
        return key

    def build_payload(self, messages: list[dict], **overrides: Any) -> dict:
        """Anthropic splits the system prompt out of the message list (offline-testable)."""
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system") or None
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") != "system"]
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": msgs,
        }
        if system:
            payload["system"] = system
        payload.update(overrides)
        return payload

    def _client(self):
        if self._client_obj is None:
            key = self._resolve_key()
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "The 'anthropic' SDK is required; install via `pip install \"driftmath[closed]\"`."
                ) from e
            self._client_obj = anthropic.Anthropic(api_key=key, timeout=self.timeout)
        return self._client_obj

    def generate(self, messages: list[dict], **gen_kwargs: Any) -> ModelResponse:
        resp = self._client().messages.create(**self.build_payload(messages, **gen_kwargs))
        raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
        text = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
        return ModelResponse(text=text, raw=raw, usage=raw.get("usage") or {})
