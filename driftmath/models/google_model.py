"""Google (Gemini) backend. Lazy ``google-genai`` import; GOOGLE_API_KEY."""

from __future__ import annotations

import os
from typing import Any

from driftmath.models.base import MissingCredentialError, Model, ModelResponse
from driftmath.models.registry import register_model


@register_model
class GoogleModel(Model):
    type = "google"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "GOOGLE_API_KEY",
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        name: str | None = None,
        **_ignored: Any,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self._api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = name or model
        self._client_obj = None

    def _resolve_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise MissingCredentialError(f"GoogleModel: environment variable {self.api_key_env} is not set.")
        return key

    def build_payload(self, messages: list[dict], **overrides: Any) -> dict:
        """Map chat messages to Gemini contents + system_instruction (offline-testable)."""
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system") or None
        contents = "\n\n".join(
            f"{m['role']}: {m['content']}" for m in messages if m.get("role") != "system"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "contents": contents,
            "system_instruction": system,
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        payload.update(overrides)
        return payload

    def _client(self):
        if self._client_obj is None:
            key = self._resolve_key()
            try:
                from google import genai
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "The 'google-genai' SDK is required; install via `pip install \"driftmath[closed]\"`."
                ) from e
            self._client_obj = genai.Client(api_key=key)
        return self._client_obj

    def generate(self, messages: list[dict], **gen_kwargs: Any) -> ModelResponse:
        client = self._client()  # resolves key (MissingCredentialError) before any google import
        from google.genai import types  # lazy

        p = self.build_payload(messages, **gen_kwargs)
        config = types.GenerateContentConfig(
            system_instruction=p.get("system_instruction"),
            max_output_tokens=p.get("max_output_tokens"),
            temperature=p.get("temperature"),
        )
        resp = client.models.generate_content(model=p["model"], contents=p["contents"], config=config)
        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        usage_dict = dict(usage.__dict__) if usage is not None and hasattr(usage, "__dict__") else {}
        return ModelResponse(text=text, raw={"text": text}, usage=usage_dict)
