"""OpenAI-compatible chat backend.

Works against any OpenAI-compatible endpoint: a local vLLM OpenAI server, OpenRouter,
or the OpenAI API itself. The ``openai`` SDK is imported lazily inside :meth:`_client`,
and the API key is resolved (and validated) *before* that import, so a missing key
raises :class:`MissingCredentialError` even when the SDK is absent.
"""

from __future__ import annotations

import json
import os
from typing import Any

from driftmath.models.base import MissingCredentialError, Model, ModelResponse
from driftmath.models.registry import register_model


@register_model
class OpenAICompatModel(Model):
    type = "openai_compat"

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        api_key_fallback: str | None = None,
        timeout: float = 120,
        max_tokens: int = 2048,
        token_parameter: str = "max_tokens",
        temperature: float | None = 0.0,
        reasoning_effort: str | None = None,
        supports_tools: bool = False,
        extra_headers: dict | None = None,
        extra_body: dict | None = None,
        name: str | None = None,
        **_ignored: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.api_key_fallback = api_key_fallback
        self.timeout = timeout
        self.max_tokens = max_tokens
        if token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("token_parameter must be 'max_tokens' or 'max_completion_tokens'")
        self.token_parameter = token_parameter
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self._supports_tools = bool(supports_tools)
        self.extra_headers = extra_headers
        self.extra_body = extra_body
        self.name = name or model
        self._client_obj = None

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    def _resolve_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = os.environ.get(self.api_key_env, "")
        if key:
            return key
        if self.api_key_fallback:  # e.g. "EMPTY" for a keyless local vLLM server
            return self.api_key_fallback
        raise MissingCredentialError(
            f"{type(self).__name__}: environment variable {self.api_key_env} is not set "
            f"(set it, or configure api_key_fallback for a local server)."
        )

    def _client(self):
        if self._client_obj is None:
            key = self._resolve_key()  # raises MissingCredentialError before importing the SDK
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover - exercised only without the SDK
                raise ImportError(
                    "The 'openai' SDK is required for OpenAI-compatible backends; "
                    "install it via `pip install \"driftmath[closed]\"`."
                ) from e
            self._client_obj = OpenAI(
                api_key=key,
                base_url=self.base_url,
                timeout=self.timeout,
                default_headers=self.extra_headers or None,
            )
        return self._client_obj

    def build_payload(self, messages: list[dict], *, tools: list[dict] | None = None, **overrides: Any) -> dict:
        """Pure construction of the chat-completions payload (offline-testable).

        Per-call ``max_tokens`` overrides (e.g. the adapter's truncation-escalation
        retry) are normalized onto ``token_parameter``, so callers never need to know
        whether this model takes ``max_tokens`` or ``max_completion_tokens``.
        ``reasoning_effort`` rides in ``extra_body`` so it reaches the request body on
        any SDK version.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            self.token_parameter: self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(overrides)
        if self.token_parameter != "max_tokens" and "max_tokens" in payload:
            payload[self.token_parameter] = payload.pop("max_tokens")
        body = dict(self.extra_body) if self.extra_body else {}
        if self.reasoning_effort:
            body.setdefault("reasoning_effort", self.reasoning_effort)
        if body:
            payload["extra_body"] = body
        if tools:
            payload["tools"] = tools
            payload.setdefault("tool_choice", "auto")
        return payload

    @staticmethod
    def _to_response(resp: Any) -> ModelResponse:
        raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        choices = raw.get("choices") or [{}]
        message = choices[0].get("message", {}) if choices else {}
        text = message.get("content") or ""
        parsed_ops = None
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {"_raw": fn.get("arguments")}
            parsed_ops = (parsed_ops or []) + [{"name": fn.get("name"), "arguments": args}]
        return ModelResponse(text=text, raw=raw, usage=raw.get("usage") or {}, parsed_ops=parsed_ops)

    def generate(self, messages: list[dict], **gen_kwargs: Any) -> ModelResponse:
        resp = self._client().chat.completions.create(**self.build_payload(messages, **gen_kwargs))
        return self._to_response(resp)

    def generate_with_tools(self, messages: list[dict], tools: list[dict], **gen_kwargs: Any) -> ModelResponse:
        if not self.supports_tools:
            raise NotImplementedError(f"{self.name} is configured with supports_tools=false")
        resp = self._client().chat.completions.create(
            **self.build_payload(messages, tools=tools, **gen_kwargs)
        )
        return self._to_response(resp)
