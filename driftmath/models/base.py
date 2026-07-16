"""Model interface.

Kept deliberately minimal and backend-agnostic so OpenAI / Anthropic / vLLM
backends can be added later without changing the systems that consume models.
A system only ever calls :meth:`Model.reset` (a priming hook; real models no-op),
:meth:`Model.generate`, and optionally :meth:`Model.generate_with_tools`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class MissingCredentialError(RuntimeError):
    """Raised when a real backend is *called* without the required API credentials.

    Always raised before any SDK import, so a missing key surfaces as this clear
    error even when the provider SDK is not installed.
    """


class ModelResponse(BaseModel):
    """One model turn.

    Attributes
    ----------
    text: free-form text output (prose state, reasoning, etc.).
    raw: backend-specific payload (for the MockModel this carries ``claimed_state``
        and a ``done`` flag).
    usage: token / step accounting.
    parsed_ops: optional list of structured operation requests ``[{"op", "args"}]``.
    """

    text: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    parsed_ops: list[dict[str, Any]] | None = None


class Model(ABC):
    """Abstract model backend."""

    name: str = "model"

    @property
    def supports_tools(self) -> bool:
        return False

    @abstractmethod
    def generate(self, messages: list[dict], **gen_kwargs: Any) -> ModelResponse:
        """Produce the next response given a message list."""
        raise NotImplementedError

    def generate_with_tools(
        self, messages: list[dict], tools: list[dict], **gen_kwargs: Any
    ) -> ModelResponse:
        """Tool-augmented generation (override in backends that support tools)."""
        raise NotImplementedError(f"{type(self).__name__} does not support tools")

    def reset(self, **kwargs: Any) -> None:
        """Per-problem priming hook. Real models ignore this; the MockModel uses
        it to load a script (gold trace + drift settings) before a solve."""
        return None
