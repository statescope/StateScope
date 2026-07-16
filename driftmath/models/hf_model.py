"""Local HuggingFace backend for small models. Lazy ``transformers``/``torch``.

Intended for small local models without a server. Weights are loaded lazily on the
first :meth:`generate` call (never at construction, so tests can instantiate it
offline without downloading anything).
"""

from __future__ import annotations

from typing import Any

from driftmath.models.base import Model, ModelResponse
from driftmath.models.registry import register_model


@register_model
class HFModel(Model):
    type = "hf"

    def __init__(
        self,
        *,
        model_id: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        max_model_len: int | None = None,
        temperature: float = 0.0,
        trust_remote_code: bool = False,
        name: str | None = None,
        **_ignored: Any,
    ) -> None:
        self.model_id = model_id
        self.dtype = dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.trust_remote_code = bool(trust_remote_code)
        self.name = name or model_id
        self._tok = None
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self):
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "transformers/torch are required for the HF backend; "
                    "install via `pip install \"driftmath[open]\"`."
                ) from e
            torch_dtype = getattr(torch, self.dtype, None)
            self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
            )
        return self._tok, self._model

    def _format(self, messages: list[dict], tok: Any) -> str:
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def generate(self, messages: list[dict], **gen_kwargs: Any) -> ModelResponse:
        tok, model = self._ensure_loaded()
        prompt = self._format(messages, tok)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        do_sample = self.temperature > 0
        out = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
        )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return ModelResponse(text=text, raw={"model_id": self.model_id}, usage={})
