"""Offline tests for truncation-escalation in the text-JSON repair loop.

A reasoning model that hits ``finish_reason == "length"`` produced empty/truncated
JSON because hidden reasoning consumed the output budget; re-asking at the same limit
fails identically. The repair loop must retry with an escalated budget instead.
"""

import json

from driftmath.adapters.repair import escalated_budget, run_text_json
from driftmath.core.state import SymbolicState
from driftmath.models.base import Model, ModelResponse

_MESSAGES = [{"role": "user", "content": "next operation"}]

_VALID = json.dumps(
    {
        "op": "bind",
        "args": {"id": "a", "formula": "5", "inputs": []},
        "claimed_state": SymbolicState().model_dump(),
        "done": False,
        "rationale": "bind a",
    }
)


def _truncated() -> ModelResponse:
    return ModelResponse(text="", raw={"choices": [{"finish_reason": "length", "message": {"content": ""}}]})


def _complete(text: str) -> ModelResponse:
    return ModelResponse(text=text, raw={"choices": [{"finish_reason": "stop", "message": {"content": text}}]})


class FakeOpenAIShapedModel(Model):
    """Truncates the first ``truncations`` calls, then answers; records gen kwargs."""

    name = "fake-openai-shaped"

    def __init__(self, truncations: int, *, token_parameter="max_completion_tokens", max_tokens=2048, answer=_VALID):
        self.truncations = truncations
        self.token_parameter = token_parameter
        self.max_tokens = max_tokens
        self.answer = answer
        self.calls: list[dict] = []

    def generate(self, messages=None, **kw) -> ModelResponse:
        self.calls.append(dict(kw))
        if len(self.calls) <= self.truncations:
            return _truncated()
        return _complete(self.answer)


class FakeBareModel(Model):
    """OpenAI-shaped truncation but no token attributes -> never receives overrides."""

    name = "fake-bare"

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, messages=None, **kw) -> ModelResponse:
        self.calls.append(dict(kw))
        return _truncated()


def test_truncated_reasoning_response_retries_with_escalated_budget():
    model = FakeOpenAIShapedModel(truncations=1)
    step = run_text_json(model, _MESSAGES, None, budget=2)
    assert step.parse_error is None
    assert step.op == "bind"
    assert step.repair_attempts == 1
    assert model.calls == [{}, {"max_tokens": 16384}]


def test_persistent_truncation_escalates_to_cap_then_stops_cleanly():
    model = FakeOpenAIShapedModel(truncations=99)
    step = run_text_json(model, _MESSAGES, None, budget=2)
    assert step.done is True and step.op is None
    assert "output-token limit" in step.parse_error
    assert model.calls == [{}, {"max_tokens": 16384}, {"max_tokens": 32768}]


def test_plain_max_tokens_backend_escalates_modestly():
    model = FakeOpenAIShapedModel(truncations=1, token_parameter="max_tokens", max_tokens=2048)
    step = run_text_json(model, _MESSAGES, None, budget=2)
    assert step.parse_error is None
    assert model.calls == [{}, {"max_tokens": 4096}]


def test_backend_without_token_attributes_never_receives_overrides():
    model = FakeBareModel()
    step = run_text_json(model, _MESSAGES, None, budget=2)
    assert step.done is True
    assert "output-token limit" in step.parse_error
    assert model.calls == [{}, {}, {}]


def test_escalated_budget_policy():
    reasoning = FakeOpenAIShapedModel(0)
    assert escalated_budget(reasoning, None) == 16384
    assert escalated_budget(reasoning, 16384) == 32768
    assert escalated_budget(reasoning, 32768) is None  # cap reached -> stop growing

    plain = FakeOpenAIShapedModel(0, token_parameter="max_tokens", max_tokens=2048)
    assert escalated_budget(plain, None) == 4096
    assert escalated_budget(plain, 4096) is None

    assert escalated_budget(FakeBareModel(), None) is None
