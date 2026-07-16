"""Offline tests for the StateScope demo server endpoints (mock model only)."""

from pathlib import Path

from apps.statescope import model_routes
from apps.statescope.examples import get_problem, list_examples
from apps.statescope.server import models_endpoint, replay_endpoint, run_endpoint

_WEB = Path(__file__).resolve().parents[1] / "apps" / "statescope" / "web"


def test_examples_list_is_populated():
    ex = list_examples()
    assert len(ex) >= 8
    for e in ex:
        for key in ("id", "title", "difficulty", "blurb", "family", "problem_text", "n_steps"):
            assert key in e


def test_run_mock_clean_both_systems_match_gold():
    out = run_endpoint({"example_id": "radical_extraneous", "system": "both", "model_key": "mock"})
    assert out["ok"]
    assert out["c"]["metrics"]["sf"] == 1.0
    assert out["d"]["metrics"]["sf"] == 1.0
    # payload shape the UI relies on
    assert out["d"]["gold_answer"]
    assert all("verify" in s for s in out["d"]["steps"])
    assert out["model_provenance"]["route"] == "mock"
    assert out["d"]["adapter_log"] == []
    # no drift -> explanation says so
    exp = out["c"]["drift_explanation"]
    assert exp["first_drift_step"] is None
    assert "No drift" in exp["explanation"]


def test_run_mock_planted_drift_contrasts_c_and_d():
    out = run_endpoint({"example_id": "two_state_recurrence", "system": "both", "model_key": "mock", "drift_step": 2})
    assert out["ok"]
    assert out["c"]["metrics"]["sf"] < 1.0  # text-state drifts
    assert out["c"]["cod"] == 2
    assert out["d"]["metrics"]["sf"] == 1.0  # ledger recovers


def test_drift_explanation_is_deterministic_and_grounded():
    out = run_endpoint({"example_id": "two_state_recurrence", "system": "c", "model_key": "mock", "drift_step": 2})
    exp = out["c"]["drift_explanation"]
    assert exp["first_drift_step"] == 2
    assert exp["first_drift_op"]
    assert exp["diverged_components"]  # names of state components, from the oracle diff
    assert exp["cas_status_at_drift"] in {"ok", "failed", "skipped"}
    assert "step 2" in exp["explanation"]
    # the components named in the text come from the diff, not from a model
    step2 = next(s for s in out["c"]["steps"] if s["index"] == 2)
    assert set(exp["diverged_components"]) == set(step2["diff"])


def test_counterfactual_replay_rederives_downstream():
    p = get_problem("linear_recurrence")
    step = next(s.index for s in p.gold_trace.steps if s.op == "bind" and s.args.get("inputs"))
    res = replay_endpoint({"example_id": "linear_recurrence", "system": "d", "step": step, "formula": "999"})
    assert res["ok"]
    assert res["result"]["metrics"]["sf"] < 1.0


# --------------------------------------------------------------------------- #
# Provider-aware research model catalog
# --------------------------------------------------------------------------- #
def test_models_endpoint_lists_all_open_proprietary_and_mock_models():
    out = models_endpoint()
    assert out["ok"]
    by_key = {m["key"]: m for m in out["models"]}
    for key in ("qwen25_1_5b", "qwen3_4b", "qwen3_8b", "qwen25_math_7b",
                "r1_distill_qwen_14b", "qwen3_14b", "qwen3_30b_a3b"):
        assert key in by_key, key
        assert by_key[key]["model_id"]
        assert by_key[key]["access"] == "open"
        assert {r["key"] for r in by_key[key]["routes"]} == {"local", "openrouter"}
    openai_keys = (
        "openai_gpt4o", "openai_gpt4o_mini", "openai_gpt5_nano", "openai_gpt5_mini",
        "openai_gpt41", "openai_gpt41_mini", "openai_gpt41_nano", "openai_gpt51",
        "openai_gpt54_mini", "openai_gpt54_nano", "openai_gpt56_luna",
    )
    for key in (*openai_keys, "anthropic_claude35_sonnet", "google_gemini15_pro"):
        assert by_key[key]["access"] == "proprietary"
        assert {r["key"] for r in by_key[key]["routes"]} == {"native", "openrouter"}
    assert {by_key[key]["model_id"] for key in openai_keys} == {
        "gpt-4o", "gpt-4o-mini", "gpt-5-nano", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini",
        "gpt-4.1-nano", "gpt-5.1", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.6-luna",
    }
    assert by_key["openai_gpt4o"]["status"] == "deprecated"
    assert by_key["openai_gpt41_nano"]["status"] == "deprecated"
    mock = by_key["mock"]
    assert mock["access"] == "debug"
    assert "debug" in mock["label"].lower() or "offline" in mock["label"].lower()


def test_provider_ui_masks_keys_and_never_exposes_server_values(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "do-not-leak-this-secret")
    payload = str(models_endpoint())
    assert "do-not-leak-this-secret" not in payload
    assert "credential_ready" in payload
    html = (_WEB / "index.html").read_text(encoding="utf-8").lower()
    assert 'type="password"' in html
    assert 'autocomplete="new-password"' in html
    assert "localstorage" not in html
    assert "session only; never saved" in html
    assert "execution route" in html


def test_request_scoped_key_overrides_environment_without_catalog_exposure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server-secret")
    model, _, _ = model_routes.make_model(
        "openai_gpt5_nano", "native", api_key="temporary-browser-secret"
    )
    assert model._resolve_key() == "temporary-browser-secret"
    assert "temporary-browser-secret" not in str(models_endpoint())


def test_env_template_is_safe_and_complete():
    root = Path(__file__).resolve().parents[1]
    template = (root / ".env.example").read_text(encoding="utf-8")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        assert f"{name}=" in template
    assert not any(line.startswith("sk-") for line in template.splitlines())


def test_run_with_unknown_model_key_returns_clean_error():
    out = run_endpoint({"example_id": "radical_extraneous", "system": "d", "model_key": "gpt4o"})
    assert out["ok"] is False
    assert "model key" in out["error"].lower()


def test_run_with_unsupported_route_returns_clean_error():
    out = run_endpoint({
        "example_id": "radical_extraneous",
        "system": "d",
        "model_key": "openai_gpt4o",
        "route_key": "local",
    })
    assert out["ok"] is False
    assert "does not support route" in out["error"]


def test_run_with_open_model_key_and_dead_endpoint_fails_cleanly(monkeypatch):
    # A configured open model with no live vLLM server must surface a clean error,
    # not crash -- and must never fall back to a hosted API.
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    out = run_endpoint({
        "example_id": "radical_extraneous", "system": "d",
        "model_key": "qwen3_4b", "route_key": "local",
        "base_url": "http://127.0.0.1:9",  # unroutable port
    })
    assert out["ok"] is False
    assert "error" in out
