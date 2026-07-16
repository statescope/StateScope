"""Offline tests for the real-model backend layer.

No network, no GPU, no API keys, no model downloads. SDK imports are lazy and are
never triggered here; credentials are only required when a backend is *called*.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from driftmath.models import model_catalog
from driftmath.models import environment as model_environment
from driftmath.models.base import MissingCredentialError
from driftmath.models.hf_model import HFModel
from driftmath.models.openai_compat import OpenAICompatModel
from driftmath.models.openrouter_model import OpenRouterModel
from driftmath.models.registry import get_model
from driftmath.models.vllm_server import (
    build_vllm_command,
    choose_max_model_len,
    choose_tensor_parallel_size,
    estimate_gpu_plan,
)

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _ROOT / "configs" / "models"
_MODEL_CONFIGS = sorted(_MODEL_DIR.glob("*.yaml"))
_CLOSED = ["closed_openai", "closed_anthropic", "closed_google"]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Config parsing + offline instantiation
# --------------------------------------------------------------------------- #
def test_there_are_model_configs():
    assert len(_MODEL_CONFIGS) >= 29


@pytest.mark.parametrize("path", _MODEL_CONFIGS, ids=lambda p: p.stem)
def test_config_parses(path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "type" in cfg


@pytest.mark.parametrize("path", _MODEL_CONFIGS, ids=lambda p: p.stem)
def test_config_instantiates_without_network_or_keys(path, monkeypatch):
    # Ensure no credentials are present; instantiation must still succeed.
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    model = get_model(str(path))
    assert model is not None
    assert hasattr(model, "generate")


# --------------------------------------------------------------------------- #
# Missing credentials -> clear error at call time (not at parse/instantiate)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _CLOSED)
def test_closed_backend_without_key_raises_missing_credential(name, monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    model = get_model(str(_MODEL_DIR / f"{name}.yaml"))
    with pytest.raises(MissingCredentialError):
        model.generate([{"role": "user", "content": "hi"}])


def test_local_vllm_config_uses_fallback_key(monkeypatch):
    # Local vLLM configs have api_key_fallback=EMPTY -> no MissingCredentialError.
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    model = get_model(str(_MODEL_DIR / "open_qwen3_4b.yaml"))
    assert model._resolve_key() == "EMPTY"  # would not raise


# --------------------------------------------------------------------------- #
# Payload building (pure, offline)
# --------------------------------------------------------------------------- #
def test_openai_compat_builds_payload():
    m = OpenAICompatModel(model="my-model", base_url="http://localhost:8000/v1", max_tokens=128, temperature=0.5)
    payload = m.build_payload([{"role": "user", "content": "hi"}])
    assert payload == {
        "model": "my-model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5,
        "max_tokens": 128,
    }
    with_tools = m.build_payload([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert with_tools["tools"] == [{"type": "function"}]
    assert with_tools["tool_choice"] == "auto"


def test_openai_compat_forwards_extra_body():
    m = OpenAICompatModel(
        model="Qwen/Qwen3-4B",
        base_url="http://localhost:8000/v1",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    payload = m.build_payload([{"role": "user", "content": "hi"}])
    assert payload["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_openai_reasoning_config_uses_completion_token_parameter():
    model = get_model(str(_MODEL_DIR / "closed_openai_gpt56_luna.yaml"))
    payload = model.build_payload([{"role": "user", "content": "hi"}])
    assert payload["model"] == "gpt-5.6-luna"
    # Reasoning models burn hidden reasoning tokens against this budget, so it must
    # be far larger than the visible JSON object (2048 caused live truncation).
    assert payload["max_completion_tokens"] == 16384
    assert "max_tokens" not in payload
    assert "temperature" not in payload


def test_openai_reasoning_config_requests_minimal_effort():
    model = get_model(str(_MODEL_DIR / "closed_openai_gpt5_nano.yaml"))
    payload = model.build_payload([{"role": "user", "content": "hi"}])
    assert payload["max_completion_tokens"] == 16384
    assert payload["extra_body"]["reasoning_effort"] == "minimal"


def test_openai_compat_normalizes_max_tokens_override_onto_token_parameter():
    m = OpenAICompatModel(model="m", token_parameter="max_completion_tokens", max_tokens=2048, temperature=None)
    payload = m.build_payload([{"role": "user", "content": "hi"}], max_tokens=16384)
    assert payload["max_completion_tokens"] == 16384
    assert "max_tokens" not in payload


def test_openai_compat_merges_reasoning_effort_into_extra_body():
    m = OpenAICompatModel(model="m", reasoning_effort="minimal", extra_body={"keep": 1})
    payload = m.build_payload([{"role": "user", "content": "hi"}])
    assert payload["extra_body"] == {"keep": 1, "reasoning_effort": "minimal"}


def test_env_refresh_picks_up_key_added_after_startup(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    monkeypatch.setattr(model_environment, "ENV_FILE", env_file)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model_environment.refresh_model_environment()
    assert not model_environment.os.environ.get("OPENAI_API_KEY")

    env_file.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    model_environment.refresh_model_environment()
    assert model_environment.os.environ["OPENAI_API_KEY"] == "test-key"


def test_env_refresh_preserves_exported_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setattr(model_environment, "ENV_FILE", env_file)
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")
    model_environment.refresh_model_environment()
    assert model_environment.os.environ["OPENAI_API_KEY"] == "process-key"


def test_openrouter_uses_env_and_base_url(monkeypatch):
    model = get_model(str(_MODEL_DIR / "openrouter_qwen.yaml"))
    assert isinstance(model, OpenRouterModel)
    assert model.api_key_env == "OPENROUTER_API_KEY"
    assert model.base_url == "https://openrouter.ai/api/v1"
    assert model.model == "qwen/qwen-2.5-72b-instruct"

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError):
        model._resolve_key()
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    assert model._resolve_key() == "router-key"


# --------------------------------------------------------------------------- #
# HF backend instantiates offline without loading weights
# --------------------------------------------------------------------------- #
def test_hf_model_does_not_load_on_init():
    m = HFModel(model_id="some/model", trust_remote_code=True)
    assert m.loaded is False
    assert m.model_id == "some/model"


# --------------------------------------------------------------------------- #
# vLLM planning + command builder
# --------------------------------------------------------------------------- #
def test_tensor_parallel_and_context_rules():
    assert choose_tensor_parallel_size(1.5, 1) == 1
    assert choose_tensor_parallel_size(7, 1) == 1
    assert choose_tensor_parallel_size(14, 1) == 1  # 14B fits a single A100 40GB
    assert choose_tensor_parallel_size(14, 2) == 2  # multi-GPU -> tensor parallel
    assert choose_max_model_len(1.5) == 32768
    assert choose_max_model_len(7) == 16384
    assert choose_max_model_len(14) == 8192


def test_gpu_plan_14b_respects_half_memory_cap():
    single = estimate_gpu_plan(14, gpu_count=1, gpu_vram_gb=40)
    assert single["tensor_parallel_size"] == 1
    assert single["gpu_memory_utilization"] == 0.5
    assert single["fits"] is False
    multi = estimate_gpu_plan(14, gpu_count=2, gpu_vram_gb=40)
    assert multi["tensor_parallel_size"] == 2
    assert multi["fits"] is True


def _cmd_for(name):
    cfg = yaml.safe_load((_MODEL_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    return build_vllm_command(cfg)


def test_vllm_command_1_5b_single_gpu():
    cmd = _cmd_for("open_qwen25_1_5b")
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen2.5-1.5B-Instruct"]
    assert "--tensor-parallel-size" in cmd and cmd[cmd.index("--tensor-parallel-size") + 1] == "1"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.5"
    assert cmd[cmd.index("--max-model-len") + 1] == "32768"
    assert "--trust-remote-code" in cmd


def test_vllm_command_7b_single_a100():
    cmd = _cmd_for("open_qwen25_7b")
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "1"
    assert cmd[cmd.index("--max-model-len") + 1] == "16384"


def test_vllm_command_14b_single_a100():
    cmd = _cmd_for("open_qwen3_14b")
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "1"
    assert cmd[cmd.index("--max-model-len") + 1] == "8192"


def test_vllm_command_14b_multi_gpu_tensor_parallel():
    cfg = yaml.safe_load((_MODEL_DIR / "open_qwen3_14b.yaml").read_text(encoding="utf-8"))
    plan = estimate_gpu_plan(cfg["catalog"]["size_b"], gpu_count=2, gpu_vram_gb=40)
    cfg["vllm"]["tensor_parallel_size"] = plan["tensor_parallel_size"]
    cmd = build_vllm_command(cfg)
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def test_catalog_covers_configs_and_paths_exist():
    assert model_catalog.CATALOG
    for info in model_catalog.CATALOG:
        assert Path(info.config_path).exists(), info.config_path
        assert info.category in model_catalog.CATEGORIES


def test_catalog_open_models_are_1b_to_30b():
    opens = model_catalog.open_models()
    assert opens
    for info in opens:
        assert 1 <= info.size_b <= 30, (info.name, info.size_b)
    assert model_catalog.get("open_qwen3_4b").size_b == 4
    assert model_catalog.get("open_qwen3_14b").size_b == 14
    assert model_catalog.get("open_qwen3_30b_a3b_instruct_2507").size_b == 30


# --------------------------------------------------------------------------- #
# CLI model overrides
# --------------------------------------------------------------------------- #
def test_parse_model_overrides_forms():
    run_eval = _load_script("run_eval")
    assert run_eval.parse_model_overrides(None, None, ["large=mock", "small=mock"]) == [
        ("large", "mock"), ("small", "mock")
    ]
    assert run_eval.parse_model_overrides(None, "a.yaml,b.yaml", None) == [
        ("a", "a.yaml"), ("b", "b.yaml")
    ]
    assert run_eval.parse_model_overrides("configs/models/open_qwen3_4b.yaml", None, None) == [
        ("open_qwen3_4b", "configs/models/open_qwen3_4b.yaml")
    ]
    assert run_eval.parse_model_overrides(None, None, None) is None


def test_run_eval_model_role_override_offline(tmp_path):
    from driftmath.io.storage import read_jsonl

    run_eval = _load_script("run_eval")
    rc = run_eval.main([
        "--experiment", str(_ROOT / "configs" / "experiments" / "smoke.yaml"),
        "--model-role", "large=mock",
        "--model-role", "small=mock",
        "--out-root", str(tmp_path),
    ])
    assert rc == 0
    rows = read_jsonl(tmp_path / "smoke" / "metrics.jsonl")
    assert {r["model"] for r in rows} == {"large", "small"}
    assert all(r["model_spec"] == "mock" for r in rows)
