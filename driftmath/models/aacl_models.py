"""The AACL demo-track model set: open, locally served models only.

This is the single source of truth for which models the AACL StateScope demo
uses. Each entry maps a short key (what ``--model`` takes on the CLI and what
the UI shows) to a committed config under ``configs/models/``. Adding a model
means adding a config file and one line here -- no core-logic changes.

No proprietary/hosted backends belong in this set: every config must be an
``openai_compat`` config pointing at a local vLLM endpoint.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from driftmath.models.local_store import MODELS_DIR, is_downloaded, local_path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "models"

# key -> config stem under configs/models/. Order = the AACL evaluation order.
AACL_MODEL_KEYS: dict[str, str] = {
    "qwen25_1_5b": "open_qwen25_1_5b",                        # Qwen/Qwen2.5-1.5B-Instruct
    "qwen3_4b": "open_qwen3_4b",                              # Qwen/Qwen3-4B
    "qwen3_8b": "open_qwen3_8b",                              # Qwen/Qwen3-8B
    "qwen25_math_7b": "open_qwen25_math_7b",                  # Qwen/Qwen2.5-Math-7B-Instruct
    "r1_distill_qwen_14b": "open_deepseek_r1_distill_qwen_14b",  # deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    "qwen3_14b": "open_qwen3_14b",                            # Qwen/Qwen3-14B
    "qwen3_30b_a3b": "open_qwen3_30b_a3b_instruct_2507",      # Qwen/Qwen3-30B-A3B-Instruct-2507
}

# Supported but not part of the required AACL set.
OPTIONAL_MODEL_KEYS: dict[str, str] = {
    "qwen3_30b_a3b_thinking": "open_qwen3_30b_a3b_thinking_2507",  # Qwen/Qwen3-30B-A3B-Thinking-2507
}

_LOCAL_PROVIDERS = {"openai_compat"}


def all_keys(include_optional: bool = True) -> list[str]:
    keys = list(AACL_MODEL_KEYS)
    if include_optional:
        keys += list(OPTIONAL_MODEL_KEYS)
    return keys


def config_path(key: str) -> Path:
    """The committed YAML config for an AACL model key."""
    stem = AACL_MODEL_KEYS.get(key) or OPTIONAL_MODEL_KEYS.get(key)
    if stem is None:
        raise KeyError(f"unknown AACL model key {key!r}; known: {all_keys()}")
    return CONFIG_DIR / f"{stem}.yaml"


def _load(key: str) -> dict:
    return yaml.safe_load(config_path(key).read_text(encoding="utf-8")) or {}


def hf_id(key: str) -> str:
    """The Hugging Face repo id behind a model key (from the config's vllm block)."""
    cfg = _load(key)
    return (cfg.get("vllm", {}) or {}).get("model_id") or (cfg.get("params", {}) or {}).get("model")


def result_slug(key: str) -> str:
    """Stable results directory slug: ``qwen3_4b`` -> ``qwen3-4b``."""
    model_name = hf_id(key).split("/")[-1].strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", model_name).strip("-")
    return slug or key.replace("_", "-")


def default_result_dir(key: str, results_root: Path | str = REPO_ROOT / "results") -> Path:
    """Default paper-run output directory for one AACL model key."""
    return Path(results_root) / result_slug(key)


def describe(key: str, models_root: Path | str = MODELS_DIR) -> dict:
    """One catalog entry (what ``/api/models`` and ``--list`` show)."""
    cfg = _load(key)
    params = cfg.get("params", {}) or {}
    cat = cfg.get("catalog", {}) or {}
    mid = hf_id(key)
    if cfg.get("type") not in _LOCAL_PROVIDERS:
        raise ValueError(f"AACL model {key!r} must be an openai_compat local config, got type={cfg.get('type')!r}")
    return {
        "key": key,
        "hf_id": mid,
        "config_path": str(config_path(key)).replace("\\", "/"),
        "size_b": cat.get("size_b"),
        "category": cat.get("category", "general"),
        "base_url": params.get("base_url"),
        "local_dir": str(local_path(mid, models_root)).replace("\\", "/"),
        "downloaded": is_downloaded(mid, models_root),
        "optional": key in OPTIONAL_MODEL_KEYS,
    }


def catalog(include_optional: bool = True, models_root: Path | str = MODELS_DIR) -> list[dict]:
    """The full AACL model catalog, in evaluation order."""
    return [describe(k, models_root) for k in all_keys(include_optional)]
