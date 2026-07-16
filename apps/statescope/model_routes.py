"""Provider-aware model catalog for the StateScope demo.

The research runtime keeps one model identity separate from its execution route.
Local open weights can run through vLLM or OpenRouter; proprietary models can use
their native provider or OpenRouter. Credentials are resolved by model backends
from server-side environment variables and are never returned by this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from driftmath.models import aacl_models
from driftmath.models.environment import refresh_model_environment
from driftmath.models.local_store import MODELS_DIR, is_downloaded, local_path
from driftmath.models.registry import get_model

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"
MOCK_KEY = "mock"

_AACL_KEYS_BY_STEM = {
    stem: key for key, stem in {**aacl_models.AACL_MODEL_KEYS, **aacl_models.OPTIONAL_MODEL_KEYS}.items()
}

_PROPRIETARY = {
    "openai_gpt4o": {
        "config": "closed_openai",
        "openrouter": "openrouter_openai",
        "organization": "OpenAI",
        "status": "deprecated",
    },
    "openai_gpt4o_mini": {
        "config": "closed_openai_gpt4o_mini",
        "openrouter_model": "openai/gpt-4o-mini",
        "organization": "OpenAI",
    },
    "openai_gpt5_nano": {
        "config": "closed_openai_gpt5_nano",
        "openrouter_model": "openai/gpt-5-nano",
        "organization": "OpenAI",
    },
    "openai_gpt5_mini": {
        "config": "closed_openai_gpt5_mini",
        "openrouter_model": "openai/gpt-5-mini",
        "organization": "OpenAI",
    },
    "openai_gpt41": {
        "config": "closed_openai_gpt41",
        "openrouter_model": "openai/gpt-4.1",
        "organization": "OpenAI",
    },
    "openai_gpt41_mini": {
        "config": "closed_openai_gpt41_mini",
        "openrouter_model": "openai/gpt-4.1-mini",
        "organization": "OpenAI",
    },
    "openai_gpt41_nano": {
        "config": "closed_openai_gpt41_nano",
        "openrouter_model": "openai/gpt-4.1-nano",
        "organization": "OpenAI",
        "status": "deprecated",
    },
    "openai_gpt51": {
        "config": "closed_openai_gpt51",
        "openrouter_model": "openai/gpt-5.1",
        "organization": "OpenAI",
    },
    "openai_gpt56_luna": {
        "config": "closed_openai_gpt56_luna",
        "openrouter_model": "openai/gpt-5.6-luna",
        "organization": "OpenAI",
    },
    "openai_gpt54_mini": {
        "config": "closed_openai_gpt54_mini",
        "openrouter_model": "openai/gpt-5.4-mini",
        "organization": "OpenAI",
    },
    "openai_gpt54_nano": {
        "config": "closed_openai_gpt54_nano",
        "openrouter_model": "openai/gpt-5.4-nano",
        "organization": "OpenAI",
    },
    "anthropic_claude35_sonnet": {
        "config": "closed_anthropic",
        "openrouter": "openrouter_anthropic",
        "organization": "Anthropic",
    },
    "google_gemini15_pro": {
        "config": "closed_google",
        "openrouter": "openrouter_google",
        "organization": "Google",
    },
}

# OpenRouter-only entries do not have a committed local-weight configuration.
_OPENROUTER_ONLY = {
    "deepseek_r1": {"config": "openrouter_deepseek", "organization": "DeepSeek", "access": "open"},
    "qwen25_72b": {"config": "openrouter_qwen", "organization": "Qwen", "access": "open"},
}


def _load(stem: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{stem}.yaml").read_text(encoding="utf-8")) or {}


def _params(stem: str) -> dict:
    return dict(_load(stem).get("params", {}) or {})


def _route(
    key: str,
    label: str,
    backend: str,
    model_id: str,
    *,
    config_stem: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    configurable_url: bool = False,
) -> dict:
    return {
        "key": key,
        "label": label,
        "backend": backend,
        "model_id": model_id,
        "config_stem": config_stem,
        "base_url": base_url,
        "configurable_url": configurable_url,
        "credential_env": api_key_env,
        "credential_ready": True if not api_key_env else bool(os.environ.get(api_key_env)),
    }


def _open_entry(path: Path) -> dict:
    stem = path.stem
    cfg = _load(stem)
    params = cfg.get("params", {}) or {}
    cat = cfg.get("catalog", {}) or {}
    model_id = params.get("model") or (cfg.get("vllm", {}) or {}).get("model_id")
    key = _AACL_KEYS_BY_STEM.get(stem, stem.removeprefix("open_"))
    downloaded = is_downloaded(model_id)
    return {
        "key": key,
        "label": model_id,
        "model_id": model_id,
        "organization": model_id.split("/", 1)[0],
        "access": "open",
        "size_b": cat.get("size_b"),
        "category": cat.get("category", "general"),
        "status": cat.get("status", "active"),
        "downloaded": downloaded,
        "local_dir": str(local_path(model_id)).replace("\\", "/"),
        "routes": [
            _route(
                "local",
                "Local vLLM",
                "openai_compat",
                model_id,
                config_stem=stem,
                base_url=params.get("base_url"),
                configurable_url=True,
            ),
            _route(
                "openrouter",
                "OpenRouter",
                "openrouter",
                model_id,
                api_key_env="OPENROUTER_API_KEY",
                base_url="https://openrouter.ai/api/v1",
            ),
        ],
    }


def _proprietary_entry(key: str, spec: dict) -> dict:
    native = _params(spec["config"])
    route_stem = spec.get("openrouter")
    routed = _params(route_stem) if route_stem else {
        "model": spec["openrouter_model"],
        "api_key_env": "OPENROUTER_API_KEY",
    }
    status = spec.get("status", "active")
    return {
        "key": key,
        "label": native["model"] + (" (deprecated)" if status == "deprecated" else ""),
        "model_id": native["model"],
        "organization": spec["organization"],
        "access": "proprietary",
        "size_b": None,
        "category": (_load(spec["config"]).get("catalog", {}) or {}).get("category", "general"),
        "status": status,
        "downloaded": False,
        "local_dir": None,
        "routes": [
            _route(
                "native",
                f"{spec['organization']} API",
                _load(spec["config"])["type"],
                native["model"],
                config_stem=spec["config"],
                api_key_env=native.get("api_key_env"),
            ),
            _route(
                "openrouter",
                "OpenRouter",
                "openrouter",
                routed["model"],
                config_stem=route_stem,
                api_key_env=routed.get("api_key_env", "OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            ),
        ],
    }


def _openrouter_only_entry(key: str, spec: dict) -> dict:
    params = _params(spec["config"])
    return {
        "key": key,
        "label": params["model"],
        "model_id": params["model"],
        "organization": spec["organization"],
        "access": spec["access"],
        "size_b": None,
        "category": (_load(spec["config"]).get("catalog", {}) or {}).get("category", "general"),
        "status": "active",
        "downloaded": False,
        "local_dir": None,
        "routes": [
            _route(
                "openrouter",
                "OpenRouter",
                "openrouter",
                params["model"],
                config_stem=spec["config"],
                api_key_env=params.get("api_key_env", "OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        ],
    }


def catalog() -> list[dict]:
    """Return every configured model, grouped by model identity rather than config file."""
    refresh_model_environment()
    models = [
        {
            "key": MOCK_KEY,
            "label": "Mock (offline debug)",
            "model_id": None,
            "organization": "StateScope",
            "access": "debug",
            "size_b": None,
            "category": "debug",
            "status": "debug",
            "downloaded": True,
            "local_dir": None,
            "routes": [_route("mock", "Offline deterministic", "mock", "mock")],
        }
    ]
    models.extend(_open_entry(path) for path in sorted(CONFIG_DIR.glob("open_*.yaml")))
    models.extend(_openrouter_only_entry(key, spec) for key, spec in _OPENROUTER_ONLY.items())
    models.extend(_proprietary_entry(key, spec) for key, spec in _PROPRIETARY.items())
    return models


def resolve(model_key: str, route_key: str | None = None) -> tuple[dict, dict]:
    model = next((item for item in catalog() if item["key"] == model_key), None)
    if model is None:
        raise KeyError(f"unknown model key {model_key!r}")
    route_key = route_key or model["routes"][0]["key"]
    route = next((item for item in model["routes"] if item["key"] == route_key), None)
    if route is None:
        raise KeyError(f"model {model_key!r} does not support route {route_key!r}")
    return model, route


def make_model(
    model_key: str,
    route_key: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
):
    """Instantiate one catalog model through one validated execution route.

    ``api_key`` is an optional request-scoped override from the masked demo field.
    It is passed to the model instance only and never included in catalog or
    provenance payloads.
    """
    refresh_model_environment()
    model_info, route = resolve(model_key, route_key)
    if model_key == MOCK_KEY:
        return get_model("mock"), model_info, route

    overrides: dict[str, Any] = {}
    if route["configurable_url"]:
        overrides["base_url"] = base_url or route["base_url"]
    if api_key and route.get("credential_env"):
        overrides["api_key"] = api_key
    stem = route.get("config_stem")
    if stem:
        model = get_model(str(CONFIG_DIR / f"{stem}.yaml"), **overrides)
    else:
        # Config-less OpenRouter route. Reasoning models burn hidden reasoning tokens
        # against the output budget and reject non-default temperatures, so they get a
        # large budget, default temperature, and a low unified reasoning effort.
        reasoning = model_info.get("category") == "reasoning"
        model = get_model(
            route["backend"],
            model=route["model_id"],
            base_url=route.get("base_url"),
            api_key_env=route.get("credential_env") or "OPENROUTER_API_KEY",
            api_key=api_key,
            max_tokens=16384 if reasoning else 2048,
            temperature=None if reasoning else 0.0,
            extra_body={"reasoning": {"effort": "low"}} if reasoning else None,
        )
    return model, model_info, route


def provenance(model_info: dict, route: dict, base_url: str | None = None) -> dict:
    """Sanitized, export-safe run provenance. No credential values are included."""
    endpoint = base_url if route["configurable_url"] and base_url else route.get("base_url")
    return {
        "model_key": model_info["key"],
        "model_id": route["model_id"],
        "access": model_info["access"],
        "route": route["key"],
        "backend": route["backend"],
        "endpoint": endpoint,
        "config": f"configs/models/{route['config_stem']}.yaml" if route.get("config_stem") else None,
    }
