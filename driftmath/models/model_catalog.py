"""Catalog of model configs with metadata, derived from ``configs/models/*.yaml``.

Building from the config files keeps the catalog in sync with what is actually
committed. Each entry records provider, backend type, model id, size, category,
recommended GPU, default context length, tool support, and the config path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"
CATEGORIES = {"general", "math", "reasoning"}


@dataclass(frozen=True)
class ModelInfo:
    name: str
    provider: str
    backend: str
    model_id: str | None
    size_b: float | None
    category: str
    recommended_gpu: str
    default_max_model_len: int | None
    supports_tools: bool
    config_path: str


def _provider(cfg: dict) -> str:
    t = cfg.get("type")
    if t == "openai_compat":
        base = (cfg.get("params", {}) or {}).get("base_url", "") or ""
        return "local-vllm" if ("localhost" in base or "127.0.0.1" in base) else "openai-compat"
    return {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google",
        "openrouter": "openrouter",
        "hf": "local-hf",
        "mock": "mock",
    }.get(t, t or "unknown")


def _info_from_config(path: Path) -> ModelInfo:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = cfg.get("params", {}) or {}
    vllm = cfg.get("vllm", {}) or {}
    cat = cfg.get("catalog", {}) or {}
    return ModelInfo(
        name=path.stem,
        provider=_provider(cfg),
        backend=cfg.get("type", "unknown"),
        model_id=params.get("model") or vllm.get("model_id"),
        size_b=cat.get("size_b"),
        category=cat.get("category", "general"),
        recommended_gpu=cat.get("recommended_gpu", "-"),
        default_max_model_len=vllm.get("max_model_len"),
        supports_tools=bool(params.get("supports_tools", cat.get("supports_tools", False))),
        config_path=str(path).replace("\\", "/"),
    )


def build_catalog(config_dir: Path | str = CONFIG_DIR) -> list[ModelInfo]:
    return [_info_from_config(p) for p in sorted(Path(config_dir).glob("*.yaml"))]


CATALOG: list[ModelInfo] = build_catalog()


def get(name: str) -> ModelInfo:
    for info in CATALOG:
        if info.name == name:
            return info
    raise KeyError(f"unknown model {name!r}; known: {[i.name for i in CATALOG]}")


def by_category(category: str) -> list[ModelInfo]:
    return [i for i in CATALOG if i.category == category]


def by_backend(backend: str) -> list[ModelInfo]:
    return [i for i in CATALOG if i.backend == backend]


def open_models() -> list[ModelInfo]:
    """Locally-served open models (vLLM or HF), 1B-14B."""
    return [i for i in CATALOG if i.provider in {"local-vllm", "local-hf"} and i.size_b is not None]


def config_path(name: str) -> str:
    return get(name).config_path
