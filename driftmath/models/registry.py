"""Model registry: ``@register_model`` + ``get_model(config_path_or_name)``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type

import yaml

from driftmath.models.base import Model

_MODELS: dict[str, Type[Model]] = {}


def register_model(cls: Type[Model]) -> Type[Model]:
    key = getattr(cls, "type", None) or getattr(cls, "name", None)
    if not key:
        raise ValueError(f"Model {cls!r} must set a 'type' (or 'name') class attribute.")
    _MODELS[key] = cls
    return cls


def get_model(config_path_or_name: str, **overrides: Any) -> Model:
    """Instantiate a model from a YAML config path or a registered type name.

    A YAML config looks like ``{type: mock, params: {...}}``. Extra keyword
    ``overrides`` take precedence over config params.
    """
    p = Path(str(config_path_or_name))
    if p.suffix.lower() in {".yaml", ".yml"} and p.exists():
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        mtype = cfg.get("type")
        params = dict(cfg.get("params", {}))
    else:
        mtype = str(config_path_or_name)
        params = {}
    params.update(overrides)
    if mtype not in _MODELS:
        raise KeyError(f"Unknown model type {mtype!r}. Registered: {sorted(_MODELS)}")
    return _MODELS[mtype](**params)


def names() -> list[str]:
    return sorted(_MODELS)
