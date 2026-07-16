"""System registry: ``@register_system`` + ``get_system(name)``."""

from __future__ import annotations

from typing import Type

from driftmath.systems.base import System

_SYSTEMS: dict[str, Type[System]] = {}


def register_system(cls: Type[System]) -> Type[System]:
    name = getattr(cls, "name", None)
    if not name or name == "system":
        raise ValueError(f"System {cls!r} must set a unique 'name' class attribute.")
    _SYSTEMS[name] = cls
    return cls


def get_system(name: str) -> System:
    if name not in _SYSTEMS:
        raise KeyError(f"Unknown system {name!r}. Registered: {sorted(_SYSTEMS)}")
    return _SYSTEMS[name]()


def names() -> list[str]:
    return sorted(_SYSTEMS)
