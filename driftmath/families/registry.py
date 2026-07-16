"""A tiny registry so families can be looked up by name from the CLI.

Concrete families register themselves with the :func:`register` class decorator;
``generate_data.py`` resolves a family by name via :func:`get`.
"""

from __future__ import annotations

from typing import Type

from driftmath.families.base import Family

_REGISTRY: dict[str, Type[Family]] = {}


def register(cls: Type[Family]) -> Type[Family]:
    """Class decorator: register a :class:`Family` subclass under its ``name``."""
    name = getattr(cls, "name", None)
    if not name or name == "base":
        raise ValueError(f"Family {cls!r} must set a unique 'name' class attribute.")
    _REGISTRY[name] = cls
    return cls


def get(name: str) -> Family:
    """Instantiate a registered family by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown family {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def names() -> list[str]:
    """Return the sorted names of all registered families."""
    return sorted(_REGISTRY)
