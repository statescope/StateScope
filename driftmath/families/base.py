"""Base class for problem families."""

from __future__ import annotations

from abc import ABC, abstractmethod

from driftmath.io.schema import Difficulty, Problem


class Family(ABC):
    """A generator of fully-specified, CAS-verifiable problems for one family.

    Concrete subclasses set a unique :attr:`name` and implement :meth:`generate`.
    """

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        n: int,
        *,
        difficulty: Difficulty | None = None,
        seed: int = 0,
    ) -> list[Problem]:
        """Return ``n`` problems for this family (implemented in a later step)."""
        raise NotImplementedError
