"""Core symbolic-state objects for DriftMath.

These are the *working* objects a solver must keep consistent across a derivation:
bindings (variable -> expression), accumulated constraints, and the current
working expression / equation / solution-candidates.

Design note: SymPy expressions are **not** stored on these models. Expressions are
held as *strings* (SymPy string form, or ``srepr`` for guaranteed round-trip), so
the models serialize to JSON trivially and stably. The semantic layer
(:mod:`driftmath.core.sym_utils`) parses these strings back into SymPy objects for
comparison. This keeps the schema independent of SymPy internals.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ItemStatus = Literal["live", "discharged"]


class StateItem(BaseModel):
    """A single entry in the symbolic state: a binding, intermediate result, or invariant.

    Attributes
    ----------
    id:
        Stable identifier (e.g. ``"u"``, ``"x"``, ``"a_3"``, or a generated id).
    expr:
        The bound expression / value as a SymPy string.
    deps:
        Ids of other :class:`StateItem`\\ s this item depends on (edges of the state DAG).
    status:
        ``"live"`` while still load-bearing, ``"discharged"`` once safely retired.
    kind:
        ``"binding" | "intermediate" | "invariant"`` (free-form, extensible).
    """

    id: str
    expr: str
    deps: list[str] = Field(default_factory=list)
    status: ItemStatus = "live"
    kind: str = "binding"


class Constraint(BaseModel):
    """A domain / assumption constraint accrued by an (often irreversible) move.

    Examples: ``"Ne(x, 0)"`` after cancelling a factor of ``x``; ``"x >= 0"`` after
    squaring both sides.
    """

    expr: str
    reason: str = ""
    deps: list[str] = Field(default_factory=list)


class SymbolicState(BaseModel):
    """The live symbolic state at one point in a derivation."""

    bindings: list[StateItem] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    current_expr: str | None = None
    current_equation: str | None = None
    candidates: list[str] = Field(default_factory=list)
    final_answer: str | None = None

    # Structural dependency nodes used only to describe the state DAG (read by
    # difficulty computation). They are deliberately NOT part of state equality:
    # state comparison is over the semantic content above, so a system need not
    # reproduce these abstract nodes to match a gold state.
    dep_nodes: list[StateItem] = Field(default_factory=list)

    # -- convenience helpers (pure bookkeeping; no SymPy here) --

    def get_binding(self, item_id: str) -> StateItem | None:
        for b in self.bindings:
            if b.id == item_id:
                return b
        return None

    def live_bindings(self) -> list[StateItem]:
        return [b for b in self.bindings if b.status == "live"]

    def binding_map(self) -> dict[str, StateItem]:
        return {b.id: b for b in self.bindings}

    def copy_state(self) -> "SymbolicState":
        return self.model_copy(deep=True)
