"""Family C -- recurrences and iterative running state.

Three synthetic sub-families, all expressed as exact chained bindings (so they reuse
:func:`driftmath.families.family_a.build_chain_trace` and are System-D replayable):

* **linear**    ``a_{n+1} = p*a_n + q``, ``a_0 = r``  -> find ``a_N``.
* **two_state** ``x_{n+1} = a*x_n + b*y_n``, ``y_{n+1} = c*x_n + d*y_n`` (fan-in 2).
* **finance**   each step applies a percentage growth and a fixed withdrawal (exact
                via ``Rational``).

Every accumulator ``a_k`` is a binding whose value is computed exactly with SymPy.
Parameters such as ``p``, ``q``, ``rate_mult``, and ``wd`` are also explicit
bindings, so every emitted problem has fan-in > 1 rather than a trivial single-lane
chain. The Euclidean algorithm is left out (optional).
"""

from __future__ import annotations

import random

from driftmath.families.base import Family
from driftmath.families.family_a import (
    BindingSpec,
    build_chain_trace,
    compute_difficulty_from_trace,
)
from driftmath.families.registry import register
from driftmath.io.schema import Problem


def _finalize(pid, specs, target, *, kind, params, problem_text, meta_extra=None) -> Problem:
    trace, values = build_chain_trace(pid, specs, target=target)
    trace.metadata.update({"subtype": "recurrence", "kind": kind, "target": target})
    meta = {
        "source": "synthetic",
        "provenance": "synthetic",
        "license": "CC0-1.0",
        "contamination_risk": "none",
        "subtype": "recurrence",
        "kind": kind,
        "target": target,
        "params": params,
    }
    if meta_extra:
        meta.update(meta_extra)
    return Problem(
        id=pid,
        family="family_c",
        problem_text=problem_text,
        gold_answer=str(values[target]),
        gold_trace=trace,
        meta=meta,
        difficulty=compute_difficulty_from_trace(trace),
    )


@register
class FamilyC(Family):
    name = "family_c"

    _KINDS = ("linear", "two_state", "finance")

    def generate(self, n: int, *, difficulty=None, seed: int = 0) -> list[Problem]:
        out: list[Problem] = []
        for i in range(n):
            rng = random.Random(seed * 10_000 + i)
            pid = f"family_c-{seed}-{i:04d}"
            kind = self._KINDS[i % len(self._KINDS)]
            out.append(getattr(self, f"_gen_{kind}")(pid, rng))
        return out

    # --------------------------------------------------------------- linear
    def _gen_linear(self, pid: str, rng: random.Random) -> Problem:
        p, q, r, N = rng.randint(2, 3), rng.randint(1, 4), rng.randint(1, 4), 5
        specs = [
            BindingSpec("p", str(p), []),
            BindingSpec("q", str(q), []),
            BindingSpec("a0", str(r), []),
        ]
        for k in range(1, N + 1):
            specs.append(BindingSpec(f"a{k}", f"p*a{k - 1} + q", [f"a{k - 1}", "p", "q"]))
        text = f"Let a_0 = {r} and a_(n+1) = {p}*a_n + {q}. Find a_{N}."
        return _finalize(
            pid, specs, f"a{N}", kind="linear",
            params={"p": p, "q": q, "r": r, "N": N}, problem_text=text,
        )

    # ------------------------------------------------------------ two_state
    def _gen_two_state(self, pid: str, rng: random.Random) -> Problem:
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        while b == a:
            b = rng.randint(1, 3)
        c, d = rng.randint(1, 3), rng.randint(1, 3)
        while d == c:
            d = rng.randint(1, 3)
        r1, N = rng.randint(1, 3), 4
        r2 = rng.randint(1, 3)
        while r2 == r1:  # distinct initials so a cross-binding swap always diverges
            r2 = rng.randint(1, 3)
        specs = [BindingSpec("x0", str(r1), []), BindingSpec("y0", str(r2), [])]
        for k in range(1, N + 1):
            specs.append(BindingSpec(f"x{k}", f"{a}*x{k - 1} + {b}*y{k - 1}", [f"x{k - 1}", f"y{k - 1}"]))
            specs.append(BindingSpec(f"y{k}", f"{c}*x{k - 1} + {d}*y{k - 1}", [f"x{k - 1}", f"y{k - 1}"]))
        text = (
            f"Let x_0={r1}, y_0={r2}, x_(n+1)={a}x_n+{b}y_n, y_(n+1)={c}x_n+{d}y_n. Find x_{N}."
        )
        return _finalize(
            pid, specs, f"x{N}", kind="two_state",
            params={"a": a, "b": b, "c": c, "d": d, "r1": r1, "r2": r2, "N": N}, problem_text=text,
        )

    # --------------------------------------------------------------- finance
    def _gen_finance(self, pid: str, rng: random.Random) -> Problem:
        principal = rng.choice([300, 700, 1100])
        rate_num = rng.choice([1, 2])  # +10% or +20%
        withdrawal = rng.choice([10, 20, 50])
        N = 4
        factor = f"(1 + {rate_num}/10)"
        specs = [
            BindingSpec("rate_mult", factor, []),
            BindingSpec("wd", str(withdrawal), []),
            BindingSpec("b0", str(principal), []),
        ]
        for k in range(1, N + 1):
            specs.append(BindingSpec(f"b{k}", f"b{k - 1}*rate_mult - wd", [f"b{k - 1}", "rate_mult", "wd"]))
        text = (
            f"A balance starts at {principal}; each step it grows by {rate_num}0% then {withdrawal} "
            f"is withdrawn. Find the balance after {N} steps."
        )
        return _finalize(
            pid, specs, f"b{N}", kind="finance",
            params={"principal": principal, "rate_num": rate_num, "withdrawal": withdrawal, "N": N},
            problem_text=text,
        )
