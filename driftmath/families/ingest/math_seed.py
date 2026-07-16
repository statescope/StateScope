"""MATH *form seeding* (template reinstantiation).

We deliberately do **not** parse free-form MATH solutions into traces. MATH is used
only to (a) ground which *forms* (radical / rational / abs / log) are worth covering
and (b) carry provenance. Each seed record names a form; we then instantiate a
*fresh* Family B problem of that form with new SymPy-generated parameters, so the
generated content is not the original MATH item (low contamination risk).
"""

from __future__ import annotations

from driftmath.families.family_b import FamilyB
from driftmath.io.datasets import load_records
from driftmath.io.schema import Problem

_FORMS = {"radical", "rational", "abs", "log"}


def load(source: dict, *, seed: int = 0) -> list[Problem]:
    """Read MATH form descriptors and reinstantiate fresh Family B problems."""
    fam = FamilyB()
    out: list[Problem] = []
    for i, rec in enumerate(load_records(source)):
        form = rec.get("form")
        if form not in _FORMS:
            continue
        p = fam.generate_template(form, seed=seed, index=i)
        p.meta.update(
            {
                "source": rec.get("source", "MATH"),
                "provenance": "template_reinstantiation",
                "license": rec.get("license", "MIT"),
                "contamination_risk": "low",  # fresh parameters, not the original item
                "original_id": rec.get("original_id"),
                "math_form": form,
                "math_subject": rec.get("subject"),
                "math_level": rec.get("level"),
            }
        )
        out.append(p)
    return out
