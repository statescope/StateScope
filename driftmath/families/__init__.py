"""Problem families and their registry.

Importing this package registers the built-in families (so ``registry.get`` and
the CLI can resolve them by name).
"""

from driftmath.families import (  # noqa: F401  (registration side effect)
    family_a,
    family_b,
    family_c,
    family_d,
)
