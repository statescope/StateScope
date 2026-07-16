"""Model abstraction and backends.

Importing this package registers all backends (their SDK imports are lazy, so this
is cheap and has no side effects beyond registration). The model catalog is *not*
imported here, to avoid reading config files on every package import.
"""

from driftmath.models.environment import refresh_model_environment

# Load repository-local credentials for every CLI/UI model entry point.
refresh_model_environment()

from driftmath.models import (  # noqa: F401  (registration side effects)
    anthropic_model,
    google_model,
    hf_model,
    mock_model,
    openai_compat,
    openai_model,
    openrouter_model,
)
