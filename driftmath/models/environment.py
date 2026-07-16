"""Safe repository-local environment loading for model backends."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def refresh_model_environment() -> None:
    """Load non-empty .env values without replacing a usable process variable.

    This is deliberately repeatable: a user may populate an initially blank .env
    after StateScope starts. A subsequent catalog refresh or run then sees the new
    value, while an explicitly exported environment variable keeps precedence.
    """
    if not ENV_FILE.is_file():
        return
    for name, value in dotenv_values(ENV_FILE).items():
        if value and not os.environ.get(name):
            os.environ[name] = value
