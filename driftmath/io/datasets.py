"""Dataset loading with explicit provenance, offline-first.

``load_records`` prefers a local JSONL fixture (so tests never touch the network)
and only lazily imports HuggingFace ``datasets`` when no local file is given. Every
returned record is annotated with its source name, split, and license.

Provenance modes used across the project:
``synthetic`` | ``program_lift`` | ``template_reinstantiation`` | ``raw_natural``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from driftmath.io.storage import read_jsonl

PROVENANCE_MODES = ("synthetic", "program_lift", "template_reinstantiation", "raw_natural")


def _load_hf(source: dict) -> list[dict]:
    """Lazy HF loader. Only reached when no local_jsonl is provided."""
    try:
        import datasets  # noqa: F401  (optional, network-bound dependency)
    except Exception as e:  # pragma: no cover - exercised only without local fixtures
        raise RuntimeError(
            "HuggingFace 'datasets' is not available and no 'local_jsonl' was given. "
            "Provide a local JSONL fixture or install datasets."
        ) from e
    from datasets import load_dataset

    hf_path = source["hf_path"]
    split = source.get("split", "test")
    config = source.get("config")
    ds = load_dataset(hf_path, config, split=split) if config else load_dataset(hf_path, split=split)
    return [dict(x) for x in ds]


def load_records(source: dict) -> list[dict]:
    """Load raw records for a source, annotated with provenance fields.

    ``source`` keys: ``name`` (required), ``local_jsonl`` (preferred path),
    ``hf_path`` / ``config`` / ``split`` (HF fallback), ``license``.
    """
    name = source.get("name")
    split = source.get("split")
    license_ = source.get("license")

    local = source.get("local_jsonl")
    if local and Path(local).exists():
        raw = read_jsonl(local)
    else:
        raw = _load_hf(source)

    for r in raw:
        r.setdefault("source", name)
        r.setdefault("split", split)
        r.setdefault("license", license_)
    return raw
