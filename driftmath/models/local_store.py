"""Repo-local model store: ``models/<org>__<name>`` directories.

Every locally downloaded Hugging Face model lives under the repo's ``models/``
directory with a stable name derived from its HF id (``Qwen/Qwen3-4B`` ->
``models/Qwen__Qwen3-4B``). Anything that needs a model path -- the download
script, the vLLM serve-command builder, the demo server -- resolves through
here, so "downloaded locally" means exactly one thing everywhere.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


def local_dir_name(hf_id: str) -> str:
    """Stable directory name for an HF id: ``Qwen/Qwen3-4B`` -> ``Qwen__Qwen3-4B``."""
    return hf_id.strip().replace("/", "__")


def local_path(hf_id: str, models_root: Path | str = MODELS_DIR) -> Path:
    """Where the model *would* live locally (whether or not it is downloaded)."""
    return Path(models_root) / local_dir_name(hf_id)


def is_downloaded(hf_id: str, models_root: Path | str = MODELS_DIR) -> bool:
    """True iff the local directory exists and holds the model config."""
    p = local_path(hf_id, models_root)
    return p.is_dir() and (p / "config.json").exists()


def resolve_model_source(hf_id: str, models_root: Path | str = MODELS_DIR) -> str:
    """The local path when downloaded, else the HF id (vLLM will pull from the hub)."""
    if is_downloaded(hf_id, models_root):
        return str(local_path(hf_id, models_root))
    return hf_id
