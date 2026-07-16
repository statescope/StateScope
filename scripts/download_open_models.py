"""Download AACL open models from Hugging Face into the repo-local ``models/`` store.

Each model lands in a stable directory named after its HF id
(``Qwen/Qwen3-4B`` -> ``models/Qwen__Qwen3-4B``), which is exactly where
``python -m driftmath.models.vllm_server --print-command`` looks for it.

Examples
--------
    python scripts/download_open_models.py --list
    python scripts/download_open_models.py --model qwen3_4b
    python scripts/download_open_models.py --model qwen3_4b,qwen3_14b
    python scripts/download_open_models.py --all-aacl
    python scripts/download_open_models.py --all-aacl --dry-run

Auth: no tokens are handled here. If a model is gated, log in first with
``huggingface-cli login`` (or set ``HF_TOKEN``); ``huggingface_hub`` picks
credentials up from the environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from driftmath.models import aacl_models
from driftmath.models.local_store import MODELS_DIR, is_downloaded, local_path


def plan_downloads(keys: list[str], models_root: Path) -> list[dict]:
    """Resolve keys to (hf_id, target dir, already-downloaded) without touching the network."""
    plans = []
    for key in keys:
        hf = aacl_models.hf_id(key)
        plans.append(
            {
                "key": key,
                "hf_id": hf,
                "target": local_path(hf, models_root),
                "downloaded": is_downloaded(hf, models_root),
            }
        )
    return plans


def download_one(hf_id: str, target: Path) -> Path:
    from huggingface_hub import snapshot_download  # lazy: only needed for real downloads

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hf_id, local_dir=str(target))
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download AACL open models into models/.")
    ap.add_argument("--model", default=None, help="AACL model key(s), comma-separated (see --list)")
    ap.add_argument("--all-aacl", action="store_true", help="download the full required AACL set")
    ap.add_argument("--include-optional", action="store_true", help="with --all-aacl, also the optional thinking variant")
    ap.add_argument("--list", action="store_true", help="list model keys and their local status")
    ap.add_argument("--dry-run", action="store_true", help="print what would be downloaded, do nothing")
    ap.add_argument("--models-dir", default=str(MODELS_DIR), help="local model store (default: repo models/)")
    ap.add_argument("--force", action="store_true", help="re-download even if already present")
    args = ap.parse_args(argv)

    models_root = Path(args.models_dir)

    if args.list:
        for entry in aacl_models.catalog(include_optional=True, models_root=models_root):
            status = "downloaded" if entry["downloaded"] else "not downloaded"
            opt = " (optional)" if entry["optional"] else ""
            print(f"{entry['key']:24s} {entry['hf_id']:44s} {status}{opt}")
        return 0

    if args.model:
        keys = [k.strip() for k in args.model.split(",") if k.strip()]
    elif args.all_aacl:
        keys = aacl_models.all_keys(include_optional=args.include_optional)
    else:
        ap.error("provide --model <key>[,<key>...], --all-aacl, or --list")

    unknown = [k for k in keys if k not in aacl_models.all_keys(include_optional=True)]
    if unknown:
        raise SystemExit(f"unknown model key(s): {unknown}; known: {aacl_models.all_keys()}")

    plans = plan_downloads(keys, models_root)
    for plan in plans:
        tag = "skip (already downloaded)" if plan["downloaded"] and not args.force else "download"
        print(f"[{tag}] {plan['key']}: {plan['hf_id']} -> {plan['target']}")
        if args.dry_run or (plan["downloaded"] and not args.force):
            continue
        download_one(plan["hf_id"], plan["target"])
        print(f"  done -> {plan['target']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
