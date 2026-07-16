"""vLLM serving helpers: GPU planning and command construction (no execution).

For **inference** vLLM shards a single model across GPUs with *tensor parallelism*
(`--tensor-parallel-size`); that is not DDP. Running many models/prompts as separate
processes would be *data-parallel scheduling* and is orthogonal to this module.

The conservative planning functions assume a single A100 40GB; on larger cards
(e.g. an MI300X 192GB) every AACL model fits on one GPU at full context, so the
per-config ``vllm:`` block is authoritative and the planner is only a fallback.
Functions here only compute plans and build the ``vllm serve ...`` argument list;
nothing is launched. When a model has been downloaded into the repo-local
``models/`` directory, the built command serves from that path (with
``--served-model-name`` kept at the canonical HF id so client configs never change).

On ROCm (MI300X) use a ROCm build of vLLM; the printed command is identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from driftmath.models.local_store import MODELS_DIR, resolve_model_source

A100_40GB = 40
MI300X_192GB = 192
_BYTES_PER_PARAM_GB = 2.0  # fp16 / bf16: ~2 GB per billion params
DEFAULT_GPU_MEMORY_UTILIZATION = 0.50


def choose_tensor_parallel_size(model_size_b: float, gpu_count: int) -> int:
    """Tensor-parallel degree. Single GPU -> 1; >8B prefers TP across GPUs when available."""
    if gpu_count <= 1:
        return 1
    if model_size_b <= 8:
        return 1
    if model_size_b <= 14:
        return 2 if gpu_count >= 2 else 1
    if model_size_b <= 32:
        return min(gpu_count, 4) if gpu_count >= 4 else 2
    return min(gpu_count, 8)


def choose_max_model_len(model_size_b: float, gpu_vram_gb: int = A100_40GB, gpu_count: int = 1) -> int:
    """Context length: smaller models get more room; larger ones less on a 40GB card."""
    if model_size_b <= 4:
        base = 32768
    elif model_size_b <= 8:
        base = 16384
    else:
        base = 8192
    # Sharding across GPUs frees KV-cache memory, so a big model can take more context.
    if gpu_count > 1 and model_size_b > 8:
        base = 16384
    return base


def estimate_gpu_plan(model_size_b: float, gpu_count: int = 1, gpu_vram_gb: int = A100_40GB) -> dict:
    """Return a serving plan: TP size, max_model_len, utilization, and a fit estimate."""
    tp = choose_tensor_parallel_size(model_size_b, gpu_count)
    max_model_len = choose_max_model_len(model_size_b, gpu_vram_gb, gpu_count)
    gpu_memory_utilization = DEFAULT_GPU_MEMORY_UTILIZATION
    weights_gb = model_size_b * _BYTES_PER_PARAM_GB
    weights_per_gpu = weights_gb / tp
    fits = weights_per_gpu < gpu_vram_gb * gpu_memory_utilization * 0.95
    return {
        "model_size_b": model_size_b,
        "gpu_count": gpu_count,
        "gpu_vram_gb": gpu_vram_gb,
        "tensor_parallel_size": tp,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "weights_gb": round(weights_gb, 1),
        "weights_per_gpu_gb": round(weights_per_gpu, 1),
        "fits": bool(fits),
        "notes": f"tp={tp}; ~{weights_per_gpu:.0f}GB weights/GPU on {gpu_vram_gb}GB"
        + ("" if fits else "; consider more GPUs or quantization"),
    }


def build_vllm_command(
    config: dict,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    models_root: Path | str = MODELS_DIR,
) -> list[str]:
    """Build the ``vllm serve`` argument list from a model config's ``vllm`` block.

    Serves from the repo-local ``models/`` copy when one exists (falling back to
    the HF id otherwise); ``--served-model-name`` stays the canonical id either way.
    """
    v = config.get("vllm", config)
    params = config.get("params", {})
    model_id = v.get("model_id") or params.get("model") or config.get("model")
    served_name = params.get("model") or model_id
    source = resolve_model_source(str(model_id), models_root)

    cmd = [
        "vllm", "serve", str(source),
        "--tensor-parallel-size", str(v.get("tensor_parallel_size", 1)),
        "--gpu-memory-utilization", str(v.get("gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION)),
        "--max-model-len", str(v.get("max_model_len", 8192)),
        "--dtype", str(v.get("dtype", "bfloat16")),
        "--host", host,
        "--port", str(port),
    ]
    if v.get("quantization"):
        cmd += ["--quantization", str(v["quantization"])]
    if v.get("trust_remote_code"):
        cmd += ["--trust-remote-code"]
    cmd += ["--served-model-name", str(served_name)]
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print the vLLM serve command for a model config.")
    ap.add_argument("--config", required=True, help="path to a model YAML config")
    ap.add_argument("--print-command", action="store_true", help="print the command (default action)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--models-dir", default=str(MODELS_DIR), help="local model store (default: repo models/)")
    args = ap.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cmd = build_vllm_command(config, host=args.host, port=args.port, models_root=args.models_dir)
    print(" ".join(cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
