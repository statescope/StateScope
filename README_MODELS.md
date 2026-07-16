# DriftMath model backends

DriftMath separates *what model* from *how it's reached*. Every model is a YAML
config under `configs/models/` of the form `{type, params, [vllm], [catalog]}`;
`get_model(path)` instantiates the backend named by `type`. **All SDK imports are
lazy and all configs instantiate offline** — credentials are only needed when a
backend is actually called (a missing key raises `MissingCredentialError`, even if
the SDK isn't installed).

Install extras as needed (none are base dependencies):

```bash
pip install "driftmath[open]"    # vllm, transformers, torch, accelerate
pip install "driftmath[closed]"  # openai, anthropic, google-genai
pip install "driftmath[data]"    # datasets, huggingface_hub
```

## Backend types

| `type` | class | env var | use |
|--------|-------|---------|-----|
| `openai_compat` | `OpenAICompatModel` | configurable (`api_key_env`) | any OpenAI-compatible server (local vLLM, OpenRouter, …) |
| `openai` | `OpenAIModel` | `OPENAI_API_KEY` | OpenAI API |
| `openrouter` | `OpenRouterModel` | `OPENROUTER_API_KEY` | OpenRouter (any hosted model) |
| `anthropic` | `AnthropicModel` | `ANTHROPIC_API_KEY` | Claude |
| `google` | `GoogleModel` | `GOOGLE_API_KEY` | Gemini |
| `hf` | `HFModel` | — | small local models via transformers |
| `mock` | `MockModel` | — | deterministic offline replay (tests/demos) |

The open `configs/models/open_*.yaml` use `openai_compat` pointed at a local vLLM
server (`base_url: http://localhost:8000/v1`, `api_key_fallback: EMPTY` so no key is
needed locally) plus a `vllm:` block describing how to serve the model.

## Local open models with vLLM (A100 40GB)

Print the serve command for any open config, then run it (needs `[open]` + a GPU):

```bash
python -m driftmath.models.vllm_server --config configs/models/open_qwen3_14b.yaml --print-command
# -> vllm serve Qwen/Qwen3-14B --tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
#    --max-model-len 8192 --dtype bfloat16 --host 0.0.0.0 --port 8000 --trust-remote-code \
#    --served-model-name Qwen/Qwen3-14B
```

GPU planning rules (default target: a single **A100 40GB**), implemented in
`driftmath/models/vllm_server.py`:

- **1B–4B**: single GPU, `max_model_len` ~32768.
- **7B–8B**: single A100 40GB, `max_model_len` ~16384.
- **>8B–14B**: prefer `tensor_parallel_size=1` on one A100 40GB if it fits (it does for
  14B in bf16, ~28GB weights), and use `tensor_parallel_size≥2` when `gpu_count>1`.
- For inference, multi-GPU = **tensor parallelism** (`--tensor-parallel-size`), *not* DDP.
  Running many models/prompts as separate processes would be *data-parallel scheduling*.

Each config's `vllm:` block exposes `tensor_parallel_size`, `gpu_memory_utilization`,
`max_model_len`, `dtype`, optional `quantization`, and optional `trust_remote_code` —
edit freely. `estimate_gpu_plan(size_b, gpu_count, gpu_vram_gb)` returns a suggested plan.

> Model context lengths above are conservative defaults; some checkpoints (e.g.
> Qwen2.5-Math, Phi-3-*-4k) cap at 4096. Edit `max_model_len` to your checkpoint.

## OpenRouter

`configs/models/openrouter_*.yaml` use `type: openrouter`, base URL
`https://openrouter.ai/api/v1`, and `OPENROUTER_API_KEY`. Model strings are
config-driven (`openai/gpt-4o`, `anthropic/claude-3.5-sonnet`, `google/gemini-pro`,
`qwen/qwen-2.5-72b-instruct`, `deepseek/deepseek-r1`). Optional `http_referer` /
`x_title` become the `HTTP-Referer` / `X-Title` headers.

```bash
export OPENROUTER_API_KEY=sk-or-...
python scripts/run_eval.py --experiment configs/experiments/gonogo_mock.yaml \
  --model-role large=configs/models/openrouter_qwen.yaml \
  --model-role small=configs/models/openrouter_deepseek.yaml
```

## Closed APIs

```bash
export OPENAI_API_KEY=...      # configs/models/closed_openai.yaml      (gpt-4o)
export ANTHROPIC_API_KEY=...   # configs/models/closed_anthropic.yaml  (claude-3.5-sonnet)
export GOOGLE_API_KEY=...      # configs/models/closed_google.yaml      (gemini-1.5-pro)
```

## Model catalog

`driftmath/models/model_catalog.py` derives a catalog (name, provider, backend,
model id, size, category, recommended GPU, default context, tool support, config
path) from the committed configs. `model_catalog.open_models()` lists the local
open models (1B–14B). Models above 14B are intentionally not in the default open
configs — add your own config to use them.

## Selecting models for an experiment

Experiment YAMLs may give `models:` as a `{label: spec}` map; the **label** is the
recorded `model` (so go/no-go roles resolve), and the **spec** (config path) is
recorded as `model_spec`. The CLI overrides the config:

```bash
# config-defined open models (large=Qwen3-14B, small=Qwen3-4B):
python scripts/run_eval.py --experiment configs/experiments/gonogo_open.yaml

# override roles on any experiment:
python scripts/run_eval.py --experiment configs/experiments/gonogo_mock.yaml \
  --model-role large=configs/models/open_qwen3_14b.yaml \
  --model-role small=configs/models/open_qwen3_4b.yaml

# ad-hoc list (labels default to the config file stem):
python scripts/run_eval.py --config configs/experiments/smoke.yaml \
  --models configs/models/open_qwen3_4b.yaml,configs/models/open_qwen3_14b.yaml
```

Then build the report (go/no-go reads `model` roles `large`/`small`):

```bash
python scripts/make_report.py --input results/gonogo_open/metrics.jsonl
```

## Systems note

System C/D currently consume the structured op/state protocol emitted by
`MockModel`. Driving them with a *real* model additionally needs a prompting +
parsing adapter (prompt the model to emit ops/state, parse its text back into the
protocol). The backends here implement `generate()` returning `ModelResponse`; the
adapter is the next integration step.
```
