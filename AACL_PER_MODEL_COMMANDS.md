# AACL Per-Model Run Commands

Run one model at a time. Use one terminal for `vllm serve ...`, and a second terminal for `run_aacl_batch.py` and `make_aacl_report.py`.

Start from the repo on the MI300X server:

```bash
cd ~/DriftMath
git pull origin main
```

By default, each single-model batch run writes to its own directory under `results/`.

## qwen25_1_5b

```bash
python scripts/download_open_models.py --model qwen25_1_5b

python -m driftmath.models.vllm_server \
  --config configs/models/open_qwen25_1_5b.yaml \
  --print-command

vllm serve models/Qwen__Qwen2.5-1.5B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name Qwen/Qwen2.5-1.5B-Instruct

python scripts/run_aacl_batch.py --model qwen25_1_5b

python scripts/make_aacl_report.py \
  --input results/qwen2.5-1.5b-instruct/metrics.jsonl \
  --out-dir results/qwen2.5-1.5b-instruct
```

## qwen3_4b

```bash
python scripts/download_open_models.py --model qwen3_4b

python -m driftmath.models.vllm_server \
  --config configs/models/open_qwen3_4b.yaml \
  --print-command

vllm serve models/Qwen__Qwen3-4B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name Qwen/Qwen3-4B

python scripts/run_aacl_batch.py --model qwen3_4b

python scripts/make_aacl_report.py \
  --input results/qwen3-4b/metrics.jsonl \
  --out-dir results/qwen3-4b
```

## qwen3_8b

```bash
python scripts/download_open_models.py --model qwen3_8b

python -m driftmath.models.vllm_server \
  --config configs/models/open_qwen3_8b.yaml \
  --print-command

vllm serve models/Qwen__Qwen3-8B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 16384 \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name Qwen/Qwen3-8B

python scripts/run_aacl_batch.py --model qwen3_8b

python scripts/make_aacl_report.py \
  --input results/qwen3-8b/metrics.jsonl \
  --out-dir results/qwen3-8b
```

## qwen25_math_7b

```bash
python scripts/download_open_models.py --model qwen25_math_7b

python -m driftmath.models.vllm_server \
  --config configs/models/open_qwen25_math_7b.yaml \
  --print-command

vllm serve models/Qwen__Qwen2.5-Math-7B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name Qwen/Qwen2.5-Math-7B-Instruct

python scripts/run_aacl_batch.py --model qwen25_math_7b

python scripts/make_aacl_report.py \
  --input results/qwen2.5-math-7b-instruct/metrics.jsonl \
  --out-dir results/qwen2.5-math-7b-instruct
```

## r1_distill_qwen_14b

```bash
python scripts/download_open_models.py --model r1_distill_qwen_14b

python -m driftmath.models.vllm_server \
  --config configs/models/open_deepseek_r1_distill_qwen_14b.yaml \
  --print-command

vllm serve models/deepseek-ai__DeepSeek-R1-Distill-Qwen-14B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

python scripts/run_aacl_batch.py --model r1_distill_qwen_14b

python scripts/make_aacl_report.py \
  --input results/deepseek-r1-distill-qwen-14b/metrics.jsonl \
  --out-dir results/deepseek-r1-distill-qwen-14b
```

## qwen3_14b

```bash
python scripts/download_open_models.py --model qwen3_14b

python -m driftmath.models.vllm_server \
  --config configs/models/open_qwen3_14b.yaml \
  --print-command

vllm serve models/Qwen__Qwen3-14B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name Qwen/Qwen3-14B

python scripts/run_aacl_batch.py --model qwen3_14b

python scripts/make_aacl_report.py \
  --input results/qwen3-14b/metrics.jsonl \
  --out-dir results/qwen3-14b
```

## qwen3_30b_a3b

```bash
python scripts/download_open_models.py --model qwen3_30b_a3b

python -m driftmath.models.vllm_server \
  --config configs/models/open_qwen3_30b_a3b_instruct_2507.yaml \
  --print-command

vllm serve models/Qwen__Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507

python scripts/run_aacl_batch.py --model qwen3_30b_a3b

python scripts/make_aacl_report.py \
  --input results/qwen3-30b-a3b-instruct-2507/metrics.jsonl \
  --out-dir results/qwen3-30b-a3b-instruct-2507
```

## Optional: qwen3_30b_a3b_thinking

```bash
python scripts/download_open_models.py --model qwen3_30b_a3b_thinking

python -m driftmath.models.vllm_server \
  --config configs/models/open_qwen3_30b_a3b_thinking_2507.yaml \
  --print-command

vllm serve models/Qwen__Qwen3-30B-A3B-Thinking-2507 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name Qwen/Qwen3-30B-A3B-Thinking-2507

python scripts/run_aacl_batch.py --model qwen3_30b_a3b_thinking

python scripts/make_aacl_report.py \
  --input results/qwen3-30b-a3b-thinking-2507/metrics.jsonl \
  --out-dir results/qwen3-30b-a3b-thinking-2507
```

## Resume And Progress

If a run stops midway, rerun the same batch command with resume and a forced progress bar:

```bash
python scripts/run_aacl_batch.py --model qwen3_4b --resume --progress on
```

Stop the current `vllm serve` process before moving to the next model.
