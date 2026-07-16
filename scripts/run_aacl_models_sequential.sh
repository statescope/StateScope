#!/usr/bin/env bash
set -euo pipefail

# Sequential AACL runner for one-GPU servers.
#
# Default behavior:
#   1. download one model into models/
#   2. serve it with vLLM on the local OpenAI-compatible endpoint
#   3. run the AACL batch into results/<model-slug>/
#   4. generate the AACL report in that same result directory
#   5. stop vLLM and delete only that downloaded model directory
#
# Usage:
#   bash scripts/run_aacl_models_sequential.sh
#   bash scripts/run_aacl_models_sequential.sh qwen3_4b
#   MODEL_KEYS="qwen3_4b qwen3_8b" bash scripts/run_aacl_models_sequential.sh
#   LIMIT=5 bash scripts/run_aacl_models_sequential.sh qwen3_4b

PYTHON="${PYTHON:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}/v1}"
MODEL_KEYS="${MODEL_KEYS:-qwen25_1_5b qwen3_4b qwen3_8b qwen25_math_7b r1_distill_qwen_14b qwen3_14b qwen3_30b_a3b}"
VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S:-1800}"
VLLM_POLL_S="${VLLM_POLL_S:-10}"
DELETE_AFTER_RUN="${DELETE_AFTER_RUN:-1}"

VLLM_PID=""

cleanup_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "stopping vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  VLLM_PID=""
}

trap cleanup_vllm EXIT

py_value() {
  local key="$1"
  local expr="$2"
  "${PYTHON}" - "$key" "$expr" <<'PY'
import sys
from driftmath.models import aacl_models
from driftmath.models.local_store import MODELS_DIR, local_path

key, expr = sys.argv[1], sys.argv[2]
if expr == "config":
    print(aacl_models.config_path(key))
elif expr == "result_dir":
    print(aacl_models.default_result_dir(key))
elif expr == "model_dir":
    print(local_path(aacl_models.hf_id(key)).resolve())
elif expr == "models_root":
    print(MODELS_DIR.resolve())
else:
    raise SystemExit(f"unknown expr: {expr}")
PY
}

wait_for_vllm() {
  local log_file="$1"
  local elapsed=0
  while (( elapsed <= VLLM_STARTUP_TIMEOUT_S )); do
    if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
      echo "vLLM ready at ${BASE_URL}"
      return 0
    fi
    if [[ -n "${VLLM_PID}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "vLLM exited before becoming ready. Last log lines:" >&2
      tail -n 120 "${log_file}" >&2 || true
      return 1
    fi
    sleep "${VLLM_POLL_S}"
    elapsed=$((elapsed + VLLM_POLL_S))
    echo "waiting for vLLM... ${elapsed}s"
  done
  echo "timed out waiting for vLLM. Last log lines:" >&2
  tail -n 120 "${log_file}" >&2 || true
  return 1
}

delete_model_dir() {
  local model_dir="$1"
  local models_root="$2"
  if [[ "${DELETE_AFTER_RUN}" != "1" ]]; then
    echo "keeping model directory because DELETE_AFTER_RUN=${DELETE_AFTER_RUN}: ${model_dir}"
    return 0
  fi
  case "${model_dir}" in
    "${models_root}"/*)
      echo "deleting downloaded model directory: ${model_dir}"
      rm -rf -- "${model_dir}"
      ;;
    *)
      echo "refusing to delete path outside models root: ${model_dir}" >&2
      return 1
      ;;
  esac
}

if (( "$#" > 0 )); then
  MODELS=("$@")
else
  read -r -a MODELS <<< "${MODEL_KEYS}"
fi

for key in "${MODELS[@]}"; do
  echo "============================================================"
  echo "running model: ${key}"
  echo "============================================================"

  config="$(py_value "${key}" config)"
  result_dir="$(py_value "${key}" result_dir)"
  model_dir="$(py_value "${key}" model_dir)"
  models_root="$(py_value "${key}" models_root)"
  mkdir -p "${result_dir}"

  "${PYTHON}" scripts/download_open_models.py --model "${key}"

  serve_cmd="$("${PYTHON}" -m driftmath.models.vllm_server \
    --config "${config}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --print-command)"

  echo "vLLM command:"
  echo "${serve_cmd}"
  echo "${serve_cmd}" > "${result_dir}/vllm_command.txt"

  vllm_log="${result_dir}/vllm.log"
  bash -lc "${serve_cmd}" > "${vllm_log}" 2>&1 &
  VLLM_PID="$!"
  echo "started vLLM pid=${VLLM_PID}; log=${vllm_log}"
  wait_for_vllm "${vllm_log}"

  run_args=(--model "${key}" --out-dir "${result_dir}" --base-url "${BASE_URL}" --resume --progress on)
  if [[ -n "${LIMIT:-}" ]]; then
    run_args+=(--limit "${LIMIT}")
  fi
  if [[ -n "${SYSTEMS:-}" ]]; then
    run_args+=(--systems "${SYSTEMS}")
  fi

  "${PYTHON}" scripts/run_aacl_batch.py "${run_args[@]}"
  "${PYTHON}" scripts/make_aacl_report.py --input "${result_dir}/metrics.jsonl" --out-dir "${result_dir}"

  cleanup_vllm
  delete_model_dir "${model_dir}" "${models_root}"
done

echo "all requested models complete"
