#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CORE8_CONFIG="${CORE8_CONFIG:-configs/eval/core8.toml}"
CORE8_MODE="${CORE8_MODE:-smoke}"
CORE8_REPORT_DIR="${CORE8_REPORT_DIR:-reports/eval/core8}"
CORE8_CHECKPOINT_ID="${CORE8_CHECKPOINT_ID:-}"
CORE8_COMPARE_TO="${CORE8_COMPARE_TO:-}"
CORE8_DTYPE="${CORE8_DTYPE:-bf16}"
CORE8_MODEL_FORMAT="${CORE8_MODEL_FORMAT:-}"

CORE8_HF_PATH="${CORE8_HF_PATH:-}"
CORE8_MEGATRON_LOAD="${CORE8_MEGATRON_LOAD:-}"
CORE8_MEGATRON_CONFIG="${CORE8_MEGATRON_CONFIG:-}"
CORE8_TOKENIZER_MODEL="${CORE8_TOKENIZER_MODEL:-data/tokenizer/o200k_base.tiktoken}"
CORE8_COMPAT="${CORE8_COMPAT:-0}"
CORE8_ROPE_BASE="${CORE8_ROPE_BASE:-}"

if [[ -z "${CORE8_MODEL_FORMAT}" ]]; then
  echo "[error] CORE8_MODEL_FORMAT must be set to hf or megatron" >&2
  exit 1
fi

extra_args=()
if [[ -n "${CORE8_CHECKPOINT_ID}" ]]; then
  extra_args+=(--checkpoint-id "${CORE8_CHECKPOINT_ID}")
fi
if [[ -n "${CORE8_COMPARE_TO}" ]]; then
  extra_args+=(--compare-to "${CORE8_COMPARE_TO}")
fi

model_args=()
if [[ "${CORE8_MODEL_FORMAT}" == "hf" ]]; then
  if [[ -z "${CORE8_HF_PATH}" ]]; then
    echo "[error] CORE8_HF_PATH is required for hf mode" >&2
    exit 1
  fi
  model_args+=(--model-format hf --hf-path "${CORE8_HF_PATH}")
elif [[ "${CORE8_MODEL_FORMAT}" == "megatron" ]]; then
  if [[ -z "${CORE8_MEGATRON_LOAD}" || -z "${CORE8_MEGATRON_CONFIG}" ]]; then
    echo "[error] CORE8_MEGATRON_LOAD and CORE8_MEGATRON_CONFIG are required for megatron mode" >&2
    exit 1
  fi
  model_args+=(--model-format megatron --megatron-load "${CORE8_MEGATRON_LOAD}" --megatron-config "${CORE8_MEGATRON_CONFIG}" --tokenizer-model "${CORE8_TOKENIZER_MODEL}")
  if [[ "${CORE8_COMPAT}" == "1" ]]; then
    model_args+=(--compat)
    if [[ -n "${CORE8_ROPE_BASE}" ]]; then
      model_args+=(--rope-base "${CORE8_ROPE_BASE}")
    fi
  fi
else
  echo "[error] CORE8_MODEL_FORMAT must be hf or megatron" >&2
  exit 1
fi

docker compose -f "$ROOT_DIR/docker/compose.yml" run --rm \
  -e CORE8_CONFIG \
  -e CORE8_MODE \
  -e CORE8_REPORT_DIR \
  -e CORE8_CHECKPOINT_ID \
  -e CORE8_COMPARE_TO \
  -e CORE8_DTYPE \
  -e CORE8_MODEL_FORMAT \
  -e CORE8_HF_PATH \
  -e CORE8_MEGATRON_LOAD \
  -e CORE8_MEGATRON_CONFIG \
  -e CORE8_TOKENIZER_MODEL \
  -e CORE8_COMPAT \
  -e CORE8_ROPE_BASE \
  vpmoe python scripts/eval/run_core8.py \
    --config "${CORE8_CONFIG}" \
    --mode "${CORE8_MODE}" \
    --report-dir "${CORE8_REPORT_DIR}" \
    --dtype "${CORE8_DTYPE}" \
    "${model_args[@]}" \
    "${extra_args[@]}"
