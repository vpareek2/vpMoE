#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEQ_LEN="${SEQ_LEN:-512}"
NPROC="${NPROC:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
LR="${LR:-1e-4}"

TOKENIZER_PATH="${TOKENIZER_PATH:-${ROOT_DIR}/data/tokenizer/o200k_base.tiktoken}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/vpmoe_dev.toml}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"
read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS_STR}"

mapfile -t MODEL_ARGS < <(
  uv run python "${ROOT_DIR}/scripts/profiling/render_vpmoe_megatron_args.py" \
    --config "${CONFIG_PATH}" \
    --seq-length "${SEQ_LEN}"
)

uv run python -m torch.distributed.run --nproc_per_node="${NPROC}" \
  "${ROOT_DIR}/vpmoe/Megatron-vpmoe/pretrain_gpt.py" \
  "${MODEL_ARGS[@]}" \
  --mock-data \
  --tokenizer-type O200kHarmonyTokenizer \
  --tokenizer-model "${TOKENIZER_PATH}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --train-iters "${TRAIN_ITERS}" \
  --lr "${LR}" --min-lr "${LR}" --lr-decay-style constant \
  --log-interval 1 --log-throughput --timing-log-level 1 \
  --eval-interval 1000000000 --eval-iters 0 --save-interval 0 \
  --bf16 --no-persist-layer-norm --transformer-impl transformer_engine \
  "${EXTRA_ARGS_ARR[@]}"
