#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROFILE_CONFIG="${PROFILE_CONFIG:-configs/vpmoe.toml}"
PROFILE_TOKENIZER="${PROFILE_TOKENIZER:-data/tokenizer/o200k_base.tiktoken}"

PROFILE_NPROC_PER_NODE="${PROFILE_NPROC_PER_NODE:-1}"
PROFILE_TP="${PROFILE_TP:-1}"
PROFILE_PP="${PROFILE_PP:-1}"
PROFILE_EP="${PROFILE_EP:-1}"

PROFILE_SEQ_LEN="${PROFILE_SEQ_LEN:-2048}"
PROFILE_MICRO_BATCH_SIZE="${PROFILE_MICRO_BATCH_SIZE:-1}"
PROFILE_GLOBAL_BATCH_SIZE="${PROFILE_GLOBAL_BATCH_SIZE:-auto}"

PROFILE_TRAIN_ITERS="${PROFILE_TRAIN_ITERS:-30}"
PROFILE_PROFILE_STEP_START="${PROFILE_PROFILE_STEP_START:-10}"
PROFILE_PROFILE_STEP_END="${PROFILE_PROFILE_STEP_END:-12}"
PROFILE_PROFILE_RANKS="${PROFILE_PROFILE_RANKS:-0}"

PROFILE_TOOL="${PROFILE_TOOL:-nsys}" # nsys|torch
PROFILE_OUT_DIR="${PROFILE_OUT_DIR:-data/profiles}"
PROFILE_RUN_NAME="${PROFILE_RUN_NAME:-vpmoe_sl${PROFILE_SEQ_LEN}_mb${PROFILE_MICRO_BATCH_SIZE}_tp${PROFILE_TP}_ep${PROFILE_EP}_pp${PROFILE_PP}_n${PROFILE_NPROC_PER_NODE}}"
PROFILE_TRANSFORMER_IMPL="${PROFILE_TRANSFORMER_IMPL:-transformer_engine}" # local|transformer_engine|inference_optimized

# Clamp profile window to a valid range for torch/NSYS profiling.
if (( PROFILE_PROFILE_STEP_END <= PROFILE_PROFILE_STEP_START )); then
  PROFILE_PROFILE_STEP_END=$((PROFILE_PROFILE_STEP_START + 1))
fi
if (( PROFILE_TRAIN_ITERS > 0 )) && (( PROFILE_PROFILE_STEP_END > PROFILE_TRAIN_ITERS )); then
  PROFILE_PROFILE_STEP_END="$PROFILE_TRAIN_ITERS"
fi
if (( PROFILE_PROFILE_STEP_START >= PROFILE_PROFILE_STEP_END )); then
  PROFILE_PROFILE_STEP_START=$((PROFILE_PROFILE_STEP_END - 1))
  if (( PROFILE_PROFILE_STEP_START < 0 )); then
    PROFILE_PROFILE_STEP_START=0
    PROFILE_PROFILE_STEP_END=1
  fi
fi

EXTRA_ENV=()
for var in CUDA_LAUNCH_BLOCKING TORCH_SHOW_CPP_STACKTRACES TORCH_CPP_LOG_LEVEL NCCL_DEBUG NVTE_DEBUG NVTE_LOG_LEVEL; do
  if [[ -n "${!var-}" ]]; then
    EXTRA_ENV+=(-e "$var")
  fi
done

docker compose -f "$ROOT_DIR/docker/compose.yml" run --rm \
  -e "PROFILE_CONFIG=$PROFILE_CONFIG" \
  -e "PROFILE_TOKENIZER=$PROFILE_TOKENIZER" \
  -e "PROFILE_NPROC_PER_NODE=$PROFILE_NPROC_PER_NODE" \
  -e "PROFILE_TP=$PROFILE_TP" \
  -e "PROFILE_PP=$PROFILE_PP" \
  -e "PROFILE_EP=$PROFILE_EP" \
  -e "PROFILE_SEQ_LEN=$PROFILE_SEQ_LEN" \
  -e "PROFILE_MICRO_BATCH_SIZE=$PROFILE_MICRO_BATCH_SIZE" \
  -e "PROFILE_GLOBAL_BATCH_SIZE=$PROFILE_GLOBAL_BATCH_SIZE" \
  -e "PROFILE_TRAIN_ITERS=$PROFILE_TRAIN_ITERS" \
  -e "PROFILE_PROFILE_STEP_START=$PROFILE_PROFILE_STEP_START" \
  -e "PROFILE_PROFILE_STEP_END=$PROFILE_PROFILE_STEP_END" \
  -e "PROFILE_PROFILE_RANKS=$PROFILE_PROFILE_RANKS" \
  -e "PROFILE_TOOL=$PROFILE_TOOL" \
  -e "PROFILE_OUT_DIR=$PROFILE_OUT_DIR" \
  -e "PROFILE_RUN_NAME=$PROFILE_RUN_NAME" \
  -e "PROFILE_TRANSFORMER_IMPL=$PROFILE_TRANSFORMER_IMPL" \
  "${EXTRA_ENV[@]}" \
  vpmoe bash -lc '
set -euo pipefail

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf "%s" "$path"
  else
    printf "/workspace/vpmoe/%s" "$path"
  fi
}

CONFIG="$(resolve_path "$PROFILE_CONFIG")"
TOKENIZER_MODEL="$(resolve_path "$PROFILE_TOKENIZER")"
OUT_DIR="$(resolve_path "$PROFILE_OUT_DIR")"

cd /workspace/vpmoe

if [[ ! -f "$CONFIG" ]]; then
  echo "[error] missing PROFILE_CONFIG: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$TOKENIZER_MODEL" ]]; then
  echo "[error] missing PROFILE_TOKENIZER: $TOKENIZER_MODEL" >&2
  exit 1
fi

if [[ "$PROFILE_PP" != "1" ]]; then
  echo "[error] vpMoE profiling currently requires PROFILE_PP=1 (value_residual constraint)." >&2
  exit 1
fi

world="$PROFILE_NPROC_PER_NODE"
denom=$((PROFILE_TP * PROFILE_PP * PROFILE_EP))
if (( world % denom != 0 )); then
  echo "[error] nproc_per_node=$world must be divisible by TP*PP*EP=$denom" >&2
  exit 1
fi
dp=$((world / denom))

global_bsz="$PROFILE_GLOBAL_BATCH_SIZE"
if [[ "$global_bsz" == "auto" ]]; then
  global_bsz=$((PROFILE_MICRO_BATCH_SIZE * dp))
fi

export O200K_HARMONY_VOCAB_PATH="$TOKENIZER_MODEL"

mapfile -t MODEL_ARGS < <(
  uv run python scripts/profiling/render_vpmoe_megatron_args.py \
    --config "$CONFIG" \
    --seq-length "$PROFILE_SEQ_LEN"
)

extra_moe_args=()
if (( PROFILE_EP > 1 )); then
  extra_moe_args+=(--moe-token-dispatcher-type alltoall)
fi

COMMON_ARGS=(
  --mock-data
  --tokenizer-type O200kHarmonyTokenizer
  --tokenizer-model "$TOKENIZER_MODEL"
  --tensor-model-parallel-size "$PROFILE_TP"
  --pipeline-model-parallel-size "$PROFILE_PP"
  --expert-model-parallel-size "$PROFILE_EP"
  "${extra_moe_args[@]}"
  --micro-batch-size "$PROFILE_MICRO_BATCH_SIZE"
  --global-batch-size "$global_bsz"
  --train-iters "$PROFILE_TRAIN_ITERS"
  --lr 1e-4 --min-lr 1e-4 --lr-decay-style constant
  --log-interval 1
  --log-throughput
  --timing-log-level 1
  --eval-interval 1000000000 --eval-iters 0
  --save-interval 0
  --bf16
  --transformer-impl "$PROFILE_TRANSFORMER_IMPL"
)

if [[ "$PROFILE_TRANSFORMER_IMPL" == "local" ]]; then
  COMMON_ARGS+=(--no-persist-layer-norm)
fi

PROFILE_ARGS=(--profile --profile-step-start "$PROFILE_PROFILE_STEP_START" --profile-step-end "$PROFILE_PROFILE_STEP_END" --profile-ranks "$PROFILE_PROFILE_RANKS")

TRAIN_CMD=(uv run python -m torch.distributed.run --nproc_per_node="$PROFILE_NPROC_PER_NODE" vpmoe/Megatron-vpmoe/pretrain_gpt.py)
TRAIN_CMD+=("${MODEL_ARGS[@]}" "${COMMON_ARGS[@]}" "${PROFILE_ARGS[@]}")

mkdir -p "$OUT_DIR"

case "$PROFILE_TOOL" in
  nsys)
    if ! command -v nsys >/dev/null 2>&1; then
      echo "[error] nsys not found in container. Set PROFILE_TOOL=torch or install nsys in the image." >&2
      exit 1
    fi
    out_prefix="$OUT_DIR/$PROFILE_RUN_NAME"
    echo "[profile] writing: ${out_prefix}.qdrep" >&2
    exec nsys profile -s none -t nvtx,cuda \
      --cudabacktrace=all --cuda-graph-trace=node --python-backtrace=cuda --wait all \
      -o "$out_prefix" --force-overwrite true \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      "${TRAIN_CMD[@]}"
    ;;
  torch)
    tb_dir="$OUT_DIR/$PROFILE_RUN_NAME.tensorboard"
    mkdir -p "$tb_dir"
    TRAIN_CMD+=("--use-pytorch-profiler" "--tensorboard-dir" "$tb_dir")
    echo "[profile] writing torch traces under: $tb_dir" >&2
    exec "${TRAIN_CMD[@]}"
    ;;
  *)
    echo "[error] PROFILE_TOOL must be one of: nsys, torch (got: $PROFILE_TOOL)" >&2
    exit 1
    ;;
esac
'
