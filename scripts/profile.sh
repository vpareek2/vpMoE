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
PROFILE_OUT_DIR="${PROFILE_OUT_DIR:-artifacts/profiles}"
PROFILE_RUN_NAME="${PROFILE_RUN_NAME:-vpmoe_sl${PROFILE_SEQ_LEN}_mb${PROFILE_MICRO_BATCH_SIZE}_tp${PROFILE_TP}_ep${PROFILE_EP}_pp${PROFILE_PP}_n${PROFILE_NPROC_PER_NODE}}"
PROFILE_TRANSFORMER_IMPL="${PROFILE_TRANSFORMER_IMPL:-transformer_engine}" # local|transformer_engine|inference_optimized
PROFILE_TORCH_RECORD_SHAPES="${PROFILE_TORCH_RECORD_SHAPES:-}"
PROFILE_TORCH_WITH_STACK="${PROFILE_TORCH_WITH_STACK:-}"

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
  -e "PROFILE_DATE_TAG=${PROFILE_DATE_TAG:-}" \
  -e "PROFILE_TIME_TAG=${PROFILE_TIME_TAG:-}" \
  -e "PROFILE_RUN_TAG=${PROFILE_RUN_TAG:-}" \
  -e "PROFILE_HW_TAG=${PROFILE_HW_TAG:-}" \
  -e "PROFILE_TORCH_RECORD_SHAPES=${PROFILE_TORCH_RECORD_SHAPES}" \
  -e "PROFILE_TORCH_WITH_STACK=${PROFILE_TORCH_WITH_STACK}" \
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

append_bool_arg() {
  local value="$1"
  local true_flag="$2"
  local false_flag="$3"
  if [[ -z "$value" ]]; then
    return 0
  fi
  case "${value,,}" in
    1|true|yes|on)
      TRAIN_CMD+=("$true_flag")
      ;;
    0|false|no|off)
      TRAIN_CMD+=("$false_flag")
      ;;
    *)
      echo "[error] invalid boolean for $true_flag/$false_flag: $value" >&2
      exit 1
      ;;
  esac
}

if [[ "$PROFILE_TOOL" == "torch" ]]; then
  append_bool_arg "$PROFILE_TORCH_RECORD_SHAPES" --profile-record-shapes --no-profile-record-shapes
  append_bool_arg "$PROFILE_TORCH_WITH_STACK" --profile-with-stack --no-profile-with-stack
fi

sanitize_path_component() {
  local value="$1"
  local lowered
  lowered="$(printf "%s" "$value" | tr "[:upper:]" "[:lower:]")"
  printf "%s" "$lowered" | tr -cs "a-z0-9._-" "_"
}

date_tag="${PROFILE_DATE_TAG:-$(date -u +%Y-%m-%d)}"
time_tag="${PROFILE_TIME_TAG:-$(date -u +%H%M%S)}"
run_tag="${PROFILE_RUN_TAG:-$time_tag}"

hw_tag="${PROFILE_HW_TAG:-}"
gpu_name=""
gpu_cc=""
if [[ -z "$hw_tag" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || true)"
  gpu_cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 || true)"
  if [[ -n "$gpu_name" ]]; then
    name_tag="$(sanitize_path_component "$gpu_name")"
    if [[ -n "$gpu_cc" ]]; then
      cc_tag="sm${gpu_cc//./}"
      hw_tag="${name_tag}_${cc_tag}"
    else
      hw_tag="$name_tag"
    fi
  fi
fi
if [[ -z "$hw_tag" ]]; then
  hw_tag="unknown"
fi

run_id="${PROFILE_RUN_NAME}_${run_tag}"
run_dir="${OUT_DIR}/${date_tag}/${hw_tag}/${PROFILE_TOOL}/${run_id}"
if ! mkdir -p "$run_dir"; then
  echo "[error] unable to create profile output dir: $run_dir" >&2
  echo "[error] set PROFILE_OUT_DIR to a writable path or fix permissions for $OUT_DIR" >&2
  exit 1
fi

git_sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
config_sha="$(sha256sum "$CONFIG" | awk "{print \$1}")"
tokenizer_sha="$(sha256sum "$TOKENIZER_MODEL" | awk "{print \$1}")"
train_cmd_str=""
printf -v train_cmd_str "%q " "${TRAIN_CMD[@]}"

export PROFILE_META_PATH="${run_dir}/meta.json"
export PROFILE_INDEX_PATH="${OUT_DIR}/index.jsonl"
export PROFILE_RUN_DIR="$run_dir"
export PROFILE_RUN_ID="$run_id"
export PROFILE_RUN_TAG="$run_tag"
export PROFILE_TIME_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export PROFILE_GIT_SHA="$git_sha"
export PROFILE_CONFIG_PATH="$CONFIG"
export PROFILE_CONFIG_SHA256="$config_sha"
export PROFILE_TOKENIZER_PATH="$TOKENIZER_MODEL"
export PROFILE_TOKENIZER_SHA256="$tokenizer_sha"
export PROFILE_TRAIN_CMD="$train_cmd_str"
export PROFILE_HW_TAG="$hw_tag"
export PROFILE_GPU_NAME="$gpu_name"
export PROFILE_GPU_CC="$gpu_cc"
export PROFILE_DP="$dp"
export PROFILE_GLOBAL_BSZ="$global_bsz"

python - <<PY
import json
import os

def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

meta = {
    "created_utc": os.environ.get("PROFILE_TIME_ISO"),
    "run_dir": os.environ.get("PROFILE_RUN_DIR"),
    "run_id": os.environ.get("PROFILE_RUN_ID"),
    "run_name": os.environ.get("PROFILE_RUN_NAME"),
    "run_tag": os.environ.get("PROFILE_RUN_TAG"),
    "profile_tool": os.environ.get("PROFILE_TOOL"),
    "profile_step_start": _as_int(os.environ.get("PROFILE_PROFILE_STEP_START")),
    "profile_step_end": _as_int(os.environ.get("PROFILE_PROFILE_STEP_END")),
    "git_sha": os.environ.get("PROFILE_GIT_SHA"),
    "config_path": os.environ.get("PROFILE_CONFIG_PATH"),
    "config_sha256": os.environ.get("PROFILE_CONFIG_SHA256"),
    "tokenizer_path": os.environ.get("PROFILE_TOKENIZER_PATH"),
    "tokenizer_sha256": os.environ.get("PROFILE_TOKENIZER_SHA256"),
    "transformer_impl": os.environ.get("PROFILE_TRANSFORMER_IMPL"),
    "seq_len": _as_int(os.environ.get("PROFILE_SEQ_LEN")),
    "micro_batch_size": _as_int(os.environ.get("PROFILE_MICRO_BATCH_SIZE")),
    "global_batch_size": _as_int(os.environ.get("PROFILE_GLOBAL_BSZ")),
    "nproc_per_node": _as_int(os.environ.get("PROFILE_NPROC_PER_NODE")),
    "tp": _as_int(os.environ.get("PROFILE_TP")),
    "pp": _as_int(os.environ.get("PROFILE_PP")),
    "ep": _as_int(os.environ.get("PROFILE_EP")),
    "dp": _as_int(os.environ.get("PROFILE_DP")),
    "hw_tag": os.environ.get("PROFILE_HW_TAG"),
    "gpu_name": os.environ.get("PROFILE_GPU_NAME"),
    "gpu_cc": os.environ.get("PROFILE_GPU_CC"),
    "train_cmd": os.environ.get("PROFILE_TRAIN_CMD", "").strip(),
}

with open(os.environ["PROFILE_META_PATH"], "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, sort_keys=True)

index = {
    "created_utc": meta["created_utc"],
    "run_id": meta["run_id"],
    "run_dir": meta["run_dir"],
    "profile_tool": meta["profile_tool"],
    "seq_len": meta["seq_len"],
    "micro_batch_size": meta["micro_batch_size"],
    "global_batch_size": meta["global_batch_size"],
    "tp": meta["tp"],
    "pp": meta["pp"],
    "ep": meta["ep"],
    "dp": meta["dp"],
    "transformer_impl": meta["transformer_impl"],
    "hw_tag": meta["hw_tag"],
}

with open(os.environ["PROFILE_INDEX_PATH"], "a", encoding="utf-8") as f:
    f.write(json.dumps(index) + "\n")
PY

case "$PROFILE_TOOL" in
  nsys)
    if ! command -v nsys >/dev/null 2>&1; then
      echo "[error] nsys not found in container. Set PROFILE_TOOL=torch or install nsys in the image." >&2
      exit 1
    fi
    out_prefix="${run_dir}/trace"
    echo "[profile] writing: ${out_prefix}.qdrep" >&2
    exec nsys profile -s none -t nvtx,cuda \
      --cudabacktrace=all --cuda-graph-trace=node --python-backtrace=cuda --wait all \
      -o "$out_prefix" --force-overwrite true \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      "${TRAIN_CMD[@]}"
    ;;
  torch)
    tb_dir="${run_dir}/tensorboard"
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
