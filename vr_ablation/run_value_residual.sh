#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VR_DATASET_DIR="${VR_DATASET_DIR:-data/megatron/vr_ablation/synth_kd_v1_100m}"
VR_TOKENIZER="${VR_TOKENIZER:-data/tokenizer/o200k_base.tiktoken}"
VR_SEQ_LEN="${VR_SEQ_LEN:-1024}"
VR_TRAIN_ITERS="${VR_TRAIN_ITERS:-305}"
VR_MICRO_BS="${VR_MICRO_BS:-1}"
VR_GLOBAL_BS="${VR_GLOBAL_BS:-32}"
VR_LOG_INTERVAL="${VR_LOG_INTERVAL:-10}"
VR_EVAL_INTERVAL="${VR_EVAL_INTERVAL:-500}"
VR_EVAL_ITERS="${VR_EVAL_ITERS:-5}"
VR_VALUE_RESIDUAL_INIT="${VR_VALUE_RESIDUAL_INIT:-1.0}"

MODEL_NUM_LAYERS="${MODEL_NUM_LAYERS:-8}"
MODEL_HIDDEN_SIZE="${MODEL_HIDDEN_SIZE:-256}"
MODEL_FFN_HIDDEN_SIZE="${MODEL_FFN_HIDDEN_SIZE:-1024}"
MODEL_NUM_HEADS="${MODEL_NUM_HEADS:-4}"


docker compose -f "$ROOT_DIR/docker/compose.yml" run --rm \
  -e VR_DATASET_DIR="$VR_DATASET_DIR" \
  -e VR_TOKENIZER="$VR_TOKENIZER" \
  -e VR_SEQ_LEN="$VR_SEQ_LEN" \
  -e VR_TRAIN_ITERS="$VR_TRAIN_ITERS" \
  -e VR_MICRO_BS="$VR_MICRO_BS" \
  -e VR_GLOBAL_BS="$VR_GLOBAL_BS" \
  -e VR_LOG_INTERVAL="$VR_LOG_INTERVAL" \
  -e VR_EVAL_INTERVAL="$VR_EVAL_INTERVAL" \
  -e VR_EVAL_ITERS="$VR_EVAL_ITERS" \
  -e VR_VALUE_RESIDUAL_INIT="$VR_VALUE_RESIDUAL_INIT" \
  -e MODEL_NUM_LAYERS="$MODEL_NUM_LAYERS" \
  -e MODEL_HIDDEN_SIZE="$MODEL_HIDDEN_SIZE" \
  -e MODEL_FFN_HIDDEN_SIZE="$MODEL_FFN_HIDDEN_SIZE" \
  -e MODEL_NUM_HEADS="$MODEL_NUM_HEADS" \
  vpmoe bash -lc '
set -euo pipefail

VR_DATASET_DIR="${VR_DATASET_DIR:-data/megatron/vr_ablation/synth_kd_v1_100m}"
VR_TOKENIZER="${VR_TOKENIZER:-data/tokenizer/o200k_base.tiktoken}"
VR_SEQ_LEN="${VR_SEQ_LEN:-1024}"
VR_TRAIN_ITERS="${VR_TRAIN_ITERS:-305}"
VR_MICRO_BS="${VR_MICRO_BS:-1}"
VR_GLOBAL_BS="${VR_GLOBAL_BS:-32}"
VR_LOG_INTERVAL="${VR_LOG_INTERVAL:-10}"
VR_EVAL_INTERVAL="${VR_EVAL_INTERVAL:-500}"
VR_EVAL_ITERS="${VR_EVAL_ITERS:-5}"
VR_VALUE_RESIDUAL_INIT="${VR_VALUE_RESIDUAL_INIT:-1.0}"

MODEL_NUM_LAYERS="${MODEL_NUM_LAYERS:-8}"
MODEL_HIDDEN_SIZE="${MODEL_HIDDEN_SIZE:-256}"
MODEL_FFN_HIDDEN_SIZE="${MODEL_FFN_HIDDEN_SIZE:-1024}"
MODEL_NUM_HEADS="${MODEL_NUM_HEADS:-4}"

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf "%s" "$path"
  else
    printf "/workspace/vpmoe/%s" "$path"
  fi
}

DATASET_DIR="$(resolve_path "$VR_DATASET_DIR")"
TOKENIZER_MODEL="$(resolve_path "$VR_TOKENIZER")"

manifest="$DATASET_DIR/manifest.json"
if [[ ! -f "$manifest" ]]; then
  echo "[error] manifest not found: $manifest" >&2
  exit 1
fi
manifest_hash="$(sha256sum "$manifest" | cut -d " " -f1 | cut -c1-8)"
cache_dir="$DATASET_DIR/cache_${VR_SEQ_LEN}_${manifest_hash}"
mkdir -p "$cache_dir"

export O200K_HARMONY_VOCAB_PATH="$TOKENIZER_MODEL"

shopt -s nullglob
train_idx=("$DATASET_DIR"/train/*_tokens.idx)
if [[ ${#train_idx[@]} -eq 0 ]]; then
  echo "[error] no train shards found under $DATASET_DIR/train" >&2
  exit 1
fi
train_prefixes=("${train_idx[@]%.idx}")

valid_idx=("$DATASET_DIR"/valid/*_tokens.idx)
valid_prefixes=()
if [[ ${#valid_idx[@]} -gt 0 ]]; then
  valid_prefixes=("${valid_idx[@]%.idx}")
fi

valid_flag=()
if [[ ${#valid_prefixes[@]} -eq 0 ]]; then
  echo "[warn] no valid shards found; running without validation."
else
  valid_flag=(--valid-data-path "${valid_prefixes[@]}")
fi

uv run python -m torch.distributed.run --nproc_per_node=1 vpmoe/Megatron-vpmoe/pretrain_gpt.py \
  --synth-kd-data \
  --train-data-path "${train_prefixes[@]}" \
  "${valid_flag[@]}" \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --num-layers "$MODEL_NUM_LAYERS" --hidden-size "$MODEL_HIDDEN_SIZE" --ffn-hidden-size "$MODEL_FFN_HIDDEN_SIZE" \
  --num-attention-heads "$MODEL_NUM_HEADS" \
  --seq-length "$VR_SEQ_LEN" --max-position-embeddings "$VR_SEQ_LEN" \
  --tokenizer-type O200kHarmonyTokenizer --tokenizer-model "$TOKENIZER_MODEL" \
  --micro-batch-size "$VR_MICRO_BS" --global-batch-size "$VR_GLOBAL_BS" \
  --train-iters "$VR_TRAIN_ITERS" --lr 1e-4 --min-lr 1e-4 --lr-decay-style constant \
  --log-interval "$VR_LOG_INTERVAL" --eval-interval "$VR_EVAL_INTERVAL" --eval-iters "$VR_EVAL_ITERS" --save-interval 0 \
  --data-cache-path "$cache_dir" \
  --transformer-impl local \
  --value-residual --value-residual-init "$VR_VALUE_RESIDUAL_INIT"
'
