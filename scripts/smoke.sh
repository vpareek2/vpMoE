#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-data/megatron/synth_kd_v1_smoke}"
SMOKE_TOKENIZER="${SMOKE_TOKENIZER:-data/tokenizer/o200k_base.tiktoken}"
SMOKE_ROWS="${SMOKE_ROWS:-1000}"
SMOKE_ROWS_PER_SHARD="${SMOKE_ROWS_PER_SHARD:-250}"
SMOKE_VALID_FRACTION="${SMOKE_VALID_FRACTION:-0.1}"
SMOKE_SEQ_LEN="${SMOKE_SEQ_LEN:-1024}"
SMOKE_TRAIN_ITERS="${SMOKE_TRAIN_ITERS:-5}"

docker compose -f "$ROOT_DIR/docker/compose.yml" run --rm \
  -e SMOKE_OUTPUT_DIR \
  -e SMOKE_TOKENIZER \
  -e SMOKE_ROWS \
  -e SMOKE_ROWS_PER_SHARD \
  -e SMOKE_VALID_FRACTION \
  -e SMOKE_SEQ_LEN \
  -e SMOKE_TRAIN_ITERS \
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

OUTPUT_DIR="$(resolve_path "$SMOKE_OUTPUT_DIR")"
TOKENIZER_MODEL="$(resolve_path "$SMOKE_TOKENIZER")"

cd /workspace/vpmoe

uv run scripts/datasets/build_synth_kd_v1.py \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --max-rows "$SMOKE_ROWS" \
  --max-rows-per-shard "$SMOKE_ROWS_PER_SHARD" \
  --valid-fraction "$SMOKE_VALID_FRACTION" \
  --output-dir "$OUTPUT_DIR" \
  --overwrite

manifest="$OUTPUT_DIR/manifest.json"
if [[ ! -f "$manifest" ]]; then
  echo "[error] manifest not found: $manifest" >&2
  exit 1
fi

manifest_hash="$(sha256sum "$manifest" | cut -d " " -f1 | cut -c1-8)"
cache_dir="$OUTPUT_DIR/cache_${SMOKE_SEQ_LEN}_${manifest_hash}"
mkdir -p "$cache_dir"

export O200K_HARMONY_VOCAB_PATH="$TOKENIZER_MODEL"

shopt -s nullglob
train_idx=("$OUTPUT_DIR"/train/*_tokens.idx)
if [[ ${#train_idx[@]} -eq 0 ]]; then
  echo "[error] no train shards found under $OUTPUT_DIR/train" >&2
  exit 1
fi
train_prefixes=("${train_idx[@]%.idx}")

valid_idx=("$OUTPUT_DIR"/valid/*_tokens.idx)
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
  --num-layers 4 --hidden-size 128 --num-attention-heads 4 --group-query-attention --num-query-groups 4 \
  --seq-length "$SMOKE_SEQ_LEN" --max-position-embeddings "$SMOKE_SEQ_LEN" \
  --tokenizer-type O200kHarmonyTokenizer --tokenizer-model "$TOKENIZER_MODEL" \
  --micro-batch-size 1 --global-batch-size 1 \
  --train-iters "$SMOKE_TRAIN_ITERS" --lr 1e-4 --min-lr 1e-4 --lr-decay-style constant \
  --log-interval 1 --eval-interval 1000000000 --eval-iters 0 --save-interval 0 \
  --data-cache-path "$cache_dir" \
  --transformer-impl local
'
