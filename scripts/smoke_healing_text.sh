#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-data/megatron/healing_text_v1}"
SMOKE_TOKENIZER="${SMOKE_TOKENIZER:-data/tokenizer/o200k_base.tiktoken}"
SMOKE_LOAD="${SMOKE_LOAD:-weights/upcycle/vpDense0-5_28}"
SMOKE_SAVE="${SMOKE_SAVE:-data/megatron/healing_text_v1_smoke}"
SMOKE_SEQ_LEN="${SMOKE_SEQ_LEN:-512}"
SMOKE_MAX_POS="${SMOKE_MAX_POS:-4096}"
SMOKE_TRAIN_ITERS="${SMOKE_TRAIN_ITERS:-2}"
SMOKE_LR="${SMOKE_LR:-1e-5}"
SMOKE_LOG_INTERVAL="${SMOKE_LOG_INTERVAL:-1}"
SMOKE_ROPE_BASE="${SMOKE_ROPE_BASE:-1000000}"

docker compose -f "$ROOT_DIR/docker/compose.yml" run --rm \
  -e "SMOKE_DATA_DIR=$SMOKE_DATA_DIR" \
  -e "SMOKE_TOKENIZER=$SMOKE_TOKENIZER" \
  -e "SMOKE_LOAD=$SMOKE_LOAD" \
  -e "SMOKE_SAVE=$SMOKE_SAVE" \
  -e "SMOKE_SEQ_LEN=$SMOKE_SEQ_LEN" \
  -e "SMOKE_MAX_POS=$SMOKE_MAX_POS" \
  -e "SMOKE_TRAIN_ITERS=$SMOKE_TRAIN_ITERS" \
  -e "SMOKE_LR=$SMOKE_LR" \
  -e "SMOKE_LOG_INTERVAL=$SMOKE_LOG_INTERVAL" \
  -e "SMOKE_ROPE_BASE=$SMOKE_ROPE_BASE" \
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

DATA_DIR="$(resolve_path "$SMOKE_DATA_DIR")"
TOKENIZER_MODEL="$(resolve_path "$SMOKE_TOKENIZER")"
LOAD_DIR="$(resolve_path "$SMOKE_LOAD")"
SAVE_DIR="$(resolve_path "$SMOKE_SAVE")"

cd /workspace/vpmoe

manifest="$DATA_DIR/manifest.json"
manifest_hash="nocache"
if [[ -f "$manifest" ]]; then
  manifest_hash="$(sha256sum "$manifest" | cut -d " " -f1 | cut -c1-8)"
fi
cache_dir="$DATA_DIR/cache_${SMOKE_SEQ_LEN}_${manifest_hash}"
mkdir -p "$cache_dir"

export O200K_HARMONY_VOCAB_PATH="$TOKENIZER_MODEL"

shopt -s nullglob
train_idx=("$DATA_DIR"/train/*_tokens.idx)
if [[ ${#train_idx[@]} -eq 0 ]]; then
  echo "[error] no train shards found under $DATA_DIR/train" >&2
  exit 1
fi
train_prefixes=("${train_idx[@]%.idx}")

valid_idx=("$DATA_DIR"/valid/*_tokens.idx)
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
  --train-data-path "${train_prefixes[@]}" \
  "${valid_flag[@]}" \
  --load "$LOAD_DIR" \
  --no-load-optim --no-load-rng --exit-on-missing-checkpoint \
  --finetune \
  --ckpt-format torch_dist \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --num-layers 28 --hidden-size 1024 --num-attention-heads 8 \
  --group-query-attention --num-query-groups 2 \
  --ffn-hidden-size 512 \
  --seq-length "$SMOKE_SEQ_LEN" --max-position-embeddings "$SMOKE_MAX_POS" \
  --position-embedding-type rope --rotary-base "$SMOKE_ROPE_BASE" \
  --tokenizer-type O200kHarmonyTokenizer --tokenizer-model "$TOKENIZER_MODEL" \
  --micro-batch-size 1 --global-batch-size 1 \
  --train-iters "$SMOKE_TRAIN_ITERS" --lr "$SMOKE_LR" --min-lr "$SMOKE_LR" --lr-decay-style constant \
  --log-interval "$SMOKE_LOG_INTERVAL" --eval-interval 1000000000 --eval-iters 0 \
  --save "$SAVE_DIR" --save-interval 1000000000 \
  --data-cache-path "$cache_dir" \
  --normalization RMSNorm --qk-layernorm --softmax-type learnable \
  --squared-relu --value-residual --value-residual-init 1.0 \
  --no-bias-gelu-fusion \
  --no-persist-layer-norm \
  --untie-embeddings-and-output-weights \
  --hetereogenous-dist-checkpoint \
  --bf16 \
  --transformer-impl local
'
