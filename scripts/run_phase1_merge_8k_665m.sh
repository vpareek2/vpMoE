#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-/data/distillation_1/phase1_mix_8k_665m}"
SEED="${SEED:-1337}"
WEIGHT_BY="${WEIGHT_BY:-assistant_tokens}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-2000}"

SYNTH_DIR="${SYNTH_DIR:-/data/distillation_1/synth_distill_phase1_high_550m}"
CODE_DIR="${CODE_DIR:-/data/distillation_1/code_opencodeinstruct_55m}"
MATH_DIR="${MATH_DIR:-/data/distillation_1/math_nemotron_v2_low_55m}"
CHAT_DIR="${CHAT_DIR:-/data/distillation_1/helpfulness_oasst2_15m}"

echo "Merging Phase 1 datasets..." >&2
echo "  synth: ${SYNTH_DIR}" >&2
echo "  code : ${CODE_DIR}" >&2
echo "  math : ${MATH_DIR}" >&2
echo "  chat : ${CHAT_DIR}" >&2
echo "  out  : ${OUT_DIR}" >&2

rm -rf "${OUT_DIR}"

PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS:-20}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}" \
python3 scripts/merge_distill_datasets.py \
  --input "synth=${SYNTH_DIR}" \
  --input "code=${CODE_DIR}" \
  --input "math=${MATH_DIR}" \
  --input "chat=${CHAT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --rows-per-shard "${ROWS_PER_SHARD}" \
  --weight-by "${WEIGHT_BY}" \
  --seed "${SEED}"

python3 scripts/count_distill_tokens.py --input-dir "${OUT_DIR}" --split all
python3 scripts/print_distill_samples.py --input-dir "${OUT_DIR}" --split train --num-samples 3 --seed 0 --raw

