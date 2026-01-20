#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
SEED="${SEED:-1337}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-2000}"
WEIGHT_BY="${WEIGHT_BY:-assistant_tokens}"
PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS:-20}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
REASONING_LEVEL="${REASONING_LEVEL:-high}"

OUT_ROOT="${OUT_ROOT:-/data/distillation_1}"
MERGE_OUT_DIR="${MERGE_OUT_DIR:-${OUT_ROOT}/phase1_mix_4k_665m}"

SYNTH_SRC_DIR="${SYNTH_SRC_DIR:-/data/raw_0/pleias_synth}"
SYNTH_SHARDS_LIST="${SYNTH_SHARDS_LIST:-${REPO_ROOT}/data/distillation_1/synth_phase1_shards.txt}"
SYNTH_PHASE1_DIR="${SYNTH_PHASE1_DIR:-/data/synth_phase1}"
SYNTH_OUT_DIR="${SYNTH_OUT_DIR:-${OUT_ROOT}/synth_distill_phase1_high_550m_4k}"

CODE_SRC_DIR="${CODE_SRC_DIR:-/datasets/nvidia__OpenCodeInstruct/data}"
CODE_OUT_DIR="${CODE_OUT_DIR:-${OUT_ROOT}/code_opencodeinstruct_55m_4k}"

MATH_SRC_FILE="${MATH_SRC_FILE:-/datasets/nvidia__Nemotron-Math-v2/data/low.jsonl}"
MATH_OUT_DIR="${MATH_OUT_DIR:-${OUT_ROOT}/math_nemotron_v2_low_55m_4k}"

CHAT_SRC_DIR="${CHAT_SRC_DIR:-/data/raw_0/OpenAssistant__oasst2}"
CHAT_OUT_DIR="${CHAT_OUT_DIR:-${OUT_ROOT}/helpfulness_oasst2_15m_4k}"

echo "Building Phase 1 (4k) distillation datasets..." >&2
echo "  max_seq_len : ${MAX_SEQ_LEN}" >&2
echo "  out_root    : ${OUT_ROOT}" >&2
echo "  merge_out   : ${MERGE_OUT_DIR}" >&2
echo "  synth_src   : ${SYNTH_SRC_DIR}" >&2
echo "  synth_list  : ${SYNTH_SHARDS_LIST}" >&2
echo "  synth_out   : ${SYNTH_OUT_DIR}" >&2
echo "  code_src    : ${CODE_SRC_DIR}" >&2
echo "  code_out    : ${CODE_OUT_DIR}" >&2
echo "  math_src    : ${MATH_SRC_FILE}" >&2
echo "  math_out    : ${MATH_OUT_DIR}" >&2
echo "  chat_src    : ${CHAT_SRC_DIR}" >&2
echo "  chat_out    : ${CHAT_OUT_DIR}" >&2

if [[ ! -d "${SYNTH_SRC_DIR}" ]]; then
  echo "Missing SYNTH source dir: ${SYNTH_SRC_DIR}" >&2
  exit 1
fi
if [[ ! -f "${SYNTH_SHARDS_LIST}" ]]; then
  echo "Missing SYNTH shard list: ${SYNTH_SHARDS_LIST}" >&2
  exit 1
fi
if [[ ! -d "${CODE_SRC_DIR}" ]]; then
  echo "Missing OpenCodeInstruct dir: ${CODE_SRC_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MATH_SRC_FILE}" ]]; then
  echo "Missing Nemotron-Math file: ${MATH_SRC_FILE}" >&2
  exit 1
fi
if [[ ! -d "${CHAT_SRC_DIR}" ]]; then
  echo "Missing OASST2 dir: ${CHAT_SRC_DIR}" >&2
  exit 1
fi

echo "Preparing SYNTH shard subset..." >&2
mkdir -p "${SYNTH_PHASE1_DIR}"
while IFS= read -r p; do
  [[ -z "${p}" ]] && continue
  b="$(basename "${p}")"
  if [[ "${p}" == /* ]]; then
    src="${p}"
  else
    src="${SYNTH_SRC_DIR}/${p}"
  fi
  ln -sf "${src}" "${SYNTH_PHASE1_DIR}/${b}"
done < "${SYNTH_SHARDS_LIST}"

echo "Building SYNTH distill dataset..." >&2
rm -rf "${SYNTH_OUT_DIR}"
PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
python3 "${REPO_ROOT}/scripts/build_synth_distill.py" \
  --input-dir "${SYNTH_PHASE1_DIR}" \
  --output-dir "${SYNTH_OUT_DIR}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --batch-size 4096 \
  --english-frac 0.75 \
  --memorization-cap 0.70 \
  --global-keep 0.20 \
  --reasoning-level "${REASONING_LEVEL}"

echo "Building OpenCodeInstruct distill dataset..." >&2
rm -rf "${CODE_OUT_DIR}"
PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
python3 "${REPO_ROOT}/scripts/build_opencode_distill.py" \
  --input-dir "${CODE_SRC_DIR}" \
  --output-dir "${CODE_OUT_DIR}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --batch-size 4096 \
  --target-total-tokens 55000000 \
  --reasoning-level "${REASONING_LEVEL}" \
  --shuffle-shards

echo "Building Nemotron-Math distill dataset..." >&2
rm -rf "${MATH_OUT_DIR}"
PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
python3 "${REPO_ROOT}/scripts/build_nemotron_math_distill.py" \
  --input "${MATH_SRC_FILE}" \
  --output-dir "${MATH_OUT_DIR}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --target-total-tokens 55000000 \
  --reasoning-level "${REASONING_LEVEL}" \
  --drop-tool-rows

echo "Building OASST2 distill dataset..." >&2
rm -rf "${CHAT_OUT_DIR}"
PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
python3 "${REPO_ROOT}/scripts/build_oasst2_distill.py" \
  --input-dir "${CHAT_SRC_DIR}" \
  --output-dir "${CHAT_OUT_DIR}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --batch-size 4096 \
  --target-total-tokens 15000000 \
  --english-frac 0.90 \
  --reasoning-level "${REASONING_LEVEL}"

echo "Merging Phase 1 datasets..." >&2
rm -rf "${MERGE_OUT_DIR}"
PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
python3 "${REPO_ROOT}/scripts/merge_distill_datasets.py" \
  --input "synth=${SYNTH_OUT_DIR}" \
  --input "code=${CODE_OUT_DIR}" \
  --input "math=${MATH_OUT_DIR}" \
  --input "chat=${CHAT_OUT_DIR}" \
  --output-dir "${MERGE_OUT_DIR}" \
  --rows-per-shard "${ROWS_PER_SHARD}" \
  --weight-by "${WEIGHT_BY}" \
  --seed "${SEED}"

python3 "${REPO_ROOT}/scripts/count_distill_tokens.py" --input-dir "${MERGE_OUT_DIR}" --split all
python3 "${REPO_ROOT}/scripts/print_distill_samples.py" --input-dir "${MERGE_OUT_DIR}" --split train --num-samples 3 --seed 0 --raw

echo "Phase 1 (4k) dataset build complete." >&2
