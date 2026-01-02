#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-weights/qwen3-0_6B}"
DONOR_MODEL_DIR="${DONOR_MODEL_DIR:-weights/gpt-oss-20b}"
OUT_DIR="${OUT_DIR:-weights/upcycle/qwen3-0_6B-o200k}"

METHOD="${METHOD:-omp}"
K="${K:-64}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -d "${BASE_MODEL_DIR}" ]]; then
  echo "Missing base model dir: ${BASE_MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DONOR_MODEL_DIR}" ]]; then
  echo "Missing donor model dir: ${DONOR_MODEL_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_DIR}")"

python3 - <<'PY'
import importlib, sys
mods = [
    "torch",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "click",
    "pydantic",
    "tqdm",
    "immutables",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    sys.stderr.write(
        "Missing Python deps for mergekit-tokensurgeon: "
        + ", ".join(missing)
        + "\n"
        + "Run `uv sync` (after updating uv.lock) and try again, or rebuild the container.\n"
    )
    sys.exit(2)
PY

export PYTHONPATH="${ROOT}/ref/mergekit:${PYTHONPATH:-}"

python3 -m mergekit.scripts.tokensurgeon \
  "${BASE_MODEL_DIR}" \
  "${DONOR_MODEL_DIR}" \
  "${OUT_DIR}" \
  --approximation-method "${METHOD}" \
  --k "${K}" \
  --device "${DEVICE}"

python3 scripts/upcycle/pad_hf_vocab.py \
  --model-dir "${OUT_DIR}" \
  --target-vocab-size 201088

echo "Wrote: ${OUT_DIR}"
