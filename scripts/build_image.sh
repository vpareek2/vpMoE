#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/build_image.sh [--push] [--platform <platform>] [--image <ref>] [--base-image <ref>]

Builds the vpMoE container image using docker/Dockerfile.
Defaults are read from docker/image.lock, then environment overrides.
EOF
}

push_image=0
platform=""
cli_image=""
cli_base_image=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push) push_image=1; shift ;;
    --platform)
      if [[ -z "${2:-}" ]]; then
        echo "error: --platform requires a value" >&2
        usage
        exit 1
      fi
      platform="$2"
      shift 2
      ;;
    --image)
      if [[ -z "${2:-}" ]]; then
        echo "error: --image requires a value" >&2
        usage
        exit 1
      fi
      cli_image="$2"
      shift 2
      ;;
    --base-image)
      if [[ -z "${2:-}" ]]; then
        echo "error: --base-image requires a value" >&2
        usage
        exit 1
      fi
      cli_base_image="$2"
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCK_FILE="${REPO_ROOT}/docker/image.lock"

if [[ -f "${LOCK_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${LOCK_FILE}"
  set +a
fi

VPMOE_IMAGE="${VPMOE_IMAGE:-ghcr.io/vpareek2/vpmoe:main}"
VPMOE_BASE_IMAGE="${VPMOE_BASE_IMAGE:-pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel}"
TRANSFORMERS_REF="${TRANSFORMERS_REF:-v4.57.6}"
TRANSFORMERS_REPO="${TRANSFORMERS_REPO:-https://github.com/huggingface/transformers.git}"

if [[ -n "${cli_image}" ]]; then
  VPMOE_IMAGE="${cli_image}"
fi
if [[ -n "${cli_base_image}" ]]; then
  VPMOE_BASE_IMAGE="${cli_base_image}"
fi

sha_tag=""
if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_sha="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
  sha_tag="${VPMOE_IMAGE%:*}:sha-${git_sha}"
fi

build_args=(
  --build-arg "BASE_IMAGE=${VPMOE_BASE_IMAGE}"
  --build-arg "TRANSFORMERS_REF=${TRANSFORMERS_REF}"
  --build-arg "TRANSFORMERS_REPO=${TRANSFORMERS_REPO}"
)

use_buildx=0
if [[ -n "${platform}" ]]; then
  use_buildx=1
  if ! docker buildx version >/dev/null 2>&1; then
    echo "error: docker buildx is required for --platform builds." >&2
    exit 1
  fi
  if [[ "${platform}" == *","* && "${push_image}" -ne 1 ]]; then
    echo "error: multi-platform builds require --push (buildx cannot --load multiple platforms)." >&2
    exit 1
  fi
fi

if [[ "${use_buildx}" -eq 1 ]]; then
  echo "Building ${VPMOE_IMAGE} with buildx..." >&2
  buildx_cmd=(
    docker buildx build
    -f "${REPO_ROOT}/docker/Dockerfile"
    "${build_args[@]}"
    -t "${VPMOE_IMAGE}"
  )
  if [[ -n "${sha_tag}" ]]; then
    buildx_cmd+=(-t "${sha_tag}")
  fi
  buildx_cmd+=(--platform "${platform}")
  if [[ "${push_image}" -eq 1 ]]; then
    buildx_cmd+=(--push)
  else
    buildx_cmd+=(--load)
  fi
  # Keep pushes small/fast; we don't need build provenance/SBOM in this repo phase.
  buildx_cmd+=(--provenance=false --sbom=false)
  buildx_cmd+=("${REPO_ROOT}")
  "${buildx_cmd[@]}"
else
  echo "Building ${VPMOE_IMAGE}..." >&2
  docker build -f "${REPO_ROOT}/docker/Dockerfile" "${build_args[@]}" -t "${VPMOE_IMAGE}" "${REPO_ROOT}"
  if [[ -n "${sha_tag}" ]]; then
    docker tag "${VPMOE_IMAGE}" "${sha_tag}"
  fi
  if [[ "${push_image}" -eq 1 ]]; then
    echo "Pushing ${VPMOE_IMAGE}..." >&2
    docker push "${VPMOE_IMAGE}"
    if [[ -n "${sha_tag}" ]]; then
      echo "Pushing ${sha_tag}..." >&2
      docker push "${sha_tag}"
    fi
  fi
fi
