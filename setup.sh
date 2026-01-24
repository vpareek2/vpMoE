#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

TRANSFORMERS_DIR="${REPO_ROOT}/src/third_party/transformers"
TRANSFORMERS_REF="${TRANSFORMERS_REF:-v4.57.6}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

ensure_dir() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    return 0
  fi
  if mkdir -p "${path}" 2>/dev/null; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo mkdir -p "${path}"
    return 0
  fi
  echo "error: cannot create ${path} (need root). Run:" >&2
  echo "  sudo mkdir -p ${path}" >&2
  exit 1
}

resolve_compose() {
  if [[ -n "${DOCKER_COMPOSE:-}" ]]; then
    echo "${DOCKER_COMPOSE}"
    return 0
  fi
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
    return 0
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
    return 0
  fi
  echo "error: docker compose not found" >&2
  exit 1
}

want_shell=0
for arg in "$@"; do
  case "${arg}" in
    --shell) want_shell=1 ;;
    *) echo "unknown arg: ${arg}" >&2; exit 1 ;;
  esac
done

need_cmd git

compose_cmd=""

ensure_dir /data
ensure_dir /datasets

is_tty=0
if [[ -t 0 && -t 1 ]]; then
  is_tty=1
fi

prompt_secret() {
  local var_name="$1"
  local label="$2"
  local default_val="${!var_name:-}"
  if [[ -n "${default_val}" ]]; then
    return 0
  fi
  if [[ "${is_tty}" -ne 1 ]]; then
    return 0
  fi
  printf "%s (leave blank to skip): " "${label}" >&2
  local value
  IFS= read -r -s value || true
  echo >&2
  if [[ -n "${value}" ]]; then
    export "${var_name}=${value}"
  fi
}

upsert_env_var() {
  local file="$1"
  local key="$2"
  local val="$3"
  if [[ -z "${val}" ]]; then
    return 0
  fi
  if [[ -f "${file}" ]]; then
    if grep -q "^${key}=" "${file}"; then
      sed -i "s#^${key}=.*#${key}=${val}#g" "${file}"
      return 0
    fi
  fi
  echo "${key}=${val}" >> "${file}"
}

ENV_FILE="${REPO_ROOT}/.env"

is_root() {
  [[ "$(id -u)" -eq 0 ]]
}

sudo_prefix() {
  if is_root; then
    echo ""
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    echo "sudo"
    return 0
  fi
  echo "error: need root or sudo to install system packages." >&2
  exit 1
}

install_docker_apt() {
  local sudo_cmd
  sudo_cmd="$(sudo_prefix)"
  ${sudo_cmd} apt-get update
  ${sudo_cmd} apt-get install -y ca-certificates curl gnupg
  ${sudo_cmd} install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | ${sudo_cmd} gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  ${sudo_cmd} chmod a+r /etc/apt/keyrings/docker.gpg
  local codename
  codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-}")"
  if [[ -z "${codename}" ]]; then
    codename="$(. /etc/os-release && echo "${VERSION_CODENAME:-}")"
  fi
  if [[ -z "${codename}" ]]; then
    echo "error: could not determine distro codename for Docker repo." >&2
    exit 1
  fi
  local distro_id
  distro_id="$(. /etc/os-release && echo "${ID}")"
  local repo_base="https://download.docker.com/linux/${distro_id}"
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] ${repo_base} \
    ${codename} stable" | ${sudo_cmd} tee /etc/apt/sources.list.d/docker.list >/dev/null
  ${sudo_cmd} apt-get update
  ${sudo_cmd} apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  if command -v systemctl >/dev/null 2>&1; then
    ${sudo_cmd} systemctl enable --now docker || true
  fi
  if ! is_root; then
    ${sudo_cmd} usermod -aG docker "${USER}" || true
  fi
}

install_nvidia_container_toolkit() {
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    local sudo_cmd
    sudo_cmd="$(sudo_prefix)"
    local distribution
    distribution="$(. /etc/os-release; echo "${ID}${VERSION_ID}")"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | ${sudo_cmd} gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
      sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
      ${sudo_cmd} tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    ${sudo_cmd} apt-get update
    ${sudo_cmd} apt-get install -y nvidia-container-toolkit
    if command -v nvidia-ctk >/dev/null 2>&1; then
      ${sudo_cmd} nvidia-ctk runtime configure --runtime=docker || true
    fi
    if command -v systemctl >/dev/null 2>&1; then
      ${sudo_cmd} systemctl restart docker || true
    fi
  fi
}

ensure_docker() {
  if command -v docker >/dev/null 2>&1; then
    return 0
  fi
  if [[ ! -f /etc/os-release ]]; then
    echo "error: docker not found and /etc/os-release missing; install docker manually." >&2
    exit 1
  fi
  . /etc/os-release
  case "${ID}" in
    ubuntu|debian)
      install_docker_apt
      ;;
    *)
      echo "error: docker not found. Unsupported distro ${ID}; install Docker manually." >&2
      exit 1
      ;;
  esac
}

ensure_docker
need_cmd docker
compose_cmd="$(resolve_compose)"

if command -v nvidia-smi >/dev/null 2>&1 || [[ -e /proc/driver/nvidia/version ]]; then
  install_nvidia_container_toolkit
fi

if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
prompt_secret "HF_TOKEN" "Hugging Face token"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
prompt_secret "WANDB_API_KEY" "Weights & Biases token"

if [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" || -n "${WANDB_API_KEY:-}" ]]; then
  upsert_env_var "${ENV_FILE}" "HF_TOKEN" "${HF_TOKEN:-}"
  upsert_env_var "${ENV_FILE}" "HUGGING_FACE_HUB_TOKEN" "${HUGGING_FACE_HUB_TOKEN:-}"
  upsert_env_var "${ENV_FILE}" "WANDB_API_KEY" "${WANDB_API_KEY:-}"
  chmod 600 "${ENV_FILE}" 2>/dev/null || true
fi

mkdir -p "${REPO_ROOT}/src/third_party"

if [[ ! -d "${TRANSFORMERS_DIR}/.git" ]]; then
  echo "Cloning transformers into ${TRANSFORMERS_DIR}..." >&2
  git clone --filter=blob:none https://github.com/huggingface/transformers.git "${TRANSFORMERS_DIR}"
fi

echo "Pinning transformers to ${TRANSFORMERS_REF}..." >&2
git -C "${TRANSFORMERS_DIR}" fetch --tags --force
git -C "${TRANSFORMERS_DIR}" checkout -q "${TRANSFORMERS_REF}"

echo "Building vpmoe container..." >&2
${compose_cmd} -f docker/compose.yml build vpmoe

echo "Starting vpmoe container..." >&2
${compose_cmd} -f docker/compose.yml up -d

cat >&2 <<EOF

Setup complete.

If you entered tokens, they are saved in .env (gitignored).
You can edit or remove .env anytime.

Enter the container:
  ${compose_cmd} -f docker/compose.yml exec vpmoe bash
EOF

if [[ "${want_shell}" -eq 1 ]]; then
  ${compose_cmd} -f docker/compose.yml exec vpmoe bash
fi
