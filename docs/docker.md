# Docker Runbook

This repo is container-first. The only supported execution path is via `docker/`.

## Base Image Selection

We use one Dockerfile (`docker/Dockerfile`) and select the NVIDIA NGC PyTorch base image via a build arg.

Default:

- `nvcr.io/nvidia/pytorch:25.12-py3`

Override (if needed):

- `VPMOE_BASE_IMAGE=nvcr.io/nvidia/pytorch:<tag>`

## Build

From the repo root:

```bash
# Default
docker compose -f docker/compose.yml build vpmoe

# Override base image
VPMOE_BASE_IMAGE=nvcr.io/nvidia/pytorch:<tag> \
  docker compose -f docker/compose.yml build vpmoe
```

## Run

```bash
docker compose -f docker/compose.yml up -d
docker compose -f docker/compose.yml exec vpmoe bash
```

For a one-off interactive shell (no background container):

```bash
docker compose -f docker/compose.yml run --rm vpmoe bash
```

If you see an NGC startup error like `No supported GPU(s) detected`, the service sets `NVIDIA_DISABLE_REQUIRE=1` to bypass the container allowlist check.

The service mounts:

- repo → `/workspace/vpmoe`
- data → `/data`
- HF cache → `/data/hf_cache` (via `HF_HOME`)

## PyTorch GPU Support

If PyTorch warns that it doesn't support your GPU's SM (compute capability), switch the base image tag. Verify inside the container:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('capability:', torch.cuda.get_device_capability())
print('arch_list:', torch.cuda.get_arch_list())
PY
```

## Python Environment

The images inherit `torch` (and related CUDA/PyTorch stack) from the chosen NGC base image.
The vpMoE venv is created with `--system-site-packages` so it can import base-image `torch` without reinstalling it.

## Adding Python Dependencies

This repo uses `uv` and a checked-in `uv.lock`. Add deps only when needed:

```bash
uv add <package>
UV_CACHE_DIR=$PWD/.uv-cache uv lock --python python3.12
```

## Common Pitfall

If you run `docker/compose.yml build vpmoe`, your shell will try to execute the YAML file and you’ll see `Permission denied`. The correct command is `docker compose -f docker/compose.yml build vpmoe`.
