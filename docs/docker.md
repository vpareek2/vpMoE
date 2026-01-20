# Docker runbook (canonical)

vpMoE is **container-first**. All supported workflows should run inside the repo container.

## Build

### Vendored Transformers (Required)

This repo expects a local Transformers source checkout at `src/third_party/transformers`, installed into the container as the `transformers` package. This keeps GPT-OSS customization in a single, reproducible codebase without import shims.

Clone (pin to the same commit/tag as the container expects):

```bash
mkdir -p src/third_party
git clone https://github.com/huggingface/transformers.git src/third_party/transformers
cd src/third_party/transformers
git checkout v4.57.6
```

```bash
docker compose -f docker/compose.yml build vpmoe
```

The container also installs the local DistillKit checkout at `src/DistillKit`
so the `distillkit` CLI is available inside the image.

Override base image (optional):

```bash
VPMOE_BASE_IMAGE=nvcr.io/nvidia/pytorch:<tag> \
  docker compose -f docker/compose.yml build vpmoe
```

## Run + attach

```bash
docker compose -f docker/compose.yml up -d
docker compose -f docker/compose.yml exec vpmoe bash
```

One-off shell:

```bash
docker compose -f docker/compose.yml run --rm vpmoe bash
```

## Local dataset mounts (recommended)

Copy the local override template:

```bash
cp docker/compose.local.example.yml docker/compose.local.yml
```

Edit `docker/compose.local.yml` to mount your host datasets into the container. A common pattern:

- mount `/path/to/datasets` (host) to `/datasets` (container, read-only)
- keep `/data` (container) as the writable workspace for derived datasets and caches

Then include both compose files when you run:

```bash
docker compose -f docker/compose.yml -f docker/compose.local.yml up -d
docker compose -f docker/compose.yml -f docker/compose.local.yml exec vpmoe bash
```

Inside the container, prefer symlinks under `/data/raw_0` that point to `/datasets/...` so the rest of the tooling has a stable path.
