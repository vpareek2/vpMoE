# Docker runbook (canonical)

vpMoE is **container-first**. All supported workflows should run inside the repo container.

## One-shot setup

From a fresh clone, run:

```bash
./setup.sh
```

This script:
- creates `/data` and `/datasets` on the host if needed,
- vendors Transformers at the pinned ref,
- builds the container, and
- starts the service.

If you need private Hugging Face or W&B access, the script will prompt for tokens
and save them to a local `.env` file (gitignored). You can also export `HF_TOKEN`
and/or `WANDB_API_KEY` beforehand to skip prompts.

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

## Canonical host mounts (no per-machine overrides)

We standardize on **host-level** mounts so every machine uses the same paths:

- **`/data` (host)** → `/data` (container, writable)
- **`/datasets` (host)** → `/datasets` (container, read‑only; optional)

`docker/compose.yml` already mounts these paths. That means **no `compose.local.yml`**
and no per-machine edits. If the paths do not exist on the host, create them once:

```bash
sudo mkdir -p /data /datasets
```

All caches, checkpoints, and derived datasets live under `/data`. Raw datasets
(if you use them) live under `/datasets`. If you train from Hugging Face datasets,
you can leave `/datasets` empty.
