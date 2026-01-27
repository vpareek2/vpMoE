# Docker runbook (canonical)

vpMoE is **container-first**. All supported workflows should run inside the repo container.

## One-shot setup (pull-only)

From a fresh clone, run:

```bash
./setup.sh
```

This script:
- creates `/data` and `/datasets` on the host if needed,
- installs Docker + Compose if missing (Ubuntu/Debian),
- installs the NVIDIA container toolkit if NVIDIA drivers are present,
- reads `docker/image.lock` for the canonical image,
- pulls the canonical image, and
- starts the service.
If Docker is missing, the script will install Docker + Compose (Ubuntu/Debian).
If NVIDIA drivers are present, it will also install the NVIDIA container toolkit.

If you need private Hugging Face or W&B access, the script will prompt for tokens
and save them to a local `.env` file (gitignored). You can also export `HF_TOKEN`
and/or `WANDB_API_KEY` beforehand to skip prompts.

## Build + publish (builder machines only)

The canonical image is defined in `docker/image.lock` and is pulled by default.
If you need to rebuild and push (e.g., bugfixes or dependency updates), use:

```bash
scripts/build_image.sh
```

To push to GHCR:

```bash
scripts/build_image.sh --push
```

If you're on an ARM64 machine (e.g., some Blackwell nodes), cross-build an amd64
image with buildx:

```bash
docker run --privileged --rm tonistiigi/binfmt --install amd64
scripts/build_image.sh --push --platform linux/amd64
```

You must be logged in to GHCR on the builder machine:

```bash
docker login ghcr.io -u <github-username>
```

If you want zero-auth pulls on new machines, set the GHCR package visibility
to **Public** after the first push.

This will:
- build from `docker/Dockerfile`,
- clone and install Transformers at the pinned ref (`TRANSFORMERS_REF`), and
- tag the image as `:main` and `:sha-<gitsha>`.

`docker/image.lock` is the single source of truth for:
- `VPMOE_IMAGE`
- `VPMOE_BASE_IMAGE` (pinned by digest)
- `TRANSFORMERS_REF`

You can override the base image and Transformers ref by exporting:

```bash
VPMOE_BASE_IMAGE=... TRANSFORMERS_REF=... scripts/build_image.sh
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

The canonical container **bind-mounts the repo** so edits to `src/` are live.
If you change dependencies (e.g., `requirements.txt`), rebuild and push a new image
with `scripts/build_image.sh`.

## Canonical host mounts (no per-machine overrides)

We standardize on **host-level** mounts so every machine uses the same paths:

- **`/data` (host)** → `/data` (container, writable)
- **`/datasets` (host)** → `/datasets` (container, read‑only; optional)

`docker/compose.yml` already mounts these paths. That means **no per-machine edits**.
If the paths do not exist on the host, create them once:

```bash
sudo mkdir -p /data /datasets
```

All caches, checkpoints, and derived datasets live under `/data`. Raw datasets
(if you use them) live under `/datasets`. If you train from Hugging Face datasets,
you can leave `/datasets` empty.
