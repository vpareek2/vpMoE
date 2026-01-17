# Docker runbook (canonical)

vpMoE is **container-first**. All supported workflows should run inside the repo container.

## Build

```bash
docker compose -f docker/compose.yml build vpmoe
```

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
