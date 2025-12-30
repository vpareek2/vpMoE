# vpMoE

Data Curation Stack: TBD

Upcycling Stack: TBD

Pre-Training Stack: Megatron-LM

Post-Training Stack: TBD

Container-first runbook: `docs/docker.md`

## Docker (Recommended)

Build:

```bash
docker compose -f docker/compose.yml build vpmoe
```

Build (override base image):

```bash
VPMOE_BASE_IMAGE=nvcr.io/nvidia/pytorch:<tag> \
  docker compose -f docker/compose.yml build vpmoe
```

Run:

```bash
docker compose -f docker/compose.yml up -d
docker compose -f docker/compose.yml exec vpmoe bash
```

One-off shell:

```bash
docker compose -f docker/compose.yml run --rm vpmoe bash
```
