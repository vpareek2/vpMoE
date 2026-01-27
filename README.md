# vpMoE

Data Curation Stack: TBD

Upcycling Stack: TBD

Pre-Training Stack: Megatron-LM

Post-Training Stack: TBD

Container-first runbook: `docs/docker.md`

## Docker (Recommended)

One-shot setup (pulls the canonical image, starts the container):

```bash
./setup.sh
```

Run:

```bash
docker compose -f docker/compose.yml up -d
docker compose -f docker/compose.yml exec vpmoe bash
```

Notes:
- The container bind-mounts the repo, so edits to `src/` are live.
- If you change Python deps, rebuild and push a new image (`scripts/build_image.sh --push`).

One-off shell:

```bash
docker compose -f docker/compose.yml run --rm vpmoe bash
```

## Teacher Probe (gpt-oss)

Run inside the container:

```bash
python3 scripts/teacher_probe.py \
  --models openai/gpt-oss-20b \
  --prompts src/data/prompts.jsonl \
  --out /data/teacher_runs.jsonl
```

Notes:
- Default base image is pinned in `docker/image.lock` (override with `VPMOE_BASE_IMAGE=...` if needed).
- For higher throughput on 20B, try `--batch-size 8` (tune per GPU / prompt lengths).
