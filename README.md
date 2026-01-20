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

## Teacher Probe (gpt-oss)

Run inside the container:

```bash
python3 scripts/teacher_probe.py \
  --models openai/gpt-oss-20b \
  --prompts src/data/prompts.jsonl \
  --out /data/teacher_runs.jsonl
```

Notes:
- Default base image is `nvcr.io/nvidia/vllm:25.12.post1-py3` (override with `VPMOE_BASE_IMAGE=...` if needed).
- For higher throughput on 20B, try `--batch-size 8` (tune per GPU / prompt lengths).
- If vLLM is available in the image, you can use `--backend vllm`.
