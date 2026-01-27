# Getting Started (Cloud H100)

This is the **copy‑paste** path to run a 2×H100 smoke test first, then scale to 8×H100.

## 1) Clone + one-shot setup

```bash
git clone <REPO_URL> vpMoE
cd vpMoE

./setup.sh
```

## 2) Start the container + attach

```bash
docker compose -f docker/compose.yml up -d
docker compose -f docker/compose.yml exec vpmoe bash
```

## 3) Inside the container: auth + dataset

```bash
hf auth login

export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p /data/distillation_1/phase1_mix_4k_665m
hf download --repo-type dataset veerpareek/vpmoe-phase1-mix-4k-665m \
  --local-dir /data/distillation_1/phase1_mix_4k_665m
```

Optional: warm the model caches (not required).

```bash
hf download veerpareek/vpmoe-20b-init --local-dir /data/hf_cache/vpmoe-20b-init
hf download openai/gpt-oss-20b --local-dir /data/hf_cache/gpt-oss-20b
```

## 4) 2×H100 smoke run

```bash
torchrun --nproc_per_node=2 -m distillkit.main \
  configs/distillkit/vpmoe_distill_1_smoke_2xh100.yaml
```

## 5) 8×H100 full run

```bash
torchrun --nproc_per_node=8 -m distillkit.main \
  configs/distillkit/vpmoe_distill_1.yaml
```

## Notes

- `docker/compose.yml` mounts host `/data` as `/data` in the container.
- The repo is bind-mounted into `/workspace/vpmoe`, so code edits are live.
- Outputs land under `/data/distill_runs/...`.
