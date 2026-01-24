# Getting Started (Cloud H100)

This is the **copy‑paste** path to run a 2×H100 smoke test first, then scale to 8×H100.

## 1) Clone + vendored Transformers

```bash
git clone <REPO_URL> vpMoE
cd vpMoE

mkdir -p ../data

mkdir -p src/third_party
git clone https://github.com/huggingface/transformers.git src/third_party/transformers
cd src/third_party/transformers
git checkout v4.57.6
cd ../../../
```

## 2) Build the container

```bash
docker compose -f docker/compose.yml build vpmoe
```

## 3) Start the container

```bash
docker compose -f docker/compose.yml up -d
docker compose -f docker/compose.yml exec vpmoe bash
```

## 4) Inside the container: auth + dataset

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

## 5) 2×H100 smoke run

```bash
torchrun --nproc_per_node=2 -m distillkit.main \
  configs/distillkit/vpmoe_distill_1_smoke_2xh100.yaml
```

## 6) 8×H100 full run

```bash
torchrun --nproc_per_node=8 -m distillkit.main \
  configs/distillkit/vpmoe_distill_1.yaml
```

## Notes

- `docker/compose.yml` mounts a host directory **sibling to the repo** as `/data`
  in the container. If you want to control that mount, adjust the compose file.
- Outputs land under `/data/distill_runs/...`.
