# Value Residual Ablation

This directory holds the value-residual (v1 mix) ablation setup.

## Dataset

- Path: `data/megatron/vr_ablation/synth_kd_v1_100m`
- Tokens: ~103.9M total (train + valid)
- Build command (already run):

```bash
docker compose -f docker/compose.yml run --rm vpmoe bash -lc "uv run scripts/datasets/build_synth_kd_v1.py \
  --tokenizer-model data/tokenizer/o200k_base.tiktoken \
  --output-dir data/megatron/vr_ablation/synth_kd_v1_100m \
  --valid-fraction 0.001 \
  --max-rows 120000 \
  --log-every 10000 \
  --overwrite"
```

## Runs

- Baseline: `vr_ablation/run_baseline.sh`
- Value residual: `vr_ablation/run_value_residual.sh`

Both scripts run inside the repo container and use:
- `num_layers=8`, `hidden_size=256`, `ffn_hidden_size=1024`, `num_attention_heads=4`
- `seq_length=1024`, `global_batch_size=32`, `train_iters=305` (~10M tokens)

Adjust via env vars (see scripts for defaults).
