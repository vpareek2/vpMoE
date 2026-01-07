# CORE-8 Eval Worklog

Purpose: keep a human-readable trail of CORE-8 evals alongside the JSON reports in `reports/eval/core8/`.

## Conventions
- **Suite**: CORE-8 (see `configs/eval/core8.toml`).
- **Mode**: `smoke` = 50 examples per task.
- **Metric**: `core8_centered_mean` (centered vs DCLM baselines in the bundle).
- **Reports**: JSON artifacts live in `reports/eval/core8/` and are tracked in git.

## Entries

### 2026-01-07

**qwen3-0_6B (HF)**  
- Core-8 centered mean: **0.322840**  
- Mode: smoke  
- Report: `reports/eval/core8/qwen3-0_6B.json`  

**qwen3-0_6B-o200k (HF)**  
- Core-8 centered mean: **0.278103**  
- Delta vs `qwen3-0_6B`: **-0.044737**  
- Mode: smoke  
- Report: `reports/eval/core8/qwen3-0_6B-o200k.json`  
- Notes: HF tokenizer emitted a warning about `fix_mistral_regex=True` (left unchanged).  

**vpDense0-5_28.compat (Megatron, post-surgery, pre-heal)**  
- Core-8 centered mean: **0.045833**  
- Delta vs `qwen3-0_6B-o200k`: **-0.232270**  
- Mode: smoke  
- Report: `reports/eval/core8/vpDense0-5_28.compat.json`  
- Notes:
  - compat eval used `--rope-base 1000000` (matches qwen3 rope_theta).
  - `--dist-ckpt-strictness ignore_all` to tolerate missing `_extra_state` keys.
  - `--finetune` used internally for compat mode to bypass missing iteration metadata.
  - bias gelu/swiglu fusions disabled for squared‑relu.

