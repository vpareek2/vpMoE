# Architecture of vpMoE

## Locked-In Config (Current)

This repo is now treating `configs/vpmoe.toml` as the **locked configuration** to build around.

### Model Hyperparameters

- Depth: **80** layers (**1** dense + **79** MoE)
- Hidden size: **1024**
- Attention: **8** heads, **GQA** with **2** KV heads (`num_query_groups=2`), `head_dim=128`
- Residual connections: **standard residual** (single residual stream; no mHC)
- Value residual mixing: **enabled** (per-layer learnable `lambda`; for layers >1, `v = lambda * v + (1 - lambda) * v1` where `v1` is layer-1 values)
- Norm: **RMSNorm** (pre-norm), plus **QK-norm**
- MLP activation: **ReLU²** (squared ReLU)
- Softmax/logit bias: **learnable per-head softmax offset** (GPT-OSS-style)
- Linear biases: **enabled** (GPT‑OSS parity), including attention projections and MLPs
- Local/global attention schedule: **3:1** local:global
- Local attention: sliding-window **TPA**, window **128**
- Global attention: full causal **GQA**
- Positional encoding: **GRAPE-M** (local) + **GRAPE-A** (global, **ALiBi special case**)

### MoE Hyperparameters

- Experts: **256**
- Routing: **DeepSeek‑V3 router** (Megatron-Core)
  - **topk=4**
  - Group-limited routing: `num_groups=8`, `group_topk=4`
  - Score function: `sigmoid`, `topk_scaling_factor=2.5`
  - Load balancing: `seq_aux_loss` with `aux_loss_coeff=1e-4`
  - Expert bias: enabled, `bias_update_rate=1e-3`
  - Router dtype: fp32
- Shared expert: **1 shared expert** (always-on), `shared_expert_size=512`
- Dense warmup: first layer uses dense FFN; remaining layers use MoE FFN
- Routed expert FFN width: `ffn_hidden_size=128`

### Tokenizer / Vocab

- Vocab size: **201088** (padded o200k Harmony)
- Untied embeddings (`untied_embeddings=True`)

### Parameter Budget (from `scripts/param_count_megatron.py`)

- Total params: **~6.02B**
- Active params (all): **~780.0M**
- Active params (non-embed): **~368.2M**

To recompute parameter counts, run `python3 scripts/param_count_megatron.py --config configs/vpmoe.toml`.

## Archive

Older exploratory writeups and ablation notes live under `docs/research/README.md`.
