# Architecture of vpMoE

## Locked-In Baseline (Current)

This repo is now treating the following as the **baseline configuration** to build around.

### Model Hyperparameters

- Depth: **80** layers (**1** dense + **79** MoE)
- Hidden size: **1024**
- Attention: **8** heads, **GQA** with **2** KV heads (`num_query_groups=2`), `head_dim=128`
- Norm: **RMSNorm** (pre-norm), plus **QK-norm**
- Softmax/logit bias: **learnable per-head softmax offset** (GPT-OSS-style)
- Local/global attention schedule: **3:1** local:global
- Local attention: sliding-window **TPA**, window **128**
- Global attention: full causal **GQA**
- Positional encoding: **GRAPE-M** (local) + **GRAPE-A** (global, **ALiBi special case**)

### MoE Hyperparameters

- Experts: **64**
- Routing: **topk=2**
- Shared expert: **1 shared expert** (always-on), `shared_expert_size=512`
- Dense warmup: first layer uses dense FFN; remaining layers use MoE FFN

### Tokenizer / Vocab

- Vocab size: **201088** (padded o200k Harmony)
- Untied embeddings (`untied_embeddings=True`)

### Parameter Budget (from `experiments/param_count.py`)

- Total params: **~6.02B**
- Active params (all): **~883.8M**
- Active params (non-embed): **~472.0M**

## Archive

Older exploratory writeups and ablation notes live under `docs/research/README.md`.
