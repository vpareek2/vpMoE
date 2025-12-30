

# VPMOE Research Progress: Pre-MoE Ablations & Megatron Setup

**Date:** December 12, 2025
**Project:** ultra-sparse mini-MoE with KD from GPT-OSS (see `docs/overview.md`)

---

## Final Ablation Stack

| Component | Winner | How Decided |
|-----------|--------|-------------|
| **Positional Encoding** | Hybrid: GRAPE-M (local) + GRAPE-A (global) | GRAPE-M beat RoPE; global ablation chose GRAPE-A over NoPE. |
| **FFN** | ReLU² | Ablated vs SwiGLU vs mhffn — better loss + torch.compile compat |
| **Attention** | Hybrid: local sliding-window TPA (W=128) + periodic global GQA (3:1) | Ablated |
| **Optimizer** | NorMuon + Polar Express | 4-way ablation (beat Muon-NS, AdamW, etc.) |
| **Init** | Zero-init output projections | muP-style, confirmed win |
| **Embed LR** | 20x adam_lr | 10x huge win (0.8 val loss drop at 150 iters), 20x sweet spot, 50x too aggressive |
| **Long Context** | YaRN | Prior decision |
| **Router** | DeepSeek-V3 style | Paper-backed, no ablation needed |

---

## Key Discoveries

### Embed LR Multiplier (Biggest Win)
- Default LR severely under-trains embeddings (sparse updates)
- 10x multiplier: val loss 4.6 vs baseline 5.4 at step 150 (0.8 improvement!)
- 20x optimal at 50M scale
- 50x too aggressive
- modded-nanogpt uses 75x at larger scale

### Implementation Details
- `embed_lr` param added to `create_muon_optimizer()` in `optimizers.py`
- Split Adam params into `embed_params` and `other_adam_params`
- LR scheduler handles embed group separately

---

## Framework Decision: Megatron-Core

### Status

This repo previously explored Megatron-Core as a training framework, but we are **not planning to use this Megatron-LM copy** for the main project going forward. Keep this section as historical context only.

### Alternatives Considered
- **TorchTitan**: DeepSeek-V3 in progress on `deepseek-v3` branch, not ready
- **Nanotron**: Limited MoE support

---

## Megatron-LM Setup (Completed)

### Environment
- **Container:** `nvcr.io/nvidia/pytorch:25.04-py3`
- **Dev GPU:** NVIDIA GB10 (DGX Spark, sm121 Blackwell)
- **Python:** 3.12

### Docker Command
```bash
docker run --runtime nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it \
  -v /home/veer/Master/research/vpmoe:/workspace/vpmoe \
  -e PIP_CONSTRAINT= \
  --name megatron-dev \
  nvcr.io/nvidia/pytorch:25.04-py3
```

### Installation
```bash
cd /workspace/vpmoe/Megatron-LM
pip install --no-build-isolation .[mlm,dev]
```

### Verified Versions
| Package | Version |
|---------|---------|
| Megatron Core | 0.16.0rc0 |
| TransformerEngine | 2.10.0 |
| PyTorch | 2.7.0a0+79aa17489c.nv25.04 |

### Container Management
```bash
# Exit
exit

# Restart
docker start -ai megatron-dev
```

---

## Config Files Created

| Config | Purpose |
|--------|---------|
| `train_normuon_pe_zeroinit_embedlr_50m.py` | 10x embed LR test |
| `train_normuon_pe_zeroinit_embedlr20x_50m.py` | 20x embed LR (winner) |
| `train_normuon_pe_zeroinit_embedlr50x_50m.py` | 50x embed LR (too aggressive) |

---

## Next Steps

See `docs/overview.md` for the current project plan and open questions.
