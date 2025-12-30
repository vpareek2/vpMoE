# Positional Encoding Plan (Minimal Ablations)

This repo already supports **GRAPE‑M** (`--position-embedding-type grapem`) in Megatron-Core.
We want a strong hybrid-attention baseline (local sliding-window + periodic global GQA) while running as few ablations as possible.
Our chosen local mechanism is sliding-window TPA; in Megatron-Core positional-encoding runs, treat local as standard SWA until TPA is ported.

## Current Constraints / Priorities

- **Budget-constrained**: minimize ablations; prefer one “good” baseline and only one follow-up.
- **Hybrid attention**: local sliding-window (W=`128`) and a **3:1** local:global ratio (periodic full-attention GQA).
- **Router**: use a **DeepSeek‑V3 style** router (aux-loss-free + expert-bias routing) rather than
  classic Switch aux loss.
- **Precision**: BF16 everywhere (router in FP32 if needed for stability).
- **Data**: data pipeline is still being wired up; for now this doc captures the architectural plan.

## Recommended Baseline: GRAPE‑M Local + GRAPE‑A Global

**Goal:** keep local layers position-aware and stable while giving global layers a strong,
cache-friendly relative bias (ALiBi-style) for long-range synchronization/mixing.

### How to implement today (no code changes)

1) Use GRAPE‑M as the base positional type:
- `--position-embedding-type grapem`
- `--grapem-learnable-freq` (optional; if you already found it wins)

2) Select global layers and turn *off* rotary on those layers:
- `--no-rope-freq <pattern>`

Where `<pattern>` is chosen so that the layers you designate as “global full attention” are also
the layers that skip rotary (NoPE on global layers).

Notes:
- In this repo, we apply **GRAPE‑A (ALiBi slopes)** only on the non-window (global) layers.

3) Enable GRAPE‑A on global (non-window) layers:
- `--grape-a`

So the full “baseline positional” recipe is:
- `--position-embedding-type grapem --grapem-learnable-freq`
- `--no-rope-freq <pattern> --grape-a`

## Prior alternative (now deprecated in this repo): GRAPE‑M Local + NoPE Global

We previously considered keeping global layers purely position-agnostic (NoPE global). After ablation,
we chose **GRAPE‑A global** instead.

## Router Plan (DeepSeek‑V3 Style)

SonicMoE uses a classic Switch-style aux loss; we do **not** want that.
Megatron-Core already exposes DSv3-ish knobs:

- Aux-loss-free: set `--moe-router-load-balancing-type none`
- Expert bias routing: `--moe-router-enable-expert-bias`
- Bias update rate: `--moe-router-bias-update-rate 1e-3` (DSv3 default)
- Typical supporting knobs:
  - `--moe-router-score-function sigmoid`
  - `--moe-router-pre-softmax` (if using that variant)
  - Optional group-limited routing (`--moe-router-num-groups`, `--moe-router-group-topk`)

## Minimal Ablation Matrix (What we will actually run)

Baseline (selected):
- **Local**: GRAPE‑M
- **Global**: GRAPE‑A / ALiBi-style additive bias (with rotary skipped on global layers)

Everything else stays fixed (optimizer, MoE structure, data schedule).
