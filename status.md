# vpMoE status (Jan 2026)

## Where we are
- **Container path is stable**: we have a CUDA 12.8 image for B200/Blackwell and have confirmed basic CUDA + torch setup.
- **Core distillation flow works end‑to‑end** in small diagnostics:
  - in‑mem updates happen (weights change during a single step)
  - checkpoints contain updated trainable weights and frozen weights stay unchanged
- **Tiny overfit sanity** works (CE‑only and CE+KL configs can overfit on tiny datasets).
- **Primary dataset** is **already built** and **must be pulled from HF**, not re‑mixed locally:
  - `veerpareek/vpmoe-phase1-mix-4k-665m`
  - download script: `scripts/download_phase1_mix.py`

## Known friction points (resolved/understood)
- **B200 flash‑attn incompatibilities** required CUDA 12.8; resolved with the CUDA 12.8 image.
- **Loss “spikiness”** in short runs is expected at tiny batch sizes / tiny dataset; does not indicate a bug.
- **Eval loss sometimes flat** when the dataset is ultra‑small; use overfit or in‑mem step tests for correctness.

## Immediate next steps
1) **Cloud smoke on 8×H100 (or 8×B200 if chosen)**
   - pull container
   - pull dataset from HF
   - run a 50–200 step smoke with real config
2) **Decide microbatch + grad‑accum**
   - run `scripts/autotune_microbatch.py` to find max per‑GPU microbatch at seq len 4096
   - choose target global batch (256/512) and compute grad‑accum
3) **Lock phase‑1 run config**
   - freeze MLPs, train attention + router
   - loss mix: CE + KL + HS cosine (Arcee mix)
   - warmup ratio ~0.025–0.05, grad clip ~0.5 (tighten to 0.3 if unstable)
   - checkpoint cadence + eval cadence appropriate for long run
4) **Full phase‑1 distillation run**
   - 665M tokens, stop early if metrics regress or instability appears

## Quick commands
### Pull dataset (HF snapshot)
```
python scripts/download_phase1_mix.py \
  --out-dir /data/distillation_1/phase1_mix_4k_665m
```

### Microbatch autotune
```
python scripts/autotune_microbatch.py \
  --seq-len 4096 \
  --target-gbs 256 512
```

### Smoke run (example)
```
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m distillkit.main configs/distillkit/vpmoe_distill_1_smoke_8xh100.yaml
```

## Pending decisions
- **H100 vs B200**: choose based on tokens/sec/$ (B200 only if >1.7× faster).
- **Final global batch** once microbatch limit known on target hardware.
