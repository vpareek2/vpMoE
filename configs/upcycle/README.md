# Upcycle configs (Milestone M0)

Canonical registry: `configs/upcycle/objects.toml`.

This folder contains the single “source of truth” configs for the upcycling path:
- `qwen3-0_6B` → `qwen3-0_6B-o200k`
- `qwen3-0_6B-o200k` → `vpDense0-5_28` (tensor surgery)
- `vpDense0-5_28` → `vpDense` (depth expansion)
- `vpDense` → `vpMoE` (dense→MoE upcycle)

Notes:
- These `.toml` files are config **artifacts**; the execution harness/wrapper that consumes them is implemented in later milestones.
- Outputs must be written under `weights/upcycle/` (gitignored) and must include provenance metadata.
- To materialize `qwen3-0_6B-o200k` today, use `scripts/upcycle/qwen3-0_6B_to_o200k.sh` (OMP via `ref/mergekit`).
