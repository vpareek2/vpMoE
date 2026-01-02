# Upcycling Runbook (Qwen3‑0.6B → vpDense0-5_28 → vpDense → vpMoE)

This is the **canonical, followable runbook** for Phase 0 upcycling initialization referenced in `docs/overview.md`.

Goal: produce a strong warmstart checkpoint for the locked vpMoE baseline (`docs/architecture.md`) **without** creating a second pretraining stack.

This runbook intentionally stages “shape/format shocks” so failures are attributable and recoverable.

## Locked invariants (do not change here)

- Student tokenizer: **o200k Harmony**, padded vocab size **201088** (`docs/architecture.md`).
- Teacher family: GPT‑OSS (**120B only**) (`docs/distillation_strategy.md`).
- Student attention geometry: **8 query heads**, **2 KV heads**, `head_dim=128` (`docs/architecture.md`).
- Local/global schedule: **3:1**, local **TPA** window **128**, GRAPE‑M/GRAPE‑A (`docs/architecture.md`).

If any step would require changing these, treat it as a spec change and update the docs first.

## Current repo status (what already exists)

- Megatron path includes:
  - `O200kHarmonyTokenizer` integration and verification (`docs/megatron.md`).
  - GRAPE + TPA plumbing and a functional smoke test (`docs/megatron.md`).
  - Dense→MoE upcycling utilities in Megatron (`vpmoe/Megatron-vpmoe/megatron/core/transformer/moe/upcycling_utils.py`).
- Missing (expected work):
  - Deterministic **Qwen → vpDense** checkpoint converter (vocab transplant + tensor surgery).
  - Deterministic SYNTH → Harmony → Megatron dataset build pipeline (`docs/megatron.md`).

## Checkpoint objects (names we will use)

We treat each named object as a concrete, reproducible artifact (checkpoint + config + provenance).

- Canonical registry: `configs/upcycle/objects.toml`.
- Canonical configs: `configs/upcycle/*.toml`.

- `qwen3-0_6B`: unmodified donor checkpoint (stored locally).
- `qwen3-0_6B-o200k`: donor with **o200k Harmony** embeddings/lm_head compatibility.
- `vpDense0-5_28`: 28-layer dense model (student-shaped), produced via Qwen→student tensor surgery.
- `vpDense`: 80-layer dense model matching our stack (depth-expanded from `vpDense0-5_28`).
- `vpMoE`: 80-layer MoE model (upcycled from `vpDense`) — **the warmstart target**.

Compat vs real is a **mode** of the same checkpoint object (two configs), not a separate checkpoint name:
- `vpDense0-5_28.compat`: closest-to-donor attention/positional settings (reduced shock).
- `vpDense0-5_28.real`: GRAPE + 3:1 schedule + TPA.

## Milestones (artifacts + exit criteria)

### M0 — Targets + configs are unambiguous

Deliverables:
- A single config for each checkpoint object above (including compat vs real).
- A provenance template: git SHA, config hash, tokenizer asset hash, dataset manifest hash.

Exit criteria:
- The “real” configs match `docs/architecture.md` invariants exactly.

### M1 — Tokenizer asset is pinned + regression tests exist

Deliverables:
- Local o200k Harmony tokenizer asset (no runtime network fetch).
- Deterministic regression tests:
  - numeric tokenization/copy prompts (must not explode token counts),
  - Harmony structure validity checks for reasoning/final spans.

Exit criteria:
- Tokenization is deterministic and numeric regressions pass for the target tokenizer.

### M2 — Qwen → vpDense0-5_28 conversion works

Deliverables:
- Converter that produces `vpDense0-5_28` from `qwen3-0_6B`:
  - vocab transplant to o200k Harmony,
  - attention tensor surgery to (8q,2kv,head_dim=128),
  - FFN width surgery to the target intermediate size,
  - deterministic output + strict shape checks.

Exit criteria:
- Checkpoint loads in our Megatron stack and runs a basic forward/loss step.

Notes:
- For training-free vocab transplant, we use OMP via `ref/mergekit` (`mergekit-tokensurgeon`).
- OMP requires a donor model that already uses the target vocabulary/tokenizer (we use `weights/gpt-oss-20b`).
- We patch `ref/mergekit` to avoid hard dependencies on `accelerate`/`peft` for tokensurgeon, and add a minimal architecture definition for GPT‑OSS (`ref/mergekit/mergekit/_data/architectures/gpt_oss.json`).

### M3 — vpDense0-5_28 (compat) healing run is stable

Deliverables:
- Short training run that stabilizes the transplanted model (not a long pretrain).

Exit criteria (minimum):
- no loss blow-ups,
- numeric regression tests still pass,
- Harmony format validity is high on a tiny held-out set (pick a target and enforce it).

### M4 — Morph to `vpDense0-5_28.real` (GRAPE + schedule + TPA)

Deliverables:
- A checkpoint-to-checkpoint config switch from compat → real (no mid-run silent toggles).

Exit criteria:
- GRAPE + 3:1 schedule + TPA run stably on a tiny job.

### M5 — Depth expand `vpDense0-5_28` → `vpDense`

Deliverables:
- Depth-expansion tool that inserts near-identity layers (zero residual contribution initially).

Exit criteria:
- “Before wake-up” forward pass is near-identical on a fixed batch,
- short wake-up run is stable.

### M6 — Upcycle `vpDense` → `vpMoE`

Deliverables:
- Dense FFN → expert replication with symmetry-breaking noise (or other minimal diversification).
- Router init that is near-uniform + load-balance loss enabled.
- Router health logging (expert histogram, collapse detection).

Exit criteria:
- No expert collapse on a small run; auxiliary losses are finite and stable.

### M7 — Hand off to Stage 1 SYNTH warmup (distillation plan)

Deliverables:
- Stage 1 run config (SYNTH CE warmstart) that consumes the warmstart checkpoint.

Exit criteria:
- Training is resumable without silent drift; logs include required provenance.

## Non-goals (to keep this small and sharp)

- Do **not** do “vpMoE28” as a required step. If needed, use it only as an opt-in debug checkpoint to validate MoE/router plumbing early.
- Do **not** introduce a second execution path outside the container (`docs/docker.md` is canonical).
- Do **not** fetch model/tokenizer assets at runtime; everything must be local and hashed.

## Operational notes

- Donor weights are expected at `weights/qwen3-0_6B` (pinned by `configs/upcycle/provenance/qwen3-0_6B.json`).
- Produced checkpoints and run artifacts live under `weights/upcycle` (repo-local, gitignored).
- Prefer failing fast with actionable errors (shape mismatches, tokenizer hash mismatch, missing provenance).
- Every produced checkpoint should embed provenance metadata sufficient to reproduce it months later.
