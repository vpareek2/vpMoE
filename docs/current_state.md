# Current State (as of 2026-01-07)

This doc captures the **current repo state** and the **next immediate step** for the Qwen3→vpDense→vpMoE upcycling path.

## Locked baseline (source of truth)

- Architecture + invariants: `docs/architecture.md`
- Distillation plan: `docs/distillation_strategy.md`
- Locked baseline highlights:
  - Teacher: GPT‑OSS‑120B only
  - Tokenizer: o200k Harmony (padded vocab **201088**)
  - Student attention geometry: **8q / 2kv**, `head_dim=128`
  - MoE target: **E=256**, **topk=4**, **shared expert size=512**

## What we have now

### Converted checkpoint (weight surgery output)

- Produced `vpDense0-5_28` at `weights/upcycle/vpDense0-5_28`.
- Converter: `scripts/upcycle/convert_qwen_to_vpdense0_5_28.py`
- Mapping summary:
  - Embeddings: Qwen3‑0.6B‑o200k → vpDense (untied output head in target).
  - Attention surgery: head mean-pooling (Q 16→8, KV 8→2), `head_dim=128`.
  - QK-norm: copies `q_norm.weight` / `k_norm.weight`.
  - Softmax offset: target-only learnable offsets are zeroed.
  - FFN surgery: **method A** (select top‑512 intermediate dims by down‑proj L2 norm; copy corresponding rows/cols).
  - Activation/stack: compat-mode assumptions (RoPE/full attention) for immediate post-surgery healing.
  - Provenance: written to `weights/upcycle/vpDense0-5_28/provenance.json`.

### Non-homogeneous distributed checkpointing (required)

The converted model uses **value residual mixing**. Layer 1 has no `value_residual_logit`, so distributed checkpointing must be **non‑homogeneous**.

- New Megatron arg: `--hetereogenous-dist-checkpoint` (wired in `vpmoe/Megatron-vpmoe/megatron/training/arguments.py`).
- Runbook note: `vpmoe/upcycle/README.md`
- Upcycle configs set:
  - `configs/upcycle/vpDense0-5_28.compat.toml`
  - `configs/upcycle/vpDense0-5_28.real.toml`

### Healing data (fineweb-edu) + Megatron dataset build

We created a reproducible “healing text” dataset for post-surgery stabilization:

- Design doc: `docs/healing_data.md`
- Materialization (HF streaming → JSONL shards + manifest):
  - Script: `scripts/datasets/materialize_fineweb_edu.py`
  - Outputs: `data/corpora/fineweb_edu/...` and a pinned manifest under `data/manifests/`
- Megatron IndexedDataset build:
  - Script: `scripts/datasets/build_healing_text_v1.py`
  - Output dataset: `data/megatron/healing_text_v1/{train,valid}/healing_text_v1_tokens.{bin,idx}`
  - Important fix: we now call `end_document()` per record so the dataset index is valid.

### Warm-up smoke (wired + verified)

We wired a small “compat healing” smoke that:
1) loads `vpDense0-5_28`,
2) builds GPTDataset on `healing_text_v1`,
3) runs a couple training steps.

- Script: `scripts/smoke_healing_text.sh`
- Notes:
  - Uses `--hetereogenous-dist-checkpoint` to match the checkpoint sharding requirements.
  - Uses `--finetune` because the converter output is model-only (no training iteration metadata / args).
  - Disables bias+GELU fusion (ReLU²) and disables persist layer norm to match the local transformer path.
  - Saved smoke checkpoint: `data/megatron/healing_text_v1_smoke` (torch_dist).

## Registry/config status

- Upcycle object registry: `configs/upcycle/objects.toml` (Qwen3‑0.6B → qwen3‑o200k → vpDense0‑5_28 → vpDense → vpMoE).
- Student dense reference config (28-layer): `configs/vpdense_28.toml`.

## Next immediate step

Run a **real compat healing run** (not just smoke) on `healing_text_v1`, starting from:
- load: `weights/upcycle/vpDense0-5_28`
- mode: compat (RoPE/full attention; reduced shock)

Operationally:
1) Do a bounded run (e.g. target ~1 epoch over the healing tokens) to stabilize post-surgery behavior.
2) Save a training checkpoint (torch_dist) that includes training metadata and can resume normally.
3) After that, drop `--finetune` and do standard resume semantics for subsequent runs.
4) Only then consider switching compat → real (GRAPE/TPA/3:1 schedule) and continue acclimation.

## Evals (recommended invariant after every train)

We should have a small, deterministic **core eval** that runs after every healing/training run to detect regressions early.
This should be:
- **Fast** (minutes, not hours), so it’s always run.
- **Pinned** (data/prompt manifest + tokenizer hash + exact decoding settings).
- **Stable metrics** (no “maybe it got better” ambiguity).

Recommended minimal core eval set:
1) **Held-out perplexity**: loss/perplexity on a pinned validation shard (e.g. `healing_text_v1` valid split).
2) **Prompt micro-suite**: a small set of curated prompts across:
   - basic math (short arithmetic/algebra),
   - factual recall (high-signal trivia),
   - reasoning with short chain (but scored on final answer only),
   - simple code/format following.
   Store as a repo-local JSONL/Harmony file with a manifest and expected answer checks (exact match / regex).
3) **Sanity invariants**: tokenizer round-trip, special token handling, and Harmony format validity checks (when relevant).

Next follow-up (not implemented yet): define a single canonical entrypoint (e.g. `scripts/eval/run_core_eval.py`) + a pinned `configs/eval/core_eval.toml` that runs the above and writes a versioned report alongside the produced checkpoint.

If you want this to be entirely “one canonical way to run it”, the next follow-up is to encode the healing run (iters/tokens, LR schedule, save cadence, logging) into a single `configs/upcycle/vpDense0-5_28.compat_heal.toml` + a single `scripts/upcycle/run_megatron.py` entrypoint that consumes it.
