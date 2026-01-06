# Data Pipeline (SYNTH KD Dataset v1)

This document is the **source of truth** for the **offline KD dataset pipeline**:

`PleIAs/SYNTH (raw) → Harmony conversations → Megatron IndexedDatasets`

It is intentionally small and explicit. If a change here conflicts with locked decisions (teacher/tokenizer/architecture), update the relevant locked docs first.

## Scope

In scope:
- Building the **training-ready** dataset for Stage 1/2/3 (CE / KD / OPD), using **SYNTH only**.
- Deterministic rendering/tokenization and **loss/span masks** required by the distillation contract.
- Megatron dataset storage conventions and provenance requirements.

Out of scope:
- Teacher scoring (Stage 2A) and OPD orchestration (Stage 3) implementation details.
- Any non-SYNTH datasets.

## Inputs

### Canonical raw-to-Harmony artifact

We treat `data/synth_harmony/` as the canonical “SYNTH→Harmony” conversion artifact.

- Format: Parquet shards, schema `synth_harmony_v1`.
- Each row contains:
  - `messages_json` (string): JSON with a `messages` array (Harmony message dicts).
  - `metadata_json` (string): JSON with stable IDs (includes `synth_id`).
  - Selected metadata columns (e.g. `synth_id`, `language`, `exercise`).

Provenance:
- `data/synth_harmony/manifest.json` is required input and must be recorded in the KD dataset manifest.

## Tokenizer (locked)

All tokenization is with the teacher-matched tokenizer:
- **Tokenizer family:** o200k Harmony
- **Padded vocab size:** `201088`
- **Local tokenizer asset:** `data/tokenizer/o200k_base.tiktoken`
- **sha256:** `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`
- **EOD token:** `<|endoftext|>` id `199999`

No runtime network fetch is permitted for tokenizer assets.

Renderer uses **openai-harmony** with `RenderConversationConfig(auto_drop_analysis=False)`.
Set `TIKTOKEN_ENCODINGS_BASE` to the directory containing `o200k_base.tiktoken`
(or set `O200K_HARMONY_VOCAB_PATH` to the file path).

## Rendering decisions (locked for v1)

We must train on **Harmony format**, and we include **format wrapper tokens** in the supervised spans.

### 1) Preserve `system` and `developer` messages (context-only)

For SYNTH KD training, we **preserve** any `system` and `developer` messages present in `messages_json`, and render them with the official Harmony renderer.

They are treated as **prompt context only**:
- They are included in `tokens`.
- They are masked out of the loss (`loss_mask=0`, `span_id=0`).

Rationale:
- This matches common chat-pretraining conventions: system/developer content conditions behavior but is not a supervised target.
- For GPT‑OSS/Harmony specifically, we want maximum fidelity to the format the teacher was trained on (no silent prompt distribution shift).

If we later decide the scaffold is truly constant and too expensive, we can move it to a **train-time injected prefix**. That is a spec change and must update this doc and the dataset manifest/provenance.

### 2) Use Harmony wrappers and channel markers

We include Harmony wrappers and channel markers (e.g. `<|start|>`, `<|message|>`, `<|channel|>analysis`, `<|end|>`, `<|return|>`) as actual tokens in the sequence, and they participate in span classification (below).

### 3) Training-style final message terminator

When the last assistant message is `channel=final`, we terminate it with `<|return|>` (training-style), not `<|end|>`.

This matches the Harmony training convention described in `ref/harmony/docs/format.md`.

### 4) Per-conversation EOD

After each serialized conversation, append a single EOD token (`199999`) as a document boundary, masked from loss.

## Supervision signals (locked for v1)

Each produced training sequence has **three aligned arrays**:

1) `tokens` (int32): token ids
2) `loss_mask` (uint8): prompt masking
3) `span_id` (uint8): reasoning vs final span labeling

### `loss_mask`

`loss_mask[t] ∈ {0,1}` applies to **labels**, not input tokens.

- `loss_mask[t] = 1` for assistant tokens only (reasoning + final), including wrappers.
- `loss_mask[t] = 0` for `system`/`developer`/`user` tokens and the appended EOD token.

Alignment rule:
- If we train with `labels = tokens[1:]`, then we store `loss_mask_for_labels = loss_mask_tokens[1:]` in the dataset sample.

### `span_id`

`span_id[t] ∈ {0,1,2}` applies to **labels**, not input tokens.

- `0`: ignore / prompt / non-assistant (`system`/`developer`/`user`)
- `1`: assistant reasoning span (Harmony `channel=analysis`)
- `2`: assistant final span (Harmony `channel=final`)

Wrappers are included: wrapper tokens inside an assistant message inherit that message’s span id.

## Output dataset layout (Megatron conventions)

We write **Megatron Core** `IndexedDataset` files (`.bin/.idx`) using `IndexedDatasetBuilder`.

Canonical root:
- `data/megatron/synth_kd_v1/`

Under it:
- `train/`
- `valid/`

Each split contains **4 shards**, aligned with the input `data/synth_harmony` shards:
- `shard_00` … `shard_03`

Each shard writes three dataset prefixes:
- `.../train/shard_00_tokens.{bin,idx}`
- `.../train/shard_00_lossmask.{bin,idx}`
- `.../train/shard_00_span.{bin,idx}`

Same for `valid/`.

Dtypes:
- `tokens`: `int32`
- `lossmask`: `uint8`
- `span`: `uint8`

Hard requirement:
- For each shard, the three datasets must have identical sequence boundaries and lengths.

## Deterministic split (locked for v1)

We split by stable ID:
- key: `synth_id`
- method: deterministic hash
- `valid_fraction = 0.001` (**0.1%**)

This produces a stable, reproducible holdout while keeping eval cost bounded.

## Training losses (span-normalized; locked)

All stages use span-normalized mixing:

- `L = w_r * mean(loss | span_id=1) + w_f * mean(loss | span_id=2)`

Defaults:
- `w_r = 1.0`
- `w_f = 1.0`

We always log:
- per-span token counts
- per-span loss values
- effective weights

## Smoke vs full builds (single pipeline)

We keep one canonical build pipeline. For early integration and debugging, we also support a **smoke build** that uses the same code path but caps work (e.g., max rows/tokens) and writes to a different output root (e.g. `data/megatron/synth_kd_v1_smoke/`).

No “second stack”: smoke is just a constrained invocation of the same builder.

## Provenance requirements (must log)

Every produced dataset root (`.../synth_kd_v1/`) must include a machine-readable manifest recording:
- git SHA
- config hash (dataset builder config)
- tokenizer asset path + sha256
- input `data/synth_harmony/manifest.json` hash
- split rule and `valid_fraction`
- shard inventory (paths, bytes, rows/seq counts)

## Megatron integration contract

Training must use a dataset implementation that:
- preserves Megatron’s standard sampling/index caching behavior (GPT-style slicing across documents),
- loads `tokens` plus aligned `loss_mask` and `span_id`,
- fails fast on any misalignment,
- exposes the extra tensors so the loss can compute span-normalized mixing.
