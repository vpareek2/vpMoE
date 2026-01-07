# Healing Data Plan (FineWeb‑Edu → Megatron)

This document defines the **reproducible, container-first** plan for the **healing** dataset used during
upcycling warmstarts (e.g., post-surgery stabilization for `vpDense0-5_28.compat` and
`vpDense0-5_28.real`).

This is intentionally **not** the KD dataset pipeline (SYNTH). KD remains specified in
`docs/data_pipeline.md`.

## Scope

In scope:
- A small, high-quality **general text** dataset for **stability/healing** runs.
- Deterministic **materialization**, **manifesting**, and **Megatron dataset build**.

Out of scope:
- SYNTH conversion/cleaning.
- Teacher scoring / OPD.

## Source (locked for v1)

- Dataset: `HuggingFaceFW/fineweb-edu` (Hugging Face Datasets)
- Split: `train`
- Revision pin: **exact HF commit hash** (recorded in the manifest; never “latest”)

We download/materialize in a single explicit step and then train purely from local files.
No training-time network fetch is permitted.

## Target size

- Target: **~200M tokens** measured under the **o200k Harmony** tokenizer.
- We may materialize more raw documents than needed; the token budget is enforced at the
  Megatron dataset build step and recorded in the output manifest.

Rationale:
- 200M is enough for short stabilization runs without creating a second long-pretraining track.

## Language policy

- **English-only if easy**:
  - If FineWeb‑Edu provides a reliable language field, filter `language == "en"` (or equivalent).
  - If no reliable field exists, **do not** introduce a heuristic language classifier in v1.
    Instead: keep all documents and record `english_filter_applied=false` in the manifest.

## Data layout (container-first)

All large artifacts are stored under the container-mounted `/data` volume.

### Raw/materialized corpus (untracked)

- Root: `/data/corpora/fineweb_edu/<hf_rev>/`
- Format: newline-delimited JSON (`.jsonl.zst` or `.jsonl`) with at least:
  - `doc_id` (stable string; if absent, we deterministically derive one)
  - `text` (string)

### Repo-tracked manifests (small, committed)

- `data/manifests/fineweb_edu_<hf_rev>.json`
  - Records HF dataset, revision, split, materialization params, file inventory + hashes.
- `data/manifests/healing_text_v1.json`
  - Points at a concrete set of materialized shard files (subset selection), plus invariants.

### Megatron outputs (untracked)

- Root: `data/megatron/healing_text_v1/`
- Includes:
  - `train/` and `valid/` splits
  - `manifest.json` capturing provenance (git sha, config hash, tokenizer hash, input manifest hash)

## Normalization (v1)

Keep text as close to source as possible:
- Remove ASCII NUL (`\u0000`) if present (hard error if pervasive).
- Do not apply aggressive Unicode normalization or punctuation rewriting in v1.

All normalization decisions must be explicit and recorded in the input manifest.

## Splits

- `valid_fraction = 0.10` (10%), deterministic by stable document id.
- Hash function: `sha256(doc_id)` and compare to threshold.
- The split rule and fraction are recorded in both the input and output manifests.

## Tokenization and tokenizer pinning (locked)

All tokenization is with the pinned teacher-matched tokenizer:
- Tokenizer family: **o200k Harmony**
- Asset: `data/tokenizer/o200k_base.tiktoken`
- sha256: `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`

## Build invariants (must hold)

For each split/shard:
- `tokens.{bin,idx}` uses `int32`
- Optional auxiliary arrays (if added later) must be perfectly aligned to `tokens`.
- Builder must be deterministic for a fixed manifest + tokenizer asset.

## Canonical workflow (design only; do not run during planning)

1) **Materialize** FineWeb‑Edu at a pinned HF revision into `/data/corpora/fineweb_edu/<hf_rev>/...`,
   and write `data/manifests/fineweb_edu_<hf_rev>.json`.
2) Select a subset (if desired) and write `data/manifests/healing_text_v1.json`.
3) Build Megatron datasets into `data/megatron/healing_text_v1/` with a build config that records:
   - token budget target (200M o200k tokens)
   - valid_fraction (10%)
   - tokenizer asset path + hash
   - input manifest hash

## Why 10% valid is acceptable here

Healing runs are short and the goal is regression detection (stability/format/numeric sanity), not
benchmarking absolute capability. A larger valid split makes early regressions cheaper to detect.
