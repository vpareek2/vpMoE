# PleIAs/SYNTH → DistillKit-ready dataset (source of truth)

This document defines the **canonical preprocessing contract** for turning the local PleIAs/SYNTH Parquet shards into a dataset that can be consumed by **DistillKit** in the same “teacher-forced distillation over existing sequences” style used in Arcee’s example runs.

For the general “how distillation works” mechanics (online vs offline, teacher-forced vs rollout), see `src/data/distillation_mechanics.md`.

## Goals

- Use **PleIAs/SYNTH** as the primary distillation corpus.
- Keep SYNTH completions **as-is**:
  - `synthetic_reasoning` (may be empty for some exercises)
  - `synthetic_answer`
- Produce compact, training-ready rows:
  - **token IDs** + **assistant-only label mask**
  - minimal metadata for stratified sampling/debug
- Avoid large, redundant “audit tables” in the derived dataset (raw Parquet remains the source of truth).

## Non-goals (for this phase)

- “Perfect” reasoning style or post-training polish (we will do real post-training later).
- Rewriting/normalizing `synthetic_reasoning` (keep verbatim).
- Encoding SYNTH `constraints` into the prompt (skip entirely; see filtering).

## Input: upstream columns (PleIAs/SYNTH)

We assume each row provides at least:

- `synth_id` (string; stable upstream ID)
- `language` (string)
- `exercise` (string)
- `model` (string; generator model name)
- `query` (string)
- `query_seed_text` (string; always present in observed shards)
- `synthetic_reasoning` (string; may be `""`)
- `synthetic_answer` (string)
- `constraints` (string; used **only** for filtering, not rendered)

## Output: training dataset schema (canonical)

One record per **unpacked example** (i.e., one conversation). We intentionally keep this schema small.

**Storage layout**

```
<output_dir>/
  train/part-00000.parquet
  validation/part-00000.parquet
  manifest.json
```

**Canonical build command**

```bash
python3 scripts/build_synth_distill.py \
  --input-dir /path/to/pleias_synth \
  --output-dir /path/to/output/synth_distill
```

```json
{
  "id": "pleias_synth:<synth_id>",
  "source": {
    "dataset": "PleIAs/SYNTH",
    "synth_id": "<synth_id>",
    "exercise": "<exercise>",
    "language": "<language>"
  },
  "input_ids": [ ... ],
  "labels": [ ... ],
  "attention_mask": [ ... ],
  "spans": {
    "assistant_token_start": 123,
    "analysis_token_count": 456,
    "final_token_count": 78
  }
}
```

### Semantics

- `input_ids`: token IDs for the full rendered conversation (**system + user + assistant**).
- `labels`: same length as `input_ids`.
  - All tokens **before** assistant output are `-100`.
  - All assistant tokens are copied from `input_ids`.
  - This makes DistillKit/TRL compute loss only on assistant tokens (teacher-forced distillation on the response).
- `attention_mask`: list of `1`s (same length as `input_ids`) for unpacked examples.
- `spans`:
  - `assistant_token_start`: index into `input_ids` where assistant output begins (first token that is not masked).
  - `analysis_token_count`: number of assistant **analysis** tokens (0 if no analysis).
  - `final_token_count`: number of assistant **final** tokens.

## Canonical prompt rendering (what becomes `input_ids`)

We render a **single-turn** conversation.

### User content template

We always include the seed text as reference/context.

```
{query}

Reference:
{query_seed_text}
```

Notes:
- We do **not** include URLs/licenses/etc. in the prompt for this phase.
- We do **not** include `constraints` in the prompt for this phase.

### Assistant content

- If `synthetic_reasoning.strip()` is non-empty:
  - assistant `analysis` = `synthetic_reasoning` (verbatim)
  - assistant `final` = `synthetic_answer` (verbatim)
- Else:
  - assistant `final` only
  - `analysis_token_count = 0`

### Tokenization / chat format

- Use the **same tokenizer/chat template** that DistillKit will use at training time (i.e., the model tokenizer).
- The resulting tokenization must be compatible with GPT‑OSS Harmony expectations.

Implementation detail (current implementation):
- Tokenize the prompt with `render_conversation_for_completion(..., Role.ASSISTANT)` to define `assistant_token_start`.
- Tokenize the full example with `render_conversation_for_training(...)` using `auto_drop_analysis=false` so analysis tokens are kept.

Implementation detail is intentionally left to the preprocessing script, but the contract is:
- The rendered tokens must preserve an explicit distinction between assistant analysis vs final **if** the tokenizer/template supports it.
- If not, we still keep `analysis_token_count`/`final_token_count` consistent with whatever representation we emit.

## Implementation overview (build_synth_distill.py)

The builder is intentionally minimal and deterministic:

1) **Scan pass (counts)** — apply the quality filters and collect counts by `exercise` and `language`.
2) **Sampling plan** — compute keep rates to hit targets:
   - memorization keep rate to cap `exercise==memorization` at ≤70%
   - non‑EN keep rate to cap non‑English at ~25–30%
3) **Build pass** — stream rows and, for each row that passes quality filters:
   - apply deterministic hash sampling for memorization and non‑EN
   - render the prompt (`query` + `Reference:` + `query_seed_text`)
   - render Harmony tokens with assistant `analysis` (if present) then `final`
   - compute `labels` mask and `spans`
   - sanitize invalid Unicode (NULs / surrogates) before Harmony rendering
   - drop if `len(input_ids) > max_seq_len`
   - assign split by deterministic hash (99/1)
4) **Write outputs** — sharded Parquet for train/validation + `manifest.json`.

`manifest.json` records config, sampling rates, token length stats, and a `git_sha` for provenance.
If Harmony rendering fails for some rows, they are dropped and counted; you can capture a sample of failures
with `--encoding-errors-log`.

## Filtering policy (cheap quality wins)

We do **not** render `constraints`, but we may use it as a filter signal.

Drop rows if any of the following are true:

1. `synthetic_answer.strip()` is empty.
2. `constraints` contains `result:Failed after maximum attempts` (case-insensitive).
3. `constraints` contains a `result:` value that is numerically/MCQ-inconsistent with the first-line answer.
   - This is a small slice but catches “obviously broken label” cases.

We do **not** otherwise “fix” answers.

## Exercise/language mixture (dataset selection)

SYNTH can be highly imbalanced by `exercise` within individual shards.

Decisions:
- Keep the dataset **English-heavy** with a small multilingual tail.
  - Target: **~70–75% English** (cap non‑EN to ~25–30%).
- Cap the memorization skew:
  - Target: **≤70%** `exercise=="memorization"` after sampling.
- Use a deterministic global downsample (`--global-keep`) when we want to hit a fixed token budget without changing shard selection.

## Train/val split + max length

- **Split:** 99% train / 1% validation (deterministic hash on `synth_id`).
- **Max sequence length (phase 1):** 8192 tokens.
  - If an example exceeds this, it is **dropped** for phase 1.

## Packing strategy (Arcee-style intent)

DistillKit/TRL operates over token sequences and can pack examples for throughput.

We will follow the “Arcee intent”:
- train on short contexts first, then longer contexts later (curriculum in `src/data/data.md`),
- rely on a single canonical preprocessing pipeline,
- keep this file as the spec for how examples are rendered/masked.

We output **unpacked examples** and let training-time packing handle throughput. This keeps
the dataset compact and avoids prepack-specific attention-mask logic in preprocessing.

## Why SYNTH works with DistillKit (and why this mirrors Arcee)

Like Arcee’s example (e.g. distilling on Tulu-style sequences), we are:
- not requiring that dataset completions were generated by the teacher,
- using the teacher as a **forward-pass signal source** over the provided tokens.

This is “teacher-forced distillation on an off-policy sequence distribution,” which is exactly what DistillKit implements.
