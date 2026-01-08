# vpMoE Eval Suite (v1)

This document defines a **lightweight, deterministic eval suite** that we run after each major checkpoint transition:

`qwen3-0.6B → qwen3-0.6B-o200k → vpDense0-5_28 → vpDense_28 → vpDense_80 → vpMoE`

The (hopelessly optimistic) goal is to keep metrics **as close as possible** across the chain and catch regressions early.

## Design goals

- **Fast**: minutes, not hours (so we actually run it after every step).
- **Deterministic**: fixed prompt sets, fixed sampling, fixed decoding, reproducible reports.
- **Comparable across the whole chain**:
  - Generation-based metrics (accuracy) work across tokenizer changes.
  - Tokenization-dependent metrics (PPL/NLL) are only compared once tokenizers match.
- **Actionable**: per-task deltas, not just one scalar.

## Suite overview

We run two tiers:

1) **CORE-8 (DCLM CORE subset)** — broad, high-signal capability canary (generation-based).
2) **PPL/NLL canary** — fast loss-based regression signals (only when tokenizers are aligned).

### Tier 1: CORE-8 (DCLM CORE subset)

We adopt the **DCLM CORE** eval bundle used by nanochat and run **8 high-signal tasks**.
This gives a broad capability scalar without maintaining our own bespoke benchmark set.

**Eval bundle**

- Source: `eval_bundle.zip` (nanochat’s CORE bundle)
- URL: `https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip`
- sha256 (pinned for v1): `90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200`
- Storage (container-first): download and unzip under the `/data` volume (untracked), e.g. `/data/eval_bundle/`.
- Contents we consume:
  - `eval_bundle/core.yaml` (task config)
  - `eval_bundle/eval_data/*.jsonl` (examples)
  - `eval_bundle/eval_meta_data.csv` (random baselines for centered scoring)

**Task selection (CORE-8)**

These are task labels from `core.yaml` (and their corresponding `dataset_uri`):

- `arc_easy` → `world_knowledge/arc_easy.jsonl`
- `arc_challenge` → `world_knowledge/arc_challenge.jsonl`
- `hellaswag` → `language_understanding/hellaswag.jsonl`
- `lambada_openai` → `language_understanding/lambada_openai.jsonl`
- `boolq` → `reading_comprehension/boolq.jsonl`
- `commonsense_qa` → `commonsense_reasoning/commonsense_qa.jsonl`
- `piqa` → `commonsense_reasoning/piqa.jsonl`
- `winogrande` → `language_understanding/winogrande.jsonl`

**Metrics**

For each task:

- `accuracy`: fraction correct (task-defined).
- `centered_accuracy`: `(acc - baseline) / (1 - baseline)` using the task baseline from `eval_meta_data.csv`.

Suite-level:

- `core8_centered_mean`: mean of centered scores across the 8 tasks.

**Deterministic sampling**

To keep results stable and cheap:

- Load the full task JSONL.
- Deterministically shuffle with RNG seed `1337`.
- Take the first `max_per_task` examples.

We define two run sizes:

- **core8_smoke**: `max_per_task = 50`
- **core8_full**: `max_per_task = 500`

Default after every step: `core8_smoke`. Run `core8_full` only on “milestone” checkpoints (e.g. after a successful healing run or before starting a long training phase).

**Decoding**

- Use the task’s ICL formatting rules (`num_fewshot`, `continuation_delimiter`, task type).
- Deterministic inference (no sampling).

### Tier 2: Loss-based canaries (PPL/NLL)

Loss-based metrics are fast and sensitive, but they depend on tokenizer alignment.

We use them only when the evaluated checkpoints share a tokenizer.

**2A) General-text stability**

- Dataset: `healing_text_v1` valid split (see `docs/healing_data.md`).
- Metric: `loss` / `ppl` (standard Megatron validation).
- Used for: regressions introduced by architecture switches, optimizer toggles, kernel changes.

**2B) SYNTH span-aware canary (optional, once Stage 1 exists)**

- Dataset: a pinned SYNTH holdout subset.
- Metric: `final-only NLL` (and optionally `reasoning-only NLL`) using `loss_mask` + `span_id`.
- Used for: catching “final answer got worse” separately from “reasoning got longer”.

## Reporting format (v1)

Every run writes one small JSON report artifact with:

- `checkpoint_id` (string; e.g. `vpDense0-5_28.compat_healed`), plus path/tag if relevant
- `git_sha`
- suite config (`core8_smoke` vs `core8_full`, seeds, decode settings)
- per-task metrics and deltas vs the previous checkpoint in the chain
- suite-level rollups

We treat this report as the canonical “did we regress?” output for each step.

**Default report location (tracked):**

- `reports/eval/core8/` (JSON reports are committed to git).

## Canonical entrypoints

- Fetch bundle: `python scripts/eval/fetch_core_bundle.py`
- Run eval (container-first wrapper): `scripts/eval/run_core8.sh`

## Pass/fail policy (v1)

For the early rung chain (vocab surgery + weight surgery + depth changes), we expect CORE-8 to
drop sharply and stay depressed until later “healing” / midtraining phases. So in v1 we treat
CORE-8 as **tracking-only**:

- Always print and persist `core8_centered_mean` and per-task metrics.
- Print deltas vs the previous rung for visibility.
- **No hard fail and no regression thresholds** until we have noise estimates and a stable baseline.

## Notes on comparability across the chain

- CORE-8 is generation-based and remains meaningful across tokenizer changes (e.g. `qwen3-0.6B` → `qwen3-0.6B-o200k`).
- PPL/NLL is **not** comparable across tokenizers; we only compare PPL after tokenizers match.
