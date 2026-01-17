# distillation_long_ctx_2 (TODO / parking lot)

This directory is reserved for the **long-context distillation phase** (stage 2).

## Why this exists

We downloaded and inspected **`nvidia/Nemotron-Competitive-Programming-v1` (InfiniByte)** and found that it is *not* a good fit for the 8k Phase 1 dataset:

- The dataset is structured as 2 messages (`user`, `assistant`), where the assistant includes both:
  - `reasoning_content` (long chain-of-thought)
  - `content` (final answer)
- In a 20k-row sample (tokenized with `o200k_base` as an approximation), `assistant_reasoning` was very large:
  - p50 ≈ 10k tokens, p90 ≈ 20k tokens, max ≈ 32k tokens
- Therefore, if we keep reasoning traces, most examples will exceed `max_seq_len=8192` and get dropped.

## Action item

Circle back after Phase 1 to decide how to use InfiniByte:

1) **Long-context stage** (preferred): raise `max_seq_len` (e.g., 32k/64k) and keep `reasoning_content` + `content`.
2) **Phase 1 compatibility**: drop `reasoning_content` (final-only) or aggressively truncate it (not recommended unless we must).

## Current stance

Treat competitive programming (especially with explicit reasoning traces) as **long-context curriculum**, not Phase 1.

## Nemotron-Math-v2 (high / long trajectories)

Nemotron-Math-v2 “high” traces also belong here (not Phase 1):

- `data/high.part_00.jsonl`, `data/high.part_01.jsonl`, `data/high.part_02.jsonl` have very long `assistant.reasoning_content` (often 10k–100k+ tokens).
- For Phase 1 (8k), use `data/low.jsonl` (or `data/medium.jsonl`) instead; for long-context training, prefer “high” (and optionally mix in some “medium”).

TODO before building the long-ctx math stage:
- Confirm the “high” files have a stable per-row identifier we can carry through preprocessing (in our EDA sample, `uuid` was missing/`None`). If needed, derive an ID deterministically (e.g., hash of `(problem, messages)`).
