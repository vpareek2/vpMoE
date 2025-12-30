# vpMoE Project Overview

## Goal

Build an **ultra-sparse mini-MoE** that is competitive with small, modern MoE baselines (e.g. **Arcee Trinity Nano 6B A1B**) while staying within a realistic independent compute budget.

## Core Constraints

- We cannot “win” by general pretraining scale (labs have effectively unbounded compute).
- We need a path that is **data- and signal-efficient**: **knowledge distillation (KD)** + careful dataset curation.
- The model should be **native-reasoning**: reasoning-style training signal is present from the start (not only bootstrapped during post-training).

## Architecture (Locked)

The baseline architecture is considered **locked** and is documented in `docs/architecture.md`.
That doc includes both the final hyperparameters and the parameter budget (total vs activated).

## Tokenizer Choice

We distill from the **GPT-OSS** family, which uses the **o200k Harmony** tokenizer. To avoid tokenizer mismatch during token-level KD and to keep formatting consistent, the student also uses o200k Harmony (padded vocab size in our config: **201088**).

## Teachers

We distill from **GPT-OSS 120B** (≈ **20B active**) only.

Rationale:

- 120B teacher provides strong targets for hard reasoning behaviors.

## Training Plan (High-Level)

### Phase 0 — Upcycling Initialization (Optional but likely)

MoE models benefit from more tokens; we do not have unlimited tokens/compute. A practical strategy is to **upcycle** from a dense checkpoint that has already seen trillions of tokens (candidate families: **Qwen** or **Gemma**) to get a starting point better than random init.

This is non-trivial because we are changing architecture (dense → fine-grained MoE, plus different attention/positional choices), but the goal is not perfect weight reuse—just a better starting point that KD can refine.

### Phase 1 — Offline KD Pretraining (Base Model)

Primary objective: produce a strong base model efficiently using KD.

Open design space:

- Offline distillation only vs a mix with normal next-token prediction (NTP)
- Teacher mixture schedule: how much 120B vs 20B, and when
- Whether to include a small amount of non-reasoning “general text” for robustness/fluency

### Phase 2 — Online / On-Policy Distillation (TBD)

We have not decided the ratio of:

- offline KD
- online/on-policy distillation
- standard NTP pretraining mixed in

On-policy distillation is appealing because it corrects *student-specific* failure modes, but it changes the training system requirements (teacher-in-the-loop throughput, orchestration, logging, etc.).

### Phase 3 — Mid-Training (Quality Data)

Once a competent base exists, mid-train on high-quality corpora to stabilize and broaden capability.

### Phase 4 — Post-Training (SFT + RL)

After the base model is strong, do post-training (SFT + RL) for instruction following, preferences, and safety alignment.

## Data Direction (Native Reasoning)

The base KD dataset should be reasoning-heavy end-to-end. One candidate is **PleIAs/SYNTH** (and/or an augmented variant). The focus is to teach structured reasoning behavior during pretraining rather than “adding thinking later”.

## Open Questions (Need Decisions)

1. What is the initial **KD recipe**: offline-only vs offline + on-policy vs offline + NTP mix?
2. What is the **upcycling source checkpoint** (if any), and how aggressively do we reuse weights?
3. What is the **first minimal dataset** for smoke tests (before full KD pipeline is ready)?
4. What are the primary **success metrics** for the base model (benchmarks and qualitative behaviors)?
