Oen — here’s a clean, “follow-this” working document that captures what we decided in this thread. It’s written as an internal spec you can drop into `docs/distillation.md` (or similar).

---

# vpMoE Distillation Plan (SYNTH-only KD)

## Scope

This document defines the **distillation strategy for the base model** using **PleIAs/SYNTH** only, with a teacher from the **GPT‑OSS** family and **Megatron‑LM** as the training backbone.

Out of scope for now:

* mid-training dataset curation
* long-context / YaRN extension schedule
* SFT + RL

Those come after the base distillation is working and profiled.

---

## Goals

1. Produce a strong **native-reasoning** base model that **emits reasoning + final** in Harmony format.
2. Maximize sample efficiency under a small independent budget by leaning on:

   * heavy upcycling init
   * SYNTH-based distillation
   * on-policy distillation (OPD-style)
3. Keep the system simple enough to iterate (avoid massive logit storage).

---

## Fixed Decisions

### Trainer

* **Megatron‑LM** for student training.

### Student

* vpMoE locked architecture (see `docs/architecture.md`; 80 layers, 256 experts top‑4, 1 shared expert, etc.)
* **Tokenizer / vocab:** same as GPT‑OSS (**o200k Harmony**, padded to 201088)
* **We emit reasoning** at inference (train reasoning channel explicitly).

### Teacher

* **GPT‑OSS‑120B** only.
* **GPU allocation default:** **1 teacher GPU + 7 student GPUs**.
* Only add more teacher replicas if profiling shows the teacher is starving the training loop.

### Dataset for KD

* **PleIAs/SYNTH only** for KD phases.
* You will curate separate mid/post-training datasets later.

---

## Data Representation

Implementation decisions for the SYNTH KD dataset build (format, masking, split, and Megatron layout) live in `docs/data_pipeline.md`.

### Convert each SYNTH row to Harmony conversation

Fields available in SYNTH:

* `query`
* `synthetic_reasoning`
* `synthetic_answer`
  (+ metadata like language, exercise type, etc.)

We serialize into Harmony/chat structure:

* **user**: `query`
* **assistant reasoning**: `synthetic_reasoning`
* **assistant final**: `synthetic_answer`

### Loss masking

* **Mask out user tokens** (no loss on prompt).
* If present, **mask out `system` and `developer` tokens** as prompt context (not supervised targets).
* Compute loss only on assistant tokens:

  * reasoning span tokens
  * final span tokens

### Span masks

Maintain:

* `mask_reasoning[t] ∈ {0,1}`
* `mask_final[t] ∈ {0,1}`
  for assistant tokens.

We will compute separate mean losses per span and mix with explicit weights.

---

## Core Loss Definitions

### Span-normalized CE

Compute two means (important: don’t average over all tokens and then “weight” — weights won’t mean what you think).

* `CE_reasoning = mean_t( -log p_s(y_t | ctx_t)  for t in reasoning span )`
* `CE_final     = mean_t( -log p_s(y_t | ctx_t)  for t in final span )`

Then:
[
L_{\text{CE}} = w_r \cdot CE_{\text{reasoning}} + w_f \cdot CE_{\text{final}}
]

**Default weights:**

* `w_r = 1.0`
* `w_f = 1.0`

(If you later see rambling reasoning / weak answers, shift toward final slightly, e.g., `w_r=0.8, w_f=1.2`.)

---

## Distillation Stages

We will do: **Vanilla SYNTH → Offline rescored SYNTH → Online OPD**, all within SYNTH.

### Stage 1 — Vanilla SYNTH warm start (offline CE)

**Purpose**

* stabilize Harmony formatting
* stabilize MoE routing / training dynamics
* prevent “on-policy on garbage” waste

**Data**

* SYNTH traces (query + reasoning + answer)

**Loss**

* `L = L_CE` (span-normalized CE on reasoning + final)

**Token target (initial default)**

* **~1B student tokens**
* Range: **0.5B–2B** depending on how strong the upcycle is

**Stop condition**

* outputs are structurally correct (Harmony reasoning + final consistently)
* training is stable (no router collapse, no divergence)
* held-out SYNTH eval stops improving quickly

---

### Stage 2 — Offline “rescored SYNTH” KD (teacher forward pass, logprobs-only)

This is where “true KD signal” starts **without** turning the teacher into a synchronous bottleneck.

#### Stage 2A — Build the rescored dataset

**Input**

* Start with a **subset** of SYNTH (recommended: 10–30% by token volume).
* Prefer:

  * longer reasoning examples
  * hardest categories you care about
  * balanced across languages/exercise types if needed

**Teacher work**

* Run GPT‑OSS‑120B **forward** on: `(prompt + SYNTH continuation)`
* Store **teacher logprob of the actual token** at each assistant-token position.

**Teacher reasoning mode**

* Use **medium** mode for rescoring (best cost/signal tradeoff).
* Record `teacher_reasoning_mode = "medium"` as metadata.

**What we store per example**

* token ids for full sequence (prompt + assistant)
* loss mask (assistant tokens only)
* span masks (reasoning vs final)
* `teacher_logp[t]` for assistant tokens (float16 is fine)
* metadata (`synth_id`, language, category, etc.)

#### Stage 2B — Train with KD-weighted loss

Because we are not storing full distributions (no top‑k logits), the most practical “distribution-ish” use of `teacher_logp` is **confidence-weighted CE**.

For assistant tokens:

* student logprob: `logp_s[t] = log p_s(y_t | ctx)`
* teacher logprob: `logp_t[t]` (stored)

Define a per-token weight such as:

* `w_t = clip(exp((logp_t[t] - m)/T), w_min, w_max)`

Where:

* `T` here is a softness factor (start with `T=1.0`)
* `m` is a centering constant (e.g., running mean of teacher logp)
* `w_min/w_max` to keep gradients sane (e.g., 0.25 to 4.0)

Then weighted CE:

* `CEw_reasoning = mean_t( w_t * (-logp_s[t]) for t in reasoning )`
* `CEw_final     = mean_t( w_t * (-logp_s[t]) for t in final )`

Stage 2 training loss:
[
L = w_r \cdot CEw_{\text{reasoning}} + w_f \cdot CEw_{\text{final}}
]

**Why this stage exists**

* improves sample efficiency vs pure CE when dataset has mixed difficulty/noise
* encourages the student to match teacher confidence patterns
* prepares the student for OPD by narrowing teacher–student gap

**Mixing**

* 50–80% batches from rescored subset (KD-weighted CE)
* 20–50% batches from vanilla SYNTH (plain CE)

**Token targets (initial default)**

* **Teacher scoring:** ~**1B tokens** worth of assistant continuations
* **Student training on Stage 2:** ~**2B student tokens**

---

### Stage 3 — Online on-policy distillation (OPD-style) on SYNTH prompts

This is the main stage you want to emphasize.

**High-level loop**

1. Sample completions from student for SYNTH prompts (`query`)
2. Teacher computes **logprobs on the sampled tokens**
3. Train student to move toward teacher on the states/tokens the student actually visits

#### Data

* Prompt pool = SYNTH `query` (and any SYNTH fields you want to add to the prompt template)
* Student generates **reasoning + final** in Harmony format

#### Teacher signal

* store teacher logprob for each sampled assistant token:

  * `logp_teacher[t] = log p_teacher(y_t | ctx_t)`
* also store `teacher_reasoning_mode` used for that rollout

#### Loss

We use an OPD/reverse‑KL style signal based on **logprob differences on sampled trajectories**.

For each sampled completion:

* student per-token logprob `logp_s[t]`
* teacher per-token logprob `logp_t[t]`

Define per-token advantage-like weight:

* `a_t = stop_grad(logp_t[t] - logp_s[t])`

Then an OPD-style loss over assistant tokens:

* `L_opd = - mean_t( a_t * logp_s[t] )`

Compute it separately for reasoning and final spans (span-normalized), then mix:
[
L_{\text{OPD}} = w_r \cdot L_{\text{OPD,reasoning}} + w_f \cdot L_{\text{OPD,final}}
]

**We will do OPD on both reasoning and final.**

#### Stability anchor

Keep a small fraction of updates as plain CE on SYNTH traces:

* **CE anchor:** 10–20% batches
* 80–90% batches OPD

This prevents drift and helps when the rollout buffer is temporarily low quality.

#### Teacher reasoning-mode curriculum

We will use the teacher’s reasoning mode as a **runtime parameter** (API parameter or equivalent control in your inference wrapper). We’ll log it with each sample.

Stage 3 is split by fraction of OPD tokens:

**Stage 3a (first ~30% of OPD tokens)**

* 70% **low**
* 30% **medium**
* rollouts per prompt: **1**

**Stage 3b (next ~40%)**

* 20% low / 60% medium / 20% high
* rollouts per prompt: **1** initially → transition to **2** near the end of 3b

**Stage 3c (final ~30%)**

* 10% low / 30% medium / 60% high
* rollouts per prompt: **2**

This curriculum controls early cost (low traces are shorter) and reserves high-effort deep reasoning for when the student is ready.

#### Throughput plan (utilization)

* Start with **1 teacher GPU + 7 training GPUs**
* Prefer an **async buffer**:

  * “sampling worker” produces student rollouts (periodic weight sync from training)
  * teacher worker scores rollouts (logprobs only)
  * trainer consumes scored rollouts
* If the buffer runs dry:

  1. reduce generation length caps early (cheap knob)
  2. reduce high-mode ratio temporarily
  3. only then consider adding a second teacher replica (and sacrificing a training GPU)

**Token target (initial default)**

* **~8B student tokens** of OPD + CE anchor mixed in
* Range: **4B–12B** depending on observed tokens/sec and convergence

---

## Initial Token Budget Targets (first run)

These are intentionally “starter defaults” you can adjust after profiling.

* Stage 1: **1B**
* Stage 2: **2B student tokens** (plus ~1B teacher-scored tokens offline)
* Stage 3: **8B OPD tokens** (+ 10–20% CE anchor inside Stage 3)

Total student tokens trained ≈ **11B** (plus offline teacher scoring).

This aligns with “OPD dominates budget” while still protecting you from early instability.

---

## What We Log (minimum required)

### Always

* global tokens/sec (training)
* GPU utilization + memory headroom
* router load balance metrics (aux loss, expert utilization histogram)
* held-out SYNTH eval loss

### Stage 2

* teacher mean logp per example
* distribution of teacher logp by span (reasoning vs final)

### Stage 3

* teacher mode counts (low/medium/high)
* average `(logp_teacher - logp_student)` by span
* success rate of format validity (Harmony structure)
* rollout lengths by mode

---

## Quality Controls

Even if you don’t train on non-SYNTH data yet, keep a tiny “canary eval” set:

* a few thousand natural prompts / plain text snippets
* evaluate occasionally (no training on it yet)
  This catches catastrophic OOD weirdness early without changing your plan.

---

## Implementation Checklist

1. **SYNTH → Harmony serializer**

   * produce token ids, loss mask, span masks
   * pack sequences for Megatron dataset format

2. **Stage 1 training**

   * verify loss masks correct (no loss on prompt tokens)
   * verify reasoning/final span weighting is correct (means computed separately)

3. **Teacher rescoring script (Stage 2A)**

   * run GPT‑OSS‑120B forward on `(prompt + continuation)`
   * store per-token teacher logp (assistant tokens only)
   * store metadata + span masks

4. **Stage 2 training**

   * implement teacher-confidence weighted CE
   * mix rescored and vanilla batches

5. **Stage 3 OPD pipeline**

   * rollout sampler (student inference)
   * teacher scorer (logprobs on sampled tokens; 1 GPU)
   * replay/buffer dataset
   * OPD loss in trainer
   * CE anchor mixing
   * reasoning mode curriculum schedule + logging
   * rollout count schedule (1 → 2)

6. **Profile & iterate**

   * if teacher bottlenecks: adjust generation caps / curriculum mix before adding GPUs
   * if training drifts: increase CE anchor %, or slow the low→high transition

---

## “Locked” Answers to Earlier Questions

* Loss is applied to **reasoning + final** (span-normalized, weighted).
* Teacher uses **1 GPU**, default **1+7 split**, profile before changing.
* Stage 3 includes a **CE anchor** for stability.
* Stage 3 uses **reasoning-mode curriculum** and transitions to **2 rollouts/prompt** later.

---

If you want, I can also produce a short **“config block”** version of this (like a YAML-ish plan with knobs and defaults) so you can track the stage schedule and ratios in one place.
