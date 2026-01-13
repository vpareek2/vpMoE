# DATASETS:

## General Reasoning
- PleIAs/SYNTH

## Math

For 80-90%:
- nvidia/Nemotron-Math-v2
- open-r1/OpenR1-Math-220k

For 10-20%:
- PrimeIntellect/Hendrycks-Math
- qwedsacf/competition_math

## Code

For 30%:
nvidia/OpenCodeReasoning-2

For 15%:
nvidia/Nemotron-Comptetitive-Programming-v1

For 35%:
nvidia/OpenCodeInstruct

For 10%:
agentica-org/DeepCoder-Preview-Dataset

For the rest - reserved for agentic maybe

## Personality

My recommended B4 KD prompt mix to start “warm + intelligent”

(Percentages are within B4, not global.)

45% UltraChat 200k (sft prompts) — primary warm-chat backbone

20% Nemotron post-train chat prompts — “friendly assistant” baseline + breadth

15% OpenAssistant best-path prompts — realism + multi-turn variety

10% Anthropic-helpful prompts (filtered) — helpfulness calibration (no red-team)

5% Deita-10k — quality anchor

5% Hermes function calling / JSON — “intelligent” structured behavior

## World Knowledge
- PleIAs/SYNTH (everything is synthetic from wikipedia)

# Confirmed Data Pipeline Doc (DistillKit BF16 Phase)

## 0) Purpose and scope

This pipeline builds a curated, continuously re-weighted prompt stream for **online knowledge distillation** (KD) from **GPT-OSS** into a student with attention surgery (SWA-128 + TPA + KDA).
We assume upstream datasets are already high-quality; our goal is **maximize learning per token** via mixture control, redundancy control, and distillation-aware data valuation.

Non-goals for this phase:

* 128K-context mastery
* document-centric training (PDF/doc QA)
* on-policy rollouts (deferred to NVIDIA post-training)

---

## 1) Distillation mode (online KD)

Training uses **online distillation**:

* For each training example, we feed the same Harmony prompt to **teacher** and **student**.
* We compute KD losses from teacher outputs/logits on-the-fly and update the student only.

Implication:

* The dataset is primarily a **prompt distribution**.
* “Targets” come from GPT-OSS during training (teacher policy), not from dataset-provided answers.

---

## 2) Bucket taxonomy (skill buckets)

We define **5 skill buckets**. Every example must belong to exactly one primary bucket.

### B1 — General Reasoning

Multi-step everyday reasoning, planning, tradeoffs, multi-hop QA, compare/contrast, causal reasoning.

### B2 — Math / Formal Reasoning

Math word problems, logic puzzles, proofs, symbolic manipulation, structured step-by-step reasoning.

### B3 — Code / Algorithmic Reasoning

Code comprehension, debugging, algorithm reasoning, complexity/invariants, code-to-explanation prompts.

### B4 — Personality / Assistant Policy

Tone, helpfulness, instruction-following, multi-turn coherency, refusal style, formatting compliance.

### B5 — World Knowledge / Explanations

Factual QA, concept explanations, “what is X” + “why” + “compare Y vs Z”, broad encyclopedic coverage.

Notes:

* “STEM explanations” that are not formal math belong in **World Knowledge** or **General Reasoning** depending on structure.
* We intentionally separate Math vs Code to avoid dominance and to preserve specialization.

---

## 3) Context-length buckets (ctx buckets)

We follow an Arcee-like strategy: **most tokens short/medium**, with a small long-context warm-up stage for KDA.

### C1: 2K–4K (primary)

### C2: 4K–8K (secondary)

### C3: 8K–16K (warm-up)

### C4: 16K–32K (tiny tail; optional)

Each example is assigned a `ctx_bucket` based on estimated packed length (prompt + expected completion).

---

## 4) Token budget and target mixture (750M total)

We allocate quotas across skill and ctx buckets. Final numbers may be tuned after proxy runs.

### 4.1 Skill-bucket target mix (global)

* B1 General Reasoning: 35–45%
* B4 Personality/Policy: 20–30%
* B2 Math/Formal: 10–20%
* B3 Code: 10–20%
* B5 World Knowledge: 10–20%

### 4.2 Context-length target mix (global)

* C1 2K–4K: ~80%
* C2 4K–8K: ~13%
* C3 8K–16K: ~6%
* C4 16K–32K: ~1% (optional)

### 4.3 Quota grid

We enforce quotas in the cross-product grid:

* (B1..B5) × (C1..C4)

This prevents the selector from collapsing into “only short math” or “only long summarization,” etc.

---

## 5) Input data strategy (high-quality sources)

We treat upstream datasets as *prompt pools* (not label pools).
Dataset-provided answers/rationales may be used only as metadata or for optional evaluation; teacher provides the KD signal online.

Primary sources (for prompt pools):

* PleIAs/SYNTH (broad reasoning/instruction)
* NVIDIA OpenMathReasoning (math)
* PrimeIntellect SYNTHETIC-2 (reasoning breadth)
* OpenThoughts / OpenR1-Math (style + procedures)
* (Optional) curated general/edu text sources for World Knowledge prompts

---

## 6) Normalization: canonical example format

All examples are stored as Harmony message arrays.

### 6.1 Canonical prompt record (PromptSpec)

* `system`: fixed system prompt for the run
* `user`: content of the task (may include context)
* Optional: multi-turn history (short; used mainly in Personality bucket)

### 6.2 Distillation record

During training, teacher generates:

* `assistant:analysis`
* `assistant:final`

We record token-span masks for analysis/final for loss weighting.

---

## 7) Redundancy control (diversity first)

Even curated datasets contain overlap. We apply redundancy control within each (bucket, ctx_bucket):

### 7.1 Clustering

* Compute prompt embeddings (or cheaper lexical embeddings).
* Cluster prompts; store `cluster_id`.

### 7.2 Diversity sampling rule

* Cap max samples per cluster per epoch.
* Always reserve a small tail for rare clusters.

Outputs:

* `cluster_id`
* `cluster_size`
* `cluster_rank` (optional)

---

## 8) Distillation-aware data valuation (no AR generation)

We build a lightweight scorer to estimate prompt training value.

### 8.1 Features (cheap)

We compute features without autoregressive generation:

* Prompt-only:

  * token length
  * bucket tags
  * lexical markers (constraints, enumerations, code fences, equations)
* Teacher forward-pass stats (optional but recommended):

  * token-level entropy/margin (teacher uncertainty) on a short continuation window
* Student–teacher disagreement stats (after student warms up):

  * forward-pass KL on short window
  * analysis-vs-final disagreement concentration

### 8.2 Score model: p(good prompt | prompt)

Train a small head (MLP) over embeddings/features to predict a continuous `utility_score ∈ [0,1]`.
Train separate heads per skill bucket (recommended) to prevent cross-domain bias.

### 8.3 Sampling policy

Do not hard-filter to only top prompts. Use probabilistic sampling:

* `sampling_weight = exp(τ * utility_score)`
* enforce per-cluster caps and per-bucket quotas
* keep a diversity tail (e.g., ε-greedy)

---

## 9) Learning-progress labeling (DataRater-spirit)

We approximate “true value” of a prompt by measuring how much it improves the student.

### 9.1 Learning-progress protocol

For a small calibration subset each cycle:

1. Snapshot student parameters.
2. Run k mini-updates on a candidate batch.
3. Measure improvement on a held-out validation slice:

   * decrease in teacher-student KL
   * improvement on a small reasoning eval set
4. Define `lp_value = Δmetric / tokens_used`.

This produces a supervised signal for the scorer.
We do this sparingly (e.g., 1–5% of batches) to keep overhead low.

### 9.2 Update cadence

* Warm-up: no learning-progress labels until student is stable.
* Then: compute LP labels periodically (e.g., every N steps or once per epoch) and re-train the scorer.

---

## 10) Curriculum schedule

### 10.1 Context curriculum

* Phase P1: C1 only (2K–4K) until stability
* Phase P2: add C2 (4K–8K)
* Phase P3: add C3 (8K–16K)
* Phase P4: optional C4 tail (16K–32K)

### 10.2 Difficulty curriculum (within bucket)

Use `utility_score` and/or difficulty proxies to:

* start mid difficulty (avoid pathological hard cases early)
* then focus more on high-utility/hard cases later
* preserve a diversity tail throughout

---

## 11) Required dataset schema (v0)

Store as Parquet (preferred) or JSONL with these fields:

### Identity

* `id` (uuid)
* `source_dataset`, `source_subset`, `source_example_id`

### Buckets

* `skill_bucket` ∈ {B1..B5}
* `ctx_bucket` ∈ {C1..C4}

### Prompt

* `messages` (Harmony JSON: system+user(+history))
* `prompt_text` (optional cached rendered text)
* `prompt_tokens_est`

### Diversity

* `embedding` (optional)
* `cluster_id`
* `near_dup_hash` (optional)

### Scores / features

* `utility_score` (float)
* `sampling_weight` (float)
* `difficulty_proxy` (float; optional)
* `teacher_entropy` / `teacher_margin` (optional)
* `student_teacher_kl` (optional)
* `lp_value` (optional)

### Distillation masks (computed after teacher output)

* `analysis_span` (token index ranges)
* `final_span` (token index ranges)

---

## 12) Quality checks (lightweight)

Even if datasets are “good,” we run:

* schema validity
* Harmony render/parse round-trip
* distribution checks:

  * tokens by bucket
  * tokens by ctx_bucket
  * cluster sizes and diversity
  * analysis/final length stats

---

## 13) Outputs for the DistillKit phase

1. Curated prompt pool dataset with bucket assignments + diversity metadata.
2. Scorer artifacts:

   * embedding model choice
   * per-bucket scorer weights
   * feature extraction config
3. Sampling manifest:

   * quota grid
   * τ temperature
   * diversity tail ε
   * curriculum schedule
4. Validation sets:

   * held-out KL slice (Harmony prompts)
   * small reasoning suite
   * format compliance suite

---

## 14) Execution plan (high-level)

1. Build prompt pool + bucket labels + ctx estimates.
2. Cluster prompts per (bucket, ctx_bucket); cap redundancy.
3. Start training with uniform sampling within quotas.
4. After warm-up, compute teacher/student features and train per-bucket scorer heads.
5. Add learning-progress labeling on a small cadence; retrain scorer periodically.
6. Maintain quotas + curricula until 750M tokens are consumed.
7. Export final BF16 checkpoint + the full data/sampling manifest for reproducibility.
