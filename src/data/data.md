# Dataset sources (draft)

We treat upstream datasets as **sequence pools**: each training example is a full Harmony sequence
(system + user + assistant). During training, the **teacher runs a forward pass over the same
`input_ids`** to provide logits (and optionally hidden states) for KD, exactly as DistillKit does.

## Global source mix (distillation phase)

### 80% — PleIAs/SYNTH (primary)

SYNTH is the default prompt source for the main distillation phase due to consistently high prompt quality.

### 20% — Anchor prompts (diversity + hardness)

We keep a small but non-trivial anchor stream to avoid single-dataset style lock-in and to cover harder / more realistic distributions.

* **8% Math-hard anchors**

  * `nvidia/Nemotron-Math-v2`
  * `PrimeIntellect/Hendrycks-Math` (small slice)
  * `qwedsacf/competition_math` (small slice)

* **6% Code anchors**

  * `nvidia/OpenCodeInstruct`
  * `agentica-org/DeepCoder-Preview-Dataset` (tiny slice; strictly capped)

* **4% Chat/helpfulness anchors**

  * `HuggingFaceH4/ultrachat_200k`
  * `OpenAssistant/oasst1`
  * `bcui19/chat-v2-anthropic-helpfulness` (small slice)

* **2% Personality / roleplay anchors**

  * (TBD: curated personality pool; keep small and high-quality)

# Confirmed Data Pipeline Doc (online KD phase)

## 0) Purpose and scope

This pipeline builds a curated, continuously re-weighted **sequence stream** for **online knowledge distillation** (KD) from **GPT-OSS** into a student with attention surgery (SWA-128 + TPA + KDA).
We assume upstream datasets are already high-quality; our goal is **maximize learning per token** via mixture control, redundancy control, and distillation-aware data valuation.

Non-goals for this phase:

* 128K-context mastery
* document-centric training (PDF/doc QA)
* on-policy rollouts (deferred to NVIDIA post-training)

---

## 1) Distillation mode (online KD)

Training uses **online distillation**:

* For each training example, we feed the same Harmony token sequence (`input_ids`) to **teacher** and **student**.
* We compute KD losses from teacher logits (and optional hidden states) on-the-fly and update the student only.
* The teacher does **not** generate new tokens during training (teacher-forced distillation over fixed sequences).

Implication:

* The dataset defines a **full sequence distribution** (prompt + assistant).
* The assistant tokens come from the dataset (e.g., SYNTH `synthetic_reasoning`/`synthetic_answer`, code `output`, etc.).
* GPT-OSS provides **soft targets** (logits/hidden states) via a forward pass over those same tokens.

---

## 2) Bucket taxonomy (skill buckets)

We define **4 skill buckets**. Every example must belong to exactly one primary bucket.

### B1 — General Reasoning

Multi-step everyday reasoning, planning, tradeoffs, multi-hop QA, compare/contrast, causal reasoning.

### B2 — Math / Formal Reasoning

Math word problems, logic puzzles, proofs, symbolic manipulation, structured step-by-step reasoning.

### B3 — Code / Algorithmic Reasoning

Code comprehension, debugging, algorithm reasoning, complexity/invariants, code-to-explanation prompts.

### B4 — Personality / Assistant Policy

Tone, helpfulness, instruction-following, multi-turn coherency, refusal style, formatting compliance.

Notes:

* “World knowledge / explanations” is covered implicitly (primarily via SYNTH) and is treated as part of **General Reasoning** for now.
* We intentionally separate Math vs Code to avoid dominance and to preserve specialization.

---

## 3) Context-length buckets (ctx buckets)

We loosely follow the Arcee/DistillKit KDA recipe: do most KD at short context for throughput/stability, then do a dedicated long-context stage so the attention stack is exposed at the target length.

### C1: 2K (primary distill)

### C2: 8K (bridge)

### C3: 32K (long-context distill / adaptation)

### C4: 64K+ (optional tail; only if we need it)

Each example is assigned a `ctx_bucket` based on estimated packed length (prompt + assistant).
For teacher-forced online KD, the dominant driver is the **dataset sequence length distribution**
(plus any preprocessing/training-time max-length caps), not generation settings like `max_new_tokens`.

---

## 4) Token budget and target mixture (draft)

We allocate quotas across skill and ctx buckets. Final numbers may be tuned after proxy runs.

### 4.1 Skill-bucket target mix (global)

* B1 General Reasoning: 50%
* B2 Math / Formal Reasoning: 17.5%
* B3 Code / Algorithmic Reasoning: 17.5%
* B4 Personality / Assistant Policy: 15%

### 4.2 Context-length target mix (curriculum-driven)

Rather than enforcing a single global mix, we schedule context lengths over time:

* **P1 (bootstrap):** C1 only (2K)
* **P2 (bridge):** C1 + small C2 (8K)
* **P3 (long-context):** primarily C3 (32K) with a small C1 tail for stability
* **P4 (optional tail):** add C4 only if evaluation indicates we need >32K robustness

### 4.3 Quota grid

We enforce quotas in the cross-product grid:

* (B1..B4) × (C1..C4)

This prevents the selector from collapsing into “only short math” or “only long summarization,” etc.

---

## 5) Input data strategy (high-quality sources)

We treat upstream datasets as **full sequence pools** (not prompt-only pools).
Dataset-provided assistant content is part of the training sequence; the teacher provides the KD signal online via forward pass.

Primary sources (for sequence pools):

* **PleIAs/SYNTH (~80%)**: the default distillation sequence stream.
* **Anchor prompts (~20%)**: a small mix of math/code/chat/personality prompts for diversity and hardness (see “Prompt sources (draft)” above).

Principle:

* Prefer a single strong “main” prompt source (SYNTH) while keeping a thin, controlled anchor stream to prevent distribution collapse.

---

## 6) Normalization: canonical example format

All examples are stored as Harmony message arrays.

### 6.1 Canonical prompt record (PromptSpec)

* `system`: fixed system prompt for the run
* `user`: content of the task (may include context)
* Optional: multi-turn history (short; used mainly in Personality bucket)

### 6.2 Distillation record

During training, we use the dataset-provided assistant spans (analysis and/or final). The teacher provides:

* per-token logits (always, in online KD)
* optional hidden states (when enabled)

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

* Phase P1: C1 only (2K) until stability
* Phase P2: add C2 (8K) as a bridge
* Phase P3: shift to C3 (32K) for long-context adaptation
* Phase P4: optional C4 tail (64K+) only if needed

### 10.2 Difficulty curriculum (within bucket)

Use `utility_score` and/or difficulty proxies to:

* start mid difficulty (avoid pathological hard cases early)
* then focus more on high-utility/hard cases later
* preserve a diversity tail throughout

### 10.3 Reasoning-effort policy (teacher runtime parameter)

We treat `reasoning_effort` as a first-class knob and keep it bounded to avoid long-tail runaway costs.

Default (most buckets):

* low: 50%
* medium: 40%
* high: 10%

Personality bucket override:

* low: 70%
* medium: 30%
* high: 0%

Caps + retry rule (draft):

* low: `max_new_tokens=2048`
* medium: `max_new_tokens=8192`
* high: try `max_new_tokens=8192`, retry once at `32768` only if `final` is missing; drop if still missing

Measured token lengths (probe):

Token stats from a small, local Harmony probe run (`openai/gpt-oss-20b`, 23 prompts × {low, medium, high} = 69 runs; output: `data/teacher_runs_20b.jsonl`):

* Totals: prompt tokens `12,510`, completion tokens `376,192` (analysis `297,694`, final `77,932`), total `388,702`
* Completion length (all modes): p50 `1,524`, p90 `10,714`, p95 `35,499`, max `65,536`
* Completion mean by `reasoning_effort`:

  * low: ~`984`
  * medium: ~`3,746`
  * high: ~`11,627` (p90 ~`40k`, max hit cap `65,536` once; one run ended with no `final`)

Implication:

* If we apply `high` broadly without caps/retry/drop rules, it will dominate token budget due to long-tail outliers (notably DeepCoder-style code prompts and hard math).

### 10.4 Token budget by stage (draft)

We follow the Arcee/DistillKit KDA writeup as a rough reference point (they mention ~300M tokens for the initial KD phase, then ~1B tokens at 32k context for long-context adaptation).

For our first full run, target **~1.3B total training tokens**, split by context stage:

* **P1 (2K): 300M tokens**
* **P2 (8K): 150M tokens** (bridge)
* **P3 (32K): 850M tokens** (long-context adaptation)
* **P4 (64K+): 0–100M tokens** (optional; only if eval indicates >32K weakness)

Sanity check (approx packed sequences processed, assuming packing fills sequences near `sequence_length`):

* P1: `300M / 2048` ≈ `146k` sequences
* P2: `150M / 8192` ≈ `18k` sequences
* P3: `850M / 32768` ≈ `26k` sequences

Notes:

* The **80% SYNTH / 20% anchors** source mix stays constant across stages; only `sequence_length` (and packing) changes.
* These are **student training tokens** (i.e., tokens consumed by the training loop). Online teacher forward-pass cost scales with the same token count; no teacher generation is assumed for the main run.

---

## 11) Required dataset schema (v0)

Store as Parquet (preferred) or JSONL with these fields:

### Identity

* `id` (uuid)
* `source_dataset`, `source_subset`, `source_example_id`

### Buckets

* `skill_bucket` ∈ {B1..B4}
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
