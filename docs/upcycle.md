> Canonical followable runbook: `vpmoe/upcycle/README.md`.
> This file is the strategy writeup and background.

## 0) What we’re starting from (facts that matter)

### Qwen3‑0.6B config (what we must adapt)

From the official HF config:

* `hidden_size = 1024`
* `num_hidden_layers = 28`
* `num_attention_heads = 16`
* `num_key_value_heads = 8`
* `head_dim = 128`
* `intermediate_size = 3072`
* `vocab_size = 151936`
* `tie_word_embeddings = true` ([Hugging Face][1])

So: hidden size matches; **depth, head layout, FFN width, vocab** do not.

### Your target constraints (why tokenizer + format alignment is non-negotiable)

OpenAI’s gpt‑oss models use **o200k_harmony** (≈201k vocab) and were trained in Harmony format. OpenAI explicitly calls out o200k_harmony and that it’s open-sourced. ([OpenAI][2])
HF also ships a `tokenizer.json` for `openai/gpt-oss-20b`, which we can use as the canonical “target tokenizer definition” for transplantation tooling. ([Hugging Face][3])

### The two key tools/papers that make this feasible

1. **Training-Free Tokenizer Transplantation via OMP** (Arcee): reconstructs embeddings for “new vocab tokens” as sparse combinations of shared anchors; integrated into `mergekit-tokensurgeon`. ([arXiv][4])
2. **Upcycling Dense → MoE** (NVIDIA): shows practical initialization + scaling tricks (incl. “virtual group” init) and finds upcycling can beat continued dense training. ([arXiv][5])

---

## 1) The target object: define “vpDense” precisely

You want a **dense variant of your final vpMoE** so that:

* all *non-FFN* weights (attn, norms, embeddings, etc.) are already aligned,
* FFN weights can be **copied/replicated into experts** later.

### vpDense definition

* Exactly your vpMoE transformer stack **except**:

  * replace each MoE FFN block with a **single dense FFN** whose structure matches your expert FFN (same activation / gating style, same intermediate dim `I_target`).
* Keep:

  * hidden size 1024,
  * your attention module shape (8 heads, 2 KV heads, head_dim 128),
  * your norm choices,
  * your local/global schedule + GRAPE/TPA (eventually).

**Important practical suggestion:** make vpDense available in 2 configs:

* `vpDense0-5_28` (28 layers) — best for *initial* Qwen import (depth matches).
* `vpDense` (80 layers) — your final depth, used after you “heal” the transplant.

This reduces risk massively vs trying to jump 28→80 immediately.

---

## 2) Step-by-step plan to turn Qwen3‑0.6B into `vpDense0-5_28`

### Step 2.1 — Tokenizer transplant: Qwen vocab → o200k_harmony vocab

Goal: create a Qwen3 checkpoint whose **embedding + lm_head** are compatible with **o200k_harmony**, without retraining.

**Method:** OMP tokenizer transplantation (Arcee), because it’s designed for exactly this “cross-tokenizer KD / reuse weights” scenario. ([arXiv][4])

**Target tokenizer source:** `openai/gpt-oss-20b` tokenizer files (HF has `tokenizer.json`). ([Hugging Face][3])

**Why this matters for you:** it aligns the student with the *teacher* tokenizer family and also aligns with your Megatron o200k Harmony path.

**Known pitfall:** the OMP paper flags **mismatched numerical tokenization schemes** as a critical failure mode for math reasoning. That means you must explicitly regression-test numbers after transplant (I’ll give a test suite below). ([arXiv][4])

Deliverable from this step:

* “Qwen3‑0.6B‑o200k” = same transformer weights, but `embed_tokens` (and tied `lm_head`) resized to ~201k.

---

### Step 2.2 — Attention head surgery: (16q, 8kv) → (8q, 2kv)

Your locked attention differs. We can do a **structured meanpool** surgery that is consistent with the standard GQA conversion guidance.

HF’s own Qwen3 docs explicitly describe constructing grouped KV heads by **meanpooling original heads within each group** when converting checkpoints. ([Hugging Face][6])
We’ll apply the same idea more aggressively:

#### Q projection (Qwen: 16 heads → you: 8 heads)

Reshape Qwen `Wq` (out_dim × in_dim) into:

* `[16, head_dim, hidden] = [16, 128, 1024]`
  Then group pairs of heads and average:
* `[8, 128, 1024]`
  Flatten back to `[1024, 1024]` (since 8×128 = 1024).

#### K/V projection (Qwen: 8 heads → you: 2 heads)

Reshape `Wk`, `Wv` into:

* `[8, 128, 1024]`
  Group 4 heads at a time and average:
* `[2, 128, 1024]`
  Flatten to `[256, 1024]` (since 2×128 = 256).

#### Output projection

Qwen’s `Wo` must map from its attention output dim to hidden. Given Qwen config has 16 heads and head_dim=128, its attention concat dim is 2048, so `Wo` is `[hidden, 2048]` and yours is `[hidden, 1024]`. ([Hugging Face][1])
Do:

* reshape `Wo` to `[hidden, 16, 128]`
* meanpool pairs over the head axis → `[hidden, 8, 128]`
* flatten → `[hidden, 1024]`.

This gives you a clean “function-preserving-ish” downmap in the same spirit as the GQA conversion note. ([Hugging Face][6])

---

### Step 2.3 — FFN surgery: Qwen `intermediate_size=3072` → your `I_target`

Qwen’s FFN width is 3072. ([Hugging Face][1])
Your vpMoE expert FFN width is whatever your config uses (call it `I_target`; I strongly suspect it’s “small” given your 6B total budget, but plug in the real value).

You need a deterministic way to reduce 3072 → `I_target` **without training**.

#### Recommended: structured neuron selection (fast, stable, reversible)

For SwiGLU-style MLPs (common in Qwen-family), you typically have:

* `gate_proj`: [3072, 1024]
* `up_proj`:   [3072, 1024]
* `down_proj`: [1024, 3072]

Pick a subset of intermediate neurons `S` of size `I_target` using a score like:

* `score[i] = ||down_proj[:, i]||_2 + ||up_proj[i, :]||_2 + ||gate_proj[i, :]||_2`

Then:

* new `gate_proj` = `gate_proj[S, :]`
* new `up_proj`   = `up_proj[S, :]`
* new `down_proj` = `down_proj[:, S]`

This preserves the “same hidden units” across all three matrices.

#### Add variance-preserving scaling (important)

When you shrink width, activations’ variance changes. A simple correction:

* multiply `down_proj` by `sqrt(3072 / I_target)` (or equivalently scale `up/gate` down)
  This is a cheap stabilizer for the first few million tokens of adaptation.

---

### Step 2.4 — Positional / attention-pattern mismatch: RoPE/full-attn → GRAPE + local schedule

Qwen uses RoPE and (by default) not sliding window. ([Hugging Face][1])
Your model uses GRAPE-M/A and a heavy local schedule.

**Do NOT flip all those switches on the very first forward pass** after surgery if you can avoid it.

Instead, in `vpDense0-5_28`, run a **two-hop morph**:

1. **Morph A (closest to Qwen):**

   * full attention everywhere
   * RoPE (temporarily)
   * no TPA
   * (keep your other invariants: hidden size, norms if compatible)

2. **Morph B (your real baseline):**

   * enable GRAPE-A on global layers, GRAPE-M on local layers
   * enable 3:1 local:global + window=128 + TPA
   * do it as a *checkpoint-to-checkpoint* config switch, not inline mid-run

This is just “don’t shock the network” engineering.

---

## 3) “Healing” training: make the transplanted vpDense0-5_28 actually useful

Even training-free transplantation is not perfect; Arcee explicitly positions this as enabling cross-tokenizer distillation / reuse, not as a magic no-op. ([arcee.ai][7])

### Do we “warm up” after every stage?

Almost — but the key is to keep each warm-up **minimal** and **targeted**:

* If a step creates a real discontinuity (tokenization semantics, tensor shapes, attention pattern, depth, routing), do a short warm-up.
* Otherwise, just run regressions and move on.

The goal is not “make it smart.” It’s “make it stable enough that the next step is meaningful.”

### Warm-up policy (what to train, and when)

#### A) After tokenizer transplant (`qwen3-0_6B → qwen3-0_6B-o200k`)

Default: **skip** training. Run numeric + special-token regressions (Section 7) and only warm up if they fail.

If you warm up, keep it extremely light:

* **Trainable:** `embed_tokens` and tied `lm_head` only.
* **Frozen:** everything else.
* **Loss:** plain NTP/CE (prompt masked; assistant tokens only).
* **LR:** very low. OMP-initialized embeddings appear sensitive; the OMP paper reports needing a much lower LR for CPT on OMP-initialized models. ([arXiv][4])
* **Stop:** numeric copy prompts stabilize; no obvious tokenization regressions.

#### B) After tensor surgery (`qwen3-0_6B-o200k → vpDense0-5_28.compat`)

This is a true discontinuity (head geometry + FFN width + sometimes positional behavior). Do a short healing run.

* **Start in compat** (full attn + RoPE, no TPA), so you only absorb *one* shock at a time (Section 2.4).
* **Trainable:** default = all parameters, but keep embeddings conservative (either frozen or low LR multiplier).
* **Loss:** plain NTP/CE with prompt masked; include Harmony-structured examples early (below).
* **Stop:** loss stops spiking; numeric regressions pass; format validity improves.

#### C) After morphing to the real stack (`vpDense0-5_28.compat → vpDense0-5_28.real`)

Treat this as another small shock (RoPE→GRAPE + schedule + TPA):

* Do a **short** stabilization run after the checkpoint-to-checkpoint config switch.
* Stop as soon as loss re-stabilizes and regressions still pass.

#### D) After depth expansion (`vpDense0-5_28 → vpDense`)

Even with near-identity init, new layers need a short “wake up”:

* **Phase 1:** train only the newly inserted layers (and their norms) for a short window.
* **Phase 2:** unfreeze all and run briefly to re-equilibrate.

#### E) After dense→MoE (`vpDense → vpMoE`)

This warm-up is effectively your existing “Stage 1 SYNTH warmup CE” (Section 6).

Practical note: upcycling literature finds that a pure “low constant LR” often plateaus; using an LR reset-style schedule can help experts diversify and improve validation loss. ([arXiv][5])

### Suggested “light training” recipes (safe defaults)

These are intentionally conservative. The warm-up goal is *stability*, not capability.

#### Tokenizer transplant warm-up (only if regressions fail)

* **Trainable:** embeddings + tied lm_head only.
* **Budget:** 10M–50M tokens.
* **LR:** start extremely low; the OMP paper reports OMP-initialized runs needing much lower LR than other methods for continued pretraining. ([arXiv][4])
* **Stop:** numeric copy prompts pass reliably; tokenization counts look normal.

#### `vpDense0-5_28.compat` healing run (required)

* **Trainable:** all params (or all-but-embeddings); keep embeddings conservative (frozen or low LR multiplier).
* **Budget:** 50M–200M tokens, or “until loss stops spiking”.
* **Schedule:** a short restart (warmup → stable → decay). Avoid a tiny constant LR that never lets the model re-equilibrate.
* **Stop:** loss stable; numeric regressions pass; Harmony format validity improving.

#### Depth expansion wake-up (`vpDense0-5_28 → vpDense`)

* **Phase 1 (new layers only):** 10M–50M tokens.
* **Phase 2 (unfreeze all):** 10M–50M tokens.
* **Stop:** loss stable and no obvious regressions vs the 28-layer checkpoint.

#### Dense→MoE stabilization (`vpDense → vpMoE`)

* Fold into Stage 1 SYNTH warm start (Section 6).
* Prefer an LR restart-style schedule over “minimum LR forever” to avoid expert collapse / lack of diversification. ([arXiv][5])

### Minimal healing objective

* plain next-token CE (NTP) is fine
* but include **Harmony-structured examples early** so the model learns your eventual formatting priors

### Minimal dataset (compute-light but high impact)

* 70–90%: “general text/code/math” (whatever you have rights to use)
* 10–30%: SYNTH prompts rendered into Harmony structure (even if you don’t fully trust the traces yet)

### What you’re trying to fix

* embedding transplant artifacts (esp. numbers)
* head meanpool artifacts
* FFN shrink artifacts
* any RoPE→GRAPE / full→local switch artifacts

### Gating criterion to move on

* loss no longer spikes after a few thousand steps
* numeric regression tests pass (below)
* format validity > 95% on a small Harmony validation set

---

## 4) Expand to `vpDense` without blowing up

Now you have `vpDense0-5_28` in good shape. Next: create an **80-layer dense checkpoint** that won’t immediately destabilize.

### The safest depth expansion trick (transformer-friendly)

For the **52 new layers**, initialize them as near-identity:

* copy RMSNorm weights from a nearby layer (or set to 1)
* set attention output projection (`o_proj`) to **zero**
* set MLP output projection (`down_proj`) to **zero**

This makes each inserted block initially contribute ~0 residual update, so the whole network behaves like the 28-layer model at initialization.

Then you train a short “wake up the new layers” phase with:

* slightly higher LR on the new layers (optional)
* normal LR on old layers

This is dramatically safer than naïvely repeating layers.

---

## 5) Convert `vpDense` → `vpMoE` (your real warmstart target)

Once you have a strong `vpDense`, you upcycle into MoE.

### Expert initialization (copy + diversify)

The NVIDIA upcycling paper finds initialization details matter (they propose “virtual group” init + weight scaling, etc.) and show upcycling can outperform continued dense training. ([arXiv][5])

For each MoE layer:

* Start every expert as a copy of the dense FFN weights
* Add tiny expert-specific noise (to break symmetry)
* Optional but good: apply a random orthogonal rotation in intermediate space per expert (keeps norms but changes basis)

### Router initialization

* initialize router weights/bias so routing is near-uniform at start
* tiny noise so top‑k ties break consistently
* keep your load-balancing aux loss turned on

This produces a MoE that behaves “like the dense model” initially, then starts specializing.

---

## 6) Only now: start your Stage 1 SYNTH warmup CE

At this point, your Stage 1 warmup CE is no longer “teach it to be a transformer.”
It’s “teach it SYNTH/Harmony behavior + stabilize routing.”

That’s exactly the best-positioning you’re aiming for.

---

## 7) Must-have regression tests (do these every checkpoint step)

These are cheap and will save you weeks.

### A) Numeric tokenization sanity (because OMP paper warns about this)

Test prompts:

* `"1 2 3 4 5"`
* `"1234567890"`
* `"1.2345e-10"`
* `"2025-08-05"`
* `"1/3 = 0.333..."`

Check:

* token counts don’t explode vs expectations
* model can copy numbers (simple “repeat the string” prompts)

(If this fails, your warmstart will be *actively harmful* for reasoning distillation later.) ([arXiv][4])

### B) Harmony formatting validity

Given a SYNTH prompt, sample:

* does it emit `reasoning` then `final` reliably?
* does it close tags correctly?

### C) Router health (after MoE conversion)

* expert usage histogram isn’t collapsed
* aux loss not diverging

---

## 8) The “plan of action” summary (what to implement, in order)

### Deliverables to assign to the team

1. **vpDense implementation**

   * `vpDense0-5_28` and `vpDense`
   * supports a “compat mode” (full attn + RoPE) for transplant healing

2. **Tokenizer transplant pipeline**

   * Use OMP transplantation (mergekit-tokensurgeon / or implement from paper)
   * Target tokenizer = HF `openai/gpt-oss-20b` tokenizer files ([Hugging Face][3])
   * Produce Qwen3‑o200k embeddings + lm_head

3. **Weight surgery script: Qwen3 → vpDense0-5_28**

   * attention head meanpool downmap (16→8 Q heads, 8→2 KV heads) ([Hugging Face][6])
   * FFN shrink 3072 → `I_target` via structured neuron selection
   * norms copied, new params zero-init

4. **Healing run**

   * short NTP mixed with Harmony/SYNTH
   * run Morph A then Morph B

5. **Depth expansion: vpDense0-5_28 → vpDense**

   * insert identity layers (zero o_proj and down_proj)

6. **Upcycle: vpDense → vpMoE**

   * replicate FFN → experts, add diversity noise
   * router uniform init + aux loss
   * adopt NVIDIA upcycling scaling ideas where possible ([arXiv][5])

7. **Now start your existing Stage 1→2→3 KD plan**

---

## 9) Why this is the best “risk/reward” version of Option B

* You’re exploiting the **hidden_size match (1024)** in Qwen3‑0.6B, which is the single biggest factor that makes weight reuse realistic. ([Hugging Face][1])
* You’re using a **principled tokenizer transplant** method that was designed for “cross-tokenizer KD / reuse weights.” ([arXiv][4])
* You’re explicitly managing the two biggest shocks:

  * **depth expansion** (28→80) via identity insertion
  * **attention pattern change** via staged morphing
* And you end at a checkpoint that’s “dense-compatible,” which aligns with the core idea behind MoE upcycling literature. ([arXiv][5])

---

If you want the next step from me: I can write a very explicit **“weight surgery spec”** (tensor shapes + exact reshape/meanpool formulas + what to do with biases + how to handle tied embeddings) in a format you can hand directly to whoever is implementing the checkpoint converter.

[1]: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json "config.json · Qwen/Qwen3-0.6B at main"
[2]: https://openai.com/index/introducing-gpt-oss/?utm_source=chatgpt.com "Introducing gpt-oss"
[3]: https://huggingface.co/openai/gpt-oss-20b/tree/main?utm_source=chatgpt.com "openai/gpt-oss-20b at main"
[4]: https://arxiv.org/abs/2506.06607?utm_source=chatgpt.com "Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit"
[5]: https://arxiv.org/abs/2410.07524?utm_source=chatgpt.com "Upcycling Large Language Models into Mixture of Experts"
[6]: https://huggingface.co/docs/transformers/en/model_doc/qwen3 "Qwen3"
[7]: https://www.arcee.ai/blog/breaking-down-model-vocabulary-barriers-with-tokenizer-transplantation "Arcee AI | Breaking Down Model Vocabulary Barriers with Tokenizer Transplantation"
