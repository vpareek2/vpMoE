Got it, Oen. Here’s a **planning doc for the DistillKit/BF16 distillation phase**—i.e., everything you need to do **before** you switch to an NVIDIA stack for NVFP4/NVFP4-robust post-training.

I’m going to assume the target student is **“GPT‑OSS‑like in every way except attention + positional handling”**:

* keep MoE, widths, norms, routers, tokenizer, Harmony output format
* **swap attention** to **SWA(128)+TPA** on the “sliding” layers and **KDA** on the “full” layers
* remove RoPE globally (no RoPE in KDA), keep **RoPE or GRAPE‑M inside SWA**

This mapping is especially clean because **gpt‑oss‑20b already alternates `sliding_attention` and `full_attention` and uses `sliding_window: 128`**. ([Hugging Face][1])

Also: **gpt‑oss must be trained/used in Harmony format**. ([GitHub][2])

---

# Distillation Phase Plan (BF16) — GPT‑OSS → Student (SWA128+TPA + KDA), using DistillKit

## 0) Outcome and success criteria

### Deliverable

A **BF16 student checkpoint** in HF format (plus config + tokenizer) that:

* speaks **Harmony** with visible `analysis` + `final`
* matches teacher behavior closely on:

  * token-level KL on held-out mixture
  * reasoning benchmark deltas (GSM8K-ish, BBH-ish, etc.)
* is stable at your intended context lengths (at minimum 2k→8k; ideally some longer-context exposure)
* is ready to become the “base” for **NVIDIA-framework post-training / NVFP4-robustness work** later

### “Pass/fail” gates

1. **Model parity gate:** student loads + forward pass + generate works with Harmony inputs
2. **Distillation gate:** KL drops consistently; no routing collapse
3. **Behavior gate:** teacher-like instruction following + reasoning style (visible analysis remains coherent)
4. **Readiness gate:** checkpoint exports cleanly and can be reloaded deterministically

---

## 1) Tooling & repo setup

### 1.1 DistillKit install + capabilities to lean on

DistillKit supports **online and offline distillation** and has **logit compression** for offline teacher outputs. ([GitHub][3])
It’s built on HF Transformers/TRL/Accelerate. ([GitHub][3])

**Action items**

* Pin a working environment:

  * PyTorch + CUDA
  * Transformers version that cleanly supports `gpt_oss` modeling + Harmony handling (use current stable / pinned commit)
  * DistillKit pinned to a commit/sha
* Decide early whether you will run:

  * **Online distillation** (teacher runs during training), or
  * **Offline distillation** (teacher logits captured + compressed)

**Recommendation**

* Start with **online using gpt‑oss‑20b** (fast iteration).
* Switch to **offline** for **gpt‑oss‑120b** (unless you have enough VRAM and want online). DistillKit’s compression exists specifically to make offline feasible. ([GitHub][3])

---

## 2) Student architecture implementation (the “attention surgery”)

### 2.1 Architecture mapping (minimal-diff, GPT‑OSS-native)

From the gpt‑oss‑20b config:

* `layer_types` alternates `sliding_attention` and `full_attention`
* `sliding_window: 128`
* `attention_bias: true` ([Hugging Face][1])

**Student mapping**

* `sliding_attention` → **SWA(128) + TPA**

  * keep **attention bias**
  * keep **RoPE or GRAPE‑M inside SWA**
* `full_attention` → **KDA**

  * **no RoPE / NoPE behavior**
  * KDA state update + recurrence

This is the cleanest “leave everything the same except attention” approach.

### 2.2 Implementation strategy

You have two viable paths:

**Path A (fastest): reuse existing KDA layer implementation**
Arcee reports they could create the student architecture quickly using the open KDA layer implementation from `flash-linear-attention` and then just “plug it in,” remove RoPE, and add a layer-type config. ([Arcee AI][4])

**Path B (your own): implement KDA + TPA kernels**
If you already have kernels or want bespoke behavior, fine—but expect more debugging before training.

**Action items**

* Implement student self-attn module(s) with a clean switch:

  * `layer_types[i] == sliding_attention` → SWA+TPA forward
  * `layer_types[i] == full_attention` → KDA forward
* Confirm shape-compatibility with gpt‑oss:

  * hidden size, head dim, GQA groups, etc.
* Confirm the student still supports:

  * caching behavior needed for training/generation
  * Harmony prompt packing
  * attention bias parity

### 2.3 Weight initialization plan (critical to avoid chaos)

Goal: copy everything that is unchanged.

**Copy directly**

* embeddings
* lm_head
* norms
* MoE experts + router
* all non-attn params

**Attention params**

* For SWA+TPA: copy teacher q/k/v/o projections as-is (windowing changes behavior but weights are a good init)
* For KDA:

  * if KDA implementation doesn’t support GQA the same way, you may need to expand/reshape weights
  * initialize KDA-unique parameters (e.g., time constants / state params) from scratch
    Arcee used this exact pattern: copy what’s compatible, initialize KDA-specific params fresh. ([Arcee AI][4])

**Gate:** run forward pass on a fixed token batch and confirm:

* no NaNs
* output logits finite
* routing logits sane (no expert collapse)

---

## 3) Data plan (Harmony + visible reasoning)

### 3.1 Non-negotiable: Harmony formatting everywhere

OpenAI explicitly says **gpt‑oss should only be used with Harmony format**. ([GitHub][2])

**Action items**

* Pick/implement a canonical “renderer” that converts a conversation into the exact Harmony text / token stream you will train on.
* Standardize:

  * system → user → assistant messages
  * `analysis` and `final` channels
  * stop tokens / EOS handling

### 3.2 Dataset sourcing

For *this distillation phase*, you want:

* a broad instruction/reasoning mixture
* plus long-context examples (to ensure KDA learns useful memory behavior)

**Two practical options**

1. **Use pre-distilled datasets** (fast bootstrap)
2. **Generate your own teacher traces** (higher quality; lets you control style, lengths, and correctness filters)

**Action items**

* Build a dataset manifest:

  * source(s)
  * license
  * Harmony conversion method
  * expected length distribution
* Create:

  * train split
  * held-out validation split (must be stable across runs)

### 3.3 Add supervision metadata for your loss

Because you’re keeping visible reasoning, you’ll want at least:

* token mask for `analysis` span
* token mask for `final` span
* optional per-token “chunk weights” for analysis (to avoid giving equal weight to filler)

**Action items**

* During preprocessing, produce for each packed sequence:

  * `input_ids`
  * `labels`
  * `analysis_mask` (0/1 per token)
  * `final_mask` (0/1 per token)
  * optional: `analysis_weight` (float per token)

> DistillKit has flexible losses, but Harmony-aware masking/weighting is usually something you implement via labels/masks/custom loss hooks. Plan for that as a small extension.

---

## 4) Teacher signal strategy (online vs offline)

DistillKit supports:

* **Online distillation** (teacher runs alongside training)
* **Offline distillation** from pre-captured teacher outputs, with logit compression. ([GitHub][3])

### 4.1 Recommended progression

**Step 1 (iteration): online with gpt‑oss‑20b**

* fewer moving parts
* fast debugging
* lets you validate loss masks + architecture quickly

**Step 2 (quality): offline or online with gpt‑oss‑120b**

* If VRAM is tight: capture logits once → reuse many times (this is what DistillKit’s compression system is for). ([GitHub][3])

### 4.2 Decide: what distribution to store (if offline)

* Dense logits are huge (vocab ~201k)
* DistillKit supports compressed sparse/top‑k + polynomial approximation to make storage feasible. ([GitHub][3])

**Action items**

* Choose compression knobs (k, exact_k, polynomial terms, etc.)
* Validate decompression numerics on a small slice (sanity-check KL stability)

---

## 5) Distillation objectives and training stages

This is the heart of the phase. The goal is to **teach the new attention stack** to behave like the teacher before you later do NVFP4 robustness work.

### 5.1 Stage structure

Arcee describes a RADLADS-inspired multi-stage approach and also notes you can collapse stages by freezing MLP and using hidden-state cosine alignment during distillation. ([Arcee AI][4])

A solid “minimal risk” plan for your attention surgery:

#### Stage A — Attention adaptation (freeze most things)

* Trainable:

  * attention modules (SWA+TPA + KDA params)
  * optionally: attention norms if needed
* Frozen:

  * embeddings, lm_head
  * MoE experts (and optionally router at first)

Loss:

* hidden-state alignment (cosine is a good default per Arcee’s report) ([Arcee AI][4])
* plus light KL + CE

Sequence length:

* start shorter (e.g., 2k) to stabilize training

**Gate:** KL/hs-loss decreases smoothly; no exploding grads; routing remains healthy.

#### Stage B — Full distillation (unfreeze MoE experts; possibly router)

* Trainable:

  * everything except maybe embeddings/lm_head (your call)
* Loss emphasis:

  * KL on outputs
  * CE component to keep “hard target” stability
  * optionally keep a smaller hs-loss term

Sequence length:

* step up to 4k–8k packed

**Gate:** downstream evals stop being “dumb fast” and start looking teacher-like.

#### Stage C — Long-context exposure (optional but recommended)

Even if you’ll do serious long-context/NVFP4 later, give KDA some long sequences now so it’s not “untrained at length”.

* length: 16k–32k (whatever is feasible)
* objective: keep KL + CE; hs-loss optional

**Gate:** no catastrophic failure when you increase length; long-context retrieval doesn’t instantly collapse.

---

### 5.2 Harmony-visible reasoning loss plan (analysis vs final)

Since you want reasoning visible:

* apply **loss to both analysis and final**, but **not necessarily equally**

**Default weighting (practical)**

* `final` gets higher weight (decision anchor)
* `analysis` gets medium weight + optional chunk weighting

**Action items**

* Implement token-masked loss:

  * `L_final = KL * final_mask`
  * `L_analysis = KL * analysis_mask * analysis_weight`
* Implement chunk weighting for analysis (simple heuristics are fine in this phase):

  * upweight math/code-like spans
  * downweight boilerplate (“let’s think…”)

> This is where you get “intelligence transfer faster” from reasoning tokens without forcing verbatim CoT in a brittle way.

---

## 6) DistillKit configuration & runs

### 6.1 What to produce

* A small set of **versioned YAML configs**:

  * `stageA_attn_only.yml`
  * `stageB_full_distill.yml`
  * `stageC_long_context.yml`

### 6.2 Core config fields you’ll standardize

* model path (student)
* teacher kind:

  * `hf` for online, or
  * `dataset` for offline prepacked logits
* loss_functions list (KL / CE / hs_cosine)
* layer mapping for hidden-state alignment (1:1 mapping is simplest if depths match)
* training_args:

  * bf16: true
  * packing: true
  * gradient checkpointing
  * save_steps/logging

DistillKit is explicitly designed to drive runs from YAML configs like this. ([Arcee AI][4])

---

## 7) Evaluation & diagnostics (must-have for this phase)

### 7.1 Distillation health metrics

Track continuously:

* token-avg KL on held-out shard
* CE / perplexity on held-out shard
* router load balance:

  * fraction per expert
  * entropy
  * % tokens routed to top experts
* “analysis length” distribution:

  * mean/median tokens in analysis vs final
* failure modes:

  * repetition
  * empty final
  * hallucinated format breaks

### 7.2 Quick behavioral eval suite

Keep a small, stable suite you can run every checkpoint:

* short math (GSM8K-like)
* short logic
* a couple long-context probes (needle-ish)
* instruction following sanity

**Gate:** student stays in Harmony format and gives usable outputs.

---

## 8) Artifacts & handoff package for the NVFP4/NVIDIA phase

Even though NVFP4 comes later, this phase should end with a clean “handoff bundle”:

### 8.1 Model artifacts

* BF16 student checkpoint
* full config.json (including your new attention layer typing)
* tokenizer files
* generation config / Harmony defaults

### 8.2 Training artifacts

* all DistillKit YAML configs
* dataset manifest + preprocessing scripts
* evaluation harness + prompt set
* notes on:

  * what froze/unfroze
  * what loss weights worked
  * any gotchas (NaNs, instability thresholds)

### 8.3 Optional but helpful

* if you captured offline logits:

  * store compression config and a tiny sample pack for reproducibility

---

## 9) Risk register (distillation-phase specific)

### Risk: KDA ↔ GQA mismatch

Mitigation:

* implement GQA-compatible KDA or carefully reshape/expand init weights
* verify forward numerics early

### Risk: MoE routing collapse early

Mitigation:

* freeze experts (and optionally router) in Stage A
* unfreeze gradually
* monitor expert load balance constantly

### Risk: Harmony formatting drift

Mitigation:

* unit tests: render → tokenize → detokenize → parse channels
* add a “format compliance” evaluator (cheap string checks)

### Risk: Overfitting to verbose analysis

Mitigation:

* weight `final` higher
* apply chunk weighting to `analysis`

---

# Checklist summary

## Must-do

* [ ] Implement student attention modules (SWA128+TPA / KDA) with GPT‑OSS-compatible shapes
* [ ] Copy/init weights cleanly
* [ ] Harmony renderer + token masks for analysis/final
* [ ] DistillKit configs for Stage A/B (+ optional long-context Stage C)
* [ ] Stable eval harness + KL tracking
* [ ] Export BF16 checkpoint + config bundle

## Nice-to-have

* [ ] Offline teacher logits for 120b using DistillKit compression
* [ ] Multi-teacher schedule: 20b online → 120b offline polish
* [ ] Early long-context exposure to prevent KDA “cold start” later

---

If you want, I can also draft **three concrete DistillKit YAMLs** (Stage A/B/C) tailored to:

* gpt‑oss teacher IDs
* Harmony-packed datasets
* your freeze patterns
* your intended KL/CE/hs_cosine weighting scheme

…and I’ll keep them consistent with what Arcee shows in their KDA distillation example config. ([Arcee AI][4])

[1]: https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json "config.json · openai/gpt-oss-20b at main"
[2]: https://github.com/openai/gpt-oss?utm_source=chatgpt.com "openai/gpt-oss: gpt-oss-120b and gpt-oss-20b are two ..."
[3]: https://github.com/arcee-ai/DistillKit "GitHub - arcee-ai/DistillKit: An Open Source Toolkit For LLM Distillation"
[4]: https://www.arcee.ai/blog/distilling-kimi-delta-attention-into-afm-4-5b-and-the-tool-we-used-to-do-it "Arcee AI | Distilling Kimi Delta Attention into AFM-4.5B"
