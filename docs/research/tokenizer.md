https://arxiv.org/pdf/2511.20849

# Tokenization & Distillation Strategy — VPMoE

Status: archived exploration. Current baseline uses the teacher-matched **o200k Harmony** tokenizer (see `docs/overview.md` and `docs/architecture.md`).

## 1. Motivation

The VPMoE model is designed to be **deep, sparse, and compute-efficient**, with most of its reasoning capacity living in depth and conditional MoE layers rather than in always-on parameters.

In this regime, **vocabulary size becomes a first-order design choice**, because:

* Embeddings and the LM head are **always active**
* Large vocabularies dominate the *active parameter budget*
* Sparse MoE capacity is crowded out by dense lexical parameters

This document defines a tokenizer and distillation strategy that:

* reduces always-on parameters
* preserves (or improves) reasoning performance
* remains compatible with large teacher models
* avoids tight tokenizer coupling

---

## 2. Key Observation

**Vocabulary size does not directly increase intelligence**, especially for small or sparse models.

Instead:

* reasoning quality is driven by **depth, training signal, and compute allocation**
* large vocabularies mainly provide lexical compression and rare-token coverage
* for KD and reasoning-heavy models, smaller vocabularies often improve data efficiency

This is supported empirically by:

* PleIAs models (Baguettotron, Monad) using ~64k vocab
* Recent work on efficient tokenizers
* Practical success of sequence-level and on-policy distillation

---

## 3. Tokenizer Design Goals

The tokenizer should:

* Minimize **always-on embedding + head parameters**
* Produce **reasonably short sequences** (but not necessarily minimal)
* Train well under **KD + mid-training**, not full pretraining
* Be **decoupled from teacher tokenization**
* Support reasoning traces and instruction-style text well

---

## 4. Target Vocabulary Size

### Baseline Target

* **50k–65k tokens**

This range provides:

* ~3× reduction in embedding/head params vs 200k vocab
* Strong token frequency statistics
* Acceptable sequence length inflation
* Good balance for reasoning + instruction workloads

### Aggressive Option (Later)

* **~32k tokens**
* Higher sequence length, but maximal parameter savings
* Only advisable after baseline stability is proven

---

## 5. Tokenizer Algorithm Options

### Option A — Standard BPE / Unigram (Safe Baseline)

* Train tokenizer on:

  * KD corpus (teacher-generated text)
  * mid-training corpus
* Simple, robust, well-understood
* Recommended for first experiments

### Option B — Length-Optimized / “Large Token” Tokenizers

* Tokenizers that explicitly optimize for **fewer tokens per character**
* Can reduce sequence length by ~10–20% at the same vocab size
* Promising for long-context + sparse models
* Should be evaluated experimentally

### Option C — Pruned Teacher Tokenizer

* Start from teacher tokenizer (e.g. GPT-OSS)
* Prune rare tokens to reach ~50k
* Preserves some lexical alignment
* More complex to implement and maintain

---

## 6. Distillation Strategy Overview

To decouple the student tokenizer from the teacher tokenizer, we use a **hybrid distillation strategy**.

### Phase 1 — Off-Policy Sequence-Level KD

* Teacher provides **text sequences**, not logits
* Student trains with standard NLL on:

  * teacher-generated text
  * mixed with mid-training / “pretraining-like” corpus
* Tokenizer mismatch is allowed because training is text-based

This phase builds:

* language modeling ability
* basic reasoning patterns
* stable representations

---

### Phase 2 — On-Policy Distillation (OPD)

* Student generates text using **its own tokenizer**
* Teacher evaluates student outputs at the **string level**

  * scoring
  * ranking
  * critique
* Student is updated via policy improvement (e.g. OPD / GRPO-style)

Key property:

* **No tokenizer alignment required**
* Teacher never needs to predict student tokens
* Intelligence transfers at the behavioral level

This phase transfers:

* reasoning style
* planning behavior
* decision heuristics

---

## 7. Why This Works Well for Sparse MoE

This setup is especially well-suited to VPMoE because:

* Sparse MoE benefits from **strong, dense supervision**
* KD reduces router instability
* Smaller vocab frees capacity for:

  * attention
  * expert FFNs
  * depth-based computation
* OPD avoids token-level coupling constraints

Net effect:

> More reasoning capacity per FLOP, fewer wasted always-on parameters.

---

## 8. Evaluation Criteria for Tokenizer Experiments

When comparing tokenizers (e.g. 200k vs 64k vs 50k):

Track:

* tokens per character
* sequence length distribution
* embedding + head parameter count
* training stability (loss curves)
* reasoning benchmarks (e.g. GSM8K, MMLU-like tasks)
* long-context degradation (with SWA)

Tokenizer changes should be judged on **end-to-end model quality**, not just compression.

---

## 9. Recommended Experiment Order

1. Train baseline VPMoE with **~64k tokenizer**
2. Compare against current **~200k tokenizer** using same KD data
3. If quality holds or improves:

   * try **~50k tokenizer**
4. Introduce OPD in post-training to recover any lost behavior
5. Only then consider more aggressive tokenizer designs

---

## 10. Summary

* Large vocabularies are not required for intelligence in sparse, reasoning-focused models
* Smaller vocabularies significantly reduce always-on parameters
* Sequence-level KD + on-policy distillation removes tokenizer matching constraints
* A **50k–65k tokenizer** is a strong, low-risk target for VPMoE
* This strategy aligns compute with reasoning rather than lexical storage

**This design intentionally prioritizes reasoning efficiency over surface-level lexical capacity.**
