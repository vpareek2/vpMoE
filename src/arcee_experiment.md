````md
# Distilling Kimi Delta Attention into AFM-4.5B  
**(and the Tool We Used to Do It)**

**Charles Goddard**  
*December 15, 2025*

Learn how Kimi Delta Attention was distilled into AFM-4.5B using knowledge distillation, long-context training, and Arcee’s open-source **DistillKit**.

---

## Overview

Moonshot AI recently released a strong paper (and model) introducing **Kimi Delta Attention (KDA)**, an extension of **Gated DeltaNet**. Results are especially compelling in the now-classic **3-to-1 interleaved local/global hybrid attention** configuration.

The pretrained model is excellent, but more importantly, Moonshot open-sourced both **training and inference kernels**. Rather than pretraining a model from scratch, I chose to **convert AFM-4.5B-Base into a hybrid KDA + full-attention transformer via knowledge distillation**, inspired by the *RADLADS* paper.

---

## Terminology

- **Full attention**: Standard global self-attention  
- **NoPE**: RoPE removed, with no replacement positional embedding scheme  

---

## Creating the Student

### Modeling Code

Thanks to **flash-linear-attention**, creating the student architecture was straightforward. Moonshot AI contributed kernels and a nearly drop-in layer implementation.

Required changes:
- Plug in KDA layers
- Remove RoPE
- Add configuration for which layers use **KDA vs. full attention**

### Weight Initialization

Most weights were copied directly from the teacher:
- MLPs
- Embeddings
- Norm layers

KDA-specific parameters (e.g. `A_log`, `dt_bias`) were initialized from scratch.

Because current KDA implementations lack true **GQA**, grouped-head projections were expanded to MHA-shaped projections by repeating weights. This increases total parameters from **4.5B → ~5B**, but avoids deeper architectural surgery.

---

## Distilling the Knowledge

The *RADLADS* pipeline proposes three stages:

1. **Attention Hidden State Alignment** (attention-only training)
2. **Full-parameter distillation**
3. **Long-context fine-tuning**

I collapsed the first two stages into **one phase**:
- MLP parameters frozen
- **Cosine loss** instead of MSE on hidden states
- Long-context fine-tuning handles MLP adjustment

### Hypothetical Distillation Config

Imagine a world where the following YAML suffices:

```yaml
project_name: distillkit-afm-kda
model: arcee-train/afm-4p5b-kdanope-untrained
trust_remote_code: true

frozen_res:
  - embed_tokens
  - lm_head
  - norm\.weight
  - ^model\.layers\.[0-9]+\.mlp\..*

output_path: /workspace/models/afm-4p5b-kda-hsd
use_flash_attention: true
sequence_length: 2048

dataset:
  train_dataset:
    repo_id: arcee-train/afm-autodistill-mix-v0
    split: train
  seed: 42

loss_functions:
  - function: cross_entropy
    weight: 0.2
  - function: kl
    weight: 0.2
    temperature: 2.0
  - function: hs_cosine
    weight: 0.6

layer_mapping: all

teacher:
  kind: hf
  path: arcee-ai/AFM-4.5B-Base
  kwargs:
    attn_implementation: flash_attention_2
    torch_dtype: bfloat16

training_args:
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 1.0e-3
  warmup_ratio: 0.025
  bf16: true
  max_grad_norm: 0.5
  optim: adamw_torch
  gradient_checkpointing: true
````

### Results

With **~300M tokens**, frozen MLPs, and hidden-state cosine loss:

* Token-averaged KL ≈ **0.2**
* Comparable to multi-stage pipelines
* Diminishing returns beyond this point

Final recipe:

* **One-phase distillation**
* **~1B tokens at 32k context**
* Learning rate: `1e-5`

---

## Student Variants Tested

1. **AFM-4.5B-KDA-NoPE (Hybrid)**

   * 3 KDA layers + 1 full-attention NoPE layer (×9)

2. **AFM-4.5B-KDA-FLP (Front-Loaded Full Attention)**

   * First 4 layers full-attention NoPE
   * Remaining layers mixed
   * Total full-attention layers: 9

3. **AFM-4.5B-KDA-Only**

   * All layers use KDA

---

## Benchmark Observations

* Teacher unsurprisingly performs best (8T tokens vs ~1B)
* **FLP performs extremely poorly**, unexpectedly
* Hybrid > pure-KDA on math benchmarks
* Knowledge benchmarks show near parity between hybrid and KDA-only

Conclusion:
Hybrid **KDA + NoPE full attention recovers significantly more performance** under equal training budgets.

---

## Long-Context Performance

Evaluated using **RULER NIAH** (exact-match retrieval).

### Key Findings

* All models: **100% single-needle retrieval up to 65k**
* **Pure KDA** degrades smoothly on multi-needle tasks
* **Hybrid model crashes past 32k**
* Likely cause: NoPE layers failing to generalize without exposure

Inference:

> KDA generalizes natively to longer contexts, while NoPE layers require training at ≥2× inference length.

---

## Why Do This?

Because **Kimi Delta Attention is absurdly fast**.

* **~8k tokens/sec per H100 (training)**
* Near-free inference
* Much smaller KV cache
* Delightful performance characteristics

Also: because it’s neat.

---

## DistillKit Update

DistillKit began as open-source scripts (Aug 2024). Internally, we evolved it into a **full-scale distillation system** supporting:

* Online & offline distillation
* Logit compression (polynomial approx + quantization + bit-packing)
* Composable losses (KL, JSD, TVD, ranking, hidden-state alignment)
* Sparse or dense distributions
* HF integration
* Scale-ready tooling

### Quick Start

```bash
git clone https://github.com/arcee-ai/distillkit.git
cd distillkit
pip install -e .
distillkit config.yaml
```

Optional teacher capture support:

```bash
pip install -e ".[capture]"
```

---

## Conclusion

Open science is good.
Post-hoc alchemy is good.
Doing weird things to open models is **very** good.

### Artifacts

* **AFM-4.5B-Base-KDA-NoPE**
* **AFM-4.5B-Base-KDA-Only**
* **DistillKit**

---

© 2026 Arcee AI. All rights reserved.

```

---