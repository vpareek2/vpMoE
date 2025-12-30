# Implementing our architecture within Megatron-LM

This document is a working list for implementing the vpMoE architecture within Megatron-LM. The reason to use this is to keep track of our progress and overall plan of implementation.

Our goal is to write code in the same style as the current Megatron-LM and not take any shortcuts when implementing.

## Tokenizer (o200k Harmony)

We must run the student with the **same tokenizer family as the teacher**: **o200k Harmony**, with **padded vocab size = 201088** (`docs/architecture.md`). Even if we pre-tokenize into Megatron `.bin/.idx`, a correct tokenizer implementation is still required for:

- Building the `.bin/.idx` datasets deterministically.
- Any online/on-policy components later (rollouts, scoring, format validation).
- Checkpoint provenance (tokenizer identity/version) and eval-time detokenization.

### Requirements

- **Exact tokenization parity** with GPT‑OSS (no “close enough” tokenizer).
- **Deterministic render → tokens**: given a Harmony conversation object, the rendered byte/string form and resulting token IDs must be stable across machines/runs.
- **Explicit IDs for special tokens** used by Harmony/o200k (BOS/EOS/EOD/PAD, etc.) and a clear definition of what we treat as document boundary (`eod` in Megatron terminology).
- **Padded vocab size** is treated as a locked invariant: `201088`. Any padding logic must be deterministic and must not change token IDs.
- **High performance implementation**

## GRAPE (positional encoding)

Our locked baseline uses **GRAPE‑M** (local layers) + **GRAPE‑A** (global layers) (`docs/architecture.md`). Megatron does not support this out of the box in our current tree, so we need to **add GRAPE positional encoding support** (and wire it into the local/global attention schedule) as part of the Megatron integration.

## Tensor Product Attention (TPA) (local sliding-window)

Our locked baseline uses **sliding-window local attention** with **TPA**, window size **128**, under a **3:1 local:global** schedule (`docs/architecture.md`). Megatron’s existing window attention is not TPA, so we need to **implement TPA for local layers** and hook it into Megatron’s attention backend selection for the windowed (local) blocks.

## Optimizer (Normuon + Polar Express)

Our baseline optimizer is **Normuon** with **Polar Express**, which Megatron does not provide out of the box. We need to **add this optimizer** (and its required scheduler/param-group behavior) to Megatron’s optimizer stack.

## Notes

- Verify `--softmax-type learnable` matches the intended GPT‑OSS-style learnable softmax/logit bias behavior (shape + where it is applied).
- Verify “always-on shared expert” semantics map cleanly to Megatron’s shared-expert implementation (and that `shared_expert_size=512` corresponds to the right intermediate size).
- Verify dense warmup + schedules compose correctly (first dense FFN, then MoE layers; plus 3:1 local/global attention schedule).
