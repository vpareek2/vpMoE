# Student Architecture (draft)

This repo’s first hybrid-attention student is **GPT‑OSS‑20B‑initialized** and follows the “Arcee KDA distillation” spirit in `src/arcee_experiment.md`, but adapted to GPT‑OSS constraints and our chosen attention mix.

## Baseline

- Base implementation: HuggingFace Transformers `v4.57.6` vendored at `src/third_party/transformers`
- Weight source / initialization: `openai/gpt-oss-20b`
- Student checkpoint dtype: **BF16 only** (no MXFP4 on the student, at rest or in-memory)
- HF integration policy (this phase): **research-first** (Option A) — use our vendored Transformers + our model code; defer “vanilla upstream AutoModel” polish until we have a final checkpoint (target export path is NeMo).

## Layer Schedule (attention types)

We repeat a 4-layer block across the entire stack:

- `TPA, KDA, KDA, KDA` (i.e. KDA:TPA = 3:1)

Interpretation:

- **TPA layers** are the *softmax attention* layers (drop-in for GPT‑OSS attention, but with Tensor Product Attention internals).
- **KDA layers** are the *linear/recurrent* layers (Kimi Delta Attention).

Config representation (source of truth):

- Keep GPT‑OSS `layer_types` as the per-layer **mask schedule** (`sliding_attention` vs `full_attention`).
- Add one explicit per-layer list (name TBD) specifying **mechanism** per layer: `{tpa,kda}`.
- KDA layers ignore `layer_types`; TPA layers use it.

## TPA Integration (option A: true TPA)

We implement **true TPA** (factorized projections + TPA attention), not a “kernel-only” swap.

Implications:

- TPA introduces new learnable parameters (A/B factor projections). These are not 1:1 compatible with GPT‑OSS `q_proj/k_proj/v_proj`.
- Our initialization strategy must therefore rely on distillation + freezing policy (do not expect perfect weight transplant).
- **TPA formulation:** full contextual TPA (factorize Q/K/V) with linear A/B factors (no factor gating; no K/V factor sharing).
- **TPA ranks:** `q_rank = 32`, `rank = 4` (interpreted as `(RQ, RK, RV) = (32, 4, 4)`; see `src/open_questions.md` for the rank math).
- **TPA value dimension:** keep `head_v_dim == head_dim == 64`.
- **Sliding window = 128** is supported from the start for TPA layers (this is the production target behavior, not a later optimization).

## Positional Encoding Policy

- **TPA:** keep GPT‑OSS RoPE behavior (theta + scaling) and apply RoPE in the TPA path.
- **KDA:** **no RoPE**. KDA is order-aware via its recurrent/conv mechanism (not via softmax+RoPE).

## “Sink” Logit Policy (GPT‑OSS attention bias)

GPT‑OSS attention uses a learned per-head “sink” logit appended before softmax (see `transformers/models/gpt_oss/modeling_gpt_oss.py` in our vendored checkout).

- **TPA:** keep sinks (sinks are meaningful only in the softmax-logit domain).
- **TPA implementation:** use the LSE-gate equivalent (flash-friendly) rather than materializing an extra sink column. See `src/open_questions.md`.
- **KDA:** **do not** add sinks. KDA has no softmax-over-positions, so a “sink logit” has nowhere to apply.

We explicitly do **not** try to invent a “KDA sink analogue” initially; if needed later, it should be treated as a new mechanism (and justified experimentally), not a port of GPT‑OSS sinks.

## Weight Initialization / Parameter Mapping

We follow the Arcee recipe from `src/arcee_experiment.md`:

- **Copy from GPT‑OSS‑20B teacher weights:** embeddings, MoE MLP (router + experts), norm layers.
- **TPA layers (true/full contextual TPA):**
  - **TPA factor projections (`W_A_*`, `W_B_*`)**: Xavier uniform (paper/reference init).
  - **`o_proj`**: copy from GPT‑OSS (same shape/semantics).
  - **`sinks`**: copy from GPT‑OSS (same shape/semantics).
  - No exact transplant exists for GPT‑OSS `q_proj/k_proj/v_proj` → TPA factors because full contextual TPA projections are bilinear in `x`.
- **KDA layers:** initialize KDA-specific dynamics parameters from scratch (not copied), e.g. `A_log`, `dt_bias` (use the reference initialization from the chosen KDA implementation).

## Distillation Losses (Arcee mix)

We use Arcee’s recommended loss mix (see `src/arcee_experiment.md`):

- Cross-entropy (student hard labels): weight `0.2`
- KL (teacher logits distillation): weight `0.2`, temperature `2.0`
- Hidden-state cosine alignment: weight `0.6`

Layer mapping: `all` (match all layers 1:1).

## MoE Distillation Policy (router + experts)

In GPT‑OSS, the “MLP” block is an MoE (`router` + `experts`). Arcee’s “freeze MLP while attention learns” principle
maps to freezing **both** router + experts early.

We use a staged unfreeze policy:

- **Phase A (short context, stabilize attention):**
  - Train the new attention stack (TPA/KDA).
  - Freeze MoE completely: `model.layers.*.mlp.router.*` and `model.layers.*.mlp.experts.*`.
  - Keep router aux loss off (do not request / train on router logits).
- **Phase B (long context, recover capacity):**
  - Unfreeze experts.
  - Keep router frozen (avoid routing churn while the representation space is still shifting).
- **Phase C (optional, late):**
  - Unfreeze router only if needed.
  - Prefer a teacher-faithful router regularizer (router distillation) over “free routing”, and add load-balancing only as
    a small guardrail if collapse appears.

Why we do **not** unfreeze experts + router together from the start:

- Representation shift from the attention swap + routing drift are two coupled instabilities.
- It becomes easy for the model to “solve” KD losses by changing routing/expert behavior instead of making the new
  attention path teacher-like.
- Early router freedom increases collapse risk (dead experts) and makes debugging ambiguous (attention vs routing).

## Attention Masking Compatibility

- **KDA training kernels only accept a 2D padding mask** (`[batch, seq]` with 0/1). They do **not** support arbitrary causal/sliding-window masks.
- **TPA** must support GPT‑OSS causal masking + sliding window behavior (window size **128**) from the start.

## Inference / Generation (early requirement)

We support **early `generate`-style evaluation** (and will benchmark after each stage), so we must implement caching and
incremental decoding for the mixed stack:

- **KDA layers:** cache is not KV; it is recurrent state (+ optional short-conv state) per layer.
- **TPA layers:** cache must support sliding-window attention efficiently (do not store unbounded history).
- The student must expose a single coherent HF-compatible caching surface so `generate` works without custom caller code.

Exact cache class / structure is tracked in `src/open_questions.md`.

## Multi-Token Prediction (MTP) (post-training)

We **do not** use MTP during the teacher-forced distillation phase; distillation is focused on matching GPT‑OSS
**next-token prediction (NTP)** behavior as closely as possible.

- MTP remains a candidate **post-training auxiliary objective** (training-only; not an inference-time multi-token decoding feature).
- Post-training MTP specifics (offsets, head design, loss weighting, masking) are tracked in `src/open_questions.md`.

## Quantization Policy (MXFP4)

GPT‑OSS checkpoints may be distributed in **MXFP4** form, but we do **not** use MXFP4 for the student.

- The student must be a pure **BF16** module graph and a pure **BF16** checkpoint.
- If the upstream GPT‑OSS‑20B weights are only available in MXFP4, we first produce a **BF16 export** and treat that
  export as the actual initialization source for all student work (training/eval/checkpointing).

## Open Questions

All unresolved decisions are tracked in `src/open_questions.md`.
