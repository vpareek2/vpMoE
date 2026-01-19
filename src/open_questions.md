# Open Questions / Decisions To Lock (WIP)

This document tracks **unresolved** design questions for the GPT‑OSS‑20B‑initialized hybrid student.

Decisions already made live in `src/architecture.md`.

## TPA (true TPA) details

- [x] **Which exact TPA formulation are we using?**
  - **Full contextual TPA** (factorize **Q, K, and V** per the TPA paper’s Eq. 3.1).
  - **Linear factors** (no sigmoid/softmax gating on factors).
  - **No non-contextual factor variants** (A/B are both contextual).
  - **No K/V factor sharing** (do not force `B_k == B_v`).
  - **RoPE stays on softmax layers**: apply RoPE in the TPA path (row-wise on B factors, as in the paper’s “direct integration”).
- [x] **TPA ranks:** choose `rank` and `q_rank` (and whether they vary by layer).
  - Locked (global, do not vary by layer initially):
    - `(RQ, RK, RV) = (32, 4, 4)`
    - Implementation naming: `q_rank = 32`, `rank = 4` (where `rank` is used for both K and V ranks).
  - **Rank math (match GPT‑OSS projection parameter counts; maximize expressiveness without increasing projection params):**
    - GPT‑OSS: `H = 64`, `D = head_dim = 64`, `KV_heads = 8`.
    - Baseline (GPT‑OSS) projection params (ignoring bias):
      - `q_proj`: `d_model * (H * D) = d_model * 4096`
      - `k_proj` / `v_proj`: `d_model * (KV_heads * D) = d_model * 512`
    - Full TPA factor projection params (ignoring bias):
      - `Q`: `W_A_q` + `W_B_q` ⇒ `d_model * RQ * (H + D) = d_model * RQ * 128`
        - Set `RQ * 128 = 4096` ⇒ `RQ = 32`
      - `K` / `V`: `W_A_k` + `W_B_k` ⇒ `d_model * RK * (H + D) = d_model * RK * 128`
        - Set `RK * 128 = 512` ⇒ `RK = 4` (same for `RV`)
- [x] **TPA value dimension:** do we keep `head_v_dim == head_dim` or use an expansion (and if so, where)?
  - Keep `head_v_dim == head_dim` (E = D = 64).
- [x] **TPA parameter init:**
  - **TPA factor projections (`W_A_*`, `W_B_*`)**: Xavier uniform (as in the TPA paper / reference code).
  - **TPA output projection (`o_proj`)**: copy from GPT‑OSS initialization source (same shape/semantics).
    - Rationale: reduces needless drift in the residual stream interface while the new attention internals learn via KD.
  - **Notes on “teacher-derived warm start”**:
    - Full contextual TPA projections are bilinear in `x` (via `A(x)^T B(x)`), so there is no exact 1:1 transplant from GPT‑OSS linear `q_proj/k_proj/v_proj`.
    - If we later want a “warm bias” toward GPT‑OSS, it must be heuristic (e.g., subspace-seeding `W_B_*`), not an exact equivalence.
- [ ] **TPA training kernel path:** deferred (likely a TPA-native flash-style kernel; decide after design is fully locked).
- [x] **Sliding window size:** TPA layers use `sliding_window = 128` (match GPT‑OSS; see `src/architecture.md`).
- [ ] **Sliding-window=128 semantics:** confirm exact masking/window semantics for both training and `generate` (must match GPT‑OSS sliding attention).

## Sink handling (TPA-only)

- [x] **Exact sink math (GPT‑OSS semantics):** implement via an LSE gate (flash-friendly).
  - GPT‑OSS eager semantics: append a per-head sink logit `s_h` as an extra “(M+1)-th key”, softmax over `M+1`, then **drop** the sink probability (no renormalization).
  - Equivalent formulation (preferred for our kernels):
    - Let masked logits over real tokens be `L[b,h,q,:]` and `logZ = logsumexp(L, dim=-1)`.
    - Define `g[b,h,q] = 1 / (1 + exp(s_h - logZ)) = sigmoid(logZ - s_h)`.
    - Compute standard attention output over real tokens: `y = softmax(L) @ V`.
    - Apply sink by scaling: `y_sink = g * y`.
  - Notes:
    - Sink is **not** masked; masking applies only to real-token logits in `L`.
    - Applying `g` after dropout/matmul is equivalent to scaling the attention weights by `g` (dropout is multiplicative).
- [x] **Kernel integration:** keep a flash-style kernel and validate equivalence.
  - Preferred integration point: kernel computes (or already has) per-(b,h,q) `logZ` / LSE; apply `g` as a post-kernel multiply on the output.
  - When we implement, add a numerical equivalence test that compares:
    - reference “append sink logit then drop sink prob” vs
    - LSE-gated output (`y_sink = g * (softmax(L) @ V)`),
    across masks (causal + sliding-window=128) and dtypes (bf16).
- [x] **Sink parameter init:** copy sink logits from GPT‑OSS initialization source for TPA layers (same shape/semantics).

## BF16-only student initialization (no MXFP4 anywhere on student)

- [x] **Reality check: HF `openai/gpt-oss-20b` is not BF16-only.**
  - The safetensors shards contain **mixed dtypes** (`BF16` + `U8`).
  - The `U8` tensors correspond to MoE expert weight storage (e.g. `model.layers.*.mlp.experts.{gate_up_proj,down_proj}_{blocks,scales}`), i.e. quantized weights.
  - Therefore, `openai/gpt-oss-20b` cannot be used as a **BF16-only** initialization source without a conversion step.
- [x] **Reality check: local `gpt-oss-20b/original` is also mixed precision.**
  - `original/model.safetensors` contains **BF16 + U8** tensors (U8 corresponds to MLP blocks/scales).
  - `original/dtypes.json` labels these as `FP4` blocks + `UE8` scales for MLP weights, confirming quantized MoE storage.
- [x] **BF16 init source:** we will produce a canonical **BF16 export** by dequantizing the MXFP4/FP4 MoE weights to BF16 and resaving a pure-BF16 checkpoint.
  - Rationale: dequantization does not introduce new distortion beyond the existing quantization error; it makes that fixed error explicit in BF16 form.
  - This BF16 export becomes the **only** supported initialization source for the student.
- [ ] **BF16 export artifact:** define the one-time “MXFP4 → BF16 export” artifact format, location, and provenance rules (hashing, config, tokenizer id).
- [ ] **Verification:** how do we assert the student checkpoint contains *no* MXFP4 blocks/scales (and no quantizer config)?

## Layer specification surface (minimal config)

- [x] **How do we specify per-layer attention mechanism?** (TPA vs KDA) and mask type (sliding vs full) with minimal new knobs.
  - Follow the Kimi-Linear pattern: store an **explicit per-layer attention mechanism schedule** as the source of truth (no inference from ratios at load time).
  - Keep GPT-OSS `layer_types` as the teacher-faithful **mask schedule** (`sliding_attention` vs `full_attention`).
  - Add exactly one new per-layer list field (name TBD, e.g. `layer_attn_impls`), length = `num_hidden_layers`, values in `{"tpa", "kda"}`.
  - Semantics:
    - If `layer_attn_impls[i] == "tpa"`: apply the mask semantics from `layer_types[i]` (sliding uses window=128).
    - If `layer_attn_impls[i] == "kda"`: ignore `layer_types[i]` (KDA uses its own chunk/recurrent semantics; only needs 2D padding mask).
  - Optional ergonomics (not source of truth): allow a compact rule/pattern (e.g. "tpa,kda,kda,kda" or "full_attention_interval=4") but **always serialize the resolved per-layer list** into the final artifact for provenance and fail-fast validation.
- [x] **How do we preserve HF compatibility?** (e.g., `from_pretrained` config round-trip, `AutoConfig` fields, etc.)
  - Choose **Option A (research-first)**: rely on our vendored HF Transformers + our model code path during distillation; do not require “vanilla upstream AutoModel” immediately.
  - Requirement: configs/checkpoints must remain **self-describing** (explicit per-layer schedule + mask schedule) so we can later export to NeMo and/or upstream a clean HF integration once the final checkpoint exists.

## Distillation schedule (beyond loss mix)

- [x] **Short‑ctx stage:** sequence length, token budget, LR, and what is frozen besides MoE (embeddings/lm_head/norms?).
  - **Sequence length:** `4096` (match GPT‑OSS `initial_context_length=4096` from the released configs).
  - **Token budget:** use the curated Phase‑1 mix as the short‑ctx budget (currently ~665M tokens per `data/distillation_1/README.md`).
  - **Dataset artifact:** rebuild Phase‑1 as a **4K-capped** dataset (no training-time slicing semantics).
  - **Freezing (Arcee-style attention-only stabilization):**
    - Freeze: embeddings, `lm_head`, norm weights, and the entire MoE MLP (router + experts).
    - Train: the new attention stack parameters (TPA + KDA internals), including copied attention-side interface params (`o_proj`, sinks).
  - **LR:** deferred; sweep later once throughput + stability are known.
- [ ] **Long‑ctx stage:** target context length(s), curriculum/ramp, LR, and unfreeze timeline.
- [x] **Long‑ctx stage:** target context length(s), curriculum/ramp, LR, and unfreeze timeline.
  - **Target context:** `32768` (32k) as the main long‑ctx adaptation stage (128k is an eval/target ceiling, not a training target in this phase).
  - **Unfreeze timeline (MoE):**
    - At the 32k stage: **unfreeze experts**.
    - Keep **router frozen** by default to avoid routing churn/collapse while representations are still shifting.
    - Optional late stage: unfreeze router only if eval/diagnostics indicate it is the bottleneck; treat as a separate decision with explicit collapse monitoring.
  - **LR:** deferred; likely smaller than short‑ctx.
- [ ] **Hidden-state alignment target:** pre-attn vs post-attn vs layer outputs; how to handle differing attention implementations.
- [x] **Hidden-state alignment target:** pre-attn vs post-attn vs layer outputs; how to handle differing attention implementations.
  - Follow DistillKit/Arcee: use the model’s standard HF `hidden_states` and align with **cosine distance** (`hs_cosine`), `layer_mapping: all`.
  - Masking: apply on the same assistant-only supervision mask as other losses (`labels != -100`).
  - Rationale: aligns at a stable per-layer interface even when attention internals differ (TPA vs KDA).

## MoE specifics (router + experts)

- [ ] **Router policy:** router distillation (teacher-faithful) vs free adaptation + load-balancing guardrails.
- [ ] **If router distillation:** what target and loss? (router logits KL, top‑k CE, etc.), temperature, and weighting.
- [ ] **Token mask for router objectives:** assistant-only vs all non-padding tokens.
- [ ] **Router aux loss usage:** do we ever enable GPT‑OSS load-balancing aux loss? If yes, when and with what coef?
- [x] **Router diagnostics (to decide whether router needs intervention):** log nmoe-style MoE health metrics.
  - Per MoE layer (computed from per-token expert assignment counts / loads):
    - **Load CV** (std/mean of expert loads)
    - **Load entropy** (entropy of normalized loads)
    - **Max-load fraction** (max expert load as % of tokens)
    - **Experts active** (count of experts with nonzero load)
    - **Dead experts** (count with zero load)
  - Aggregates across layers:
    - mean/std of load CV
    - mean/min of load entropy
    - total dead-expert count
  - Optional (if available): router bias / logit scale stats (range / norm) to detect drift.

## KDA constraints we must design around

- [x] **Mask plumbing:** KDA accepts only a 2D padding mask; define the canonical mask representation flowing through the model.
  - Canonical mask surface through the model: `attention_mask` is always **2D padding mask** `[batch, seq]` (1 = real token, 0 = padding).
  - TPA layers derive the required **causal/sliding** mask internally from the 2D padding mask (as GPT‑OSS does), keyed by `layer_types[i]`.
  - KDA layers receive only the 2D padding mask (or `None` if no padding); they must never be passed a 4D / `[B,S,S]` attention mask.
- [x] **Training mode:** KDA training supports only chunk mode; define how we enforce this and fail fast otherwise.
  - Training: enforce `mode == "chunk"` for KDA layers; fail fast if a config requests recurrent mode while `training=True`.
- [x] **KDA hyperparams:** confirm defaults for `use_short_conv`, `conv_size`, `allow_neg_eigval`, etc., and whether they are global or per-layer.
  - Default policy: match Kimi-Linear reference defaults; treat these as **global** (no per-layer overrides initially).
  - Defaults (from the KDA reference implementation):
    - `mode = "chunk"` (training; inference may use `"fused_recurrent"` when supported)
    - `use_short_conv = true`
    - `conv_size = 4`
    - `conv_bias = false`
    - `allow_neg_eigval = false`
    - `expand_v = 1.0`

## Inference / `generate` (early requirement)

- [x] **Unified cache design:** how do we represent a mixed cache (TPA cache + per-layer KDA recurrent/conv state) in an HF-compatible way?
  - Follow Arcee’s KDA+full-attn template: a **single** `past_key_values` cache object holding **per-layer state dicts**, keyed by `layer_idx`.
  - Per-layer state schema (conceptual):
    - **TPA layers:** `attn_state=(K, V)` (KV cache).
    - **KDA layers:** `recurrent_state` (+ optional `conv_state` when `use_short_conv=true`).
  - Cache should support conversion from legacy tuple/list into the cache class (as Arcee does via `Cache.from_legacy_cache`) so HF `generate` callers work.
- [x] **Sliding window in decode:** ensure TPA decode obeys window=128 (no unbounded cache growth).
  - Match GPT‑OSS masking semantics:
    - For TPA layers whose `layer_types[i] == "sliding_attention"`: KV cache is **bounded** to window=128.
    - For TPA layers whose `layer_types[i] == "full_attention"`: KV cache is **unbounded** (no forced window), as in GPT‑OSS.
- [x] **`prepare_inputs_for_generation`:** define the input/positioning semantics for incremental decode across mixed layers.
  - Follow Arcee: do **not** implement a custom `prepare_inputs_for_generation` initially; rely on HF `GenerationMixin`.
  - Requirement: our `forward(...)` must accept `past_key_values` + `cache_position` and remain compatible with HF generation slicing.
  - Masking/positions: construct GPT‑OSS-style per-layer masks internally from the 2D padding mask + `cache_position`.
- [x] **Benchmarks:** define the minimal evaluation set we run after each stage (and the exact commands/configs).
  - Follow Arcee’s evaluation shortlist for short-context quality:
    - MMLU
    - ARC-Challenge
    - HellaSwag
    - GSM8K
  - Long-context evaluation: RULER NIAH (needle-in-a-haystack), including multi-needle variants.

## Multi-Token Prediction (MTP) (post-training; not in distillation v1)

- [x] **Decision:** do **not** use MTP during the teacher-forced distillation phase.
  - Goal of this phase is to match GPT‑OSS **next-token prediction (NTP)** behavior as closely as possible.
  - GPT‑OSS does not provide a clean “teacher target” for MTP-style `t → t+k` predictions (teacher logits at `t+k` are conditioned on intervening tokens).
- [ ] **Post-training follow-up:** once we have a solid distilled checkpoint (and a stable eval harness), revisit MTP as an auxiliary objective.
  - Define: offsets, head design/tying, loss weight/schedule, and assistant-only masking rules.
