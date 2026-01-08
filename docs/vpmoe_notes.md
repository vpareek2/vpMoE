# vpMoE — Design Notes (Working Doc)

Last updated: 2026-01-07

This file is the **design discussion scratchpad** for vpMoE. It complements the locked specs in:
- `docs/architecture.md`
- `docs/distillation_strategy.md`

## 0) Locked Baseline (Confirm / Don’t Drift)

- Teacher: **GPT‑OSS‑120B**
- Tokenizer: **o200k Harmony**, padded vocab size **201088**
- Base KD dataset: **PleIAs/SYNTH**
- Student architecture: **as specified in `docs/architecture.md`**

If we change any of the above, treat it as a **spec change** and update the locked docs first.

## 1) Decision Surface (Checklist)

For each item: record the final decision, rationale, and any constraints.

### A) Tokenization + Format

- Harmony format: special tokens + exact prompt/assistant span rules
- Reasoning vs final spans: representation + masking/weighting rules
- Stop sequences + truncation policy
- Sequence lengths: train vs eval (and any schedule)
- Packing policy (if any): boundaries + invariants

### B) Model Architecture (Dense + MoE)

- Dense baselines to track: qwen3‑0.6B → qwen o200k → vpdense 0.5/28 → vpdense 28 → vpdense 80 → vpMoE
- Attention: heads, head_dim, GQA/query_groups, QK‑norm, value residuals, softmax_type
- Positional encoding: GRAPE/GRAPEm knobs + any fusion constraints
- Bias policy: `add_bias_linear`, `add_qkv_bias` (and backend constraints, e.g. TPA)
- Untied embeddings: on/off

### C) Attention Backends + Schedule

- Local/global schedule (e.g. 3:1) and window semantics
- TPA: on/off, rank knobs, constraints with other features
- Flash attention policy (when allowed)

### Decision: Attention schedule + backends (baseline)

- Windowed local attention: `window_size=128` (**locked**)
- Local/global schedule: `window_attn_skip_freq=4` (3:1) (**locked**)
- Local layers: use **TPA** QKV projections + sliding-window masking.
- Global layers: use standard **full causal** attention.
- TPA ranks (`tpa_rank`, `tpa_q_rank`): treat as **tunable** later (not locked today).
- FlashAttention: plan to enable later for speed (once TE/optimizer path is sorted).

### Clarification: where softmax offset + biases apply

- `softmax_type="learnable"` (learnable softmax offset / “sink”) is a **per-layer** attention parameter and is applied in both
  windowed and full-attention layers; the window schedule only changes the mask (see `megatron/core/transformer/dot_product_attention.py`).
- Linear biases (`add_bias_linear=true`, `add_qkv_bias=true`) are **global** module construction settings: they apply in both local and
  global layers (there is no “SWA-only bias” mode).

### D) MoE (vpMoE only)

- Experts/topk/shared expert/dense warmup semantics
- Router: DSv3 router knobs (LB type, aux coeff, score fn, groups, expert bias, dtype)
- Dispatch + parallelism: EP degree, token dispatcher, constraints
- Perf toggles: grouped GEMM / permute fusion / router fusion (and what happens if TE is unusable)

### Decision: MoE shape + router (baseline)

- Experts: `num_experts=256` (**locked**)
- Routing: `topk=4` (**locked**)
- Shared expert: enabled, `shared_expert_size=512` (**locked**)
- Dense warmup: `first_layer_dense=true` (**locked**)
- Routed expert FFN width: `ffn_hidden_size=128` (**locked**)
- DSv3 router (Megatron-Core DSv2/DSv3-style):
  - `router_load_balancing_type="seq_aux_loss"`
  - `aux_loss_coeff=1e-4`
  - `router_score_function="sigmoid"`
  - `router_topk_scaling_factor=2.5`
  - `router_num_groups=8`
  - `router_group_topk=4`
  - `router_enable_expert_bias=true`
  - `router_bias_update_rate=1e-3`
  - `router_dtype="fp32"`
- EP: start with `EP=1` (tunable later; when `EP>1`, use all-to-all dispatch).
- Perf toggles: keep `grouped_gemm=true`, `permute_fusion=true`, `router_fusion=true` in configs; may be gated by TE availability in practice.

### E) Objective + Distillation

- KD losses: KL vs CE mix, temperature(s), hard label mix
- Masking: prompt tokens masked; reasoning vs final handling
- Span mixing: **span-normalized** (per-span mean then weighted)
- Teacher scoring / OPD (if used): rollout lengths, sampling params, logging contract

### Decision: Distillation stages (current)

- Stage 1 (warm KD): student **CE only** on SYNTH/Harmony tokens (no teacher logits yet).
- Stage 2 (offline KD): store **top‑K teacher logits** (small run / subset due to storage cost).
- Stage 3 (OPD): sampling/scoring params TBD; teacher reasoning mode curriculum: **medium → high**.

### Decision: Masking invariants (need to be correct)

Industry standard for instruction SFT is to compute loss only on assistant/completion tokens (mask prompt/input tokens by setting labels to `-100` / `loss_mask=0`). We do this explicitly via Harmony rendering + a per-token `lossmask` sidecar, and we keep reasoning vs final spans via `span_id`.

Concretely, mainstream SFT tooling exposes this as a first-class option:
- TRL: `completion_only_loss` and `assistant_only_loss` (labels set to `-100` outside completion/assistant spans).
- Axolotl: `train_on_inputs: false` (“mask out or include the human's prompt from the training labels”).
- Torchtune: `train_on_input=False` masks prompt labels by replacing them with `-100` (and points to alpaca-lora’s implementation).
- Alpaca-lora reference: when `train_on_inputs` is false, it sets prompt labels to `-100` and trains only on response labels.

Note: some trainers default to “train on inputs” (i.e., include prompt tokens in the loss) for certain dataset formats, so it’s important we explicitly lock this in our data pipeline rather than assume defaults.

For vpMoE KD, we keep this invariant because:
- It matches the intended conditional modeling objective (predict assistant given the prompt).
- It avoids spending loss budget on user/system text that we neither want to imitate nor distill.
- It makes masking behavior independent of downstream trainer defaults (Megatron/HF/etc.).

References:
- TRL SFT docs (`assistant_only_loss`, `completion_only_loss`): https://huggingface.co/docs/trl/v0.22.1/sft_trainer
- Axolotl config reference (`train_on_inputs`): https://docs.axolotl.ai/docs/config-reference.html
- Torchtune alpaca dataset (`train_on_input` masking): https://docs.pytorch.org/torchtune/0.5/_modules/torchtune/datasets/_alpaca.html
- Alpaca-lora implementation (`train_on_inputs` → `-100` labels): https://github.com/tloen/alpaca-lora/blob/main/finetune.py

### Decision: F) Data + Curriculum (SYNTH KD stage)

- Next datasets (midtrain/posttrain): **TBD** (for this stage we only support SYNTH, per `docs/data_pipeline.md`).
- `seq_len`: **2048** for the current SYNTH KD stage (we can extend later via YaRN / long-context work).
- Format validity: **strict reject** on invalid Harmony structure during dataset build (fail fast).
  - “Auto-repair” would mean mutating examples to “make them parse” (e.g., dropping malformed messages, inventing missing channel markers, patching role/channel fields).
  - We do **not** do this implicitly in the build pipeline. If we ever want repair, it should be a separate, versioned transform that writes a new dataset + manifest recording exact edits.
- Filtering (content): for SYNTH we treat it as **trusted** for now; do **structure validation only**.
- Determinism/manifests: follow existing repo manifests:
  - Input: `data/synth_harmony/manifest.json` (written by `vpmoe/data_curation/harmony/write_synth_harmony_manifest.py`).
  - Output: `data/megatron/synth_kd_v1/manifest.json` (written by `scripts/datasets/build_synth_kd_v1.py`), including `git_sha`, `config_hash`, tokenizer sha, input manifest sha, split rule, and shard inventory.


### F) Data + Curriculum

- Phase datasets + weights (beyond PleIAs/SYNTH)
- Filtering + format validation strictness
- Sharding/manifests + determinism/resume offsets
- Length schedule + packing

### G) Optimizer + LR + Stability

- Optimizer: `normuon` (Normuon + aux AdamW path) vs `adam`
- Normuon knobs: momentum, beta2, eps, Polar Express safety factor / coeff table
- Aux AdamW knobs: aux LR policy, weight decay, betas/eps (if exposed)
- LR schedule: warmup/decay/min_lr
- Stability: grad clip, qk-clip, loss scale, etc.

### Decision: Optimizer family (baseline)

- Optimizer: **Normuon + Polar Express** as the default optimizer for vpMoE training.
- Parameter routing:
  - 2D non-embedding matrices → **Normuon**
  - embeddings/output + 1D params (norm scales) + biases → **AdamW**
- Polar Express: use the default coefficient table with safety factor `2e-2`.

This matches the current Megatron integration contract in `docs/megatron.md`.

### Note: Hyperparams not locked yet

We will do hyperparameter sweeps **after profiling**. For now we only lock the optimizer *family* and routing;
exact LR schedule, AdamW betas/eps/decay, and grad clipping are **tunable**.

Reference check (nmoe):
- `ref/nmoe/nmoe/train.py` (main MoE training loop) does **not** do gradient clipping before optimizer step.
- `ref/nmoe/nmoe/data/train.py` (HYDRA head training) clips the **head parameters only** with `clip_grad_norm_(..., 1.0)`.

### Next questions to lock (G)

1) AdamW side: keep Megatron defaults (`--weight-decay 0.01`, `--adam-beta1 0.9`, `--adam-beta2 0.999`, `--adam-eps 1e-8`), or do you want different values?
2) Grad clipping: **lock** `--clip-grad 1.0` for now (baseline safety rail; revisit in sweeps after profiling).
3) LR schedule for Stage 1 SYNTH warm KD:
   - do we want a simple **cosine decay** (recommended) vs constant LR,
   - and roughly what `--train-iters` / token budget should we size it for?
4) Normuon aux LR: do we set `--normuon-aux-lr` explicitly (e.g. equal to `--lr`), or keep it unset so it follows `--lr`?

### H) Precision + Kernels

- BF16/FP16/FP8 policy per phase
- Transformer impl: local vs TransformerEngine; fusion toggles policy
- Determinism contract (what we guarantee)

### Decision: Precision + kernels (baseline)

- Precision: **BF16** for training activations/weights by default.
  - Allow **FP32** where required for numeric stability / spec parity (e.g., router dtype `fp32`, some reductions/norms as implemented).
  - We are not enabling FP8/NVFP4 training in the baseline path today.
- Kernels / transformer implementation:
  - `--transformer-impl local` is the correctness + smoke-test baseline.
  - `--transformer-impl transformer_engine` is the intended performance backend for real runs, but we will diagnose TE issues before relying on it.
  - No “silent downshift”: if a run requests TE-only fusions, the job should fail fast with an actionable error rather than silently disabling them.

### Next questions to lock (H)

1) Determinism posture: **best-effort determinism** (we prioritize resumability + logged provenance; we do not require fully deterministic kernels everywhere).
2) TE perf toggles: **defer** TE-dependent fusions (`grouped_gemm`, `permute_fusion`, `router_fusion`) until TE is stable; baseline configs keep them **off** for now.

### I) Parallelism + Batch Geometry

- DP/TP/PP/EP degrees (current dev + target cluster)
- Micro/global batch sizes + grad accumulation
- Activation checkpointing policy

### Decision: Parallelism + batch geometry (baseline)

- Dev bringup (GB10): `TP=1`, `PP=1`, `EP=1` (pure DDP if multi-GPU; single-GPU otherwise).
- Target (8×H100/B200): start with `DP=8`, `TP=1`, `PP=1`, `EP=1` (keep topology simple; tune later).
- EP: **tunable later** (we may use `EP>1` after we profile and validate stability/throughput).
- Activation checkpointing: **off initially** (bringup + profiling first; revisit if memory-bound).

### J) Checkpointing + Resume

- Provenance: git sha, config hash, tokenizer id/version, dataset manifest hash
- Resume invariants: RNG + dataloader shard/offset captured
- Save cadence + naming

### Decision: Checkpointing + resume (baseline)

- Implementation: use **Megatron’s default checkpointing/resume** mechanisms (no custom stack).
- Save cadence/retention: **TBD** until we size total steps / wall-clock; avoid locking early.
- Invariants (must hold even with “defaults”):
  - Resume must not silently drift (RNG state + dataloader/shard/offset captured).
  - Checkpoints must log provenance (git SHA, config hash, tokenizer identity/hash, dataset manifest hash).

### K) Eval + Diagnostics

- Lightweight eval suite: tasks + sampling + decode params + scoring
- PPL corpora choice(s) and sampling policy
- Regression thresholds + reporting format (reports in git)

### Decision: Eval + diagnostics posture (baseline)

- Decode/sampling settings for “final” eval are **TBD** (we will lock these after the first real training runs).
- CORE-8 in the rung chain is **tracking-only** (we expect drops during surgery; no pass/fail thresholds yet).
- Reports remain repo-tracked so we can visualize recovery later.

### L) Profiling + Perf

- What to measure: fwd/bwd/optimizer/comm, memory, kernel hotspots
- Tooling: nsys/ncu/torch profiler, capture windows, NVTX ranges
- Perf acceptance targets for “ready to train”

### Note: MoE health metrics (tracking-only; W&B)

We added nmoe-style router health metrics to Megatron and log them to W&B at `--log-interval`:

- Aggregate (`moe/router_agg/*`): `mean_cv_pct`, `std_cv_pct`, `mean_entropy`, `min_entropy`, `mean_max_load_pct`,
  `dead_experts_count`, `experts_active_mean`, `expert_bias_range`.
- Per-layer (`moe/router/layer_{i}/*`, gated by `--moe-per-layer-logging`): `cv_pct`, `entropy`, `max_load_pct`,
  `experts_active`, `dead_experts`, `bias_range`.

Implementation:
- Cache tokens-per-expert each forward: `vpmoe/Megatron-vpmoe/megatron/core/transformer/moe/router.py`
- Compute + log: `vpmoe/Megatron-vpmoe/megatron/core/transformer/moe/moe_utils.py` and `vpmoe/Megatron-vpmoe/megatron/training/training.py`

### Decision: Profiling contract (baseline)

We do profiling to quantify *step time* and identify bottlenecks (not to re-decide architecture).

- Fixed geometry for baseline profiling:
  - `seq_len=2048`
  - `micro_batch_size=1`
  - `TP=1`, `PP=1`, `EP=1` (DP varies with GPU count)
  - Data: `--mock-data` (stable + fast)
- Canonical runs:
  1) Single GPU torch profiler (operator breakdown): `PROFILE_TOOL=torch PROFILE_TRANSFORMER_IMPL=local scripts/profile.sh`
  2) Single GPU nsys (kernel timeline): `PROFILE_TOOL=nsys PROFILE_TRANSFORMER_IMPL=transformer_engine scripts/profile.sh` (only once TE is stable)
  3) 8×GPU nsys (comm/overlap): same as (2) with `PROFILE_NPROC_PER_NODE=8` (and `DP=8`)
- Artifacts policy:
  - Track only small **summary JSON** outputs (tokens/s, ms/step, peak mem, router_agg snapshot) under `reports/perf/`.
  - Do not commit raw traces (`.qdrep`, torch profiler traces).
- No A/B perf ablations in the baseline plan (architecture is already selected); we only measure the end-to-end step.

## 2) Discussion Log

Add entries as we lock decisions.

### 2026-01-07

- Topic: Harmony format + masking + packing (KD baseline)
- Decision:
  - Harmony rendering: use the repo’s `ref/harmony` defaults (assistant channels: `analysis` + `final`).
  - Loss masking: supervise **assistant tokens only** (mask user/system/developer).
  - Reasoning vs final: supervise **both** spans (current default weights `w_r=1.0`, `w_f=1.0`; may change with curriculum).
  - Packing: allow multi-conversation packing; require attention/position resets at EOD boundaries.
- Rationale:
  - This matches common instruction/KD practice while keeping the data pipeline simple and deterministic.
- Follow-ups (need explicit decisions):
  - Confirm canonical training flags for packing: `--reset-attention-mask --reset-position-ids` (and optionally `--eod-mask-loss`).

### Decision: KD span weights (warming / base KD)

We supervise both assistant spans (analysis + final) with span-normalized weighting:

- `--kd-span-weight-reasoning 1.0`
- `--kd-span-weight-final 1.0`

Rationale: maximize pressure to learn reasoning patterns in the initial KD phase. We may revise these weights as part of the learning curriculum discussion (e.g., down-weight reasoning later, or schedule weights by phase).

### Decision: Structured truncation (KD data)

For any overlong Harmony conversation (rendered token length > `seq_len`), we apply **structured truncation**:

- Hard requirement: preserve Harmony structure (message boundaries + required special tokens).
- Preserve the **last assistant `final` message** in full. If the final message alone cannot fit, **drop** the example (fail fast).
- If still over budget, include as much as fits from the preceding assistant `analysis` message by taking a **suffix** of its content (keep the `analysis` header; truncate content from the left).
- If still over budget, truncate the prompt context by taking a **suffix** of the latest user message content (keep the `user` header; truncate content from the left).

This keeps the supervised target (final) intact while maintaining the most recent context and preserving format tokens.

### Decision: Rung Ladder (qwen → vpDense → vpMoE)

We will track capability degradation across this ladder:

1) **Qwen3-0.6B (HF)**: baseline init checkpoint from HuggingFace.
2) **Qwen3-0.6B-o200k**: same model, **vocab surgery only** to o200k Harmony via mergekit `tokensurgeon`.
   - Implementation note: we use OMP transplantation (`scripts/upcycle/qwen3-0_6B_to_o200k.sh`, `configs/upcycle/qwen3-0_6B-o200k.toml`)
     with a GPT‑OSS donor model (k=64).
3) **vpDense0-5_28 (shape-only, 28L)**: “halfway” rung = **student-shaped dense** model at 28 layers (same as Qwen depth),
   with vp geometry (hidden/head dims, vocab, intermediate size) but **feature-minimal**:
   - No GRAPE/GRAPEm (use vanilla RoPE)
   - No TPA / window schedule (use standard attention)
   - No value residuals
   - `softmax_type="vanilla"` (no learnable offset)
   - Activation switch only: replace SwiGLU with **ReLU²** (squared ReLU)
4) **vpDense_28 (full stack, 28L)**: full vpDense architecture/features but only 28 layers:
   - local/global attention schedule + TPA windowed attention
   - GRAPE‑M locally + GRAPE‑A globally
   - value residuals
   - learnable softmax offset (`softmax_type="learnable"`)
5) **vpDense_80 (full stack, 80L)**: expand dense model from 28 → 80 layers (method TBD).
6) **vpMoE (full, 80L)**: swap dense FFNs to the locked MoE design (experts/topk/shared expert + DSv3 router).

Notes:
- Embeddings are **untied** in the student path (confirmed by configs / upcycle).
- TPA supports **attention bias** (QKV + output projection) for GPT‑OSS parity; this is **not** about the learnable softmax offset.

### Decision: GPT‑OSS attention bias parity (keep 3:1 schedule)

- Keep the **3:1 local/global** schedule (`window_attn_skip_freq=4`) unchanged.
- Enable **linear biases globally** to match GPT‑OSS (`add_bias_linear=true`, `add_qkv_bias=true`).

This matches GPT‑OSS attention bias behavior while preserving our local‑attention swap (TPA).
