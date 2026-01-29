# vpMoE build checklist (GPT‑OSS → vpMoE)

This is the **execution checklist** for implementing vpMoE by starting from GPT‑OSS’ HF implementation and iterating
toward the target architecture via **`trust_remote_code=True`** (Arcee-style). It is meant to be **ticked off as we go**
while building `modeling_vpmoe.py`, not as a speculative design doc.

Background/reference (keep, but don’t treat as the checklist):
- `src/architecture.md` (decisions already locked)
- `src/open_questions.md` (remaining “not yet decided” items)
- `src/data/distillation_mechanics.md` (distillation semantics contract)
- `docs/docker.md` (container runbook; canonical way to run)

## 0) Invariants (do not violate)

- **Teacher-forced distillation** (DistillKit semantics): teacher forward pass over the same `input_ids`; no rollouts.
- **Loss masking**: supervision is only on assistant tokens (dataset sets `labels=-100` outside assistant spans).
- **Tokenizer / format**: Harmony o200k (`openai_harmony` / `HARMONY_GPT_OSS`), padded vocab size **201088**.
- **Student checkpoint dtype**: **BF16-only** (no MXFP4 blocks/scales in the student checkpoint).
- **One canonical execution path**: the container setup (`docs/docker.md`); no parallel host install path.

## 1) Deliverable surface (what must work end-to-end)

### 1.1 HF load path (Arcee-style)

- [ ] Student checkpoint directory contains:
  - `config.json` with `auto_map` entries (HF remote-code mapping).
  - `configuration_vpmoe.py` and `modeling_vpmoe.py` (plus small helper modules if needed).
  - model weights (`model.safetensors` shards) + tokenizer artifacts.
- [ ] DistillKit loads student with:
  - `trust_remote_code=True`
  - `AutoModelForCausalLM.from_pretrained(student_path, ...)`

### 1.2 Forward signature compatibility

- [ ] `forward(...)` supports the HF/DistillKit hot-path kwargs:
  - `input_ids`, `attention_mask` (2D padding mask), `labels`
  - `past_key_values`, `cache_position`, `use_cache`
  - `output_hidden_states`, `output_attentions`, `output_router_logits`
  - `return_dict`
- [ ] `generate(...)` works via `GenerationMixin` without custom `prepare_inputs_for_generation` (Arcee strategy).

## 2) Config contract (minimal knobs; explicit schedules)

- [ ] Keep GPT‑OSS `layer_types` as the teacher-faithful **mask schedule**:
  - values are `sliding_attention` vs `full_attention`.
- [ ] Add explicit per-layer attention mechanism schedule:
  - `layer_attn_impls: List[Literal["tpa","kda"]]` of length `num_hidden_layers`.
  - Source of truth is the explicit list (not an inferred ratio).
- [ ] KDA hyperparams are global (match Kimi defaults; see `src/open_questions.md`), plus “training is chunk-only”.
- [ ] TPA hyperparams are global (ranks, window size, sink enabled; see `src/architecture.md`).

## 3) Model deltas (GPT‑OSS → vpMoE)

This is the “what changes” list when copying GPT‑OSS as the starting point.

### 3.1 Keep as GPT‑OSS

- [ ] Embeddings / LM head shape + tying policy.
- [ ] RMSNorm semantics (GPT‑OSS uses a specific RMSNorm behavior).
- [ ] MoE MLP structure (router + experts) as a baseline (router/expert training policy is staged).

### 3.2 Replace attention stack (primary change)

- [ ] Introduce a per-layer switch:
  - if `layer_attn_impls[i] == "tpa"`: use vpMoE TPA attention.
  - if `layer_attn_impls[i] == "kda"`: use KDA attention (FLa implementation).

## 4) Masking + positions (must match GPT‑OSS where applicable)

### 4.1 Two kinds of “masking” (don’t mix them up)

- **Loss masking**: dataset sets `labels=-100` for non-assistant tokens; DistillKit applies losses where `labels >= 0`.
- **Attention masking**: logits masking uses `-inf`/large negative values inside attention kernels (causal/sliding).

### 4.2 Canonical attention mask surface

- [ ] Model only accepts `attention_mask` as a **2D padding mask** `[B, S]` (1=token, 0=pad).
- [ ] TPA derives causal/sliding behavior internally from 2D mask + `layer_types[i]`.
- [ ] KDA receives only the 2D padding mask (never a 4D / `[B,S,S]` mask); fail fast otherwise.

### 4.3 Sliding window behavior

- [ ] Sliding window is **128** for `layer_types[i] == "sliding_attention"`.
- [ ] Exact training + decode semantics match GPT‑OSS (confirm and test; tracked in `src/open_questions.md`).

## 5) Cache contract (single coherent `past_key_values`)

- [ ] Unify caching under a single cache object (Arcee-style), supporting conversion from legacy tuple/list.
- [ ] Per-layer cache state:
  - TPA: `attn_state=(K,V)` (bounded to window=128 for sliding layers; unbounded for full layers).
  - KDA: `recurrent_state` (+ optional `conv_state`) per layer.
- [ ] HF `cache_position` support (incremental decode; mask construction keyed off it).

## 6) Sink semantics (TPA-only; GPT‑OSS faithful)

- [ ] Implement GPT‑OSS sink behavior (per-head sink logit, appended then dropped) via the LSE-gate equivalent:
  - `y_sink = sigmoid(logZ - s_h) * (softmax(L) @ V)` (see `src/open_questions.md`).
- [ ] Numerical equivalence test vs reference “append sink then drop sink prob” (bf16).

## 7) Initialization + BF16-only checkpointing

- [x] Produce one canonical **MXFP4/FP4 → BF16 export** of GPT‑OSS‑20B MoE weights and treat it as the init source.
  - Script: `scripts/export_gpt_oss_mxfp4_to_bf16.py` (writes `bf16_export_manifest.json` and runs validation).
- [ ] Define export artifact rules (format, location, provenance: hashes, tokenizer id/version, config hash) — tracked in `src/open_questions.md`.
- [x] Build the student init checkpoint from the BF16 export.
  - Script: `scripts/init_vpmoe_from_gpt_oss.py --teacher /data/gpt-oss-20b-bf16 --output-dir weights/vpmoe-20b-init`
  - Copies embeddings, norms, router/expert weights, plus `o_proj` + sinks on TPA layers.
  - Leaves TPA factor projections and all KDA params at their initial Xavier/random init.
- [x] Initialization rules (student):
  - TPA factor projections: Xavier init.
  - Copy attention-side interface params from init source: `o_proj` + sink params (TPA layers).
  - Do not attempt 1:1 transplant of GPT‑OSS q/k/v projections into full contextual TPA (not equivalent).

## 8) Distillation schedule (what we run first)

- [ ] Stage 1 (short ctx): `seq_len=4096`, Phase‑1 dataset (~665M tokens), Arcee-style freezing (attention stack only).
  - Eval cadence: target ~50M tokens between evals; for 8 GPUs with `per_device_train_batch_size=1` and `gradient_accumulation_steps=2`, set `eval_steps≈3125` (scale proportionally with world size/accumulation).
  - Flash attention: for H100, prefer the FA3 sinks kernel from the Hugging Face hub (`kernels-community/vllm-flash-attn3`); keep FA2 as fallback if the kernel is unavailable.
  - Kernels: enable `use_kernels=true` for the **student** to use RMSNorm/MegaBlocks kernels; **do not** enable kernels on the MXFP4 teacher because those kernels force BF16 paths.
  - Optimizer note: HF does not expose `torch.optim.Muon` via `optim=...` in our pinned Transformers, so DistillKit in this repo
    supports opting into Muon via `training_args.optim_args: "muon"` (optionally with `momentum/nesterov/ns_steps/eps`), while keeping
    `training_args.optim` set to any valid HF optimizer name (e.g. `adamw_torch`). Muon is applied to **2D parameters only** (as torch enforces);
    all non‑2D trainable params (biases, norm weights, sink logits, etc.) use AdamW.
  - Muon stability note: Muon is tuned for **large global batch**. Expect spiky loss/grad norms in tiny-batch smoke runs. For real runs, raise
    global batch (e.g., increase `gradient_accumulation_steps`), use a longer warmup (e.g., `warmup_ratio=0.05–0.1`), and consider a slightly
    lower peak LR.
  - Moonlight-aligned Muon params: use nonzero weight decay and per-matrix update scaling. Example:
    `optim_args: "muon,lr=0.02,wd=0.1,update_scale=0.2,momentum=0.95,nesterov=true,ns_steps=5,adamw_lr=3e-4,adamw_wd=0.1"`.
  - Smoke baseline (2026‑01‑29, 2×B200, `use_kernels=false`, `flash_attention_2`, seq_len=4096, steps=200): train_runtime=1207.85s,
    steps/s=0.166, samples/s=0.331, train_loss=3.4137, eval weighted token avg=3.4323.
- [ ] Stage 2 (long ctx): target `seq_len=32768` (32k), unfreeze experts, keep router frozen; revisit router only if diagnostics demand.

## 9) Diagnostics + evaluation (gates)

- [ ] Log MoE health metrics (nmoe-style) to detect collapse/bias and to decide whether router intervention is needed.
- [ ] Run short-context eval shortlist: MMLU, ARC-Challenge, HellaSwag, GSM8K.
- [ ] Run long-context eval: RULER NIAH (incl multi-needle variants).

## 10) Explicitly deferred (do not implement “by accident”)

- [ ] **TPA training kernel path** (flash-style kernel): defer until correctness + surfaces are stable.
- [ ] **Sliding-window exact semantics** (training + `generate`): confirm vs GPT‑OSS and then lock with tests.
- [ ] **Router policy** (distill vs free + aux losses): defer until the router is actually unfrozen / intervention is required.
- [ ] **MTP**: post-training only; not part of distillation v1.
