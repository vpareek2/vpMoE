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

### Megatron-LM integration (what we will run)

- **Status:** implemented and test-verified (core tokenizer path only).
- **Tokenizer type:** `O200kHarmonyTokenizer` (Megatron-integrated, no extra stack).
- **Tokenizer asset:** a **local** `o200k_base.tiktoken` file (no network fetch at runtime).
- **Hash check:** `sha256=446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`.
- **Special tokens + IDs:** must match `o200k_harmony` exactly (Harmony guide + `tiktoken` public encoding).
- **EOD:** `<|endoftext|>` (id `199999`).
- **Padded vocab size:** hard-locked at `201088` (fail fast on any mismatch).

Example (training / preprocess):

```bash
--tokenizer-type O200kHarmonyTokenizer \
--tokenizer-model data/tokenizer/o200k_base.tiktoken
```

**Legacy tokenizer path is disabled by default.** If you intentionally need the legacy
builder for non-core tokenizers, pass `--legacy-tokenizer`. Our training path should
use the core tokenizer stack.

### Data pipeline (SYNTH → Harmony → Megatron datasets)

- **Status:** TODO (needs implementation + end-to-end testing).
- Goal: deterministic `.bin/.idx` dataset builds from Harmony conversations using `O200kHarmonyTokenizer`, including loss masks and reasoning/final span masks per `docs/distillation_strategy.md`.

## GRAPE (positional encoding)

Our locked baseline uses **GRAPE‑M** (local layers) + **GRAPE‑A** (global layers) (`docs/architecture.md`). Megatron does not support this out of the box in our current tree, so we need to **add GRAPE positional encoding support** (and wire it into the local/global attention schedule) as part of the Megatron integration.

For clarity, in our baseline **GRAPE‑A means the ALiBi special case** (a head-wise linear distance bias); we are not locking in GRAPE‑AP or gated GRAPE‑A variants unless explicitly updated in the architecture spec.

- **Status:** implemented (GRAPE‑M as `position-embedding-type=grapem`, GRAPE‑A via `--grape-a`).
- **Constraints:** GRAPE‑M does not support flash decode/flashinfer yet; GRAPE‑A is training‑only (no packed sequences or inference).
- **Config:** set `--no-rope-freq` to skip GRAPE‑M on global layers when `--grape-a` is enabled.

## Tensor Product Attention (TPA) (local sliding-window)

Our locked baseline uses **sliding-window local attention** with **TPA**, window size **128**, under a **3:1 local:global** schedule (`docs/architecture.md`). Megatron’s existing window attention is not TPA, so we need to **implement TPA for local layers** and hook it into Megatron’s attention backend selection for the windowed (local) blocks.

- **Status:** implemented via `--use-tpa` with `--tpa-rank/--tpa-q-rank` (local layers only).
- **Constraints:** requires `--window-size` set and `num_query_groups >= tensor_model_parallel_size`.

### Smoke test (end-to-end, mock data)

We keep an **opt-in** functional smoke test that runs a tiny `pretrain_gpt.py` job exercising
GRAPE‑M + GRAPE‑A + window schedule + TPA. Enable it explicitly:

```bash
VPMOE_RUN_SMOKE=1 O200K_HARMONY_VOCAB_PATH=data/tokenizer/o200k_base.tiktoken \
  uv run pytest -q vpmoe/Megatron-vpmoe/tests/functional_tests/python_test_utils/test_smoke_grape_tpa_pretrain.py
```

### Sliding-window semantics (avoid off-by-one)

We treat the configured window size (**128**) as **window span length in tokens, including the current token**.

For a query position `i`, allowed key positions are:

`j ∈ [max(0, i - (W - 1)), i]` where `W=128`.

Equivalently, we mask if:

- `j > i` (future / causal)
- `i - j >= W` (too far in the past)

This makes the maximum lookback distance `W-1` (127 for `W=128`), and avoids the common “`W` means `W+1` tokens” integration bug seen in some backends.

## Optimizer (Normuon + Polar Express)

Our baseline optimizer is **Normuon** with **Polar Express**.

- **Status:** implemented (Normuon for 2D non-embedding weights; AdamW for everything else).
- **Routing:** any parameter with `is_embedding_or_output_parameter` stays on AdamW; all other 2D matrices use Normuon; 1D/bias params use AdamW.
- **LR policy:** Normuon uses `--lr`; AdamW uses `--normuon-aux-lr` if set, otherwise `--lr`. Embedding/output can be overridden via `--decoupled-lr`.
- **Polar Express:** uses the default coefficient table (5 iters, safety factor `2e-2`). For a different safety factor, pass `--polar-express-coeffs-path` with a JSON list of `(a,b,c)` triples computed for your setting.
- **Constraints:** no distributed optimizer / CPU offload / precision-aware optimizer (DDP is supported).

Example:

```bash
--optimizer normuon \
--normuon-aux-lr 1e-4 \
--normuon-momentum 0.95 \
--normuon-beta2 0.95 \
--polar-express-safety-factor 2e-2
```

## Notes

- Verify `--softmax-type learnable` matches the intended GPT‑OSS-style learnable softmax/logit bias behavior (shape + where it is applied).
- Verify “always-on shared expert” semantics map cleanly to Megatron’s shared-expert implementation (and that `shared_expert_size=512` corresponds to the right intermediate size).
- Verify dense warmup + schedules compose correctly (first dense FFN, then MoE layers; plus 3:1 local/global attention schedule).
