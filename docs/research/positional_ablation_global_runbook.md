# Global Positional Ablation (4096 ctx): NoPE vs GRAPE-A (Global Layers Only)

Goal: pick the better **global-layer** positional strategy while keeping **local layers = GRAPE‑M** under the fixed hybrid schedule:
- Local: sliding-window attention `W=128`
- Global: full attention every 4th layer (3:1 local:global)

This is intentionally a **small proxy ablation** (single GPU, single seed) to choose between:
- **Global NoPE** (no rotary / no position signal on global layers)
- **Global GRAPE‑A** (ALiBi-style additive bias on global layers)

**Outcome:** we choose **GRAPE‑A on global layers** and **GRAPE‑M on local layers**.

---

## 1) Fixed flags (all runs)

### Hybrid local/global schedule
- `--seq-length 4096`
- `--window-size 128,0`
- `--window-attn-skip-freq 4`  (=> layers 4,8,12,… are global/full-attn)

### Local positional baseline
- `--position-embedding-type grapem`
- `--grapem-learnable-freq` (only if this is your GRAPE‑M winner setting)

Notes:
- In this Megatron tree, `--grape-a` is implemented as **ALiBi slopes** and is applied only on **non-window** layers (i.e. global layers under the schedule above).
- `--no-rope-freq` disables rotary on selected layers by setting `rotary_pos_emb = None` on those layers.

---

## 2) Ablation flags (what differs)

### Run A: Global NoPE
- Add: `--no-rope-freq 4`

### Run B: Global GRAPE-A (additive-only on global)
- Add: `--no-rope-freq 4 --grape-a`

Why keep `--no-rope-freq 4` in both runs?
- It makes the comparison “**NoPE vs additive bias**” on global layers (instead of “rotary+additive vs rotary-off”).

Optional anchor (only if you want it):
- **GRAPE‑M everywhere:** omit `--no-rope-freq` and omit `--grape-a`.

---

## 3) Suggested minimal budget (single dev GPU)

Pick a small model (e.g. ~124M-ish GPT-2 scale or smaller) and a token budget that’s big enough to separate loss curves.

Recommended starting point per run:
- Target tokens: **200M**
- Seeds: **1** (if results are close, repeat only the top-2 configs with a 2nd seed)

### Step count calculator
Tokens per optimizer step:
`tokens_per_step = global_batch_size * seq_length`  (single GPU, DP=1)

Example (recommended to start):
- `seq_length = 4096`
- `global_batch_size = 8`
- `tokens_per_step = 8 * 4096 = 32768`
- Steps for 200M tokens:
  - `train_iters = ceil(200_000_000 / 32768) = 6104`

If you need to reduce batch size:
- `global_batch_size = 4` ⇒ `train_iters ≈ 12208`

---

## 4) Smoke test (do this before spending tokens)

Before the real run, do a short 20–50 iteration check to ensure GRAPE‑A is actually supported by your build:
- Run both configs with `--mock-data` and `--train-iters 50`.
- If `--grape-a` fails, fix TE/version/build before running the real ablation.

---

## 5) Metrics to record

### Primary
- Validation loss trajectory (same eval set & cadence for both runs)

### Secondary (cheap and relevant to “global positional”)
- A simple **passkey/needle** retrieval probe at 4k context (tie-breaker if losses are close).
- Stability signals (loss spikes, grad norm) if you already log them.

---

## 6) Decision rule

Pick the winner by:
1) Lower validation loss at matched tokens *and* better loss trajectory, then
2) Use the passkey/needle probe as a tie-breaker if val loss deltas are small.

Operational note:
- In this repo, GRAPE‑A has limitations in some inference modes (e.g., certain flash-decode/dynamic batching paths). If the quality difference is negligible, **Global NoPE** is the safer operational choice.
