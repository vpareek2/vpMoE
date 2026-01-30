# Debug configs (local confidence tests)

These configs are for **learning-signal sanity checks** (overfit / fixed-batch eval),
not throughput benchmarking.

## Tiny dataset (required)

The default smoke configs expect the full on-disk dataset at:

- `/data/distillation_1/phase1_mix_4k_665m`

On a fresh machine (or a dev box), you may not have that dataset mounted. For
debugging, create a tiny DistillKit-ready dataset:

```bash
python scripts/make_tiny_distill_dataset.py \
  --output-dir /data/distillation_debug/overfit_tiny64_s512 \
  --tokenizer openai/gpt-oss-20b \
  --train-n 64 \
  --val-n 16
```

For realistic **memory** testing at a target sequence length (e.g. 4096), you can
force examples to an exact token length:

```bash
python scripts/make_tiny_distill_dataset.py \
  --output-dir /data/distillation_debug/autotune_tiny_4k \
  --tokenizer openai/gpt-oss-20b \
  --train-n 256 \
  --val-n 64 \
  --sequence-length 4096
```

Single-example dataset (for a strict must-pass memorization test):

```bash
python scripts/make_tiny_distill_dataset.py \
  --output-dir /data/distillation_debug/overfit_1example \
  --tokenizer openai/gpt-oss-20b \
  --train-n 1 \
  --val-n 1
```

Tiny 8-example dataset (no repeats; makes loss movement easier to see):

```bash
python scripts/make_tiny_distill_dataset.py \
  --output-dir /data/distillation_debug/overfit_tiny8_s256 \
  --tokenizer openai/gpt-oss-20b \
  --train-n 8 \
  --val-n 8
```

## Overfit run (logits-only)

Strict must-pass (CE-only, 1 example, eval on train):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m distillkit.main configs/debug/overfit_1example_s256_ce_only.yaml
```

Must-pass KD path (CE+KL, 1 example, eval on train):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m distillkit.main configs/debug/overfit_1example_s256_ce_kl.yaml
```

Recommended first pass (CE-only, conservative optimizer settings):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m distillkit.main configs/debug/overfit_tiny64_s512_ce_only_safe.yaml
```

Faster variant (shorter seq + fewer steps):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m distillkit.main configs/debug/overfit_tiny64_s256_ce_only_fast.yaml
```

Then (if stable) move to CE+KL:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m distillkit.main configs/debug/overfit_tiny64_s512.yaml
```

Short "mirror" run (same loss mix + scheduler shape as the real run, but cheap):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m distillkit.main configs/debug/overfit_tiny64_s256_mirror_short.yaml
```

Mirror run on the **8 unique examples** (recommended for clarity):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m distillkit.main configs/debug/overfit_tiny8_s256_mirror_short.yaml
```

Mirror run with a larger effective batch (recommended for a smoother trend):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m distillkit.main configs/debug/overfit_tiny8_s256_mirror_bsz8.yaml
```

Success criteria:
- training CE/KL should decrease on this tiny dataset (strong signal), and
- the run should be stable (no NaNs / inf / exploding grad norms).

## Autotune microbatch (new machine preflight)

On a new GPU node, first determine the largest `per_device_train_batch_size`
that fits at the target context length, then set `gradient_accumulation_steps`
to hit a large global batch (Muon is happier at large global batch).

Run inside the repo container:

```bash
python scripts/autotune_microbatch.py \
  --base-config configs/distillkit/vpmoe_distill_1_smoke_2xh100.yaml \
  --nproc 8 \
  --sequence-length 4096 \
  --candidates 4,2,1 \
  --targets 256,512
```
