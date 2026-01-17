# Distillation mechanics (Arcee / DistillKit style) — source of truth

This doc captures **how distillation actually works** in the Arcee/DistillKit setup we’re following, so we don’t accidentally drift into “teacher generates completions” assumptions.

## Terms (be precise)

- **Teacher-forced distillation** (what DistillKit does in “online” mode):
  - We have a fixed token sequence `input_ids` (system + user + assistant).
  - We run **student forward** on that same sequence.
  - We run **teacher forward** on that same sequence.
  - We compute distillation losses (e.g., KL) from teacher logits vs student logits **at each token position** (masked to assistant tokens).
  - The teacher does **not** generate new tokens during training.

- **Rollout / generation distillation** (not what DistillKit “online” is):
  - Teacher generates continuations on prompts (sampling/greedy).
  - Those generated tokens become targets for student.
  - This is a different pipeline (higher variance, additional knobs like `max_new_tokens`, stop conditions, etc.).

- **Online vs offline (DistillKit’s meaning)**:
  - **Online**: teacher logits (and optional hidden states) are computed **during training** via a teacher forward pass.
  - **Offline**: teacher signals are pre-captured in the dataset (compressed sparse distributions), so training does not run the teacher model.

## What DistillKit computes (from code, not vibes)

DistillKit’s training loop (high-level):

1. **Student forward pass** on `input_ids` (+ `attention_mask`, `labels`).
2. **Teacher signal** acquisition:
   - Online: `teacher_model(input_ids, attention_mask, ...)` → teacher logits (and optionally hidden states).
   - Offline: decompress per-token teacher distributions from dataset fields.
3. Compute one or more losses (weighted mixture), typically:
   - `cross_entropy` (standard LM loss) on `labels`.
   - `kl` (distribution distillation) between teacher and student distributions.
   - Optional: hidden-state alignment (e.g., cosine) when teacher is online.

Key implementation details:

- **Masking**: loss is applied where `labels >= 0`.
  - DistillKit builds `valid_mask = (labels >= 0)` and passes it to every loss.
  - This is how we enforce “prompt tokens are context only; assistant tokens are trained.”
- **Teacher doesn’t know our intent unless it’s in the tokens**:
  - Online teacher is conditioned only by the tokens in `input_ids` and any `teacher_kwargs` passed to the HF forward call.
  - There is no “reasoning level” knob passed implicitly by DistillKit.

Code references:
- `src/DistillKit/distillkit/trainer.py` (`compute_loss`, `total_distillation_loss`)
- `src/DistillKit/distillkit/signals.py` (`OnlineSignalSource.get_signal`, `OfflineSignalSource.get_signal`)
- `src/DistillKit/examples/afm_test.yml` (example “online” configuration)

## GPT‑OSS “Reasoning” knob: what controls it in this setup

For GPT‑OSS + Harmony, “Reasoning: {level}” is part of the **system message content**.

In DistillKit’s teacher-forced distillation:
- The teacher sees that control only if it is **embedded in `input_ids`**.
- Therefore, to enforce “always high,” we bake `Reasoning: high` into the system header for every example.

If we ever want the knob to remain meaningful (low/medium/high behaviors), we must:
- include all desired reasoning levels in the training distribution, and
- condition them explicitly via system message content (not via DistillKit config).

## What this implies for our dataset format

To run teacher-forced distillation cleanly, each dataset example should provide:

- `input_ids`: full sequence (system + user + assistant).
- `labels`: same length as `input_ids`.
  - Set prompt/system/user tokens to `-100`.
  - Copy assistant tokens from `input_ids` for the supervised span(s).
- `attention_mask`: 1/0 mask (unpacked examples can be all 1s).
- Optional: span metadata (useful for debugging/weighting):
  - `assistant_token_start`, `analysis_token_count`, `final_token_count`.

The important property is the **label mask contract**: distillation losses only “count” where labels are not `-100`.

## Curriculum: what “curriculum” means here

Because we are teacher-forcing on a fixed sequence distribution:
- A “curriculum” is primarily about **what sequences we train on** (mixture, length, difficulty), not about generation hyperparameters.
- If we change “Reasoning: low → medium → high,” we are changing the *conditioning tokens* seen by teacher and student.
  - That’s only desirable if we care about preserving a controllable knob.
  - If we want “hyper-intelligent always,” keep it **high** throughout.

## Common pitfalls

- Confusing “online distillation” with “teacher generates completions online.”
- Setting a “reasoning level” in config but not in tokens: it won’t do anything unless it affects `input_ids`.
- Training without masking prompt tokens: the model will learn to reproduce system/user scaffolding.
- Expecting packing to teach long-context: packing improves throughput, not long-range retrieval/reasoning.

