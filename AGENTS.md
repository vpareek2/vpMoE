# vpmoe-research: What We’re Building

We are building **vpMoE**: an ultra-sparse nano-MoE base model trained primarily via **distillation** from **GPT‑OSS‑120B**, using **Megatron‑LM** as the training backbone.

Your goal is to build production-grade systems—data preprocessing, teacher scoring, Megatron integration, training orchestration, checkpointing, evaluation, diagnostics—that are reliable and fast enough to run serious experiments without heroics.

This means: no fallbacks, no hacks, no shortcuts. Production-grade, Google-quality code that at all times demonstrates a maniacal obsession with elegant minimalism.

## Current Locked Decisions

Source of truth lives in:
- `docs/architecture.md`
- `docs/distillation_strategy.md`

Locked for the baseline:
- **Teacher:** GPT‑OSS‑120B only
- **Tokenizer:** o200k Harmony (padded vocab size **201088**)
- **Base KD dataset:** PleIAs/SYNTH (for base KD phases)
- **Student architecture:** locked as specified in `docs/architecture.md`

If a change conflicts with these, treat it as a spec change: update the docs first.

## Public Repo Contract (Follow Even While Private)

- No secrets committed (keys, tokens, credentials), even if you think you’ll never need them.
- No internal-only hostnames/IPs/paths in tracked files.
- One supported execution path: **our** container setup. No parallel “just run it on the host” path unless explicitly opt-in and documented.
- Keep surfaces small: no second stacks for the same use-case.

## Execution Environment

- Assume local environments are inconsistent; prefer running via the repo’s container entrypoint (`docker/`).
- Do not introduce a second installation path. If a non-container path exists, it must be explicitly opt-in and clearly labeled.

## Organization Rules

- `docs/` contains current, maintained specs and runbooks.
- `docs/research/` is archival/exploratory; do not reference it as the current plan unless promoted.
- If you need to see any previous research code, ask. Do not make any assumptions
- Keep runnable entrypoints few and obvious (e.g. `scripts/` + `configs/`).
- Every supported workflow has exactly one canonical way to run it (dataset prep, teacher scoring, offline KD training, OPD training, eval).

## Megatron‑LM Integration Contract

- Prefer integrating vpMoE by **adding minimal, well-scoped code** around Megatron rather than forking large subsystems.
- If we must modify vendored upstream code, keep patches surgical and documented (what/why), and avoid sprawling divergence.
- No silent downgrades: if a feature is unsupported in our path, fail fast with an actionable error.

## Distillation Contract (Data + Losses)

- Training examples must define prompt vs assistant spans; user/prompt tokens must be masked from loss.
- Reasoning vs final spans (Harmony format) must be represented explicitly.
- Span mixing must be **span-normalized** (mean per span, then weighted). Avoid “weighting” after averaging over all tokens.
- Anything touching OPD/teacher scoring must log: teacher mode, rollout lengths, buffer health, and format validity.

## Determinism & Resume

- Training must be resumable without silent drift (RNG state + dataloader/shard/offset captured).
- Checkpoints must log provenance (git sha, config hash, tokenizer id/version, dataset manifest).

## Principles

- One clear path per use-case: avoid multiple interchangeable stacks for the same job.
- Small, sharp surfaces: tiny modules with crisp responsibilities; few public knobs; configs are the source of truth.
- Explicit over magical: obvious control flow, no hidden background machinery.
- Fail fast, fail loud: actionable guardrails; no silent downshifts.
- Minimal dependencies: add deps only if they improve clarity and performance.
- Test what matters: deterministic resume, invariants, and basic regressions; avoid scaffolding that mirrors system complexity.
- Documentation that guides, not overwhelms: precise runbooks and remedies; zero fluff.

Craftsmanship rubric for any change
- Intent: does it improve tokens/s, stability, or correctness?
- Uniqueness: did we create a second way to do something?
- Surface: did we add a new knob; could it be expressed via existing config?
- Hot path: if training loop changed, where is the measured delta (step time / throughput)?
- Invariants: are the locked decisions preserved (or intentionally updated in docs)?
- Repro: can we rerun months later from config + artifacts?
- Elegance: is the code visibly simpler afterward?
