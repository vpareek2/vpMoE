# High-Performance Optimizer: Problem Statement & Requirements

## Context

Current profiling on GB10 (sm121) shows the optimizer dominating step time.
In a representative run, the optimizer inner step is ~35s with ~3.4M GPU ops,
and kernel/API time is dominated by kernel launch overhead, allocator churn,
and small elementwise kernels. Forward/backward compute is not the bottleneck.

This doc defines the problem and requirements for a high-performance optimizer
implementation to unblock training throughput on GB10 and ensure scalability
on H100/B200 nodes.

## Problem Statement

We need a production-grade, high-performance implementation of the current
optimizer behavior (Normuon + auxiliary AdamW) that dramatically reduces
optimizer step time and GPU operation count, without changing model semantics
or training results.

The current implementation is correct but too slow: it spends the majority of
iteration time in optimizer-related kernels and allocator overhead, which
does not improve with faster attention/MLP kernels. This makes profiling and
training on GB10 impractical and will remain a dominant cost even on B200
unless addressed.

## Requirements

### Correctness & Semantics

- Must preserve the exact optimizer semantics:
  - Normuon on 2D non-embedding parameters.
  - AdamW on all other parameters.
  - Decoupled weight decay behavior must match current code.
- Must not change training results beyond acceptable numerical drift.
- Must maintain determinism/resume behavior per repo invariants.

### Performance

- Must materially reduce optimizer wall time and GPU op count.
- Must materially reduce kernel launch overhead in the optimizer step.
- Must reduce or eliminate allocator churn during optimizer step.
- Performance gains must hold for:
  - GB10 (sm121, local transformer impl)
  - H100/B200 (TE enabled)

### Compatibility

- Must integrate cleanly with Megatron training loop and existing configs.
- Must keep public interfaces stable (no breaking changes to training scripts).
- Must work with current mixed-precision settings (bf16 in training runs).

### Observability

- Must expose clear timing for optimizer step (existing timers are fine).
- Must allow profiling with nsys/torch profiler without crashing the run.
- Must provide a way to compare old vs new optimizer performance.

### Constraints

- No silent feature downgrades.
- No parallel optimizer stacks for the same use-case.
- Minimal new dependencies; add only if justified.

## Acceptance Criteria

- Optimizer step time is no longer the dominant portion of iteration time in
  profiling runs at the current model size.
- Significant reduction in GPU op count and kernel launch overhead during
  optimizer step, confirmed via NVTX + nsys.
- No regression in training stability or loss behavior.

## Non-Goals

- Changing model architecture, tokenizer, or distillation plan.
- Altering training data pipeline or loss definitions.
- Switching to a different optimizer family.
