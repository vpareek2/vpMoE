## Status (2026-01-09)

### Where we are
- GPT-OSS parity knobs are locked in configs (including `attention_dropout = 0.0`).
- Full 6.01B config runs on GB10 with **local/unfused** backend; TE fused is unstable on sm121.
- GRAPE setup behaves as intended: GRAPE-M on windowed layers, GRAPE-A on global layers via `no_rope_freq`.
- Normuon is wired end-to-end and active in logs.

### Key findings
- Optimizer dominates step time on GB10 (Normuon step ~60–65% of iteration).
- GPU kernels are largely elementwise + reductions; optimizer is bandwidth/launch-overhead heavy.
- FlashAttention is disabled by `softmax_type=learnable` + post-scale bias; fused attention works only for SWA but crashes on sm121.

### Recent artifacts
- Nsight Systems capture: `artifacts/profiles/migrated_2026-01-08/gb10_sm121/nsys/vpmoe_sl512_mb1_tp1_ep1_pp1_n1.nsys-rep`
- New traces now land under `artifacts/profiles/<date>/<hw>/<tool>/<run_id>/`.
- Torch profiler traces in `artifacts/profiles/migrated_2026-01-08/gb10_sm121/torch/*/*.pt.trace.json` are CPU-only; new runs capture CUDA when available.

### Blockers / risks
- sm121 (GB10) TE fused attention is unstable; performance work should move to H100/B200.
- Optimizer step is the primary bottleneck on GB10.

### Next goals (ordered)
1) **Optimizer deep-dive**: filter nsys report to optimizer NVTX range and identify top kernels.
2) **Healing run**: proceed with weight-surgery model healing after optimizer baseline captured.
3) **Perf pass on H100/B200**: re-run TE fused attention and profiler once hardware is available.

### Notes / reminders
- `scripts/profile.sh` uses `PROFILE_CONFIG` (not `CONFIG_PATH`).
- Renderer emits `--window-size (W-1,0)`; keep `window_size` in TOML consistent with desired tuple.
