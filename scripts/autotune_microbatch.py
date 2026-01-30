#!/usr/bin/env python3
"""
Auto-tune per-device microbatch size on a new machine (e.g. 8xB200) by running
1 training step through DistillKit and catching OOMs.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass

import yaml


_OOM_RE = re.compile(r"(out of memory|CUDA error: out of memory|CUDNN_STATUS_ALLOC_FAILED)", re.IGNORECASE)


@dataclass(frozen=True)
class TrialResult:
    per_device_train_batch_size: int
    ok: bool
    wall_s: float
    output_tail: str


def _run(cmd: list[str], *, env: dict[str, str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    return p.returncode, p.stdout


def _ensure_tiny_dataset(
    *,
    dataset_dir: str,
    tokenizer: str,
    train_n: int,
    val_n: int,
    sequence_length: int,
    env: dict[str, str],
) -> None:
    if os.path.isdir(dataset_dir):
        return
    os.makedirs(os.path.dirname(dataset_dir) or ".", exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/make_tiny_distill_dataset.py",
        "--output-dir",
        dataset_dir,
        "--tokenizer",
        tokenizer,
        "--train-n",
        str(train_n),
        "--val-n",
        str(val_n),
        "--sequence-length",
        str(sequence_length),
    ]
    print("creating tiny dataset:", " ".join(cmd), flush=True)
    rc, out = _run(cmd, env=env)
    if rc != 0:
        print(out, file=sys.stderr)
        raise SystemExit(rc)


def _make_probe_config(
    *,
    base_cfg_path: str,
    out_cfg_path: str,
    out_run_dir: str,
    dataset_dir: str,
    per_device_train_batch_size: int,
    nproc: int,
    force_teacher_device_map: str,
) -> None:
    cfg = yaml.safe_load(open(base_cfg_path))

    # Force teacher onto GPU to avoid CPU tensors flowing into Triton/kernels.
    cfg.setdefault("teacher", {}).setdefault("kwargs", {})["device_map"] = force_teacher_device_map

    # Point to our tiny dataset and ensure we don't do any slow dataset transforms.
    cfg.setdefault("dataset", {}).setdefault("train_dataset", {})["disk_path"] = dataset_dir
    cfg["dataset"]["train_dataset"]["split"] = "train"
    cfg.setdefault("dataset", {}).setdefault("eval_dataset", {})["disk_path"] = dataset_dir
    cfg["dataset"]["eval_dataset"]["split"] = "validation"
    cfg["dataset"]["num_eval_samples"] = 0

    cfg["output_path"] = out_run_dir

    ta = cfg.setdefault("training_args", {})
    ta.setdefault("dataset_kwargs", {})["skip_prepare_dataset"] = True

    # Make the probe fast and side-effect free.
    ta["max_steps"] = 1
    ta["num_train_epochs"] = 1
    ta["per_device_train_batch_size"] = per_device_train_batch_size
    ta["gradient_accumulation_steps"] = 1
    ta["dataloader_num_workers"] = 0
    ta["dataloader_pin_memory"] = True
    ta["logging_steps"] = 1
    ta["report_to"] = []

    # Avoid writing checkpoints during the probe.
    ta["save_strategy"] = "no"
    ta["evaluation_strategy"] = "no"
    ta["eval_steps"] = 0

    # Keep LR schedule out of the picture for a memory probe.
    ta["lr_scheduler_type"] = "constant"
    ta["warmup_ratio"] = 0.0
    ta.pop("lr_scheduler_kwargs", None)

    # Ensure consistent DDP batch accounting (avoid partial batches).
    ta["dataloader_drop_last"] = True

    # Make sure the run dir can be reused.
    ta["overwrite_output_dir"] = True

    yaml.safe_dump(cfg, open(out_cfg_path, "w"), sort_keys=False)


def _tail(s: str, n_lines: int = 80) -> str:
    lines = s.splitlines()
    return "\n".join(lines[-n_lines:])


def _trial(
    *,
    base_cfg_path: str,
    dataset_dir: str,
    per_device_train_batch_size: int,
    nproc: int,
    force_teacher_device_map: str,
    env: dict[str, str],
) -> TrialResult:
    out_cfg = f"/tmp/vpmoe_autotune_bsz{per_device_train_batch_size}.yaml"
    out_run = f"/tmp/vpmoe_autotune_run_bsz{per_device_train_batch_size}"

    _make_probe_config(
        base_cfg_path=base_cfg_path,
        out_cfg_path=out_cfg,
        out_run_dir=out_run,
        dataset_dir=dataset_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        nproc=nproc,
        force_teacher_device_map=force_teacher_device_map,
    )

    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        "-m",
        "distillkit.main",
        out_cfg,
    ]
    print("probe:", " ".join(cmd), flush=True)
    t0 = time.time()
    rc, out = _run(cmd, env=env)
    dt = time.time() - t0

    if rc == 0:
        return TrialResult(per_device_train_batch_size=per_device_train_batch_size, ok=True, wall_s=dt, output_tail=_tail(out))
    if _OOM_RE.search(out):
        return TrialResult(per_device_train_batch_size=per_device_train_batch_size, ok=False, wall_s=dt, output_tail=_tail(out))
    # Non-OOM failure: surface output.
    print(out, file=sys.stderr, flush=True)
    raise SystemExit(rc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        default="configs/distillkit/vpmoe_distill_1_smoke_2xh100.yaml",
        help="Stage-1 distill config; we override dataset + training_args for the probe.",
    )
    ap.add_argument("--nproc", type=int, default=8, help="GPUs to use for the probe.")
    ap.add_argument("--sequence-length", type=int, default=4096)
    ap.add_argument("--dataset-dir", default="/data/distillation_debug/autotune_tiny_4k")
    ap.add_argument("--tokenizer", default="openai/gpt-oss-20b")
    ap.add_argument(
        "--candidates",
        default="4,2,1",
        help="Comma-separated per_device_train_batch_size candidates to try (descending is recommended).",
    )
    ap.add_argument("--force-teacher-device-map", default="cuda")
    ap.add_argument(
        "--targets",
        default="256,512",
        help="Comma-separated target global batches to suggest grad_accum for.",
    )
    args = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "disabled")

    candidates = [int(x) for x in args.candidates.split(",") if x.strip()]
    if not candidates:
        raise SystemExit("no candidates provided")

    # Ensure dataset is large enough that DDP doesn't hit a partial batch.
    train_n = max(256, max(candidates) * args.nproc * 8)
    val_n = max(64, max(candidates) * args.nproc * 2)
    _ensure_tiny_dataset(
        dataset_dir=args.dataset_dir,
        tokenizer=args.tokenizer,
        train_n=train_n,
        val_n=val_n,
        sequence_length=args.sequence_length,
        env=env,
    )

    results: list[TrialResult] = []
    chosen: TrialResult | None = None
    for b in candidates:
        r = _trial(
            base_cfg_path=args.base_config,
            dataset_dir=args.dataset_dir,
            per_device_train_batch_size=b,
            nproc=args.nproc,
            force_teacher_device_map=args.force_teacher_device_map,
            env=env,
        )
        results.append(r)
        print(f"result: per_device_train_batch_size={b} ok={r.ok} wall_s={r.wall_s:.2f}", flush=True)
        if r.ok:
            chosen = r
            break
        print("OOM (tail):\n" + r.output_tail, flush=True)

    if chosen is None:
        raise SystemExit(f"no candidate fit: {candidates}")

    micro = chosen.per_device_train_batch_size
    world = args.nproc
    denom = micro * world
    targets = [int(x) for x in args.targets.split(",") if x.strip()]

    print("\n== recommended grad_accum (given chosen per-device batch) ==", flush=True)
    for tgt in targets:
        ga = max(1, (tgt + denom - 1) // denom)
        actual = denom * ga
        print(f"target_global_batch={tgt} -> gradient_accumulation_steps={ga} (actual_global_batch={actual})", flush=True)

    print("\n== chosen ==\n", flush=True)
    print(chosen.output_tail, flush=True)


if __name__ == "__main__":
    main()

