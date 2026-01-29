#!/usr/bin/env python3
"""
Run a tiny DistillKit smoke (2 steps, save checkpoints) and verify that:
  - a known-trainable parameter changes between checkpoint-1 and checkpoint-2
  - a known-frozen parameter does NOT change between checkpoint-1 and checkpoint-2
  - (optional) the model saved at output_path matches the last checkpoint

This is meant to be the "trust but verify" script you can run on any new machine
to confirm that training is actually updating weights and that checkpoints are
not silently saving stale weights.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import torch
import yaml
from safetensors.torch import safe_open


@dataclass(frozen=True)
class DiffResult:
    name: str
    max_abs: float


def _load_tensor(root: str, name: str) -> torch.Tensor:
    index_path = os.path.join(root, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(index_path)
    idx = json.load(open(index_path))
    shard = idx["weight_map"][name]
    with safe_open(os.path.join(root, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _find_checkpoints(output_dir: str) -> list[tuple[int, str]]:
    checkpoints: list[tuple[int, str]] = []
    for entry in os.listdir(output_dir):
        if not entry.startswith("checkpoint-"):
            continue
        path = os.path.join(output_dir, entry)
        if not os.path.isdir(path):
            continue
        try:
            step = int(entry.split("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def _diff(a_dir: str, b_dir: str, name: str) -> DiffResult:
    a = _load_tensor(a_dir, name).float()
    b = _load_tensor(b_dir, name).float()
    return DiffResult(name=name, max_abs=(b - a).abs().max().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/distillkit/vpmoe_distill_1_smoke_2xh100.yaml",
        help="Base config to use; will be overridden to 2 steps + checkpoint saving.",
    )
    ap.add_argument("--nproc", type=int, default=2, help="GPUs/processes per node.")
    ap.add_argument(
        "--output-path",
        default=None,
        help="Where to write the run. Defaults to /data/distill_runs/verify_<ts>.",
    )
    ap.add_argument("--steps", type=int, default=2, help="Number of training steps to run.")
    ap.add_argument("--save-steps", type=int, default=1, help="Checkpoint save interval.")
    ap.add_argument("--clean-output", action="store_true", help="Delete output_path if it exists.")
    ap.add_argument("--disable-wandb", action="store_true")
    ap.add_argument("--force-teacher-device-map", default="cuda")
    ap.add_argument(
        "--trainable-param",
        default="model.layers.0.self_attn.W_A_q.weight",
        help="A parameter that must update across checkpoints.",
    )
    ap.add_argument(
        "--frozen-param",
        default="model.layers.0.mlp.experts.gate_up_proj",
        help="A parameter that should remain frozen across checkpoints.",
    )
    ap.add_argument(
        "--min-trainable-diff",
        type=float,
        default=0.0,
        help="Minimum required max-abs diff for trainable param between checkpoints.",
    )
    ap.add_argument(
        "--max-frozen-diff",
        type=float,
        default=0.0,
        help="Maximum allowed max-abs diff for frozen param between checkpoints.",
    )
    args = ap.parse_args()

    if args.output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output_path = f"/data/distill_runs/verify_{ts}"

    if args.disable_wandb:
        os.environ.setdefault("WANDB_DISABLED", "true")
        os.environ.setdefault("WANDB_MODE", "disabled")

    cfg_dict = yaml.safe_load(open(args.config))

    # Force teacher onto GPU in multi-proc to avoid CPU tensors flowing into Triton/kernels.
    cfg_dict.setdefault("teacher", {}).setdefault("kwargs", {})["device_map"] = args.force_teacher_device_map

    # Make output explicit and reproducible.
    cfg_dict["output_path"] = args.output_path

    ta = cfg_dict.setdefault("training_args", {})
    ta["max_steps"] = args.steps
    ta["save_strategy"] = "steps"
    ta["save_steps"] = args.save_steps
    ta["save_total_limit"] = max(2, args.steps)
    ta["logging_steps"] = 1
    ta["eval_steps"] = 0
    ta["report_to"] = []
    ta["overwrite_output_dir"] = True
    ta["lr_scheduler_type"] = "constant"
    ta["warmup_ratio"] = 0.0
    ta.pop("lr_scheduler_kwargs", None)

    # Write a temporary config so we don't mutate the checked-in YAML.
    tmp_cfg = f"/tmp/vpmoe_verify_{os.getpid()}.yaml"
    yaml.safe_dump(cfg_dict, open(tmp_cfg, "w"), sort_keys=False)

    if os.path.exists(args.output_path):
        if args.clean_output:
            shutil.rmtree(args.output_path)
        else:
            raise SystemExit(
                f"output_path exists: {args.output_path}\n"
                f"Pass --clean-output or use a different --output-path."
            )

    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.nproc}",
        "-m",
        "distillkit.main",
        tmp_cfg,
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    checkpoints = _find_checkpoints(args.output_path)
    if len(checkpoints) < 2:
        raise SystemExit(
            f"expected >=2 checkpoints under {args.output_path}, found: {[c[0] for c in checkpoints]}"
        )

    ck1 = checkpoints[0][1]
    ck2 = checkpoints[-1][1]

    print("checkpoints:", checkpoints[0][0], "->", checkpoints[-1][0], flush=True)

    trainable = _diff(ck1, ck2, args.trainable_param)
    frozen = _diff(ck1, ck2, args.frozen_param)
    print("trainable diff:", trainable.max_abs, trainable.name, flush=True)
    print("frozen diff:", frozen.max_abs, frozen.name, flush=True)

    # If the run saved a final model at output_path, it should match the last checkpoint.
    final_index = os.path.join(args.output_path, "model.safetensors.index.json")
    if os.path.isfile(final_index):
        final_trainable = _diff(ck2, args.output_path, args.trainable_param)
        print("final vs last checkpoint diff:", final_trainable.max_abs, final_trainable.name, flush=True)

    ok = True
    if trainable.max_abs <= args.min_trainable_diff:
        ok = False
        print(
            f"FAIL: trainable param did not change (<= {args.min_trainable_diff}).",
            flush=True,
        )
    if frozen.max_abs > args.max_frozen_diff:
        ok = False
        print(
            f"FAIL: frozen param changed (> {args.max_frozen_diff}).",
            flush=True,
        )

    if not ok:
        raise SystemExit(2)
    print("PASS: checkpoints contain updated weights.", flush=True)


if __name__ == "__main__":
    main()
