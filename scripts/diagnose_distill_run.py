#!/usr/bin/env python3
"""
Comprehensive diagnostic for DistillKit training.

Runs a single-step train (optionally) and reports:
- environment + CUDA
- config summary
- dataset sanity
- teacher device placement
- optimizer param groups / LR
- in-memory weight deltas
- saved-vs-in-memory deltas (optional)
"""

import argparse
import os
import sys
import json
import shutil
from typing import Any

import yaml
import torch
from safetensors.torch import safe_open

from distillkit.configuration import DistillationRunConfig
from distillkit.main import (
    collate_packed_batch,
    create_signal_source,
    load_data,
    load_student_model,
    load_tokenizer,
)
from distillkit.signals import OnlineSignalSource
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.trainer import DistillationTrainer
from trl import SFTConfig


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_rank0() -> bool:
    return _rank() == 0


def _print(*args: Any) -> None:
    if _is_rank0():
        print(*args, flush=True)


def _env_report() -> None:
    _print("== env ==")
    _print("python:", sys.version.split()[0])
    _print("torch:", torch.__version__)
    _print("torch cuda:", torch.version.cuda)
    _print("cuda available:", torch.cuda.is_available())
    _print("device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        try:
            _print("device name:", torch.cuda.get_device_name(0))
            _print("capability:", torch.cuda.get_device_capability(0))
        except Exception as exc:
            _print("device info error:", exc)
    try:
        import transformers  # noqa: F401
        _print("transformers:", transformers.__version__)
    except Exception as exc:
        _print("transformers import error:", exc)
    try:
        import flash_attn  # noqa: F401
        _print("flash_attn:", flash_attn.__version__)
    except Exception as exc:
        _print("flash_attn import error:", exc)
    du = shutil.disk_usage("/data") if os.path.isdir("/data") else None
    if du:
        _print(
            "disk /data: total=%s used=%s free=%s"
            % (
                f"{du.total/1e9:.1f}G",
                f"{du.used/1e9:.1f}G",
                f"{du.free/1e9:.1f}G",
            )
        )


def _summarize_config(cfg: dict) -> None:
    _print("== config ==")
    ta = cfg.get("training_args", {})
    _print("output_path:", cfg.get("output_path"))
    _print("optim_args:", ta.get("optim_args"))
    _print("lr:", ta.get("learning_rate"), "wd:", ta.get("weight_decay"))
    _print("scheduler:", ta.get("lr_scheduler_type"), "warmup_ratio:", ta.get("warmup_ratio"))
    _print("max_steps:", ta.get("max_steps"), "save_strategy:", ta.get("save_strategy"))
    _print("model_kwargs:", cfg.get("model_kwargs", {}))
    teacher_kwargs = (cfg.get("teacher", {}) or {}).get("kwargs", {})
    _print("teacher kwargs:", teacher_kwargs)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and "device_map" not in teacher_kwargs:
        _print("WARN: teacher.kwargs.device_map missing in multi-proc; teacher may stay on CPU.")


def _dataset_report(cfg: DistillationRunConfig) -> tuple:
    tokenizer = load_tokenizer(cfg)
    ds_train, ds_eval = load_data(cfg.dataset, tokenizer)
    if _is_rank0():
        try:
            ex = ds_train[0]
            lbl = torch.tensor(ex["labels"])
            _print("== dataset ==")
            _print("train len:", len(ds_train), "eval len:", len(ds_eval) if ds_eval else 0)
            _print("seq len:", len(ex["input_ids"]))
            _print("label -100 ratio:", float((lbl == -100).float().mean()))
        except Exception as exc:
            _print("dataset report error:", exc)
    return tokenizer, ds_train, ds_eval


def _teacher_device_report(signal_source) -> None:
    if not _is_rank0():
        return
    if not isinstance(signal_source, OnlineSignalSource):
        _print("teacher: offline/dataset")
        return
    teacher = signal_source.teacher_model
    devices = set()
    try:
        for p in teacher.parameters():
            devices.add(str(p.device))
            break
    except Exception as exc:
        _print("teacher device check error:", exc)
        return
    _print("teacher device sample:", list(devices))


def _load_base_snapshot(model_id: str) -> str | None:
    hf_home = os.environ.get("HF_HOME", "/data/hf_cache")
    snap_root = os.path.join(
        hf_home, "hub", f"models--{model_id.replace('/', '--')}", "snapshots"
    )
    if not os.path.isdir(snap_root):
        return None
    snaps = sorted(os.listdir(snap_root))
    if not snaps:
        return None
    return os.path.join(snap_root, snaps[-1])


def _load_tensor(root: str, name: str) -> torch.Tensor:
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    shard = idx["weight_map"][name]
    with safe_open(os.path.join(root, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/distillkit/vpmoe_distill_1_smoke_2xh100.yaml")
    ap.add_argument("--output", default="/tmp/diag_full")
    ap.add_argument("--param", default="model.layers.0.self_attn.W_A_q.weight")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--save-dir", default="/tmp/diag_full_saved")
    ap.add_argument("--disable-wandb", action="store_true")
    ap.add_argument("--force-teacher-device-map", default=None)
    args = ap.parse_args()

    if args.disable_wandb:
        os.environ.setdefault("WANDB_DISABLED", "true")
        os.environ.setdefault("WANDB_MODE", "disabled")

    _env_report()

    cfg_dict = yaml.safe_load(open(args.config))
    _summarize_config(cfg_dict)

    if args.force_teacher_device_map is not None:
        cfg_dict.setdefault("teacher", {}).setdefault("kwargs", {})["device_map"] = args.force_teacher_device_map

    # Override for diagnostic run
    cfg_dict["output_path"] = args.output
    cfg_dict.setdefault("training_args", {})
    ta = cfg_dict["training_args"]
    ta["max_steps"] = args.steps
    ta["save_strategy"] = "no"
    ta["eval_steps"] = 0
    ta["logging_steps"] = 1
    ta["lr_scheduler_type"] = "constant"
    ta["warmup_ratio"] = 0.0
    ta["report_to"] = []

    cfg = DistillationRunConfig.model_validate(cfg_dict)

    tokenizer, ds_train, ds_eval = _dataset_report(cfg)
    tokenizer_vocab_size = max(
        len(tokenizer.get_vocab()),
        max(tokenizer.get_vocab().values()) + 1,
    )

    model = load_student_model(cfg, tokenizer_vocab_size)
    signal_source = create_signal_source(cfg, tokenizer_vocab_size)
    _teacher_device_report(signal_source)

    hsm = None
    if cfg.layer_mapping is not None:
        if not isinstance(signal_source, OnlineSignalSource):
            raise RuntimeError("Hidden state distillation requires online teacher.")
        teacher_hidden_size = signal_source.teacher_model.config.hidden_size
        mapping = (
            [(i, i) for i in range(model.config.num_hidden_layers)]
            if cfg.layer_mapping == "all"
            else cfg.layer_mapping
        )
        hsm = HiddenStateMapping(
            student=model,
            teacher_hidden_size=teacher_hidden_size,
            layer_mapping=mapping,
            force_projection=cfg.force_hidden_state_projection,
        )

    config_kwargs = dict(cfg.training_args)
    dataset_kwargs = config_kwargs.pop("dataset_kwargs", {})
    if cfg.dataset.prepacked:
        dataset_kwargs["skip_prepare_dataset"] = True
    max_length = config_kwargs.pop("max_length", cfg.sequence_length)

    training_arguments = SFTConfig(
        **config_kwargs,
        max_length=max_length,
        output_dir=cfg.output_path,
        dataset_kwargs=dataset_kwargs,
    )

    trainer = DistillationTrainer(
        model=model,
        config=cfg,
        signal_source=signal_source,
        hidden_state_mapping=hsm,
        true_vocab_size=tokenizer_vocab_size,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        args=training_arguments,
        data_collator=collate_packed_batch if cfg.dataset.prepacked else None,
        processing_class=None if cfg.dataset.prepacked else tokenizer,
    )

    named = dict(trainer.model.named_parameters())
    if args.param not in named:
        raise SystemExit(f"Param not found: {args.param}")
    p = named[args.param]
    before = p.detach().clone()

    trainer.train()

    after = p.detach().clone()
    _print("== results ==")
    _print("in-mem diff:", (after - before).abs().max().item())
    _print(
        "param in optimizer:",
        any(p is q for g in trainer.optimizer.param_groups for q in g["params"]),
    )
    _print("lr groups:", [g["lr"] for g in trainer.optimizer.param_groups[:2]])

    if args.save and _is_rank0():
        os.makedirs(args.save_dir, exist_ok=True)
        trainer.save_model(args.save_dir)
        saved = _load_tensor(args.save_dir, args.param)
        _print("in-mem vs saved diff:", (after.cpu() - saved).abs().max().item())

        base = _load_base_snapshot(cfg.train_model)
        if base:
            try:
                base_t = _load_tensor(base, args.param)
                _print("base vs saved diff:", (saved - base_t).abs().max().item())
            except Exception as exc:
                _print("base snapshot diff error:", exc)
        else:
            _print("base snapshot not found in cache.")


if __name__ == "__main__":
    main()
