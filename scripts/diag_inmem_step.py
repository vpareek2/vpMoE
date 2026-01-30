#!/usr/bin/env python3
"""
Run a single in-memory train step (no checkpoint save) and report whether
weights actually change. This isolates Trainer/optimizer behavior without
disk writes.
"""

import argparse
import os
from dataclasses import dataclass

import yaml

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


@dataclass
class _ParamProbe:
    name: str
    dtype: str
    requires_grad: bool
    grad_max_abs: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/distillkit/vpmoe_distill_1_smoke_2xh100.yaml",
    )
    parser.add_argument("--output", default="/tmp/diag_inmem")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--param",
        default="model.layers.0.self_attn.W_A_q.weight",
        help="Parameter name to diff before/after the train step.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B logging for this diagnostic run.",
    )
    args = parser.parse_args()

    if args.disable_wandb:
        os.environ.setdefault("WANDB_DISABLED", "true")
        os.environ.setdefault("WANDB_MODE", "disabled")

    cfg_dict = yaml.safe_load(open(args.config))
    cfg_dict["output_path"] = args.output
    cfg_dict["training_args"]["max_steps"] = args.max_steps
    cfg_dict["training_args"]["save_strategy"] = "no"
    cfg_dict["training_args"]["eval_steps"] = 0
    cfg_dict["training_args"]["logging_steps"] = 1
    cfg_dict["training_args"]["lr_scheduler_type"] = "constant"
    cfg_dict["training_args"]["warmup_ratio"] = 0.0
    cfg_dict["training_args"]["report_to"] = []

    cfg = DistillationRunConfig.model_validate(cfg_dict)

    tokenizer = load_tokenizer(cfg)
    ds_train, ds_eval = load_data(cfg.dataset, tokenizer)

    tokenizer_vocab_size = max(
        len(tokenizer.get_vocab()),
        max(tokenizer.get_vocab().values()) + 1,
    )

    model = load_student_model(cfg, tokenizer_vocab_size)
    signal_source = create_signal_source(cfg, tokenizer_vocab_size)

    hsm = None
    if cfg.layer_mapping is not None:
        if not isinstance(signal_source, OnlineSignalSource):
            raise RuntimeError("Hidden state distillation requires online teacher.")
        teacher_hidden_size = signal_source.teacher_model.config.hidden_size
        if cfg.layer_mapping == "all":
            mapping = [(i, i) for i in range(model.config.num_hidden_layers)]
        else:
            mapping = cfg.layer_mapping
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
    probe = _ParamProbe(
        name=args.param,
        dtype=str(p.dtype),
        requires_grad=bool(p.requires_grad),
    )

    def _capture_grad(g):
        # Trainer may zero grads after the optimizer step; capture during backward.
        try:
            probe.grad_max_abs = float(g.detach().abs().max().item())
        except Exception:
            probe.grad_max_abs = None
        return g

    if p.requires_grad:
        p.register_hook(_capture_grad)
    before = p.detach().clone()

    trainer.train()

    after = p.detach().clone()
    print("in-mem diff:", (after - before).abs().max().item())
    print(
        "param:",
        probe.name,
        "dtype:",
        probe.dtype,
        "requires_grad:",
        probe.requires_grad,
        "grad_max_abs:",
        probe.grad_max_abs,
    )
    print(
        "param in optimizer:",
        any(p is q for g in trainer.optimizer.param_groups for q in g["params"]),
    )
    lrs = [float(g.get("lr", 0.0)) for g in trainer.optimizer.param_groups[:4]]
    upd = [
        float(g.get("update_scale", 0.0)) for g in trainer.optimizer.param_groups[:4]
    ]
    print("lr groups:", lrs)
    print("update_scale groups:", upd)


if __name__ == "__main__":
    main()
