# Copyright 2024 Charles O. Goddard

import torch
from transformers.trainer_pt_utils import get_parameter_names
from transformers import (
    PreTrainedModel,
)
from trl import SFTTrainer

from distillkit.configuration import DistillationRunConfig, LossFunctionConfig
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs import ALL_LOSS_CLASSES, LossFunctionBase
from distillkit.signals import OnlineSignalSource, SignalSource, TeacherSignal


def create_loss_func(cfg: LossFunctionConfig) -> LossFunctionBase:
    for cls in ALL_LOSS_CLASSES:
        if cfg.function.value == cls.name():
            return cls(
                **cfg.model_dump(exclude=["function", "weight"], exclude_none=True)
            )
    raise RuntimeError(f"Unknown loss function '{cfg.function}'")


class DistillationTrainer(SFTTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        config: DistillationRunConfig,
        signal_source: SignalSource,
        true_vocab_size: int,
        *args,
        hidden_state_mapping: HiddenStateMapping | None = None,
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        self.true_vocab_size = true_vocab_size
        self.config = config

        self.loss_functions = [create_loss_func(lfc) for lfc in config.loss_functions]
        self.need_hidden_states = any(
            lf.requires_hidden_states() for lf in self.loss_functions
        )

        self.signal_source = signal_source
        self.hidden_state_mapping = hidden_state_mapping

        if self.need_hidden_states and not self.signal_source.supports_hidden_states():
            raise ValueError(
                "Configuration requests hidden state loss, but the provided Teacher "
                "(Offline/Dataset) does not support hidden states."
            )

        if (self.hidden_state_mapping is None) and self.need_hidden_states:
            raise ValueError(
                "Must define a hidden state mapping to use hidden state losses."
            )

        if isinstance(self.signal_source, OnlineSignalSource):
            self.signal_source.teacher_model = self.signal_source.teacher_model.to(
                self.accelerator.device
            )

        self.model_accepts_loss_kwargs = False

    @staticmethod
    def _parse_muon_optim_args(value: str | None) -> dict:
        """
        Parse TrainingArguments.optim_args when using Muon.

        HF does not expose torch.optim.Muon via TrainingArguments.optim, so we
        opt into Muon by setting:
          optim_args: "muon" (defaults)
          optim_args: "muon,momentum=0.95,nesterov=true,ns_steps=5,eps=1e-7"
        """

        if not value:
            return {}
        raw = value.strip()
        if not raw:
            return {}
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts or parts[0].lower() != "muon":
            return {}

        out: dict = {}
        for part in parts[1:]:
            if "=" not in part:
                raise ValueError(f"Invalid optim_args token '{part}'. Expected key=value.")
            key, val = part.split("=", 1)
            key = key.strip()
            val = val.strip()
            if not key:
                raise ValueError(f"Invalid optim_args token '{part}'. Empty key.")

            if key in {"momentum", "eps"}:
                out[key] = float(val)
            elif key in {"ns_steps"}:
                out[key] = int(val)
            elif key in {"nesterov"}:
                vv = val.lower()
                if vv in {"true", "1", "yes", "y"}:
                    out[key] = True
                elif vv in {"false", "0", "no", "n"}:
                    out[key] = False
                else:
                    raise ValueError(f"Invalid boolean for {key} in optim_args: '{val}'.")
            else:
                raise ValueError(
                    f"Unsupported Muon optim_args key '{key}'. "
                    "Supported keys: momentum, nesterov, ns_steps, eps."
                )
        return out

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        muon_kwargs = self._parse_muon_optim_args(getattr(self.args, "optim_args", None))
        if not muon_kwargs:
            return super().create_optimizer()

        # No-decay policy: biases + norm weights.
        # Note: transformers.ALL_LAYERNORM_LAYERS does not include RMSNorm in 4.57.6,
        # so we also exclude any parameter names containing ".norm." / ending in "norm.weight".
        decay_param_names = set(
            n
            for n in get_parameter_names(self.model, [torch.nn.LayerNorm])
            if not n.endswith(".bias")
        )

        weight_decay = float(getattr(self.args, "weight_decay", 0.0))
        lr = float(getattr(self.args, "learning_rate", 1e-3))

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                    and (n not in decay_param_names)
                    and (not n.endswith(".bias"))
                    and (".norm." not in n)
                    and (not n.endswith("norm.weight"))
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                    and (
                        (n in decay_param_names)
                        or n.endswith(".bias")
                        or (".norm." in n)
                        or n.endswith("norm.weight")
                    )
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.Muon(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
            **muon_kwargs,
        )
        return self.optimizer

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"]
        if self.config.dataset.eos_label_token_ids:
            inputs["labels"] = inputs["labels"].clone()
            for tok_id in self.config.dataset.eos_label_token_ids:
                inputs["labels"][inputs["labels"] == tok_id] = (
                    self.model.config.eos_token_id
                )

        student_model = model.module if hasattr(model, "module") else model
        student_outputs = student_model(
            **{
                k: inputs[k]
                for k in ["input_ids", "attention_mask", "labels"]
                if k in inputs
            },
            return_dict=True,
            output_hidden_states=self.need_hidden_states,
            **kwargs,
        )
        if student_outputs.logits.shape[-1] != self.true_vocab_size:
            # truncate any extra logits from padding
            student_outputs.logits = student_outputs.logits[..., : self.true_vocab_size]

        total_loss = self.total_distillation_loss(
            student_outputs,
            inputs,
            num_items_in_batch=None,
        )
        return (total_loss, student_outputs) if return_outputs else total_loss

    def total_distillation_loss(
        self, student_outputs, inputs, num_items_in_batch: int | None = None
    ):
        valid_mask = (inputs["labels"] >= 0).unsqueeze(-1)
        signal: TeacherSignal = self.signal_source.get_signal(
            inputs,
            return_hidden_states=self.need_hidden_states,
        )

        losses = []
        loss_fns = []
        weights = []
        for idx, loss_fn in enumerate(self.loss_functions):
            cfg = self.config.loss_functions[idx]
            loss = loss_fn(
                student_outputs,
                signal,
                mask=valid_mask,
                hidden_state_mapping=self.hidden_state_mapping,
                num_items_in_batch=num_items_in_batch,
            )
            losses.append(loss)
            loss_fns.append(cfg.function.value)
            weights.append(cfg.weight)

        total_loss = 0.0
        for loss, weight in zip(losses, weights):
            total_loss += loss * weight
        total_loss = total_loss / sum(weights)
        self.log(
            {
                f"distillation_loss/{idx + 1}_{loss_fn}": loss.item()
                for idx, (loss, loss_fn) in enumerate(zip(losses, loss_fns))
            }
        )
        return total_loss
