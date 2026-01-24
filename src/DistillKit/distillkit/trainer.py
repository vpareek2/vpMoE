# Copyright 2024 Charles O. Goddard

import logging

import torch
from transformers.trainer_pt_utils import get_parameter_names
from transformers import (
    PreTrainedModel,
    TrainerCallback,
)
from transformers.modeling_outputs import CausalLMOutput
from trl import SFTTrainer

from distillkit.configuration import DistillationRunConfig, LossFunctionConfig
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs import ALL_LOSS_CLASSES, LossFunctionBase
from distillkit.signals import OnlineSignalSource, SignalSource, TeacherSignal

LOG = logging.getLogger(__name__)


class DistillEvalCallback(TrainerCallback):
    def __init__(self, trainer: "DistillationTrainer") -> None:
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        eval_steps = getattr(args, "eval_steps", None)
        if not eval_steps or eval_steps <= 0:
            return control
        if state.global_step == 0 or (state.global_step % eval_steps) != 0:
            return control
        if self.trainer.eval_dataset is None:
            return control
        metrics = self.trainer.evaluate_distill_subset()
        if metrics:
            self.trainer.log(metrics)
        return control


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
            teacher = self.signal_source.teacher_model
            device_map = getattr(teacher, "hf_device_map", None)
            if device_map:
                LOG.info(
                    "Teacher model uses a device_map; leaving it on dispatched devices."
                )
            else:
                teacher = teacher.to(self.accelerator.device)
            self.signal_source.teacher_model = teacher

        if self.eval_dataset is not None and getattr(self.args, "eval_steps", None):
            self.add_callback(DistillEvalCallback(self))

        self.model_accepts_loss_kwargs = False

    @staticmethod
    def _parse_muon_optim_args(value: str | None) -> dict | None:
        """
        Parse TrainingArguments.optim_args when using Muon.

        HF does not expose torch.optim.Muon via TrainingArguments.optim, so we
        opt into Muon by setting:
          optim_args: "muon" (defaults)
          optim_args: "muon,momentum=0.95,nesterov=true,ns_steps=5,eps=1e-7"
        """

        if not value:
            return None
        raw = value.strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts or parts[0].lower() != "muon":
            return None

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
        if muon_kwargs is None:
            return super().create_optimizer()

        # Muon only supports 2D parameters (torch enforces this), so we run Muon
        # on matrix-shaped weights and fall back to AdamW for everything else
        # (biases, norms, sink logits, etc.).
        weight_decay = float(getattr(self.args, "weight_decay", 0.0))
        lr = float(getattr(self.args, "learning_rate", 1e-3))

        # No-decay policy: biases + norm weights.
        # Note: transformers.ALL_LAYERNORM_LAYERS does not include RMSNorm in 4.57.6,
        # so we also exclude any parameter names containing ".norm." / ending in "norm.weight".
        decay_param_names = set(
            n
            for n in get_parameter_names(self.model, [torch.nn.LayerNorm])
            if not n.endswith(".bias")
        )

        embedding_param_names = set()
        lm_head_param_names = set()
        for module_name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                for pn, _ in module.named_parameters(recurse=False):
                    full = f"{module_name}.{pn}" if module_name else pn
                    embedding_param_names.add(full)
            if module_name == "lm_head" or module_name.endswith(".lm_head"):
                for pn, _ in module.named_parameters(recurse=False):
                    full = f"{module_name}.{pn}" if module_name else pn
                    lm_head_param_names.add(full)

        muon_decay = []
        muon_no_decay = []
        adamw_decay = []
        adamw_no_decay = []
        embed_head_tensors = 0
        embed_head_elems = 0
        non2d_tensors = 0
        non2d_elems = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            is_no_decay = (
                (name in decay_param_names)
                or name.endswith(".bias")
                or (".norm." in name)
                or name.endswith("norm.weight")
            )
            is_embedding_or_head = (name in embedding_param_names) or (
                name in lm_head_param_names
            )
            is_2d = (param.ndim == 2) and (not is_embedding_or_head)

            if is_embedding_or_head:
                embed_head_tensors += 1
                embed_head_elems += param.numel()

            if is_2d:
                (muon_no_decay if is_no_decay else muon_decay).append(param)
            else:
                non2d_tensors += 1
                non2d_elems += param.numel()
                (adamw_no_decay if is_no_decay else adamw_decay).append(param)

        if not muon_decay and not muon_no_decay:
            raise ValueError(
                "Muon requested via optim_args, but no trainable 2D parameters were found."
            )
        def _summarize(params):
            return len(params), sum(p.numel() for p in params)

        muon_decay_n, muon_decay_e = _summarize(muon_decay)
        muon_no_decay_n, muon_no_decay_e = _summarize(muon_no_decay)
        adamw_decay_n, adamw_decay_e = _summarize(adamw_decay)
        adamw_no_decay_n, adamw_no_decay_e = _summarize(adamw_no_decay)

        if self.is_world_process_zero():
            LOG.info(
                "Muon enabled via optim_args. muon=%d tensors/%d elems "
                "(decay=%d/%d, no_decay=%d/%d); adamw=%d tensors/%d elems "
                "(decay=%d/%d, no_decay=%d/%d). embed/lm_head=%d/%d elems; non2d=%d/%d elems.",
                muon_decay_n + muon_no_decay_n,
                muon_decay_e + muon_no_decay_e,
                muon_decay_n,
                muon_decay_e,
                muon_no_decay_n,
                muon_no_decay_e,
                adamw_decay_n + adamw_no_decay_n,
                adamw_decay_e + adamw_no_decay_e,
                adamw_decay_n,
                adamw_decay_e,
                adamw_no_decay_n,
                adamw_no_decay_e,
                embed_head_tensors,
                embed_head_elems,
                non2d_tensors,
                non2d_elems,
            )

        muon_param_groups = []
        if muon_decay:
            muon_param_groups.append({"params": muon_decay, "weight_decay": weight_decay})
        if muon_no_decay:
            muon_param_groups.append({"params": muon_no_decay, "weight_decay": 0.0})

        muon_opt = torch.optim.Muon(
            muon_param_groups,
            lr=lr,
            weight_decay=weight_decay,
            **muon_kwargs,
        )

        adamw_param_groups = []
        if adamw_decay:
            adamw_param_groups.append({"params": adamw_decay, "weight_decay": weight_decay})
        if adamw_no_decay:
            adamw_param_groups.append({"params": adamw_no_decay, "weight_decay": 0.0})

        if not adamw_decay and not adamw_no_decay:
            self.optimizer = muon_opt
            return self.optimizer

        adamw_opt = torch.optim.AdamW(
            adamw_param_groups,
            lr=lr,
            betas=(float(self.args.adam_beta1), float(self.args.adam_beta2)),
            eps=float(self.args.adam_epsilon),
            weight_decay=0.0,
        )

        class _MuonAdamWComposite(torch.optim.Optimizer):
            def __init__(self, muon: torch.optim.Optimizer, adamw: torch.optim.Optimizer):
                # Register param_groups with the base class for HF/Accelerate compatibility.
                merged_groups = list(muon.param_groups) + list(adamw.param_groups)
                super().__init__(merged_groups, defaults={})
                self._muon = muon
                self._adamw = adamw

            @torch.no_grad()
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
                self._muon.step()
                self._adamw.step()
                return loss

            def zero_grad(self, set_to_none: bool = True):
                self._muon.zero_grad(set_to_none=set_to_none)
                self._adamw.zero_grad(set_to_none=set_to_none)

            def state_dict(self):
                return {
                    "composite": "muon+adamw",
                    "muon": self._muon.state_dict(),
                    "adamw": self._adamw.state_dict(),
                }

            def load_state_dict(self, state_dict):
                if not isinstance(state_dict, dict):
                    raise TypeError("Invalid optimizer state dict.")
                self._muon.load_state_dict(state_dict.get("muon", {}))
                self._adamw.load_state_dict(state_dict.get("adamw", {}))

        self.optimizer = _MuonAdamWComposite(muon_opt, adamw_opt)
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

    def _distill_losses_for_batch(
        self,
        student_outputs: CausalLMOutput,
        inputs: dict[str, torch.Tensor],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        valid_mask = (inputs["labels"] >= 0).unsqueeze(-1)
        signal: TeacherSignal = self.signal_source.get_signal(
            inputs,
            return_hidden_states=self.need_hidden_states,
        )
        losses: dict[str, torch.Tensor] = {}
        for idx, loss_fn in enumerate(self.loss_functions):
            cfg = self.config.loss_functions[idx]
            losses[cfg.function.value] = loss_fn(
                student_outputs,
                signal,
                mask=valid_mask,
                hidden_state_mapping=self.hidden_state_mapping,
                num_items_in_batch=num_items_in_batch,
            )
        return losses

    @torch.no_grad()
    def evaluate_distill_subset(self) -> dict[str, float]:
        if self.eval_dataset is None:
            return {}
        self.model.eval()
        device = self.accelerator.device
        total_tokens = torch.tensor(0.0, device=device)
        sums = {
            "cross_entropy": torch.tensor(0.0, device=device),
            "kl": torch.tensor(0.0, device=device),
            "hs_cosine": torch.tensor(0.0, device=device),
        }

        dataloader = self.get_eval_dataloader()
        for inputs in dataloader:
            inputs = self._prepare_inputs(inputs)
            if "labels" not in inputs:
                inputs["labels"] = inputs["input_ids"]
            if self.config.dataset.eos_label_token_ids:
                inputs["labels"] = inputs["labels"].clone()
                for tok_id in self.config.dataset.eos_label_token_ids:
                    inputs["labels"][inputs["labels"] == tok_id] = (
                        self.model.config.eos_token_id
                    )

            labels = inputs["labels"]
            token_count = (labels >= 0).sum()
            if token_count.item() == 0:
                continue

            student_model = self.model.module if hasattr(self.model, "module") else self.model
            student_outputs = student_model(
                **{
                    k: inputs[k]
                    for k in ["input_ids", "attention_mask", "labels"]
                    if k in inputs
                },
                return_dict=True,
                output_hidden_states=self.need_hidden_states,
            )
            if student_outputs.logits.shape[-1] != self.true_vocab_size:
                student_outputs.logits = student_outputs.logits[..., : self.true_vocab_size]

            losses = self._distill_losses_for_batch(
                student_outputs,
                inputs,
                num_items_in_batch=token_count,
            )
            for key in sums:
                if key in losses:
                    sums[key] += losses[key] * token_count
            total_tokens += token_count

        total_tokens = self.accelerator.reduce(total_tokens, reduction="sum")
        for key in sums:
            sums[key] = self.accelerator.reduce(sums[key], reduction="sum")

        metrics: dict[str, float] = {}
        if total_tokens.item() > 0:
            for key in sums:
                metrics[f"eval_distill/{key}_token_avg"] = (
                    sums[key] / total_tokens
                ).item()
            weighted = 0.0
            weight_sum = 0.0
            for idx, cfg in enumerate(self.config.loss_functions):
                name = cfg.function.value
                if name in sums:
                    weighted += (sums[name] / total_tokens) * cfg.weight
                    weight_sum += cfg.weight
            if weight_sum > 0:
                metrics["eval_distill/weighted_token_avg"] = (weighted / weight_sum).item()

        self.model.train()
        return metrics

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
