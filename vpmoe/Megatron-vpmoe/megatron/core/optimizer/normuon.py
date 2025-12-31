# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from dataclasses import astuple
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from megatron.core.optimizer.optimizer_config import OptimizerConfig, ParamKey


DEFAULT_POLAR_EXPRESS_SAFETY_FACTOR = 2e-2
DEFAULT_POLAR_EXPRESS_EPS = 1e-6
DEFAULT_POLAR_EXPRESS_COEFFS: Tuple[Tuple[float, float, float], ...] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def _matches(param: torch.nn.Parameter, param_name: str, param_key: ParamKey) -> bool:
    if isinstance(param_key.name, str):
        target_names = [param_key.name]
    else:
        target_names = list(param_key.name)
    for target_name in target_names:
        if param_name in target_name:
            return True

    if isinstance(param_key.attr, str):
        target_attrs = [param_key.attr]
    else:
        target_attrs = list(param_key.attr)
    for target_attr in target_attrs:
        if getattr(param, target_attr, False):
            return True

    return False


def _load_polar_express_coeffs(path: str | Path) -> Tuple[Tuple[float, float, float], ...]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Polar Express coeffs must be a non-empty list.")
    coeffs: List[Tuple[float, float, float]] = []
    for idx, triple in enumerate(payload):
        if (
            not isinstance(triple, (list, tuple))
            or len(triple) != 3
            or not all(isinstance(x, (int, float)) for x in triple)
        ):
            raise ValueError(f"Polar Express coeffs at index {idx} must be a triple of numbers.")
        coeffs.append((float(triple[0]), float(triple[1]), float(triple[2])))
    return tuple(coeffs)


def select_polar_express_coeffs(config: OptimizerConfig) -> Tuple[Tuple[float, float, float], ...]:
    if getattr(config, "polar_express_coeffs_path", None):
        return _load_polar_express_coeffs(config.polar_express_coeffs_path)

    if (
        abs(getattr(config, "polar_express_safety_factor", DEFAULT_POLAR_EXPRESS_SAFETY_FACTOR)
            - DEFAULT_POLAR_EXPRESS_SAFETY_FACTOR)
        > 1e-12
    ):
        raise ValueError(
            "polar_express_safety_factor differs from the default table. "
            "Provide --polar-express-coeffs-path for a matching coefficient set."
        )

    return DEFAULT_POLAR_EXPRESS_COEFFS


def polar_express_sign(
    matrix: torch.Tensor,
    coeffs: Sequence[Tuple[float, float, float]],
    safety_factor: float = DEFAULT_POLAR_EXPRESS_SAFETY_FACTOR,
    eps: float = DEFAULT_POLAR_EXPRESS_EPS,
) -> torch.Tensor:
    if matrix.ndim < 2:
        raise ValueError("Polar Express requires a tensor with at least 2 dimensions.")

    transposed = False
    work = matrix
    if work.size(-2) > work.size(-1):
        work = work.transpose(-2, -1)
        transposed = True

    work_dtype = torch.bfloat16 if work.is_cuda else torch.float32
    work = work.to(dtype=work_dtype)
    denom = work.norm(dim=(-2, -1), keepdim=True) * (1.0 + safety_factor) + eps
    work = work / denom

    for a, b, c in coeffs:
        xx_t = work @ work.transpose(-2, -1)
        b_term = b * xx_t + c * (xx_t @ xx_t)
        work = a * work + b_term @ work

    if transposed:
        work = work.transpose(-2, -1)

    return work.to(dtype=matrix.dtype)


def _normuon_variance_reduction(
    update: torch.Tensor,
    second_momentum: torch.Tensor,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    update_fp32 = update.float()
    v_mean = update_fp32.square().mean(dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)

    scale = torch.rsqrt(second_momentum + eps)
    scaled = update_fp32 * scale

    norm_before = update_fp32.norm(dim=(-2, -1), keepdim=True)
    norm_after = scaled.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    scaled = scaled * (norm_before / norm_after)

    return scaled.to(dtype=update.dtype)


def normuon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    second_momentum: torch.Tensor,
    coeffs: Sequence[Tuple[float, float, float]],
    *,
    mu: float,
    beta2: float,
    safety_factor: float,
    eps: float,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - mu)
    if nesterov:
        update = grad.lerp(momentum, mu)
    else:
        update = momentum

    update = polar_express_sign(
        update,
        coeffs=coeffs,
        safety_factor=safety_factor,
        eps=DEFAULT_POLAR_EXPRESS_EPS,
    )
    update = _normuon_variance_reduction(update, second_momentum, beta2, eps)

    scale = math.sqrt(max(1.0, grad.shape[-2] / grad.shape[-1]))
    return update.mul(scale)


def _select_group_lr(config: OptimizerConfig, use_normuon: bool) -> Tuple[float, float]:
    max_lr = config.lr
    min_lr = config.min_lr
    if not use_normuon:
        aux_lr = getattr(config, "normuon_aux_lr", None)
        aux_min_lr = getattr(config, "normuon_aux_min_lr", None)
        if aux_lr is not None:
            max_lr = aux_lr
        if aux_min_lr is not None:
            min_lr = aux_min_lr
    return max_lr, min_lr


def build_normuon_param_groups(
    model_chunks: Iterable[torch.nn.Module],
    config: OptimizerConfig,
    config_overrides: Optional[dict[ParamKey, OptimizerConfig]],
) -> List[dict]:
    params_map: dict[Tuple, List[torch.nn.Parameter]] = {}
    configs_map: dict[Tuple, Tuple[OptimizerConfig, bool]] = {}

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            uses_default_config = False
            if config_overrides is None:
                config_for_param = config
                uses_default_config = True
            else:
                config_for_param = None
                for param_key in config_overrides:
                    if _matches(param, name, param_key):
                        config_for_param = config_overrides[param_key]
                        break
                if config_for_param is None:
                    config_for_param = config
                    uses_default_config = True

            use_normuon = (
                param.ndim == 2 and not getattr(param, "is_embedding_or_output_parameter", False)
            )
            no_wd = name.endswith(".bias") or len(param.shape) == 1
            wd_mult = 0.0 if no_wd else 1.0
            is_expert_parallel = not getattr(param, "allreduce", True)

            config_for_param_copy = copy.deepcopy(config_for_param)
            config_for_param_copy.timers = None
            config_tuple = astuple(config_for_param_copy)
            key = (use_normuon, wd_mult, is_expert_parallel, config_tuple)
            params_map.setdefault(key, []).append(param)
            configs_map.setdefault(key, (config_for_param, uses_default_config))

    params_key = list(params_map.keys())
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered_params_key = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_params_key, params_key)
        for keys in gathered_params_key:
            for key in keys:
                if key not in params_key:
                    params_key.append(key)

    param_groups: List[dict] = []
    for key in params_key:
        use_normuon, wd_mult, is_expert_parallel, _ = key
        params = params_map.get(key, [])
        config_for_param, uses_default_config = configs_map.get(key, (config, True))

        group_max_lr, group_min_lr = _select_group_lr(config_for_param, use_normuon)

        param_group = {
            "params": params,
            "wd_mult": wd_mult,
            "lr_mult": 1.0,
            "is_expert_parallel": is_expert_parallel,
            "is_decoupled_lr": False,
            "default_config": uses_default_config,
            "max_lr": group_max_lr,
            "min_lr": group_min_lr,
            "lr": group_max_lr,
            "weight_decay": config_for_param.weight_decay * wd_mult,
            "use_normuon": use_normuon,
        }

        if use_normuon:
            param_group["momentum"] = config_for_param.normuon_momentum
            param_group["beta2"] = config_for_param.normuon_beta2
            param_group["normuon_eps"] = config_for_param.normuon_eps
        else:
            param_group["betas"] = (config_for_param.adam_beta1, config_for_param.adam_beta2)
            param_group["eps"] = config_for_param.adam_eps

        param_groups.append(param_group)

    return param_groups


class NormuonWithAuxAdam(torch.optim.Optimizer):
    """Normuon for matrix weights + AdamW for all other parameters."""

    def __init__(
        self,
        param_groups: List[dict],
        *,
        coeffs: Sequence[Tuple[float, float, float]],
        safety_factor: float,
        nesterov: bool = True,
        eps: float = 1e-10,
    ) -> None:
        self._coeffs = tuple(coeffs)
        self._safety_factor = float(safety_factor)
        self._nesterov = bool(nesterov)
        self._normuon_eps = float(eps)
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            lr = group["lr"]
            weight_decay = group.get("weight_decay", 0.0)
            if group.get("use_normuon", False):
                mu = group["momentum"]
                beta2 = group["beta2"]
                eps = group.get("normuon_eps", self._normuon_eps)
                for p in params:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("Normuon does not support sparse gradients.")
                    if p.grad.ndim != 2:
                        raise ValueError("Normuon only supports 2D parameter tensors.")
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    if "second_momentum_buffer" not in state:
                        state["second_momentum_buffer"] = torch.zeros(
                            p.shape[0],
                            1,
                            device=p.device,
                            dtype=torch.float32,
                        )
                    update = normuon_update(
                        p.grad,
                        state["momentum_buffer"],
                        state["second_momentum_buffer"],
                        self._coeffs,
                        mu=mu,
                        beta2=beta2,
                        safety_factor=self._safety_factor,
                        eps=eps,
                        nesterov=self._nesterov,
                    )
                    if weight_decay:
                        p.mul_(1.0 - lr * weight_decay)
                    p.add_(update, alpha=-lr)
            else:
                betas = group["betas"]
                eps = group["eps"]
                for p in params:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients.")
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.lerp_(p.grad, 1 - betas[0])
                    exp_avg_sq.lerp_(p.grad.square(), 1 - betas[1])
                    bias_correction1 = 1.0 - betas[0] ** state["step"]
                    bias_correction2 = 1.0 - betas[1] ** state["step"]
                    update = exp_avg / bias_correction1
                    update = update / (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                    if weight_decay:
                        p.mul_(1.0 - lr * weight_decay)
                    p.add_(update, alpha=-lr)

        return loss
