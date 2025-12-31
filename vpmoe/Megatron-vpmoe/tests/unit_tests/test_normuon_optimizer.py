# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import torch.nn as nn

from megatron.core.optimizer.normuon import (
    DEFAULT_POLAR_EXPRESS_COEFFS,
    _normuon_variance_reduction,
    build_normuon_param_groups,
    polar_express_sign,
)
from megatron.core.optimizer.optimizer_config import NormuonOptimizerConfig, ParamKey


def test_polar_express_sign_finite_and_shape():
    torch.manual_seed(123)
    matrix = torch.randn(4, 3)
    out = polar_express_sign(matrix, DEFAULT_POLAR_EXPRESS_COEFFS)
    assert out.shape == matrix.shape
    assert torch.isfinite(out).all()
    assert out.abs().sum().item() > 0.0


def test_normuon_variance_reduction_preserves_norm():
    torch.manual_seed(321)
    update = torch.randn(6, 4)
    second_momentum = torch.zeros(6, 1, dtype=torch.float32)
    norm_before = update.float().norm(dim=(-2, -1), keepdim=True)
    scaled = _normuon_variance_reduction(update, second_momentum, beta2=0.95, eps=1e-10)
    norm_after = scaled.float().norm(dim=(-2, -1), keepdim=True)
    assert torch.allclose(norm_before, norm_after, rtol=1e-5, atol=1e-6)


def test_build_normuon_param_groups_routing():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = nn.Parameter(torch.randn(4, 4))
            self.vector = nn.Parameter(torch.randn(4))
            self.embed = nn.Parameter(torch.randn(8, 4))
            self.embed.is_embedding_or_output_parameter = True

    model = Dummy()
    config = NormuonOptimizerConfig(
        lr=1e-3,
        min_lr=0.0,
        normuon_aux_lr=1e-4,
        normuon_aux_min_lr=0.0,
    )
    groups = build_normuon_param_groups([model], config, config_overrides=None)

    normuon_params = []
    aux_params = []
    for group in groups:
        if group.get("use_normuon", False):
            normuon_params.extend(group["params"])
        else:
            aux_params.extend(group["params"])

    assert model.matrix in normuon_params
    assert model.vector in aux_params
    assert any(param is model.embed for param in aux_params)

    aux_group = next(
        group for group in groups if any(param is model.vector for param in group["params"])
    )
    assert aux_group["wd_mult"] == 0.0
    assert aux_group["max_lr"] == 1e-4


def test_build_normuon_param_groups_decoupled_override():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = nn.Parameter(torch.randn(4, 4))
            self.embed = nn.Parameter(torch.randn(8, 4))
            self.embed.is_embedding_or_output_parameter = True

    model = Dummy()
    base = NormuonOptimizerConfig(lr=1e-3, min_lr=0.0, normuon_aux_lr=1e-4, normuon_aux_min_lr=0.0)
    decoupled = NormuonOptimizerConfig(lr=5e-5, min_lr=5e-5, normuon_aux_lr=None, normuon_aux_min_lr=None)
    config_overrides = {ParamKey(attr="is_embedding_or_output_parameter"): decoupled}

    groups = build_normuon_param_groups([model], base, config_overrides=config_overrides)
    embed_group = next(group for group in groups if any(param is model.embed for param in group["params"]))
    assert embed_group["use_normuon"] is False
    assert embed_group["max_lr"] == 5e-5
    assert embed_group["min_lr"] == 5e-5
