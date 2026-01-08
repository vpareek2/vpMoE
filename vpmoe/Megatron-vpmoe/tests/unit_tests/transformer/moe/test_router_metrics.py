import math

import torch

from megatron.core.transformer.moe.moe_utils import compute_router_stats


def test_compute_router_stats_basic():
    tokens = torch.tensor([10, 10, 0, 0], dtype=torch.float32)
    bias = torch.tensor([-1.0, 2.0, 0.0, 1.0], dtype=torch.float32)
    stats = compute_router_stats(tokens, expert_bias=bias)

    assert stats["experts_active"] == 2.0
    assert stats["dead_experts"] == 2.0
    assert math.isclose(stats["entropy"], math.log(2.0), rel_tol=1e-6)
    assert math.isclose(stats["cv_pct"], 100.0, rel_tol=1e-6)
    assert math.isclose(stats["max_load_pct"], 50.0, rel_tol=1e-6)
    assert math.isclose(stats["bias_range"], 3.0, rel_tol=1e-6)


def test_compute_router_stats_zero_tokens():
    tokens = torch.zeros(4, dtype=torch.float32)
    stats = compute_router_stats(tokens)

    assert stats["entropy"] == 0.0
    assert stats["cv_pct"] == 0.0
    assert stats["max_load_pct"] == 0.0
    assert stats["experts_active"] == 0.0
    assert stats["dead_experts"] == 4.0
