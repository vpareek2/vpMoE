# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


def build_alibi_slopes(num_heads: int) -> list[float]:
    """Return head-wise slopes for ALiBi."""

    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-2 ** -(torch.log2(torch.tensor(float(n))) - 3))
        ratio = start
        return [float(start * (ratio**i)) for i in range(n)]

    if (num_heads & (num_heads - 1)) == 0:
        return _get_slopes_power_of_2(num_heads)

    closest_power_of_2 = 2 ** (num_heads.bit_length() - 1)
    slopes = _get_slopes_power_of_2(closest_power_of_2)
    slopes += build_alibi_slopes(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
    return slopes


class AlibiBias(nn.Module):
    """ALiBi attention bias helper.

    Produces a bias tensor with shape (1, H, T, T) where bias[i, j] = -dist * slope_h.
    Future positions are clamped to distance 0 (bias 0), relying on causal masks elsewhere.
    """

    def __init__(
        self,
        num_heads: Optional[int] = None,
        slopes: Optional[Iterable[float]] = None,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        if slopes is None:
            if num_heads is None:
                raise ValueError("Either num_heads or slopes must be provided.")
            slopes = build_alibi_slopes(num_heads)
        slopes_tensor = torch.tensor(list(slopes), dtype=torch.float32)
        if learnable:
            self.slopes = nn.Parameter(slopes_tensor)
        else:
            self.register_buffer("slopes", slopes_tensor, persistent=False)
        self._cached_seq_len = None
        self._cached_bias = None

    def _compute_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        arange = torch.arange(seq_len, device=device)
        dist = (arange.view(seq_len, 1) - arange.view(1, seq_len)).clamp_min(0).float()
        slopes = self.slopes.to(device=device, dtype=torch.float32).view(1, -1, 1, 1)
        return -dist.view(1, 1, seq_len, seq_len) * slopes

    def get_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.slopes.requires_grad or self._cached_seq_len != seq_len or self._cached_bias is None:
            bias = self._compute_bias(seq_len, device=device)
            if not self.slopes.requires_grad:
                self._cached_seq_len = seq_len
                self._cached_bias = bias
        else:
            bias = self._cached_bias
            if bias.device != device:
                bias = bias.to(device=device)
        return bias.to(dtype=dtype)
