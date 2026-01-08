# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    set_tensor_model_parallel_attributes,
)
from megatron.core.utils import divide, get_pg_size, get_tensor_model_parallel_group_if_none

try:
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear

    HAVE_TE = True
except ImportError:
    TEColumnParallelLinear = None
    TELinear = None
    HAVE_TE = False


class TPALinearQKV(nn.Module):
    """TPA QKV projection that matches Megatron's mixed QKV layout."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()

        if gather_output:
            raise ValueError("TPALinearQKV does not support gather_output=True.")
        if skip_bias_add:
            raise ValueError("TPALinearQKV does not support skip_bias_add.")
        if config.tpa_rank is None or config.tpa_q_rank is None:
            raise ValueError("TPA requires config.tpa_rank and config.tpa_q_rank to be set.")

        self.config = config
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        world_size = get_pg_size(self.tp_group)
        self.tpa_rank = int(config.tpa_rank)
        self.tpa_q_rank = int(config.tpa_q_rank)
        self.use_te_linears = bool(getattr(config, "transformer_impl", None) == "transformer_engine")
        if self.use_te_linears and not HAVE_TE:
            raise ImportError(
                "TPA requested TE linears (transformer_impl=transformer_engine), but Transformer Engine "
                "extensions are not available."
            )

        self.num_attention_heads = int(config.num_attention_heads)
        self.num_query_groups = (
            int(config.num_query_groups) if config.num_query_groups is not None else self.num_attention_heads
        )
        if self.num_attention_heads % self.num_query_groups != 0:
            raise ValueError(
                "TPA requires num_attention_heads to be divisible by num_query_groups."
            )
        if self.num_query_groups < world_size:
            raise ValueError(
                "TPA requires num_query_groups >= tensor_model_parallel_size for correct sharding."
            )

        self.num_attention_heads_per_partition = divide(self.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.num_query_groups, world_size)

        self.head_dim = int(config.kv_channels)
        expected_output = self.head_dim * self.num_attention_heads + 2 * self.head_dim * self.num_query_groups
        if output_size != expected_output:
            raise ValueError(
                f"TPALinearQKV expected output_size={expected_output}, got {output_size}."
            )
        self.output_size_per_partition = divide(expected_output, world_size)

        LinearA = TEColumnParallelLinear if self.use_te_linears else ColumnParallelLinear
        self.linear_a_q = LinearA(
            input_size,
            self.num_attention_heads * self.tpa_q_rank,
            config=config,
            init_method=init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=is_expert,
            tp_comm_buffer_name='tpa_a_q',
            tp_group=self.tp_group,
        )
        self.linear_a_k = LinearA(
            input_size,
            self.num_query_groups * self.tpa_rank,
            config=config,
            init_method=init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=is_expert,
            tp_comm_buffer_name='tpa_a_k',
            tp_group=self.tp_group,
        )
        self.linear_a_v = LinearA(
            input_size,
            self.num_query_groups * self.tpa_rank,
            config=config,
            init_method=init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=is_expert,
            tp_comm_buffer_name='tpa_a_v',
            tp_group=self.tp_group,
        )

        # B projections are shared across heads; gather to full output.
        if self.use_te_linears:
            assert TELinear is not None

            def _dup_linear(name: str, out_dim: int) -> nn.Module:
                return TELinear(
                    input_size=input_size,
                    output_size=out_dim,
                    parallel_mode="duplicated",
                    config=config,
                    init_method=init_method,
                    bias=False,
                    skip_bias_add=False,
                    skip_weight_param_allocation=False,
                    tp_comm_buffer_name=name,
                    is_expert=is_expert,
                    symmetric_ar_type=config.symmetric_ar_type,
                    tp_group=None,
                )

            self.linear_b_q = _dup_linear("tpa_b_q", self.tpa_q_rank * self.head_dim)
            self.linear_b_k = _dup_linear("tpa_b_k", self.tpa_rank * self.head_dim)
            self.linear_b_v = _dup_linear("tpa_b_v", self.tpa_rank * self.head_dim)
        else:
            gather_b = world_size > 1
            self.linear_b_q = ColumnParallelLinear(
                input_size,
                self.tpa_q_rank * self.head_dim,
                config=config,
                init_method=init_method,
                gather_output=gather_b,
                bias=False,
                skip_bias_add=False,
                is_expert=is_expert,
                tp_comm_buffer_name='tpa_b_q',
                tp_group=self.tp_group,
            )
            self.linear_b_k = ColumnParallelLinear(
                input_size,
                self.tpa_rank * self.head_dim,
                config=config,
                init_method=init_method,
                gather_output=gather_b,
                bias=False,
                skip_bias_add=False,
                is_expert=is_expert,
                tp_comm_buffer_name='tpa_b_k',
                tp_group=self.tp_group,
            )
            self.linear_b_v = ColumnParallelLinear(
                input_size,
                self.tpa_rank * self.head_dim,
                config=config,
                init_method=init_method,
                gather_output=gather_b,
                bias=False,
                skip_bias_add=False,
                is_expert=is_expert,
                tp_comm_buffer_name='tpa_b_v',
                tp_group=self.tp_group,
            )

        if bias:
            if config.use_cpu_initialization:
                self.bias = nn.Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = nn.Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
            if config.perform_initialization:
                with torch.no_grad():
                    self.bias.zero_()
            setattr(
                self.bias,
                "allreduce",
                not (is_expert and config.expert_model_parallel_size > 1),
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, hidden_states):
        # hidden_states: [s, b, h]
        if hidden_states.dtype != self.config.params_dtype:
            hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        seq_len, batch_size, _ = hidden_states.shape

        a_q, _ = self.linear_a_q(hidden_states)
        a_k, _ = self.linear_a_k(hidden_states)
        a_v, _ = self.linear_a_v(hidden_states)
        b_q, _ = self.linear_b_q(hidden_states)
        b_k, _ = self.linear_b_k(hidden_states)
        b_v, _ = self.linear_b_v(hidden_states)

        a_q = a_q.view(seq_len, batch_size, self.num_attention_heads_per_partition, self.tpa_q_rank)
        a_k = a_k.view(seq_len, batch_size, self.num_query_groups_per_partition, self.tpa_rank)
        a_v = a_v.view(seq_len, batch_size, self.num_query_groups_per_partition, self.tpa_rank)
        b_q = b_q.view(seq_len, batch_size, self.tpa_q_rank, self.head_dim)
        b_k = b_k.view(seq_len, batch_size, self.tpa_rank, self.head_dim)
        b_v = b_v.view(seq_len, batch_size, self.tpa_rank, self.head_dim)

        q = torch.matmul(
            a_q.view(seq_len * batch_size, self.num_attention_heads_per_partition, self.tpa_q_rank),
            b_q.view(seq_len * batch_size, self.tpa_q_rank, self.head_dim),
        ).div_(self.tpa_q_rank)
        q = q.view(seq_len, batch_size, self.num_attention_heads_per_partition, self.head_dim)

        k = torch.matmul(
            a_k.view(seq_len * batch_size, self.num_query_groups_per_partition, self.tpa_rank),
            b_k.view(seq_len * batch_size, self.tpa_rank, self.head_dim),
        ).div_(self.tpa_rank)
        k = k.view(seq_len, batch_size, self.num_query_groups_per_partition, self.head_dim)

        v = torch.matmul(
            a_v.view(seq_len * batch_size, self.num_query_groups_per_partition, self.tpa_rank),
            b_v.view(seq_len * batch_size, self.tpa_rank, self.head_dim),
        ).div_(self.tpa_rank)
        v = v.view(seq_len, batch_size, self.num_query_groups_per_partition, self.head_dim)

        heads_per_group = divide(
            self.num_attention_heads_per_partition, self.num_query_groups_per_partition
        )
        q = q.view(seq_len, batch_size, self.num_query_groups_per_partition, heads_per_group, self.head_dim)
        q = q.reshape(seq_len, batch_size, self.num_query_groups_per_partition, heads_per_group * self.head_dim)

        mixed_qkv = torch.cat([q, k, v], dim=3).reshape(seq_len, batch_size, -1)
        if self.bias is not None:
            mixed_qkv = mixed_qkv + self.bias
        return mixed_qkv, None

    def backward_dw(self) -> None:
        for module in (
            self.linear_a_q,
            self.linear_a_k,
            self.linear_a_v,
            self.linear_b_q,
            self.linear_b_k,
            self.linear_b_v,
        ):
            if hasattr(module, "backward_dw"):
                module.backward_dw()
