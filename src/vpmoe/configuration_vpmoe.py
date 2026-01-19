# coding=utf-8
# Copyright 2025 The vpMoE authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vpMoE configuration (GPT‑OSS + hybrid attention schedule)."""

from __future__ import annotations

from typing import Iterable, List, Optional

from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig


_ATTN_IMPLS = {"tpa", "kda"}
_DEFAULT_ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 32.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "truncate": False,
    "original_max_position_embeddings": 4096,
}


def _as_list(value: Optional[Iterable[str]]) -> Optional[List[str]]:
    if value is None:
        return None
    return list(value)


class VpMoEConfig(GptOssConfig):
    model_type = "vpmoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    auto_map = {
        "AutoConfig": "configuration_vpmoe.VpMoEConfig",
        "AutoModel": "modeling_vpmoe.VpMoEModel",
        "AutoModelForCausalLM": "modeling_vpmoe.VpMoEForCausalLM",
    }

    def __init__(
        self,
        num_hidden_layers: int = 36,
        num_local_experts: int = 128,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        head_dim: int = 64,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        sliding_window: int = 128,
        rope_theta: float = 150000.0,
        tie_word_embeddings: bool = False,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        rope_scaling: Optional[dict] = None,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 4,
        router_aux_loss_coef: float = 0.9,
        output_router_logits: bool = False,
        use_cache: bool = True,
        layer_types: Optional[List[str]] = None,
        # vpMoE additions
        layer_attn_impls: Optional[Iterable[str]] = None,
        tpa_q_rank: int = 32,
        tpa_rank: int = 4,
        kda_mode: str = "chunk",
        kda_use_short_conv: bool = True,
        kda_conv_size: int = 4,
        kda_conv_bias: bool = False,
        kda_allow_neg_eigval: bool = False,
        kda_expand_v: float = 1.0,
        kda_num_v_heads: Optional[int] = None,
        **kwargs,
    ):
        if rope_scaling is None:
            rope_scaling = dict(_DEFAULT_ROPE_SCALING)

        # Resolve per-layer attention schedule.
        layer_attn_impls = _as_list(layer_attn_impls)
        if layer_attn_impls is None:
            layer_attn_impls = ["tpa"] * num_hidden_layers
        if len(layer_attn_impls) != num_hidden_layers:
            raise ValueError(
                "layer_attn_impls must have length num_hidden_layers; "
                f"got {len(layer_attn_impls)} vs {num_hidden_layers}."
            )
        invalid = sorted(set(layer_attn_impls) - _ATTN_IMPLS)
        if invalid:
            raise ValueError(
                "layer_attn_impls may only contain "
                f"{sorted(_ATTN_IMPLS)}; got invalid entries {invalid}."
            )

        # TPA config validation (locked defaults for v1).
        if tpa_q_rank <= 0 or tpa_rank <= 0:
            raise ValueError("tpa_q_rank and tpa_rank must be positive integers.")
        if tpa_q_rank != 32 or tpa_rank != 4:
            raise ValueError(
                "tpa_q_rank/tpa_rank are locked for v1: expected "
                "tpa_q_rank=32, tpa_rank=4."
            )

        # KDA config validation (Kimi/Arcee defaults).
        if kda_mode not in {"chunk", "fused_recurrent"}:
            raise ValueError("kda_mode must be 'chunk' or 'fused_recurrent'.")
        if kda_conv_size <= 0:
            raise ValueError("kda_conv_size must be > 0.")
        if kda_expand_v <= 0:
            raise ValueError("kda_expand_v must be > 0.")
        if kda_num_v_heads is None:
            kda_num_v_heads = num_attention_heads

        self.layer_attn_impls = layer_attn_impls
        self.tpa_q_rank = tpa_q_rank
        self.tpa_rank = tpa_rank
        self.kda_mode = kda_mode
        self.kda_use_short_conv = kda_use_short_conv
        self.kda_conv_size = kda_conv_size
        self.kda_conv_bias = kda_conv_bias
        self.kda_allow_neg_eigval = kda_allow_neg_eigval
        self.kda_expand_v = kda_expand_v
        self.kda_num_v_heads = kda_num_v_heads

        super().__init__(
            num_hidden_layers=num_hidden_layers,
            num_local_experts=num_local_experts,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_scaling=rope_scaling,
            attention_dropout=attention_dropout,
            num_experts_per_tok=num_experts_per_tok,
            router_aux_loss_coef=router_aux_loss_coef,
            output_router_logits=output_router_logits,
            use_cache=use_cache,
            layer_types=layer_types,
            **kwargs,
        )


__all__ = ["VpMoEConfig"]
