#!/usr/bin/env python3
"""Analytic parameter counter for vpMoE.

This script does NOT import Megatron or torch. It uses closed-form math based
on the architecture config to return an instant count.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


def _load_toml(path: str) -> Dict[str, Any]:
    try:
        import tomllib  # py3.11+
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "TOML parsing requires Python 3.11+ (tomllib).\n"
            f"Original error: {e}"
        )
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid TOML root in {path}: expected a table.")
    return data


def _get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _human(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def _mlp_params(
    hidden: int,
    ffn: int,
    *,
    gated: bool,
    bias: bool,
) -> int:
    if gated:
        # linear_fc1 output is 2 * ffn for GLU variants
        fc1_out = 2 * ffn
        fc1 = hidden * fc1_out + (fc1_out if bias else 0)
        fc2 = ffn * hidden + (hidden if bias else 0)
        return fc1 + fc2
    fc1 = hidden * ffn + (ffn if bias else 0)
    fc2 = ffn * hidden + (hidden if bias else 0)
    return fc1 + fc2


def _attention_params(
    hidden: int,
    num_heads: int,
    num_query_groups: int,
    head_dim: int,
    *,
    use_tpa: bool,
    tpa_rank: int | None,
    tpa_q_rank: int | None,
    add_bias_linear: bool,
    add_qkv_bias: bool,
) -> int:
    # Output projection always exists.
    out_proj = hidden * hidden + (hidden if add_bias_linear else 0)

    if use_tpa:
        if tpa_rank is None or tpa_q_rank is None:
            raise SystemExit("TPA enabled but tpa_rank/tpa_q_rank not set.")
        # See megatron/core/transformer/tpa.py for exact shapes.
        a_q = hidden * (num_heads * tpa_q_rank)
        a_k = hidden * (num_query_groups * tpa_rank)
        a_v = hidden * (num_query_groups * tpa_rank)
        b_q = hidden * (tpa_q_rank * head_dim)
        b_k = hidden * (tpa_rank * head_dim)
        b_v = hidden * (tpa_rank * head_dim)
        # TPA forbids biases for QKV.
        return out_proj + a_q + a_k + a_v + b_q + b_k + b_v

    # Standard QKV: output dim = hidden + 2 * num_query_groups * head_dim
    qkv_out = hidden + 2 * num_query_groups * head_dim
    qkv = hidden * qkv_out
    qkv_bias = qkv_out if (add_bias_linear or add_qkv_bias) else 0
    return out_proj + qkv + qkv_bias


def _norm_params(hidden: int, *, normalization: str, include_bias: bool) -> int:
    # RMSNorm has weight only; LayerNorm has weight+bias.
    if normalization == "RMSNorm":
        return hidden
    if normalization == "LayerNorm":
        return hidden * (2 if include_bias else 1)
    # L2Norm has no params.
    if normalization == "L2Norm":
        return 0
    raise SystemExit(f"Unsupported normalization: {normalization}")


def _qk_norm_params(head_dim: int, normalization: str) -> int:
    # q/k layernorm always uses weight+bias for LayerNorm, weight-only for RMSNorm.
    if normalization == "LayerNorm":
        per = head_dim * 2
    elif normalization == "RMSNorm":
        per = head_dim
    else:
        per = 0
    # q + k
    return 2 * per


def _grapem_params(
    head_dim: int,
    num_heads: int,
    *,
    share_across_heads: bool,
    rotary_percent: float = 1.0,
) -> int:
    dim = int(head_dim * rotary_percent)
    base = dim // 2
    if share_across_heads:
        return base
    return base * num_heads


def _softmax_offset_params(num_layers: int, num_heads: int, softmax_type: str) -> int:
    if softmax_type != "learnable":
        return 0
    return num_layers * num_heads


def _count_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    vocab = int(_get(cfg, "tokenizer", "vocab_size", default=201_088))
    num_layers = int(_get(cfg, "model", "num_layers", default=80))
    hidden = int(_get(cfg, "model", "hidden_size", default=1024))
    num_heads = int(_get(cfg, "model", "num_attention_heads", default=8))
    num_query_groups = int(_get(cfg, "model", "num_query_groups", default=2))
    head_dim = int(_get(cfg, "model", "head_dim", default=128))
    ffn = int(_get(cfg, "model", "ffn_hidden_size"))
    normalization = str(_get(cfg, "model", "normalization", default="RMSNorm"))
    qk_layernorm = bool(_get(cfg, "model", "qk_layernorm", default=True))
    softmax_type = str(_get(cfg, "model", "softmax_type", default="learnable"))
    add_bias_linear = bool(_get(cfg, "model", "add_bias_linear", default=True))
    add_qkv_bias = bool(_get(cfg, "model", "add_qkv_bias", default=False))
    gated = bool(_get(cfg, "model", "gated_linear_unit", default=False))
    untied = bool(_get(cfg, "model", "untied_embeddings", default=True))

    activation = str(_get(cfg, "model", "activation", default="squaredrelu"))
    use_tpa = bool(_get(cfg, "tpa", "use_tpa", default=True))
    tpa_rank = _get(cfg, "tpa", "tpa_rank", default=None)
    tpa_q_rank = _get(cfg, "tpa", "tpa_q_rank", default=None)

    num_experts = int(_get(cfg, "moe", "num_experts", default=64))
    topk = int(_get(cfg, "moe", "topk", default=2))
    shared_expert = bool(_get(cfg, "moe", "shared_expert", default=True))
    shared_expert_size = int(_get(cfg, "moe", "shared_expert_size", default=512))
    first_layer_dense = bool(_get(cfg, "moe", "first_layer_dense", default=True))

    position_embedding_type = str(
        _get(cfg, "attention", "position_embedding_type", default="grapem")
    )
    grapem_learnable_freq = bool(_get(cfg, "grape", "grapem_learnable_freq", default=True))
    grapem_share_across_heads = bool(_get(cfg, "grape", "grapem_share_across_heads", default=True))
    grape_a = bool(_get(cfg, "grape", "grape_a", default=False))
    grape_a_learnable = bool(_get(cfg, "grape", "grape_a_learnable", default=False))

    # Embeddings + output head
    embeddings = vocab * hidden
    output = vocab * hidden if untied else 0

    # Norms: per layer input/post + final layernorm
    per_layer_norms = 2 * _norm_params(hidden, normalization=normalization, include_bias=True)
    final_norm = _norm_params(hidden, normalization=normalization, include_bias=True)

    # QK norm params (per layer) if enabled.
    qk_norm = _qk_norm_params(head_dim, normalization) if qk_layernorm else 0

    # Softmax offset params (per layer) if learnable.
    softmax_offset = _softmax_offset_params(num_layers, num_heads, softmax_type)

    # GRAPE-M params (single module).
    grapem_params = 0
    if position_embedding_type == "grapem":
        grapem_params = _grapem_params(
            head_dim,
            num_heads,
            share_across_heads=grapem_share_across_heads,
            rotary_percent=1.0,
        )

    # GRAPE-A is ALiBi in our Megatron wiring. By default it uses a non-trainable buffer.
    grape_a_params = num_heads if (grape_a and grape_a_learnable) else 0

    # Local/global schedule affects *whether* TPA is used: in our Megatron wiring, TPA applies
    # to local (windowed) layers only; global layers use standard QKV.
    window_size = int(_get(cfg, "attention", "window_size", default=0))
    window_attn_skip_freq = _get(cfg, "attention", "window_attn_skip_freq", default=None)
    has_window = window_size and window_size > 0
    if has_window and window_attn_skip_freq is None:
        local_layers = num_layers
        global_layers = 0
    elif has_window and isinstance(window_attn_skip_freq, int) and window_attn_skip_freq > 0:
        global_layers = num_layers // window_attn_skip_freq
        local_layers = num_layers - global_layers
    else:
        local_layers = 0
        global_layers = num_layers

    # Attention params per layer, split by local/global behavior.
    attn_std_per_layer = _attention_params(
        hidden,
        num_heads,
        num_query_groups,
        head_dim,
        use_tpa=False,
        tpa_rank=None,
        tpa_q_rank=None,
        add_bias_linear=add_bias_linear,
        add_qkv_bias=add_qkv_bias,
    )
    attn_local_per_layer = (
        _attention_params(
            hidden,
            num_heads,
            num_query_groups,
            head_dim,
            use_tpa=True,
            tpa_rank=tpa_rank,
            tpa_q_rank=tpa_q_rank,
            add_bias_linear=add_bias_linear,
            add_qkv_bias=add_qkv_bias,
        )
        if use_tpa and local_layers > 0
        else attn_std_per_layer
    )

    # MLP params.
    dense_mlp = _mlp_params(hidden, ffn, gated=gated, bias=add_bias_linear)
    shared_mlp = (
        _mlp_params(hidden, shared_expert_size, gated=gated, bias=add_bias_linear)
        if shared_expert
        else 0
    )

    # Router params (per MoE layer).
    router = hidden * num_experts + (num_experts if add_bias_linear else 0)

    # Layer counts.
    dense_layers = 1 if first_layer_dense else 0
    moe_layers = num_layers - dense_layers

    # Expert params per MoE layer.
    expert_per_layer = num_experts * _mlp_params(hidden, ffn, gated=gated, bias=add_bias_linear)

    totals = {
        "embeddings": embeddings,
        "output": output,
        "attention_local": attn_local_per_layer * local_layers,
        "attention_global": attn_std_per_layer * global_layers,
        "norms": per_layer_norms * num_layers + final_norm,
        "qk_norms": qk_norm * num_layers,
        "softmax_offset": softmax_offset,
        "grapem": grapem_params,
        "grape_a": grape_a_params,
        "dense_mlp": dense_mlp * dense_layers,
        "moe_experts": expert_per_layer * moe_layers,
        "moe_router": router * moe_layers,
        "moe_shared_expert": shared_mlp * moe_layers,
    }

    total = sum(totals.values())
    experts_total = totals["moe_experts"]
    activated = int(total - experts_total + experts_total * (topk / num_experts))

    return {
        "total_params": total,
        "activated_params_estimate": activated,
        "by_category": dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True)),
        "assumptions": {
            "mlp_gated": gated,
            "activation": activation,
            "normalization": normalization,
            "qk_layernorm": qk_layernorm,
            "softmax_type": softmax_type,
            "use_tpa": use_tpa,
            "local_layers": local_layers,
            "global_layers": global_layers,
            "add_bias_linear": add_bias_linear,
            "add_qkv_bias": add_qkv_bias,
            "first_layer_dense": first_layer_dense,
            "grapem_requires_grad": grapem_learnable_freq,
            "grape_a": grape_a,
            "grape_a_learnable": grape_a_learnable,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analytic vpMoE parameter counter.")
    p.add_argument(
        "--config",
        default="configs/vpmoe.toml",
        help="TOML config path (defaults to configs/vpmoe.toml).",
    )
    p.add_argument("--json-out", default=None, help="Optional path to write JSON results.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.config):
        raise SystemExit(f"Config not found: {args.config}")
    cfg = _load_toml(args.config)
    counts = _count_from_config(cfg)

    total = counts["total_params"]
    activated = counts["activated_params_estimate"]
    title = "TOTAL PARAMS / ACTIVE PARAMS"
    bar = "=" * max(24, len(title) + 8)
    print(bar)
    print(f"  {title}")
    print(bar)
    print(f"  total_params:            {total:>14}  ({_human(total)})")
    print(f"  activated_params_est:    {activated:>14}  ({_human(activated)})")
    print("")

    print("BREAKDOWN (share of total)")
    print("-" * 24)
    name_w = max(len(k) for k in counts["by_category"]) if counts["by_category"] else 10
    for k, v in counts["by_category"].items():
        pct = (v / total * 100.0) if total else 0.0
        print(f"  {k:<{name_w}}  {v:>14}  {_human(v):>8}  {pct:>6.2f}%")

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"script": os.path.relpath(__file__), "counts": counts, "config": args.config},
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
        print(f"wrote_json: {out_path}")


if __name__ == "__main__":
    main()
