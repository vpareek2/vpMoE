#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _load_toml(path: Path) -> Dict[str, Any]:
    import tomllib

    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid TOML root: {path}")
    return payload


def _get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def _as_int(x: Any, *, name: str) -> int:
    _require(isinstance(x, int), f"{name} must be an int, got {type(x).__name__}")
    return int(x)


def _as_bool(x: Any, *, name: str) -> bool:
    _require(isinstance(x, bool), f"{name} must be a bool, got {type(x).__name__}")
    return bool(x)


def _as_float(x: Any, *, name: str) -> float:
    if isinstance(x, int):
        return float(x)
    _require(isinstance(x, float), f"{name} must be a float, got {type(x).__name__}")
    return float(x)


def _as_str(x: Any, *, name: str) -> str:
    _require(isinstance(x, str), f"{name} must be a str, got {type(x).__name__}")
    return str(x)


@dataclass(frozen=True)
class VpmoeConfig:
    vocab_size: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    ffn_hidden_size: int

    value_residual: bool
    value_residual_init: float
    normalization: str
    qk_layernorm: bool
    softmax_type: str
    add_bias_linear: bool
    add_qkv_bias: bool
    activation: str
    untied_embeddings: bool

    max_sequence_length: int
    position_embedding_type: str
    attention_dropout: float
    window_size: int
    window_attn_skip_freq: Optional[Union[int, List[int]]]

    use_tpa: bool
    tpa_rank: int
    tpa_q_rank: int

    grape_a: bool
    grapem_learnable_freq: bool
    grapem_share_across_heads: bool
    grapem_log_freq_scale: float

    num_experts: int
    topk: int
    router_load_balancing_type: Union[str, List[str]]
    aux_loss_coeff: float
    router_score_function: str
    router_topk_scaling_factor: Optional[float]
    router_pre_softmax: bool
    router_num_groups: Optional[int]
    router_group_topk: Optional[int]
    router_enable_expert_bias: bool
    router_bias_update_rate: float
    router_dtype: Optional[str]
    grouped_gemm: bool
    permute_fusion: bool
    router_fusion: bool
    shared_expert: bool
    shared_expert_size: int
    first_layer_dense: bool

    optimizer: str
    normuon_momentum: float
    normuon_beta2: float
    normuon_eps: float
    normuon_aux_lr: Optional[float]
    normuon_aux_min_lr: Optional[float]
    polar_express_safety_factor: float
    polar_express_coeffs_path: Optional[str]


def _parse_cfg(cfg: Dict[str, Any]) -> VpmoeConfig:
    vocab_size = _as_int(_get(cfg, "tokenizer", "vocab_size"), name="tokenizer.vocab_size")

    num_layers = _as_int(_get(cfg, "model", "num_layers"), name="model.num_layers")
    hidden_size = _as_int(_get(cfg, "model", "hidden_size"), name="model.hidden_size")
    num_attention_heads = _as_int(
        _get(cfg, "model", "num_attention_heads"), name="model.num_attention_heads"
    )
    num_query_groups = _as_int(
        _get(cfg, "model", "num_query_groups"), name="model.num_query_groups"
    )
    ffn_hidden_size = _as_int(_get(cfg, "model", "ffn_hidden_size"), name="model.ffn_hidden_size")

    value_residual = _as_bool(_get(cfg, "model", "value_residual"), name="model.value_residual")
    value_residual_init = _as_float(
        _get(cfg, "model", "value_residual_init", default=1.0), name="model.value_residual_init"
    )
    normalization = _as_str(_get(cfg, "model", "normalization"), name="model.normalization")
    qk_layernorm = _as_bool(_get(cfg, "model", "qk_layernorm"), name="model.qk_layernorm")
    softmax_type = _as_str(_get(cfg, "model", "softmax_type"), name="model.softmax_type")
    add_bias_linear = _as_bool(_get(cfg, "model", "add_bias_linear"), name="model.add_bias_linear")
    add_qkv_bias = _as_bool(_get(cfg, "model", "add_qkv_bias"), name="model.add_qkv_bias")
    activation = _as_str(_get(cfg, "model", "activation"), name="model.activation")
    untied_embeddings = _as_bool(
        _get(cfg, "model", "untied_embeddings"), name="model.untied_embeddings"
    )

    max_sequence_length = _as_int(
        _get(cfg, "attention", "max_sequence_length"), name="attention.max_sequence_length"
    )
    position_embedding_type = _as_str(
        _get(cfg, "attention", "position_embedding_type"), name="attention.position_embedding_type"
    )
    attention_dropout = _as_float(
        _get(cfg, "attention", "attention_dropout", default=0.0),
        name="attention.attention_dropout",
    )
    window_size = _as_int(_get(cfg, "attention", "window_size"), name="attention.window_size")

    window_attn_skip_freq = _get(cfg, "attention", "window_attn_skip_freq", default=None)
    if window_attn_skip_freq is not None:
        if isinstance(window_attn_skip_freq, int):
            pass
        elif isinstance(window_attn_skip_freq, list) and all(
            isinstance(v, int) for v in window_attn_skip_freq
        ):
            pass
        else:
            raise SystemExit(
                "attention.window_attn_skip_freq must be int, list[int], or null "
                f"(got {type(window_attn_skip_freq).__name__})"
            )

    use_tpa = _as_bool(_get(cfg, "tpa", "use_tpa"), name="tpa.use_tpa")
    tpa_rank = _as_int(_get(cfg, "tpa", "tpa_rank"), name="tpa.tpa_rank")
    tpa_q_rank = _as_int(_get(cfg, "tpa", "tpa_q_rank"), name="tpa.tpa_q_rank")

    grape_a = _as_bool(_get(cfg, "grape", "grape_a"), name="grape.grape_a")
    grapem_learnable_freq = _as_bool(
        _get(cfg, "grape", "grapem_learnable_freq"), name="grape.grapem_learnable_freq"
    )
    grapem_share_across_heads = _as_bool(
        _get(cfg, "grape", "grapem_share_across_heads"), name="grape.grapem_share_across_heads"
    )
    grapem_log_freq_scale = _as_float(
        _get(cfg, "grape", "grapem_log_freq_scale", default=16.0), name="grape.grapem_log_freq_scale"
    )

    num_experts = _as_int(_get(cfg, "moe", "num_experts"), name="moe.num_experts")
    topk = _as_int(_get(cfg, "moe", "topk"), name="moe.topk")

    router_load_balancing_type = _get(cfg, "moe", "router_load_balancing_type", default="aux_loss")
    if isinstance(router_load_balancing_type, str):
        pass
    elif isinstance(router_load_balancing_type, list) and all(
        isinstance(v, str) for v in router_load_balancing_type
    ):
        pass
    else:
        raise SystemExit(
            "moe.router_load_balancing_type must be str, list[str], or null "
            f"(got {type(router_load_balancing_type).__name__})"
        )

    aux_loss_coeff = _as_float(_get(cfg, "moe", "aux_loss_coeff", default=0.0), name="moe.aux_loss_coeff")
    router_score_function = _as_str(
        _get(cfg, "moe", "router_score_function", default="softmax"), name="moe.router_score_function"
    )
    router_topk_scaling_factor = _get(cfg, "moe", "router_topk_scaling_factor", default=None)
    if router_topk_scaling_factor is not None:
        router_topk_scaling_factor = _as_float(
            router_topk_scaling_factor, name="moe.router_topk_scaling_factor"
        )

    router_pre_softmax = _as_bool(
        _get(cfg, "moe", "router_pre_softmax", default=False), name="moe.router_pre_softmax"
    )
    router_num_groups = _get(cfg, "moe", "router_num_groups", default=None)
    if router_num_groups is not None:
        router_num_groups = _as_int(router_num_groups, name="moe.router_num_groups")
    router_group_topk = _get(cfg, "moe", "router_group_topk", default=None)
    if router_group_topk is not None:
        router_group_topk = _as_int(router_group_topk, name="moe.router_group_topk")

    router_enable_expert_bias = _as_bool(
        _get(cfg, "moe", "router_enable_expert_bias", default=False),
        name="moe.router_enable_expert_bias",
    )
    router_bias_update_rate = _as_float(
        _get(cfg, "moe", "router_bias_update_rate", default=1e-3), name="moe.router_bias_update_rate"
    )
    router_dtype = _get(cfg, "moe", "router_dtype", default=None)
    if router_dtype is not None:
        router_dtype = _as_str(router_dtype, name="moe.router_dtype")
        _require(router_dtype in {"fp32", "fp64"}, "moe.router_dtype must be one of: fp32, fp64")

    grouped_gemm = _as_bool(_get(cfg, "moe", "grouped_gemm", default=False), name="moe.grouped_gemm")
    permute_fusion = _as_bool(
        _get(cfg, "moe", "permute_fusion", default=False), name="moe.permute_fusion"
    )
    router_fusion = _as_bool(
        _get(cfg, "moe", "router_fusion", default=False), name="moe.router_fusion"
    )
    shared_expert = _as_bool(_get(cfg, "moe", "shared_expert"), name="moe.shared_expert")
    shared_expert_size = _as_int(
        _get(cfg, "moe", "shared_expert_size"), name="moe.shared_expert_size"
    )
    first_layer_dense = _as_bool(
        _get(cfg, "moe", "first_layer_dense"), name="moe.first_layer_dense"
    )

    optimizer = _as_str(_get(cfg, "optimizer", "name", default="normuon"), name="optimizer.name")
    _require(
        optimizer in {"adam", "sgd", "normuon"},
        "optimizer.name must be one of: adam, sgd, normuon",
    )
    normuon_momentum = _as_float(
        _get(cfg, "optimizer", "normuon_momentum", default=0.95),
        name="optimizer.normuon_momentum",
    )
    normuon_beta2 = _as_float(
        _get(cfg, "optimizer", "normuon_beta2", default=0.95),
        name="optimizer.normuon_beta2",
    )
    normuon_eps = _as_float(
        _get(cfg, "optimizer", "normuon_eps", default=1e-10),
        name="optimizer.normuon_eps",
    )
    normuon_aux_lr = _get(cfg, "optimizer", "normuon_aux_lr", default=None)
    if normuon_aux_lr is not None:
        normuon_aux_lr = _as_float(normuon_aux_lr, name="optimizer.normuon_aux_lr")
    normuon_aux_min_lr = _get(cfg, "optimizer", "normuon_aux_min_lr", default=None)
    if normuon_aux_min_lr is not None:
        normuon_aux_min_lr = _as_float(
            normuon_aux_min_lr, name="optimizer.normuon_aux_min_lr"
        )
    polar_express_safety_factor = _as_float(
        _get(cfg, "optimizer", "polar_express_safety_factor", default=2e-2),
        name="optimizer.polar_express_safety_factor",
    )
    polar_express_coeffs_path = _get(cfg, "optimizer", "polar_express_coeffs_path", default=None)
    if polar_express_coeffs_path is not None:
        polar_express_coeffs_path = _as_str(
            polar_express_coeffs_path, name="optimizer.polar_express_coeffs_path"
        )

    _require(
        hidden_size == num_attention_heads * _as_int(_get(cfg, "model", "head_dim"), name="model.head_dim"),
        "Expected model.head_dim * model.num_attention_heads == model.hidden_size",
    )

    if grape_a:
        _require(
            window_attn_skip_freq is not None,
            "grape.grape_a=true requires attention.window_attn_skip_freq to define global layers.",
        )

    if router_num_groups is not None:
        _require(
            num_experts % router_num_groups == 0,
            "moe.router_num_groups must divide moe.num_experts",
        )
        _require(
            router_group_topk is not None,
            "moe.router_group_topk must be set when moe.router_num_groups is set",
        )
        _require(
            topk % int(router_group_topk) == 0,
            "moe.router_group_topk must divide moe.topk",
        )

    return VpmoeConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        value_residual=value_residual,
        value_residual_init=value_residual_init,
        normalization=normalization,
        qk_layernorm=qk_layernorm,
        softmax_type=softmax_type,
        add_bias_linear=add_bias_linear,
        add_qkv_bias=add_qkv_bias,
        activation=activation,
        untied_embeddings=untied_embeddings,
        max_sequence_length=max_sequence_length,
        position_embedding_type=position_embedding_type,
        attention_dropout=attention_dropout,
        window_size=window_size,
        window_attn_skip_freq=window_attn_skip_freq,
        use_tpa=use_tpa,
        tpa_rank=tpa_rank,
        tpa_q_rank=tpa_q_rank,
        grape_a=grape_a,
        grapem_learnable_freq=grapem_learnable_freq,
        grapem_share_across_heads=grapem_share_across_heads,
        grapem_log_freq_scale=grapem_log_freq_scale,
        num_experts=num_experts,
        topk=topk,
        router_load_balancing_type=router_load_balancing_type,
        aux_loss_coeff=aux_loss_coeff,
        router_score_function=router_score_function,
        router_topk_scaling_factor=router_topk_scaling_factor,
        router_pre_softmax=router_pre_softmax,
        router_num_groups=router_num_groups,
        router_group_topk=router_group_topk,
        router_enable_expert_bias=router_enable_expert_bias,
        router_bias_update_rate=router_bias_update_rate,
        router_dtype=router_dtype,
        grouped_gemm=grouped_gemm,
        permute_fusion=permute_fusion,
        router_fusion=router_fusion,
        shared_expert=shared_expert,
        shared_expert_size=shared_expert_size,
        first_layer_dense=first_layer_dense,
        optimizer=optimizer,
        normuon_momentum=normuon_momentum,
        normuon_beta2=normuon_beta2,
        normuon_eps=normuon_eps,
        normuon_aux_lr=normuon_aux_lr,
        normuon_aux_min_lr=normuon_aux_min_lr,
        polar_express_safety_factor=polar_express_safety_factor,
        polar_express_coeffs_path=polar_express_coeffs_path,
    )


def _window_tuple(window_size: int) -> str:
    _require(window_size > 0, "window_size must be > 0 when window attention is enabled")
    # See docs/megatron.md: window size W includes current token => max lookback is W-1.
    return f"({window_size - 1},0)"


def _pattern_first_dense(num_layers: int) -> str:
    _require(num_layers >= 1, "num_layers must be >= 1")
    if num_layers == 1:
        return "([0])"
    # 1 dense layer followed by (num_layers-1) expert layers.
    return f"([0]+[1]*{num_layers - 1})"


def render_args(cfg: VpmoeConfig, *, seq_length: int) -> List[str]:
    _require(seq_length > 0, "seq_length must be > 0")
    _require(seq_length <= cfg.max_sequence_length, "seq_length exceeds max_sequence_length")

    args: List[str] = []

    # Tokenizer / vocab.
    args.extend(["--vocab-size", str(cfg.vocab_size)])
    args.extend(["--padded-vocab-size", str(cfg.vocab_size)])

    # Network size.
    args.extend(["--num-layers", str(cfg.num_layers)])
    args.extend(["--hidden-size", str(cfg.hidden_size)])
    args.extend(["--num-attention-heads", str(cfg.num_attention_heads)])
    args.append("--group-query-attention")
    args.extend(["--num-query-groups", str(cfg.num_query_groups)])
    args.extend(["--ffn-hidden-size", str(cfg.ffn_hidden_size)])

    # Sequence length.
    args.extend(["--seq-length", str(seq_length)])
    args.extend(["--max-position-embeddings", str(cfg.max_sequence_length)])

    # Model features.
    args.extend(["--normalization", cfg.normalization])
    if cfg.qk_layernorm:
        args.append("--qk-layernorm")
    args.extend(["--softmax-type", cfg.softmax_type])

    if not cfg.add_bias_linear:
        args.append("--disable-bias-linear")
    if cfg.add_qkv_bias:
        args.append("--add-qkv-bias")

    if cfg.activation == "squaredrelu":
        args.append("--squared-relu")
        # Megatron constrains bias-activation fusion to specific activations.
        # SquaredReLU must disable these fusions.
        args.append("--no-bias-gelu-fusion")
        args.append("--no-bias-swiglu-fusion")
    elif cfg.activation == "swiglu":
        args.append("--swiglu")
    elif cfg.activation == "gelu":
        pass
    else:
        raise SystemExit(f"Unsupported activation: {cfg.activation}")

    if cfg.value_residual:
        args.append("--value-residual")
        args.extend(["--value-residual-init", str(cfg.value_residual_init)])

    if cfg.untied_embeddings:
        args.append("--untie-embeddings-and-output-weights")

    # Optimizer.
    args.extend(["--optimizer", cfg.optimizer])
    if cfg.optimizer == "normuon":
        args.extend(["--normuon-momentum", str(cfg.normuon_momentum)])
        args.extend(["--normuon-beta2", str(cfg.normuon_beta2)])
        args.extend(["--normuon-eps", str(cfg.normuon_eps)])
        if cfg.normuon_aux_lr is not None:
            args.extend(["--normuon-aux-lr", str(cfg.normuon_aux_lr)])
        if cfg.normuon_aux_min_lr is not None:
            args.extend(["--normuon-aux-min-lr", str(cfg.normuon_aux_min_lr)])
        args.extend(
            ["--polar-express-safety-factor", str(cfg.polar_express_safety_factor)]
        )
        if cfg.polar_express_coeffs_path is not None:
            args.extend(
                ["--polar-express-coeffs-path", str(cfg.polar_express_coeffs_path)]
            )

    # Attention / positional encoding.
    args.extend(["--position-embedding-type", cfg.position_embedding_type])
    args.extend(["--attention-dropout", str(cfg.attention_dropout)])
    if cfg.position_embedding_type == "grapem":
        # Megatron GPT model disallows RoPE fusion with GRAPE-M.
        args.append("--no-rope-fusion")
        if cfg.grape_a:
            args.append("--grape-a")
            if isinstance(cfg.window_attn_skip_freq, int):
                args.extend(["--no-rope-freq", str(cfg.window_attn_skip_freq)])
        if cfg.grapem_learnable_freq:
            args.append("--grapem-learnable-freq")
        else:
            args.append("--no-grapem-learnable-freq")
        if cfg.grapem_share_across_heads:
            args.append("--grapem-share-across-heads")
        else:
            args.append("--grapem-per-head")
        args.extend(["--grapem-log-freq-scale", str(cfg.grapem_log_freq_scale)])

    if cfg.window_size > 0:
        args.extend(["--window-size", _window_tuple(cfg.window_size)])
        if cfg.window_attn_skip_freq is not None:
            args.extend(["--window-attn-skip-freq", str(cfg.window_attn_skip_freq)])

    if cfg.use_tpa:
        args.append("--use-tpa")
        args.extend(["--tpa-rank", str(cfg.tpa_rank)])
        args.extend(["--tpa-q-rank", str(cfg.tpa_q_rank)])

    # MoE.
    args.extend(["--num-experts", str(cfg.num_experts)])
    args.extend(["--moe-router-topk", str(cfg.topk)])
    args.extend(["--moe-ffn-hidden-size", str(cfg.ffn_hidden_size)])

    if isinstance(cfg.router_load_balancing_type, str):
        args.extend(["--moe-router-load-balancing-type", cfg.router_load_balancing_type])
        if cfg.router_load_balancing_type != "none" and cfg.aux_loss_coeff != 0.0:
            args.extend(["--moe-aux-loss-coeff", str(cfg.aux_loss_coeff)])
    else:
        args.extend(["--moe-router-load-balancing-type", *cfg.router_load_balancing_type])
        if cfg.aux_loss_coeff != 0.0:
            args.extend(["--moe-aux-loss-coeff", str(cfg.aux_loss_coeff)])

    args.extend(["--moe-router-score-function", cfg.router_score_function])
    if cfg.router_pre_softmax:
        args.append("--moe-router-pre-softmax")
    if cfg.router_num_groups is not None:
        args.extend(["--moe-router-num-groups", str(cfg.router_num_groups)])
        args.extend(["--moe-router-group-topk", str(int(cfg.router_group_topk))])
    if cfg.router_topk_scaling_factor is not None:
        args.extend(["--moe-router-topk-scaling-factor", str(cfg.router_topk_scaling_factor)])
    if cfg.router_enable_expert_bias:
        args.append("--moe-router-enable-expert-bias")
        args.extend(["--moe-router-bias-update-rate", str(cfg.router_bias_update_rate)])
    if cfg.router_dtype is not None:
        args.extend(["--moe-router-dtype", cfg.router_dtype])
    if cfg.grouped_gemm:
        args.append("--moe-grouped-gemm")
    if cfg.permute_fusion:
        args.append("--moe-permute-fusion")
    if cfg.router_fusion:
        args.append("--moe-router-fusion")

    if cfg.shared_expert:
        args.extend(["--moe-shared-expert-intermediate-size", str(cfg.shared_expert_size)])
    if cfg.first_layer_dense:
        args.extend(["--moe-layer-freq", _pattern_first_dense(cfg.num_layers)])

    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render vpMoE TOML into Megatron pretrain_gpt.py args.")
    ap.add_argument("--config", type=Path, default=Path("configs/vpmoe.toml"))
    ap.add_argument("--seq-length", type=int, default=None)
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    _require(args.config.exists(), f"Missing config: {args.config}")
    cfg = _parse_cfg(_load_toml(args.config))
    seq_length = int(args.seq_length) if args.seq_length is not None else cfg.max_sequence_length
    rendered = render_args(cfg, seq_length=seq_length)
    for item in rendered:
        print(item)


if __name__ == "__main__":
    main()
