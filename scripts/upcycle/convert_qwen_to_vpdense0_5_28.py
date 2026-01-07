#!/usr/bin/env python3
"""Convert Qwen3-0.6B-o200k (HF) into vpDense0-5_28 (Megatron, compat)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vpmoe" / "Megatron-vpmoe"))
sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class StudentSpec:
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    ffn_hidden_size: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert Qwen3-0.6B-o200k (HF) into vpDense0-5_28 (Megatron compat)."
    )
    ap.add_argument(
        "--config",
        default="configs/upcycle/vpDense0-5_28.convert.toml",
        help="Conversion config TOML.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output dir.")
    return ap.parse_args()


def load_toml(path: Path) -> Dict[str, object]:
    import tomllib

    return tomllib.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def try_git_sha(root: Path) -> Optional[str]:
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out if out else None
    except Exception:
        return None


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"Output dir exists: {path} (use --overwrite)")
        # Refuse to delete if not within repo.
        if REPO_ROOT not in path.resolve().parents and path.resolve() != REPO_ROOT:
            raise SystemExit(f"Refusing to overwrite outside repo: {path}")
        for child in path.iterdir():
            if child.is_dir():
                for sub in child.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                for sub in sorted(child.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
                child.rmdir()
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def write_training_layout(output_dir: Path, iteration: int = 0) -> None:
    """Create Megatron training-compatible checkpoint layout."""
    tracker = output_dir / "latest_checkpointed_iteration.txt"
    tracker.write_text(f"{iteration}\n")

    iter_dir = output_dir / f"iter_{iteration:07d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    candidates = ["common.pt", ".metadata", "metadata.json"]
    candidates.extend([p.name for p in output_dir.glob("__*.distcp")])

    for name in candidates:
        src = output_dir / name
        if not src.exists():
            continue
        dst = iter_dir / name
        if dst.exists():
            continue
        dst.symlink_to(Path("..") / name)


def load_hf_index(model_dir: Path) -> Dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())["weight_map"]
    # Fallback to single-file
    single = model_dir / "model.safetensors"
    require(single.exists(), "Missing model.safetensors or model.safetensors.index.json")
    return {}


def load_hf_tensor(model_dir: Path, weight_map: Dict[str, str], key: str):
    from safetensors import safe_open

    if weight_map:
        shard = model_dir / weight_map[key]
    else:
        shard = model_dir / "model.safetensors"
    with safe_open(shard.as_posix(), framework="pt") as f:
        return f.get_tensor(key)


def meanpool_heads(weight, num_src: int, num_tgt: int, head_dim: int, *, dim: int):
    # dim=0 for [out, in] where out = num_heads*head_dim
    # dim=1 for [out, in] where in = num_heads*head_dim (output proj)
    if num_src % num_tgt != 0:
        raise ValueError(f"Cannot mean-pool heads: {num_src} -> {num_tgt}")
    if dim == 0:
        hidden = weight.shape[1]
        w = weight.view(num_src, head_dim, hidden)
        group = num_src // num_tgt
        w = w.view(num_tgt, group, head_dim, hidden).mean(dim=1)
        return w.reshape(num_tgt * head_dim, hidden)
    if dim == 1:
        hidden = weight.shape[0]
        w = weight.view(hidden, num_src, head_dim)
        group = num_src // num_tgt
        w = w.view(hidden, num_tgt, group, head_dim).mean(dim=2)
        return w.reshape(hidden, num_tgt * head_dim)
    raise ValueError("dim must be 0 or 1")


def topk_by_down_norm(down_w, k: int):
    import torch

    scores = down_w.float().pow(2).sum(dim=0).sqrt()
    # Stable-ish tie break by index.
    idx = torch.arange(scores.shape[0], dtype=scores.dtype, device=scores.device)
    scores = scores + idx * 1e-8
    return torch.topk(scores, k, largest=True).indices


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    require(cfg_path.exists(), f"Missing config: {cfg_path}")
    cfg = load_toml(cfg_path)

    source_dir = Path(cfg["inputs"]["source_dir"])
    output_dir = Path(cfg["outputs"]["output_dir"])
    require(source_dir.exists(), f"Missing source dir: {source_dir}")

    student = cfg["student"]
    spec = StudentSpec(
        num_layers=int(student["num_layers"]),
        hidden_size=int(student["hidden_size"]),
        num_attention_heads=int(student["num_attention_heads"]),
        num_key_value_heads=int(student["num_key_value_heads"]),
        head_dim=int(student["head_dim"]),
        vocab_size=int(student["vocab_size_padded"]),
        ffn_hidden_size=int(student["intermediate_size"]),
    )

    # Validate HF config.
    hf_cfg = json.loads((source_dir / "config.json").read_text())
    require(hf_cfg.get("num_hidden_layers") == spec.num_layers, "Unexpected num_hidden_layers")
    require(hf_cfg.get("hidden_size") == spec.hidden_size, "Unexpected hidden_size")
    require(hf_cfg.get("num_attention_heads") == 16, "Unexpected HF num_attention_heads")
    require(hf_cfg.get("num_key_value_heads") == 8, "Unexpected HF num_key_value_heads")
    require(hf_cfg.get("intermediate_size") == 3072, "Unexpected HF intermediate_size")
    require(hf_cfg.get("vocab_size") == spec.vocab_size, "Unexpected HF vocab_size")
    require(hf_cfg.get("tie_word_embeddings") is True, "Expected tied embeddings in HF")
    rope_theta = float(hf_cfg.get("rope_theta", 10000))
    rms_eps = float(hf_cfg.get("rms_norm_eps", 1e-5))

    ensure_empty_dir(output_dir, args.overwrite)

    # Distributed init (single-rank).
    import torch
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    seed = int(cfg.get("determinism", {}).get("seed", 1234))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_parallel_cuda_manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0) if device.type == "cuda" else None

    # Build Megatron model (compat).
    from megatron.core.activations import squared_relu
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=spec.num_layers,
        hidden_size=spec.hidden_size,
        num_attention_heads=spec.num_attention_heads,
        num_query_groups=spec.num_key_value_heads,
        kv_channels=spec.head_dim,
        ffn_hidden_size=spec.ffn_hidden_size,
        normalization="RMSNorm",
        layernorm_epsilon=rms_eps,
        qk_layernorm=True,
        softmax_type="learnable",
        add_bias_linear=True,
        add_qkv_bias=False,
        gated_linear_unit=False,
        activation_func=squared_relu,
        value_residual=True,
        value_residual_init=1.0,
        use_tpa=False,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        perform_initialization=True,
    )

    layer_spec = get_gpt_layer_local_spec(
        qk_layernorm=True,
        normalization="RMSNorm",
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=spec.vocab_size,
        max_sequence_length=4096,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=rope_theta,
    ).to(device)
    model.eval()

    weight_map = load_hf_index(source_dir)

    # Embeddings and output head (untied in target).
    embed_w = load_hf_tensor(source_dir, weight_map, "model.embed_tokens.weight")
    model.embedding.word_embeddings.weight.data.copy_(embed_w.to(device=device, dtype=model.embedding.word_embeddings.weight.dtype))
    model.output_layer.weight.data.copy_(embed_w.to(device=device, dtype=model.output_layer.weight.dtype))

    # Final norm.
    final_norm = load_hf_tensor(source_dir, weight_map, "model.norm.weight")
    model.decoder.final_layernorm.weight.data.copy_(final_norm.to(device=device, dtype=model.decoder.final_layernorm.weight.dtype))

    # Per-layer mapping.
    selection_indices: Dict[str, List[int]] = {}
    for i in range(spec.num_layers):
        prefix = f"model.layers.{i}"
        layer = model.decoder.layers[i]

        # Layer norms.
        in_ln = load_hf_tensor(source_dir, weight_map, f"{prefix}.input_layernorm.weight")
        layer.input_layernorm.weight.data.copy_(in_ln.to(device=device, dtype=layer.input_layernorm.weight.dtype))

        post_ln = load_hf_tensor(source_dir, weight_map, f"{prefix}.post_attention_layernorm.weight")
        layer.pre_mlp_layernorm.weight.data.copy_(post_ln.to(device=device, dtype=layer.pre_mlp_layernorm.weight.dtype))

        # Attention projections.
        wq = load_hf_tensor(source_dir, weight_map, f"{prefix}.self_attn.q_proj.weight")
        wk = load_hf_tensor(source_dir, weight_map, f"{prefix}.self_attn.k_proj.weight")
        wv = load_hf_tensor(source_dir, weight_map, f"{prefix}.self_attn.v_proj.weight")
        wo = load_hf_tensor(source_dir, weight_map, f"{prefix}.self_attn.o_proj.weight")
        q_norm = load_hf_tensor(source_dir, weight_map, f"{prefix}.self_attn.q_norm.weight")
        k_norm = load_hf_tensor(source_dir, weight_map, f"{prefix}.self_attn.k_norm.weight")

        wq_s = meanpool_heads(wq, 16, 8, spec.head_dim, dim=0)
        wk_s = meanpool_heads(wk, 8, 2, spec.head_dim, dim=0)
        wv_s = meanpool_heads(wv, 8, 2, spec.head_dim, dim=0)
        qkv = torch.cat([wq_s, wk_s, wv_s], dim=0)

        layer.self_attention.linear_qkv.weight.data.copy_(
            qkv.to(device=device, dtype=layer.self_attention.linear_qkv.weight.dtype)
        )
        if getattr(layer.self_attention.linear_qkv, "bias", None) is not None:
            layer.self_attention.linear_qkv.bias.data.zero_()

        wo_s = meanpool_heads(wo, 16, 8, spec.head_dim, dim=1)
        layer.self_attention.linear_proj.weight.data.copy_(
            wo_s.to(device=device, dtype=layer.self_attention.linear_proj.weight.dtype)
        )
        if getattr(layer.self_attention.linear_proj, "bias", None) is not None:
            layer.self_attention.linear_proj.bias.data.zero_()

        # QK layernorm weights.
        if getattr(layer.self_attention, "q_layernorm", None) is not None:
            require(
                layer.self_attention.q_layernorm.weight.shape == q_norm.shape,
                f"q_norm shape mismatch at layer {i}",
            )
            layer.self_attention.q_layernorm.weight.data.copy_(
                q_norm.to(
                    device=device,
                    dtype=layer.self_attention.q_layernorm.weight.dtype,
                )
            )
            if getattr(layer.self_attention.q_layernorm, "bias", None) is not None:
                layer.self_attention.q_layernorm.bias.data.zero_()
        if getattr(layer.self_attention, "k_layernorm", None) is not None:
            require(
                layer.self_attention.k_layernorm.weight.shape == k_norm.shape,
                f"k_norm shape mismatch at layer {i}",
            )
            layer.self_attention.k_layernorm.weight.data.copy_(
                k_norm.to(
                    device=device,
                    dtype=layer.self_attention.k_layernorm.weight.dtype,
                )
            )
            if getattr(layer.self_attention.k_layernorm, "bias", None) is not None:
                layer.self_attention.k_layernorm.bias.data.zero_()

        # QK norm and softmax offset are target-only: keep defaults but zero softmax offset.
        core_attn = layer.self_attention.core_attention
        if hasattr(core_attn, "softmax_offset") and core_attn.softmax_offset is not None:
            core_attn.softmax_offset.data.zero_()

        # MLP surgery (A).
        up = load_hf_tensor(source_dir, weight_map, f"{prefix}.mlp.up_proj.weight")
        down = load_hf_tensor(source_dir, weight_map, f"{prefix}.mlp.down_proj.weight")
        sel = topk_by_down_norm(down, spec.ffn_hidden_size)
        selection_indices[str(i)] = sel.cpu().tolist()

        fc1 = up.index_select(0, sel)
        fc2 = down.index_select(1, sel)

        layer.mlp.linear_fc1.weight.data.copy_(
            fc1.to(device=device, dtype=layer.mlp.linear_fc1.weight.dtype)
        )
        layer.mlp.linear_fc2.weight.data.copy_(
            fc2.to(device=device, dtype=layer.mlp.linear_fc2.weight.dtype)
        )
        if getattr(layer.mlp.linear_fc1, "bias", None) is not None:
            layer.mlp.linear_fc1.bias.data.zero_()
        if getattr(layer.mlp.linear_fc2, "bias", None) is not None:
            layer.mlp.linear_fc2.bias.data.zero_()

    # Save checkpoint (distributed format).
    from megatron.core import dist_checkpointing

    metadata = {
        "dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True),
        "non_homogeneous_layers": True,
    }
    sharded_sd = model.sharded_state_dict(metadata=metadata)
    dist_checkpointing.save(
        sharded_state_dict=sharded_sd,
        checkpoint_dir=str(output_dir),
        content_metadata={"vpmoe_format": "vpDense0-5_28"},
    )

    # Write provenance.
    sel_hashes = {}
    for layer, idxs in selection_indices.items():
        blob = (",".join(str(x) for x in idxs)).encode("utf-8")
        sel_hashes[layer] = hashlib.sha256(blob).hexdigest()
    provenance = {
        "git_sha": try_git_sha(REPO_ROOT),
        "source_dir": str(source_dir),
        "source_config": "config.json",
        "source_config_sha256": sha256_file(source_dir / "config.json"),
        "source_rope_theta": rope_theta,
        "source_rms_norm_eps": rms_eps,
        "source_index_sha256": sha256_file(source_dir / "model.safetensors.index.json")
        if (source_dir / "model.safetensors.index.json").exists()
        else None,
        "student_spec": spec.__dict__,
        "seed": seed,
        "dist_checkpoint": {"non_homogeneous_layers": True},
        "ffn_selection_method": "A_weight_norm",
        "ffn_selection_indices": selection_indices,
        "ffn_selection_hashes": sel_hashes,
        "output_dir": str(output_dir),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    write_training_layout(output_dir)
    print(f"Wrote: {output_dir}")


if __name__ == "__main__":
    main()
