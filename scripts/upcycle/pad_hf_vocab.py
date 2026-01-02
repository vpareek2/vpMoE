#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class VocabPadPlan:
    model_dir: Path
    base_weights_file: Path
    base_is_sharded: bool
    pad_weights_file: Path
    shard0_file: Path
    shard1_file: Path
    index_file: Path
    config_file: Path
    target_vocab_size: int
    embed_key: str
    lm_head_key: str


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def _build_plan(model_dir: Path, target_vocab_size: int) -> VocabPadPlan:
    base_weights_file = model_dir / "model.safetensors"
    shard0_file = model_dir / "model-00000-of-00002.safetensors"
    base_is_sharded = False
    if base_weights_file.exists():
        base_is_sharded = False
    elif shard0_file.exists():
        base_weights_file = shard0_file
        base_is_sharded = True
    else:
        raise FileNotFoundError(
            f"Missing {base_weights_file.name} and {shard0_file.name}"
        )

    return VocabPadPlan(
        model_dir=model_dir,
        base_weights_file=base_weights_file,
        base_is_sharded=base_is_sharded,
        pad_weights_file=model_dir / "model.vocabpad.safetensors",
        shard0_file=shard0_file,
        shard1_file=model_dir / "model-00001-of-00002.safetensors",
        index_file=model_dir / "model.safetensors.index.json",
        config_file=model_dir / "config.json",
        target_vocab_size=target_vocab_size,
        embed_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
    )


def pad_hf_vocab(model_dir: str, target_vocab_size: int) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    plan = _build_plan(Path(model_dir), target_vocab_size)
    if not plan.config_file.exists():
        raise FileNotFoundError(f"Missing {plan.config_file}")

    cfg = _load_json(plan.config_file)
    cfg_vocab_size = int(cfg.get("vocab_size", 0) or 0)
    if cfg_vocab_size <= 0:
        raise ValueError(f"Unexpected config.vocab_size={cfg.get('vocab_size')}")

    index = None
    if plan.index_file.exists():
        try:
            index = _load_json(plan.index_file)
        except Exception:
            index = None

    def _resolve_tensor_path(key: str) -> Path:
        if index:
            mapped = index.get("weight_map", {}).get(key)
            if mapped:
                candidate = plan.model_dir / mapped
                if candidate.exists():
                    return candidate
        if plan.shard1_file.exists():
            return plan.shard1_file
        return plan.base_weights_file

    def _load_tensor(key: str) -> torch.Tensor:
        for path in (_resolve_tensor_path(key), plan.base_weights_file):
            if not path.exists():
                continue
            with safe_open(path.as_posix(), framework="pt") as st:
                if key in st.keys():
                    return st.get_tensor(key)
        raise KeyError(f"Missing tensor {key} in {plan.model_dir}")

    def _load_tensor_from_file(path: Path, key: str) -> torch.Tensor:
        with safe_open(path.as_posix(), framework="pt") as st:
            if key not in st.keys():
                raise KeyError(f"Missing tensor {key} in {path.name}")
            return st.get_tensor(key)

    with safe_open(plan.base_weights_file.as_posix(), framework="pt") as st:
        keys = list(st.keys())

    embed = _load_tensor(plan.embed_key)
    lm_head = _load_tensor(plan.lm_head_key)

    if embed.ndim != 2 or lm_head.ndim != 2:
        raise ValueError("Expected 2D embedding weights")
    if embed.shape[0] != lm_head.shape[0]:
        raise ValueError(
            f"embed/lm_head vocab mismatch: {embed.shape[0]} vs {lm_head.shape[0]}"
        )
    if embed.shape[1] != lm_head.shape[1]:
        raise ValueError(
            f"embed/lm_head hidden mismatch: {embed.shape[1]} vs {lm_head.shape[1]}"
        )

    current_vocab_size = int(embed.shape[0])
    if cfg_vocab_size != current_vocab_size:
        print(
            f"[pad_hf_vocab] config vocab_size={cfg_vocab_size} "
            f"but weights have {current_vocab_size} rows"
        )

    base_embed = _load_tensor_from_file(plan.base_weights_file, plan.embed_key)
    base_vocab_size = int(base_embed.shape[0])

    have_pad_artifact = plan.pad_weights_file.exists() or plan.shard1_file.exists()
    needs_sharded_layout = (base_vocab_size != plan.target_vocab_size) or have_pad_artifact

    if current_vocab_size == plan.target_vocab_size and not needs_sharded_layout:
        print(f"[pad_hf_vocab] vocab already padded: {current_vocab_size}")
        if cfg_vocab_size != plan.target_vocab_size:
            cfg["vocab_size"] = plan.target_vocab_size
            _write_json(plan.config_file, cfg)
        return

    if current_vocab_size > plan.target_vocab_size:
        raise ValueError(
            f"Refusing to shrink vocab: old={current_vocab_size} target={plan.target_vocab_size}"
        )

    pad_rows = plan.target_vocab_size - current_vocab_size
    if pad_rows > 0:
        print(
            f"[pad_hf_vocab] padding vocab {current_vocab_size} -> "
            f"{plan.target_vocab_size} (+{pad_rows})"
        )

        def _pad_matrix(mat: torch.Tensor) -> torch.Tensor:
            mean = mat.to(torch.float32).mean(dim=0, keepdim=True).to(mat.dtype)
            pad = mean.expand(pad_rows, -1).contiguous()
            return torch.cat([mat, pad], dim=0)

        embed_padded = _pad_matrix(embed)
        lm_head_padded = _pad_matrix(lm_head)
    else:
        embed_padded = embed
        lm_head_padded = lm_head

    if needs_sharded_layout:
        tmp_pad = plan.model_dir / "model.vocabpad.safetensors.tmp"
        save_file(
            {
                plan.embed_key: embed_padded.contiguous(),
                plan.lm_head_key: lm_head_padded.contiguous(),
            },
            tmp_pad.as_posix(),
        )

    # Transformers expects either a single `model.safetensors` file *or* a sharded
    # layout with `model.safetensors.index.json` and shard files. It will ignore
    # the index if `model.safetensors` exists.
    #
    # So, after padding, we convert to a 2-shard layout:
    # - shard0: original weights (renamed from model.safetensors)
    # - shard1: padded vocab tensors (model.embed_tokens.weight + lm_head.weight)
    if needs_sharded_layout:
        def _write_filtered_shard0(src: Path, dst: Path) -> None:
            with safe_open(src.as_posix(), framework="pt") as st:
                tensors = {
                    k: st.get_tensor(k)
                    for k in st.keys()
                    if k not in (plan.embed_key, plan.lm_head_key)
                }
            save_file(tensors, dst.as_posix())

        tmp_shard0 = plan.model_dir / "model-00000-of-00002.safetensors.tmp"
        _write_filtered_shard0(plan.base_weights_file, tmp_shard0)
        tmp_shard0.replace(plan.shard0_file)

        if not plan.base_is_sharded and plan.base_weights_file.exists():
            plan.base_weights_file.unlink()

        tmp_pad.replace(plan.shard1_file)

    # Create a sharded index mapping embed + lm_head to shard1, everything else to shard0.
    weight_map: Dict[str, str] = {}
    for k in keys:
        weight_map[k] = plan.shard0_file.name
    weight_map[plan.embed_key] = plan.shard1_file.name
    weight_map[plan.lm_head_key] = plan.shard1_file.name

    if needs_sharded_layout:
        total_size = plan.shard0_file.stat().st_size + plan.shard1_file.stat().st_size
        _write_json(
            plan.index_file,
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
        )

        cfg["vocab_size"] = plan.target_vocab_size
        _write_json(plan.config_file, cfg)
        print(
            f"[pad_hf_vocab] wrote {plan.index_file.name} with shards "
            f"{plan.shard0_file.name}, {plan.shard1_file.name}"
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--target-vocab-size", type=int, required=True)
    args = ap.parse_args()

    pad_hf_vocab(args.model_dir, args.target_vocab_size)
