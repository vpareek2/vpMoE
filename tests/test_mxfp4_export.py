# coding=utf-8
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from vpmoe.mxfp4_export import validate_bf16_checkpoint  # noqa: E402


def _write_config(tmp_path: Path, data: dict) -> None:
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")


def _write_safetensors(tmp_path: Path, tensors: dict, name: str = "model.safetensors") -> None:
    save_file(tensors, str(tmp_path / name))


def test_validate_bf16_checkpoint_ok(tmp_path: Path) -> None:
    _write_config(tmp_path, {"torch_dtype": "bfloat16"})
    _write_safetensors(
        tmp_path,
        {
            "model.layers.0.mlp.experts.gate_up_proj": torch.zeros((2, 2), dtype=torch.bfloat16),
            "model.rotary_emb.inv_freq": torch.zeros((2,), dtype=torch.float32),
        },
    )
    issues = validate_bf16_checkpoint(tmp_path)
    assert issues == []


def test_validate_bf16_checkpoint_flags_quantization_config(tmp_path: Path) -> None:
    _write_config(tmp_path, {"quantization_config": {"quant_method": "mxfp4"}})
    _write_safetensors(
        tmp_path,
        {
            "model.layers.0.mlp.experts.gate_up_proj": torch.zeros((2, 2), dtype=torch.bfloat16),
        },
    )
    issues = validate_bf16_checkpoint(tmp_path)
    kinds = {issue.kind for issue in issues}
    assert "quantization_config" in kinds


def test_validate_bf16_checkpoint_flags_blocks_and_dtype(tmp_path: Path) -> None:
    _write_config(tmp_path, {"torch_dtype": "bfloat16"})
    _write_safetensors(
        tmp_path,
        {
            "model.layers.0.mlp.experts.gate_up_proj_blocks": torch.zeros((1,), dtype=torch.uint8),
        },
    )
    issues = validate_bf16_checkpoint(tmp_path)
    kinds = {issue.kind for issue in issues}
    assert "quantized_key" in kinds
    assert "dtype" in kinds
