# coding=utf-8
"""Utilities to export GPT-OSS MXFP4 checkpoints to BF16 and validate outputs."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from safetensors import safe_open


@dataclass(frozen=True)
class ValidationIssue:
    kind: str
    message: str
    key: Optional[str] = None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _allowed_fp32_key(key: str, allow_fp32_keys: Iterable[str]) -> bool:
    for suffix in allow_fp32_keys:
        if key.endswith(suffix):
            return True
    return False


def validate_bf16_checkpoint(
    checkpoint_dir: Path,
    *,
    allow_fp32_keys: Iterable[str] = ("rotary_emb.inv_freq",),
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    checkpoint_dir = Path(checkpoint_dir)

    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        issues.append(
            ValidationIssue(kind="missing_config", message="Missing config.json in checkpoint directory.")
        )
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    kind="invalid_config",
                    message=f"config.json is not valid JSON: {exc}",
                )
            )
        else:
            if "quantization_config" in config:
                issues.append(
                    ValidationIssue(
                        kind="quantization_config",
                        message="config.json still contains quantization_config; expected BF16-only export.",
                    )
                )
            torch_dtype = config.get("torch_dtype")
            if torch_dtype and torch_dtype not in {"bfloat16", "bf16"}:
                issues.append(
                    ValidationIssue(
                        kind="torch_dtype",
                        message=f"config.json torch_dtype={torch_dtype!r} is not bfloat16/bf16.",
                    )
                )

    safetensors_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not safetensors_files:
        issues.append(
            ValidationIssue(
                kind="missing_weights",
                message="No .safetensors files found in checkpoint directory.",
            )
        )
        return issues

    for weights_path in safetensors_files:
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith("_blocks") or key.endswith("_scales"):
                    issues.append(
                        ValidationIssue(
                            kind="quantized_key",
                            message="Found MXFP4 blocks/scales tensor in BF16 export.",
                            key=key,
                        )
                    )
                dtype = f.get_slice(key).get_dtype()
                if dtype != "BF16":
                    if dtype == "F32" and _allowed_fp32_key(key, allow_fp32_keys):
                        continue
                    issues.append(
                        ValidationIssue(
                            kind="dtype",
                            message=f"Tensor dtype {dtype} is not BF16.",
                            key=key,
                        )
                    )
    return issues


def export_mxfp4_to_bf16(
    model_id: str,
    output_dir: Path,
    *,
    revision: Optional[str] = None,
    device_map: Optional[str] = "auto",
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    local_files_only: bool = False,
    max_shard_size: str = "5GB",
    save_tokenizer: bool = True,
    overwrite: bool = False,
    manifest_name: str = "bf16_export_manifest.json",
) -> dict:
    import shutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, __version__ as tf_version

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory {output_dir} is not empty. Use --overwrite to replace it."
            )
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_cfg = Mxfp4Config(dequantize=True)
    resolved_dtype = torch_dtype
    if isinstance(torch_dtype, str):
        key = torch_dtype.lower()
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "auto": "auto",
        }
        if key not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype!r}")
        resolved_dtype = dtype_map[key]
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quant_cfg,
        torch_dtype=resolved_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )

    if hasattr(model.config, "quantization_config"):
        delattr(model.config, "quantization_config")
    if hasattr(model, "hf_quantizer"):
        delattr(model, "hf_quantizer")
    if hasattr(model, "quantization_method"):
        delattr(model, "quantization_method")
    model.is_quantized = False

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        tokenizer.save_pretrained(output_dir)

    manifest = {
        "source": {"model_id": model_id, "revision": revision},
        "exported_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "transformers_version": tf_version,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "artifacts": {},
    }

    config_path = output_dir / "config.json"
    if config_path.exists():
        manifest["artifacts"]["config.json"] = {"sha256": _sha256_file(config_path)}

    index_path = output_dir / "model.safetensors.index.json"
    if index_path.exists():
        manifest["artifacts"]["model.safetensors.index.json"] = {
            "sha256": _sha256_file(index_path)
        }

    tokenizer_path = output_dir / "tokenizer.json"
    if tokenizer_path.exists():
        manifest["artifacts"]["tokenizer.json"] = {"sha256": _sha256_file(tokenizer_path)}

    manifest_path = output_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return manifest


__all__ = ["ValidationIssue", "export_mxfp4_to_bf16", "validate_bf16_checkpoint"]
