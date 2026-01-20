#!/usr/bin/env python3
"""
Initialize vpMoE student weights from a BF16 GPT-OSS checkpoint.

Copies:
- embeddings, final norm, lm_head
- per-layer RMSNorm weights
- MoE router + expert weights
- for TPA layers only: attention o_proj + sinks

Leaves:
- TPA factor projections (W_A_*, W_B_*) at Xavier init
- all KDA parameters at their model init
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as tf_version

from vpmoe.configuration_vpmoe import VpMoEConfig
from vpmoe.modeling_vpmoe import VpMoEForCausalLM
from vpmoe.mxfp4_export import validate_bf16_checkpoint


@contextmanager
def _default_dtype(dtype: torch.dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _sha256_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_torch_dtype(dtype: str) -> torch.dtype | str:
    key = dtype.lower()
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
        raise ValueError(f"Unsupported torch_dtype: {dtype!r}")
    return dtype_map[key]


def _git_sha(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _copy_param(dst: torch.Tensor, src: torch.Tensor, name: str) -> None:
    if dst.shape != src.shape:
        raise ValueError(f"Shape mismatch for {name}: {tuple(dst.shape)} vs {tuple(src.shape)}")
    dst.copy_(src.detach().to(device=dst.device, dtype=dst.dtype))


def _copy_linear(dst: torch.nn.Linear, src: torch.nn.Linear, name: str) -> None:
    _copy_param(dst.weight, src.weight, f"{name}.weight")
    if dst.bias is None and src.bias is None:
        return
    if dst.bias is None or src.bias is None:
        raise ValueError(f"Bias mismatch for {name}: dst={dst.bias is not None}, src={src.bias is not None}")
    _copy_param(dst.bias, src.bias, f"{name}.bias")


def _validate_teacher_checkpoint(path: Path) -> None:
    issues = validate_bf16_checkpoint(path)
    if issues:
        messages = []
        for issue in issues:
            location = f" [{issue.key}]" if issue.key else ""
            messages.append(f"[{issue.kind}]{location} {issue.message}")
        raise ValueError(
            "Teacher checkpoint failed BF16 validation:\n" + "\n".join(messages)
        )


def _ensure_config_match(teacher_config, student_config, keys: Iterable[str]) -> None:
    mismatches = []
    for key in keys:
        t_val = getattr(teacher_config, key, None)
        s_val = getattr(student_config, key, None)
        if t_val != s_val:
            mismatches.append((key, t_val, s_val))
    if mismatches:
        details = ", ".join(f"{k}: teacher={t} student={s}" for k, t, s in mismatches)
        raise ValueError(f"Teacher/student config mismatch: {details}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize vpMoE from BF16 GPT-OSS checkpoint.")
    parser.add_argument(
        "--teacher",
        type=str,
        default="/data/gpt-oss-20b-bf16",
        help="Path or model id for the BF16 GPT-OSS checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "weights" / "vpmoe-20b-init",
        help="Output directory for the initialized student checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "src" / "vpmoe" / "config.json",
        help="vpMoE config.json to instantiate the student.",
    )
    parser.add_argument(
        "--code-source",
        type=Path,
        default=REPO_ROOT / "src" / "vpmoe" / "ckpt_skeleton",
        help="Directory containing configuration_vpmoe.py/modeling_vpmoe.py/__init__.py to copy.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-shard-size", type=str, default="5GB")
    parser.add_argument("--skip-tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--skip-teacher-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip validating teacher checkpoint for BF16-only tensors.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    output_dir = args.output_dir
    if output_dir.exists():
        if not args.overwrite and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory {output_dir} is not empty. Use --overwrite to replace it."
            )
        if args.overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_path = Path(args.teacher)
    if teacher_path.exists() and teacher_path.is_dir() and not args.skip_teacher_validation:
        _validate_teacher_checkpoint(teacher_path)

    teacher_dtype = _resolve_torch_dtype(args.torch_dtype)

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        revision=args.revision,
        torch_dtype=teacher_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    teacher.eval()

    config = VpMoEConfig.from_pretrained(args.config)
    config.torch_dtype = "bfloat16"

    with _default_dtype(torch.bfloat16):
        student = VpMoEForCausalLM(config)
    student.eval()

    _ensure_config_match(
        teacher.config,
        student.config,
        keys=(
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "num_local_experts",
            "vocab_size",
            "attention_bias",
        ),
    )

    copied = []
    with torch.no_grad():
        _copy_param(
            student.model.embed_tokens.weight,
            teacher.model.embed_tokens.weight,
            "model.embed_tokens.weight",
        )
        copied.append("model.embed_tokens.weight")

        _copy_param(student.model.norm.weight, teacher.model.norm.weight, "model.norm.weight")
        copied.append("model.norm.weight")

        _copy_param(student.lm_head.weight, teacher.lm_head.weight, "lm_head.weight")
        copied.append("lm_head.weight")

        num_layers = student.config.num_hidden_layers
        for idx in range(num_layers):
            s_layer = student.model.layers[idx]
            t_layer = teacher.model.layers[idx]

            _copy_param(
                s_layer.input_layernorm.weight,
                t_layer.input_layernorm.weight,
                f"model.layers.{idx}.input_layernorm.weight",
            )
            copied.append(f"model.layers.{idx}.input_layernorm.weight")

            _copy_param(
                s_layer.post_attention_layernorm.weight,
                t_layer.post_attention_layernorm.weight,
                f"model.layers.{idx}.post_attention_layernorm.weight",
            )
            copied.append(f"model.layers.{idx}.post_attention_layernorm.weight")

            _copy_param(
                s_layer.mlp.router.weight,
                t_layer.mlp.router.weight,
                f"model.layers.{idx}.mlp.router.weight",
            )
            copied.append(f"model.layers.{idx}.mlp.router.weight")

            _copy_param(
                s_layer.mlp.router.bias,
                t_layer.mlp.router.bias,
                f"model.layers.{idx}.mlp.router.bias",
            )
            copied.append(f"model.layers.{idx}.mlp.router.bias")

            _copy_param(
                s_layer.mlp.experts.gate_up_proj,
                t_layer.mlp.experts.gate_up_proj,
                f"model.layers.{idx}.mlp.experts.gate_up_proj",
            )
            copied.append(f"model.layers.{idx}.mlp.experts.gate_up_proj")

            _copy_param(
                s_layer.mlp.experts.gate_up_proj_bias,
                t_layer.mlp.experts.gate_up_proj_bias,
                f"model.layers.{idx}.mlp.experts.gate_up_proj_bias",
            )
            copied.append(f"model.layers.{idx}.mlp.experts.gate_up_proj_bias")

            _copy_param(
                s_layer.mlp.experts.down_proj,
                t_layer.mlp.experts.down_proj,
                f"model.layers.{idx}.mlp.experts.down_proj",
            )
            copied.append(f"model.layers.{idx}.mlp.experts.down_proj")

            _copy_param(
                s_layer.mlp.experts.down_proj_bias,
                t_layer.mlp.experts.down_proj_bias,
                f"model.layers.{idx}.mlp.experts.down_proj_bias",
            )
            copied.append(f"model.layers.{idx}.mlp.experts.down_proj_bias")

            if s_layer.attn_impl == "tpa":
                _copy_linear(
                    s_layer.self_attn.o_proj,
                    t_layer.self_attn.o_proj,
                    f"model.layers.{idx}.self_attn.o_proj",
                )
                copied.append(f"model.layers.{idx}.self_attn.o_proj.weight")
                if s_layer.self_attn.o_proj.bias is not None:
                    copied.append(f"model.layers.{idx}.self_attn.o_proj.bias")
                _copy_param(
                    s_layer.self_attn.sinks,
                    t_layer.self_attn.sinks,
                    f"model.layers.{idx}.self_attn.sinks",
                )
                copied.append(f"model.layers.{idx}.self_attn.sinks")

    student.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    if not args.skip_tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.teacher,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
                fix_mistral_regex=True,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                args.teacher,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
            )
        tokenizer.save_pretrained(output_dir)

    code_source = args.code_source
    for filename in ("configuration_vpmoe.py", "modeling_vpmoe.py", "__init__.py"):
        src_path = code_source / filename
        if not src_path.exists():
            raise FileNotFoundError(f"Missing code file to copy: {src_path}")
        shutil.copy2(src_path, output_dir / filename)

    manifest = {
        "source": {"teacher": args.teacher, "revision": args.revision},
        "created_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "transformers_version": tf_version,
        "torch_version": torch.__version__,
        "git_sha": _git_sha(REPO_ROOT),
        "config_source": {
            "path": str(args.config),
            "sha256": _sha256_file(Path(args.config)),
        },
        "copied_keys": copied,
        "artifacts": {},
    }

    teacher_manifest_path = teacher_path / "bf16_export_manifest.json"
    if teacher_manifest_path.exists():
        try:
            manifest["source"]["bf16_export_manifest"] = json.loads(
                teacher_manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            manifest["source"]["bf16_export_manifest"] = "<invalid json>"

    for artifact_name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "configuration_vpmoe.py",
        "modeling_vpmoe.py",
        "__init__.py",
    ):
        artifact_path = output_dir / artifact_name
        if artifact_path.exists():
            manifest["artifacts"][artifact_name] = {"sha256": _sha256_file(artifact_path)}

    manifest_path = output_dir / "vpmoe_init_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Initialized vpMoE student checkpoint at {output_dir}")
    print(f"Copied {len(copied)} tensors from teacher")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
