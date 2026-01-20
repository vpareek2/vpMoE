#!/usr/bin/env python3
"""
Export GPT-OSS MXFP4 checkpoints to BF16 and validate the result.

This uses Transformers' Mxfp4Config(dequantize=True) loading path, then saves a
pure-BF16 safetensors checkpoint (no *_blocks/_scales).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from vpmoe.mxfp4_export import export_mxfp4_to_bf16, validate_bf16_checkpoint  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GPT-OSS MXFP4 -> BF16 checkpoint.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(REPO_ROOT / "src" / "third_party" / "gpt-oss-20b"),
        help="Model id or local path to the GPT-OSS MXFP4 checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for BF16 checkpoint.",
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
        "--validate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only validate an existing BF16 checkpoint at --output-dir.",
    )
    parser.add_argument(
        "--skip-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip validation after export.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.validate_only:
        issues = validate_bf16_checkpoint(args.output_dir)
        if issues:
            for issue in issues:
                location = f" [{issue.key}]" if issue.key else ""
                print(f"[{issue.kind}]{location} {issue.message}", file=sys.stderr)
            return 1
        print("BF16 checkpoint validation: OK")
        return 0

    export_mxfp4_to_bf16(
        args.model,
        args.output_dir,
        revision=args.revision,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        max_shard_size=args.max_shard_size,
        save_tokenizer=not args.skip_tokenizer,
        overwrite=args.overwrite,
    )

    if not args.skip_validation:
        issues = validate_bf16_checkpoint(args.output_dir)
        if issues:
            for issue in issues:
                location = f" [{issue.key}]" if issue.key else ""
                print(f"[{issue.kind}]{location} {issue.message}", file=sys.stderr)
            return 2
        print("BF16 checkpoint validation: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
