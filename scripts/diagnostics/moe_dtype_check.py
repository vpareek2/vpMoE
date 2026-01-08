#!/usr/bin/env python3
"""Diagnose dtype promotion in MoE per-token scaling with bias."""

from __future__ import annotations

import argparse

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check dtype promotion when per-token scaling is fp32."
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (default: cuda if available)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    output = torch.randn(
        args.seq_len, args.batch, args.hidden_size, device=device, dtype=torch.bfloat16
    )
    output_bias = torch.randn(args.hidden_size, device=device, dtype=torch.bfloat16)

    # Simulate router_dtype=fp32 -> per-token scale in fp32.
    per_token_scale_fp32 = torch.rand(
        args.seq_len, args.batch, device=device, dtype=torch.float32
    )
    per_token_scale_bf16 = per_token_scale_fp32.to(torch.bfloat16)

    scaled_fp32 = output_bias.unsqueeze(0) * per_token_scale_fp32.unsqueeze(-1)
    result_fp32 = output + scaled_fp32

    scaled_bf16 = output_bias.unsqueeze(0) * per_token_scale_bf16.unsqueeze(-1)
    result_bf16 = output + scaled_bf16

    print("device:", device)
    print("output dtype:", output.dtype)
    print("output_bias dtype:", output_bias.dtype)
    print("per_token_scale fp32 dtype:", per_token_scale_fp32.dtype)
    print("scaled fp32 dtype:", scaled_fp32.dtype)
    print("result fp32 dtype:", result_fp32.dtype)
    print("per_token_scale bf16 dtype:", per_token_scale_bf16.dtype)
    print("scaled bf16 dtype:", scaled_bf16.dtype)
    print("result bf16 dtype:", result_bf16.dtype)

    # Show a safe cast-back pattern.
    result_cast = output + scaled_fp32.to(output.dtype)
    print("result with explicit cast dtype:", result_cast.dtype)


if __name__ == "__main__":
    main()
