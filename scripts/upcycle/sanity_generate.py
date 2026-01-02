#!/usr/bin/env python3
from __future__ import annotations

import argparse


def pick_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick generation sanity check for upcycled HF checkpoints.")
    ap.add_argument(
        "--model-dir",
        default="weights/upcycle/qwen3-0_6B-o200k",
        help="Path to the HF model directory.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    device = pick_device(torch)
    model = model.to(device).eval()

    prompts = [
        "Hello! In one sentence: what is the capital of France?",
        "Repeat exactly: 2025-08-05",
        "Write a short Python function that adds two integers.",
    ]

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        attention_mask = torch.ones_like(input_ids)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        print("PROMPT:", prompt)
        print("OUT:", tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True))
        print()


if __name__ == "__main__":
    main()
