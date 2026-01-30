#!/usr/bin/env python3
"""
Create a tiny DistillKit-ready dataset on disk for local debugging / overfit tests.

Output matches the repo's preprocessing contract (see `src/data/synth_preprocess.md`):
  - input_ids: full sequence (system + user + assistant)
  - labels: -100 for non-assistant tokens, copied from input_ids for assistant tokens
  - attention_mask: 1s
  - spans: {assistant_token_start, analysis_token_count, final_token_count}

We use the tokenizer's chat template (Harmony) and request an assistant tokens mask
to compute the label masking precisely.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import datasets
import transformers


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_examples(n: int) -> list[dict[str, Any]]:
    # Keep these short and deterministic; we want a "must overfit" sanity check.
    base = [
        ("Compute 2 + 2.", "4"),
        ("What is the capital of France?", "Paris."),
        ("Reverse the string 'stressed'.", "desserts"),
        ("If x=3, what is 2x+1?", "7"),
        ("List the first 5 primes.", "2, 3, 5, 7, 11"),
        ("Write 'hello' in uppercase.", "HELLO"),
        ("What is 10 * 10?", "100"),
        ("Return only the word: cat", "cat"),
    ]
    out: list[dict[str, Any]] = []
    for i in range(n):
        q, a = base[i % len(base)]
        out.append(
            {
                "id": f"tiny:{i}",
                "source": {"dataset": "tiny_debug", "i": i},
                "system": "You are a helpful assistant.",
                "user": q,
                "assistant": a,
            }
        )
    return out


def _tokenize_row(tok: transformers.PreTrainedTokenizer, row: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["user"]},
        {"role": "assistant", "content": row["assistant"]},
    ]

    # Many chat templates (including Harmony variants) do not expose an explicit
    # "assistant tokens mask". To compute the supervised region robustly, we:
    #   1) tokenize the prompt (system+user) with add_generation_prompt=True
    #      (this includes the assistant prefix tokens used for generation)
    #   2) tokenize the full example (system+user+assistant completion)
    #   3) label-mask everything in (1), supervise everything after it
    prompt_messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["user"]},
    ]

    prompt_ids = tok.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        return_tensors=None,
    )
    full_ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        return_tensors=None,
    )

    # Ensure prompt_ids is a prefix of full_ids. If it isn't (template mismatch),
    # fall back to the longest common prefix to avoid producing misaligned labels.
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        common += 1
    assistant_token_start = common

    input_ids = full_ids
    labels = [-100] * len(input_ids)
    for i in range(assistant_token_start, len(input_ids)):
        labels[i] = input_ids[i]

    final_token_count = len(input_ids) - assistant_token_start

    return {
        "id": row["id"],
        "source": row["source"],
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "spans": {
            "assistant_token_start": int(assistant_token_start),
            "analysis_token_count": 0,
            "final_token_count": int(final_token_count),
        },
    }


def _pad_or_truncate_to_length(
    *,
    tok: transformers.PreTrainedTokenizer,
    ex: dict[str, Any],
    sequence_length: int,
) -> dict[str, Any]:
    """Force example to an exact token length without changing the supervised region.

    NOTE: We keep attention_mask=1 for padded tokens to avoid "unpadding" paths
    in flash-attn integrations; this is intended for realistic memory testing.
    Any tokens appended past the assistant completion are label-masked (-100),
    so they do not contribute to the loss.
    """

    if sequence_length <= 0:
        return ex

    input_ids = ex["input_ids"]
    labels = ex["labels"]
    attn = ex["attention_mask"]

    if len(input_ids) != len(labels) or len(input_ids) != len(attn):
        raise ValueError("example fields are misaligned")

    assistant_token_start = int(ex["spans"]["assistant_token_start"])

    # Truncate first if needed.
    if len(input_ids) > sequence_length:
        input_ids = input_ids[:sequence_length]
        labels = labels[:sequence_length]
        attn = attn[:sequence_length]
        if assistant_token_start >= sequence_length:
            raise ValueError(
                f"sequence_length too small: assistant_token_start={assistant_token_start} >= {sequence_length}"
            )

    # Pad with a "real" token ID (we use EOS) and keep attention_mask=1.
    # These tokens are after the assistant completion, so they must be label-masked.
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no eos_token_id; cannot pad")

    if len(input_ids) < sequence_length:
        pad_n = sequence_length - len(input_ids)
        input_ids = input_ids + [int(eos_id)] * pad_n
        attn = attn + [1] * pad_n
        labels = labels + [-100] * pad_n

    ex["input_ids"] = input_ids
    ex["labels"] = labels
    ex["attention_mask"] = attn

    # Update span lengths to reflect truncation only. Padding stays masked.
    final_token_count = 0
    for v in labels[assistant_token_start:]:
        if v == -100:
            break
        final_token_count += 1
    ex["spans"]["final_token_count"] = int(final_token_count)
    return ex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tokenizer", default="openai/gpt-oss-20b")
    ap.add_argument("--train-n", type=int, default=64)
    ap.add_argument("--val-n", type=int, default=16)
    ap.add_argument(
        "--sequence-length",
        type=int,
        default=0,
        help="If >0, force all examples to an exact token length by truncating/padding.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _ensure_dir(args.output_dir)

    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    train_rows = _make_examples(args.train_n)
    val_rows = _make_examples(args.val_n)

    train = [
        _pad_or_truncate_to_length(tok=tok, ex=_tokenize_row(tok, r), sequence_length=args.sequence_length)
        for r in train_rows
    ]
    val = [
        _pad_or_truncate_to_length(tok=tok, ex=_tokenize_row(tok, r), sequence_length=args.sequence_length)
        for r in val_rows
    ]

    d_train = datasets.Dataset.from_list(train)
    d_val = datasets.Dataset.from_list(val)

    d = datasets.DatasetDict({"train": d_train, "validation": d_val})
    d.save_to_disk(args.output_dir)

    print(f"wrote tiny dataset: {args.output_dir}")
    print(f"train={len(d_train)} validation={len(d_val)}")


if __name__ == "__main__":
    main()
