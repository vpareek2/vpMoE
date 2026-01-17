#!/usr/bin/env python3
"""
Lightweight EDA for JSONL datasets (streaming).

Designed for large files: reads a bounded sample without loading the dataset
into memory. Supports optional token stats via tiktoken (if installed).

Example:
  python3 scripts/eda_jsonl.py \
    --input /datasets/nvidia__Nemotron-Competitive-Programming-v1/data/infinibyte.part_00.jsonl \
    --input /datasets/nvidia__Nemotron-Competitive-Programming-v1/data/infinibyte.part_01.jsonl \
    --sample-rows 20000 \
    --tiktoken-encoding o200k_base \
    --show-samples 2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclasses.dataclass
class RunningStats:
    count: int = 0
    total: int = 0
    min_val: Optional[int] = None
    max_val: int = 0

    def update(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.max_val = max(self.max_val, value)
        if self.min_val is None:
            self.min_val = value
        else:
            self.min_val = min(self.min_val, value)

    def mean(self) -> float:
        return (self.total / self.count) if self.count else 0.0


def _percentile(sorted_vals: Sequence[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    idx = int((pct / 100.0) * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield lineno, line


def _load_tiktoken(encoding_name: str):
    try:
        import tiktoken  # type: ignore
    except Exception as exc:
        raise SystemExit(f"tiktoken not available but --tiktoken-encoding was set: {exc}") from exc
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        raise SystemExit(f"Failed to load tiktoken encoding '{encoding_name}': {exc}") from exc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Streaming EDA for JSONL datasets.")
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Path to a JSONL file. Repeatable.",
    )
    p.add_argument(
        "--sample-rows",
        type=int,
        default=20000,
        help="Total number of rows to sample across all inputs (approx).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling within each file.",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Max chars to print per field in samples.",
    )
    p.add_argument(
        "--show-samples",
        type=int,
        default=0,
        help="Print N sampled rows (truncated).",
    )
    p.add_argument(
        "--tiktoken-encoding",
        type=str,
        default=None,
        help="If set, compute token length stats with tiktoken (e.g. o200k_base).",
    )
    return p.parse_args()


def _reservoir_sample(
    rng: random.Random, items: Iterable[Any], k: int
) -> List[Any]:
    """
    Reservoir sampling: uniformly sample k items from a stream of unknown length.
    """
    sample: List[Any] = []
    for i, item in enumerate(items, start=1):
        if i <= k:
            sample.append(item)
            continue
        j = rng.randint(1, i)
        if j <= k:
            sample[j - 1] = item
    return sample


def _extract_messages_fields(obj: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extract (user_content, assistant_reasoning, assistant_content) from common
    message-based JSONL rows.

    Expected shape:
      {"messages":[{"role":"user","content":...}, ..., {"role":"assistant","content":...,"reasoning_content":...}], ...}

    Notes:
    - `reasoning_content` is optional; when missing, we return "".
    - We pick the first user message and the last assistant message.
    """
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("missing/invalid messages")

    user_text: Optional[str] = None
    assistant_msg: Optional[Dict[str, Any]] = None

    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user" and user_text is None:
            c = m.get("content")
            if isinstance(c, str):
                user_text = c
        if role == "assistant":
            assistant_msg = m

    if user_text is None or assistant_msg is None:
        raise ValueError("missing user/assistant messages")

    assistant_content = assistant_msg.get("content")
    reasoning_content = assistant_msg.get("reasoning_content", "")
    if not isinstance(assistant_content, str):
        raise ValueError("assistant content not a string")
    if not isinstance(reasoning_content, str):
        reasoning_content = ""

    return user_text, reasoning_content, assistant_content


def main() -> int:
    args = _parse_args()

    paths = [Path(p).expanduser().resolve() for p in args.input]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Input file(s) not found: {missing}")

    enc = _load_tiktoken(args.tiktoken_encoding) if args.tiktoken_encoding else None

    rng = random.Random(args.seed)
    per_file = max(1, math.ceil(args.sample_rows / max(1, len(paths))))

    total_rows_seen = 0
    total_rows_parsed = 0
    json_errors = 0
    schema_keys: Counter[str] = Counter()

    # Stats for InfiniByte-like message structure.
    user_char: List[int] = []
    assistant_char: List[int] = []
    reasoning_char: List[int] = []
    user_tok: List[int] = []
    assistant_tok: List[int] = []
    reasoning_tok: List[int] = []

    # Keep some sample rows for printing.
    raw_samples: List[Dict[str, Any]] = []

    for path in paths:
        # Reservoir sample *lines* to avoid scanning the whole file to pick random rows.
        # But JSONL lines are independent; we can sample raw lines then parse.
        line_rng = random.Random(rng.randint(0, 2**31 - 1))
        sampled_lines = _reservoir_sample(line_rng, (line for _, line in _iter_jsonl(path)), per_file)

        for raw in sampled_lines:
            total_rows_seen += 1
            try:
                obj = json.loads(raw)
            except Exception:
                json_errors += 1
                continue
            if not isinstance(obj, dict):
                continue
            total_rows_parsed += 1
            for k in obj.keys():
                schema_keys[k] += 1

            try:
                u, r, a = _extract_messages_fields(obj)
            except Exception:
                # This script is meant for quick inspection; we only track InfiniByte fields for now.
                continue

            user_char.append(len(u))
            assistant_char.append(len(a))
            reasoning_char.append(len(r))
            if enc is not None:
                user_tok.append(len(enc.encode(u)))
                assistant_tok.append(len(enc.encode(a)))
                reasoning_tok.append(len(enc.encode(r)))

            if args.show_samples and len(raw_samples) < args.show_samples:
                raw_samples.append(
                    {
                        "uuid": obj.get("uuid"),
                        "license": obj.get("license"),
                        "user": u,
                        "assistant_reasoning": r,
                        "assistant": a,
                    }
                )

    def _print_len_stats(name: str, vals: List[int]) -> None:
        if not vals:
            print(f"{name}: no samples")
            return
        vals_sorted = sorted(vals)
        print(
            f"{name}: p50={_percentile(vals_sorted, 50)} "
            f"p90={_percentile(vals_sorted, 90)} "
            f"p95={_percentile(vals_sorted, 95)} "
            f"p99={_percentile(vals_sorted, 99)} "
            f"max={vals_sorted[-1]}"
        )

    print(f"inputs: {len(paths)}")
    for p in paths:
        print(f"- {p}")
    print()
    print(f"sample_target_rows: {args.sample_rows} (per_file={per_file})")
    print(f"sampled_lines: {total_rows_seen} parsed_json: {total_rows_parsed} json_errors: {json_errors}")
    print()

    if schema_keys:
        print("top-level keys (sample):")
        for k, v in schema_keys.most_common(20):
            print(f"  {k}: {v}")
        print()

    print("char length stats (sampled):")
    _print_len_stats("user", user_char)
    _print_len_stats("assistant_reasoning", reasoning_char)
    _print_len_stats("assistant", assistant_char)
    print()

    if enc is not None:
        print(f"token length stats with tiktoken({args.tiktoken_encoding}) (sampled):")
        _print_len_stats("user", user_tok)
        _print_len_stats("assistant_reasoning", reasoning_tok)
        _print_len_stats("assistant", assistant_tok)
        print()

    if raw_samples:
        print("=" * 120)
        for i, s in enumerate(raw_samples, start=1):
            print(f"sample {i}")
            print(f"uuid: {s.get('uuid')}  license: {s.get('license')}")
            print("user:")
            print(_truncate(str(s.get("user", "")), args.max_chars))
            print("\nassistant_reasoning:")
            print(_truncate(str(s.get("assistant_reasoning", "")), args.max_chars))
            print("\nassistant:")
            print(_truncate(str(s.get("assistant", "")), args.max_chars))
            print("=" * 120)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
