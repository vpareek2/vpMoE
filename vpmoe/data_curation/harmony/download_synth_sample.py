#!/usr/bin/env python3
"""Download a small SYNTH sample for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and save small SYNTH shards for inspection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synth_sample"),
        help="Directory to write JSONL samples and metadata.",
    )
    parser.add_argument(
        "--dataset",
        default="PleIAs/SYNTH",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=3,
        help="Number of virtual shards to partition the split into.",
    )
    parser.add_argument(
        "--shard-indices",
        default="0,1,2",
        help="Comma-separated shard indices to materialize.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200,
        help="Max rows to save per shard (set -1 for full shard).",
    )
    parser.add_argument(
        "--streaming",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use streaming mode to avoid full dataset download.",
    )
    return parser.parse_args()


def iter_rows(dataset: Iterable, max_rows: int) -> Iterable[dict]:
    if max_rows < 0:
        yield from dataset
        return
    if hasattr(dataset, "take"):
        yield from dataset.take(max_rows)
        return
    for idx, row in enumerate(dataset):
        if idx >= max_rows:
            break
        yield row


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_indices = [int(s.strip()) for s in args.shard_indices.split(",") if s.strip()]
    if not shard_indices:
        raise SystemExit("No shard indices provided.")

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
    )

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "streaming": args.streaming,
        "num_shards": args.num_shards,
        "shard_indices": shard_indices,
        "max_rows": args.max_rows,
        "features": getattr(dataset, "features", None),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    for shard_idx in shard_indices:
        shard = dataset.shard(num_shards=args.num_shards, index=shard_idx)
        out_path = output_dir / f"synth_shard_{shard_idx:02d}.jsonl"
        count = 0
        with out_path.open("w", encoding="utf-8") as handle:
            for row in iter_rows(shard, args.max_rows):
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                count += 1
        print(f"Wrote {count} rows to {out_path}")

    print(f"Metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
