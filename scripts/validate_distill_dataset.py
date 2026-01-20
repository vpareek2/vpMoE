#!/usr/bin/env python3
"""Validate token counts for a save_to_disk DistillKit dataset with progress."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm


@dataclass
class SplitStats:
    rows: int
    total_tokens: int
    total_assistant_tokens: int
    max_tokens: int
    max_assistant_tokens: int

    @property
    def mean_tokens(self) -> float:
        return self.total_tokens / self.rows if self.rows else 0.0

    @property
    def mean_assistant_tokens(self) -> float:
        return self.total_assistant_tokens / self.rows if self.rows else 0.0


def _summarize_split(ds: Dataset, name: str, batch_size: int) -> SplitStats:
    total_rows = ds.num_rows
    total_tokens = 0
    total_assistant = 0
    max_tokens = 0
    max_assistant = 0

    with tqdm(total=total_rows, desc=name, unit="row", dynamic_ncols=True) as pbar:
        for batch in ds.to_iterable_dataset().iter(batch_size=batch_size):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            for ids, lbls in zip(input_ids, labels):
                seq_len = len(ids)
                if seq_len > max_tokens:
                    max_tokens = seq_len
                total_tokens += seq_len

                assistant_tokens = 0
                for x in lbls:
                    if x != -100:
                        assistant_tokens += 1
                if assistant_tokens > max_assistant:
                    max_assistant = assistant_tokens
                total_assistant += assistant_tokens

            pbar.update(len(input_ids))

    return SplitStats(
        rows=total_rows,
        total_tokens=total_tokens,
        total_assistant_tokens=total_assistant,
        max_tokens=max_tokens,
        max_assistant_tokens=max_assistant,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate token counts for a save_to_disk dataset.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "all"],
    )
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ds = load_from_disk(args.input_dir)

    if isinstance(ds, DatasetDict):
        split_names = ["train", "validation"] if args.split == "all" else [args.split]
        missing = [s for s in split_names if s not in ds]
        if missing:
            raise SystemExit(f"Missing splits in dataset: {missing}")
        splits: Dict[str, Dataset] = {s: ds[s] for s in split_names}
    elif isinstance(ds, Dataset):
        if args.split != "all":
            raise SystemExit("Single dataset found; omit --split or use --split all.")
        splits = {"all": ds}
    else:
        raise SystemExit("Unsupported dataset type loaded from disk.")

    results: Dict[str, SplitStats] = {}
    for name, split_ds in splits.items():
        results[name] = _summarize_split(split_ds, name, args.batch_size)

    if "train" in results and "validation" in results:
        train = results["train"]
        val = results["validation"]
        all_stats = SplitStats(
            rows=train.rows + val.rows,
            total_tokens=train.total_tokens + val.total_tokens,
            total_assistant_tokens=train.total_assistant_tokens + val.total_assistant_tokens,
            max_tokens=max(train.max_tokens, val.max_tokens),
            max_assistant_tokens=max(train.max_assistant_tokens, val.max_assistant_tokens),
        )
        results["all"] = all_stats

    for name in ["train", "validation", "all"]:
        if name not in results:
            continue
        stats = results[name]
        print(f"{name}:")
        print(f"  rows: {stats.rows}")
        print(f"  total_tokens: {stats.total_tokens}")
        print(f"  total_assistant_tokens: {stats.total_assistant_tokens}")
        print(f"  mean_tokens: {stats.mean_tokens:.2f}")
        print(f"  mean_assistant_tokens: {stats.mean_assistant_tokens:.2f}")
        print(f"  max_tokens: {stats.max_tokens}")
        print(f"  max_assistant_tokens: {stats.max_assistant_tokens}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
