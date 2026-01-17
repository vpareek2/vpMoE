#!/usr/bin/env python3
"""
Lightweight EDA for Parquet datasets.

Example:
  python3 scripts/eda_parquet.py \
    --input-dir ~/Master/datasets/nvidia__OpenCodeInstruct/data \
    --sample-shards 5 \
    --sample-rows-per-shard 4000 \
    --columns id input output domain unit_tests tests_execution_status average_test_score \
    --text-cols input output unit_tests \
    --cat-cols domain tests_execution_status \
    --token-encoding o200k_base \
    --show-samples 2
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick EDA for parquet datasets.")
    p.add_argument("--input-dir", type=str, default=None, help="Directory of parquet shards.")
    p.add_argument("--input-file", type=str, default=None, help="Single parquet file.")
    p.add_argument("--sample-shards", type=int, default=5)
    p.add_argument("--sample-rows-per-shard", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Columns to load (default: all).",
    )
    p.add_argument(
        "--text-cols",
        nargs="*",
        default=None,
        help="Columns to treat as text for length/token stats.",
    )
    p.add_argument(
        "--cat-cols",
        nargs="*",
        default=None,
        help="Columns to show top value counts for.",
    )
    p.add_argument(
        "--token-encoding",
        type=str,
        default=None,
        help="tiktoken encoding name (e.g. o200k_base) for token stats.",
    )
    p.add_argument("--show-samples", type=int, default=0, help="Print N sample rows.")
    p.add_argument("--truncate-chars", type=int, default=400)
    p.add_argument("--random-sample", action="store_true", help="Randomly sample rows.")
    return p.parse_args()


def _maybe_set_tiktoken_env() -> None:
    # Prefer local encodings cache if present.
    if "TIKTOKEN_ENCODINGS_BASE" not in os.environ:
        for candidate in [
            Path("/data/hf_cache/tiktoken-encodings"),
            Path("data/hf_cache/tiktoken-encodings"),
        ]:
            if candidate.exists():
                os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(candidate.resolve())
                break

    if "TIKTOKEN_CACHE_DIR" not in os.environ:
        for candidate in [
            Path("/data/hf_cache/tiktoken-encodings"),
            Path("data/hf_cache/tiktoken-encodings"),
        ]:
            if candidate.exists():
                os.environ["TIKTOKEN_CACHE_DIR"] = str(candidate.resolve())
                break


def _iter_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.parquet"))


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _len_stats(series, name: str):
    s = series.fillna("").astype(str)
    q = s.str.len().quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    mx = int(s.str.len().max())
    print(
        f"{name:24s} p50={int(q[0.5])} p90={int(q[0.9])} p95={int(q[0.95])} "
        f"p99={int(q[0.99])} max={mx}"
    )


def _token_stats(series, name: str, enc):
    s = series.fillna("").astype(str)
    tok_lens = s.map(lambda x: len(enc.encode(x)))
    q = tok_lens.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    mx = int(tok_lens.max())
    print(
        f"{name:24s} p50={int(q[0.5])} p90={int(q[0.9])} p95={int(q[0.95])} "
        f"p99={int(q[0.99])} max={mx}"
    )


def _print_samples(df, n: int, truncate_chars: int, seed: int):
    if n <= 0:
        return
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    for i, row in enumerate(sample.itertuples(index=False)):
        print("\n" + "=" * 100)
        print(f"sample {i + 1}")
        for col, val in zip(sample.columns, row):
            if isinstance(val, str):
                print(f"{col}:\n{_truncate(val, truncate_chars)}")
            else:
                print(f"{col}: {val}")


def main() -> int:
    args = _parse_args()
    if not args.input_dir and not args.input_file:
        raise SystemExit("Provide --input-dir or --input-file.")

    import pyarrow.parquet as pq
    import pandas as pd

    rng = random.Random(args.seed)

    if args.input_file:
        files = [Path(args.input_file).expanduser().resolve()]
    else:
        input_dir = Path(args.input_dir).expanduser().resolve()
        files = _iter_files(input_dir)
    if not files:
        raise SystemExit("No parquet files found.")

    # Basic schema + row count from first file
    pf = pq.ParquetFile(files[0])
    print("first shard:", files[0].name)
    print("rows:", pf.metadata.num_rows)
    print("schema:", pf.schema)

    # Sample shards
    if args.sample_shards > 0 and len(files) > args.sample_shards:
        files = files[: args.sample_shards]

    cols = args.columns
    dfs = []
    for p in files:
        table = pq.read_table(p, columns=cols)
        if args.sample_rows_per_shard and table.num_rows > args.sample_rows_per_shard:
            if args.random_sample:
                df = table.to_pandas()
                df = df.sample(
                    n=args.sample_rows_per_shard, random_state=rng.randint(0, 2**31 - 1)
                )
            else:
                df = table.slice(0, args.sample_rows_per_shard).to_pandas()
        else:
            df = table.to_pandas()
        df["__shard"] = p.name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nsample rows: {len(df)} from shards: {len(files)}")

    # Missingness
    print("\nmissingness:")
    miss_cols = cols if cols else [c for c in df.columns if c != "__shard"]
    for c in miss_cols:
        miss = float(df[c].isna().mean())
        print(f"{c:24s} missing_frac={miss:.3f}")

    # Value counts
    if args.cat_cols:
        print("\nTop values:")
        for c in args.cat_cols:
            print(f"\n{c}:")
            print(df[c].value_counts().head(10))

    # Length stats
    if args.text_cols:
        print("\nchar length stats:")
        for c in args.text_cols:
            _len_stats(df[c], c)

    # Token stats
    if args.token_encoding:
        _maybe_set_tiktoken_env()
        import tiktoken

        enc = tiktoken.get_encoding(args.token_encoding)
        print("\ntoken length stats:")
        for c in args.text_cols or []:
            _token_stats(df[c], c, enc)

    _print_samples(df, args.show_samples, args.truncate_chars, args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
