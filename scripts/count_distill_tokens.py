#!/usr/bin/env python3
"""Summarize token counts for a DistillKit-ready dataset (Parquet)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _require(name: str):
    try:
        import importlib

        return importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f"Missing required module '{name}': {exc}") from exc


def _iter_batches(dataset, columns: List[str], batch_size: int):
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pydict()


def _estimate_rows_in_dir(path: Path) -> Optional[int]:
    pq = _require("pyarrow.parquet")
    if not path.exists():
        return None
    total = 0
    saw_any = False
    for shard in sorted(path.glob("*.parquet")):
        saw_any = True
        try:
            pf = pq.ParquetFile(str(shard))
            md = pf.metadata
            if md is not None:
                total += int(md.num_rows)
        except Exception:
            return None
    return total if saw_any else None


def _summarize_split(
    path: Path,
    batch_size: int,
    *,
    by_language: bool,
    by_exercise: bool,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    tqdm = _require("tqdm").tqdm
    ds = _require("pyarrow.dataset")
    dataset = ds.dataset(str(path), format="parquet")

    total_rows = 0
    total_tokens = 0
    total_assistant = 0

    breakdown: Dict[str, Any] = {}
    tokens_by_lang: Dict[str, int] = defaultdict(int)
    assistant_by_lang: Dict[str, int] = defaultdict(int)
    rows_by_lang: Counter = Counter()
    tokens_by_ex: Dict[str, int] = defaultdict(int)
    assistant_by_ex: Dict[str, int] = defaultdict(int)
    rows_by_ex: Counter = Counter()

    columns = ["input_ids", "spans"]
    if by_language or by_exercise:
        columns.append("source")

    expected_rows = _estimate_rows_in_dir(path)
    with tqdm(total=expected_rows, desc=path.name, unit="row", dynamic_ncols=True) as pbar:
        for batch in _iter_batches(dataset, columns, batch_size):
            input_ids = batch.get("input_ids", [])
            spans = batch.get("spans", [])
            sources = batch.get("source", [])
            if not sources:
                sources = [None] * len(input_ids)

            for ids, span, source in zip(input_ids, spans, sources):
                total_rows += 1
                seq_tokens = len(ids)
                total_tokens += seq_tokens
                if isinstance(span, dict):
                    assistant_tokens = int(span.get("analysis_token_count", 0)) + int(
                        span.get("final_token_count", 0)
                    )
                    total_assistant += assistant_tokens
                else:
                    assistant_tokens = 0

                if isinstance(source, dict):
                    lang = str(source.get("language") or "unknown").strip().lower()
                    ex = str(source.get("exercise") or "unknown").strip().lower()
                else:
                    lang = "unknown"
                    ex = "unknown"

                if by_language:
                    rows_by_lang[lang] += 1
                    tokens_by_lang[lang] += seq_tokens
                    assistant_by_lang[lang] += assistant_tokens

                if by_exercise:
                    rows_by_ex[ex] += 1
                    tokens_by_ex[ex] += seq_tokens
                    assistant_by_ex[ex] += assistant_tokens

            pbar.update(len(input_ids))

    mean_tokens = total_tokens / total_rows if total_rows else 0.0
    mean_assistant = total_assistant / total_rows if total_rows else 0.0
    summary = {
        "rows": total_rows,
        "total_tokens": total_tokens,
        "total_assistant_tokens": total_assistant,
        "mean_tokens": mean_tokens,
        "mean_assistant_tokens": mean_assistant,
    }

    if by_language:
        breakdown["by_language"] = {
            "rows": dict(rows_by_lang),
            "total_tokens": dict(tokens_by_lang),
            "total_assistant_tokens": dict(assistant_by_lang),
        }
    if by_exercise:
        breakdown["by_exercise"] = {
            "rows": dict(rows_by_ex),
            "total_tokens": dict(tokens_by_ex),
            "total_assistant_tokens": dict(assistant_by_ex),
        }

    return summary, breakdown


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count tokens in a distill dataset.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "all"],
    )
    parser.add_argument("--by-language", action="store_true", help="Print language breakdowns.")
    parser.add_argument("--by-exercise", action="store_true", help="Print exercise breakdowns.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    splits = ["train", "validation"] if args.split == "all" else [args.split]

    results = {}
    breakdowns = {}
    for split in splits:
        path = input_dir / split
        if not path.exists():
            raise SystemExit(f"Split not found: {path}")
        results[split], breakdowns[split] = _summarize_split(
            path,
            args.batch_size,
            by_language=args.by_language,
            by_exercise=args.by_exercise,
        )

    # Aggregate if needed
    if len(results) > 1:
        total_rows = sum(r["rows"] for r in results.values())
        total_tokens = sum(r["total_tokens"] for r in results.values())
        total_assistant = sum(r["total_assistant_tokens"] for r in results.values())
        results["all"] = {
            "rows": total_rows,
            "total_tokens": total_tokens,
            "total_assistant_tokens": total_assistant,
            "mean_tokens": total_tokens / total_rows if total_rows else 0.0,
            "mean_assistant_tokens": total_assistant / total_rows if total_rows else 0.0,
        }
        if args.by_language or args.by_exercise:
            all_breakdown: Dict[str, Any] = {}

            if args.by_language:
                rows = Counter()
                tokens = Counter()
                assistant = Counter()
                for split in ["train", "validation"]:
                    split_bd = breakdowns.get(split, {}).get("by_language", {})
                    rows.update(split_bd.get("rows", {}))
                    tokens.update(split_bd.get("total_tokens", {}))
                    assistant.update(split_bd.get("total_assistant_tokens", {}))
                all_breakdown["by_language"] = {
                    "rows": dict(rows),
                    "total_tokens": dict(tokens),
                    "total_assistant_tokens": dict(assistant),
                }

            if args.by_exercise:
                rows = Counter()
                tokens = Counter()
                assistant = Counter()
                for split in ["train", "validation"]:
                    split_bd = breakdowns.get(split, {}).get("by_exercise", {})
                    rows.update(split_bd.get("rows", {}))
                    tokens.update(split_bd.get("total_tokens", {}))
                    assistant.update(split_bd.get("total_assistant_tokens", {}))
                all_breakdown["by_exercise"] = {
                    "rows": dict(rows),
                    "total_tokens": dict(tokens),
                    "total_assistant_tokens": dict(assistant),
                }

            breakdowns["all"] = all_breakdown

    for split, stats in results.items():
        print(f"{split}:")
        print(f"  rows: {stats['rows']}")
        print(f"  total_tokens: {stats['total_tokens']}")
        print(f"  total_assistant_tokens: {stats['total_assistant_tokens']}")
        print(f"  mean_tokens: {stats['mean_tokens']:.2f}")
        print(f"  mean_assistant_tokens: {stats['mean_assistant_tokens']:.2f}")

        breakdown = breakdowns.get(split) or {}
        if args.by_language and "by_language" in breakdown:
            lang_rows = breakdown["by_language"]["rows"]
            lang_tokens = breakdown["by_language"]["total_tokens"]
            lang_assistant = breakdown["by_language"]["total_assistant_tokens"]
            en_rows = int(lang_rows.get("en", 0))
            en_tokens = int(lang_tokens.get("en", 0))
            en_assistant = int(lang_assistant.get("en", 0))
            print(
                f"  language: en_rows={en_rows} ({(en_rows / stats['rows'] * 100.0) if stats['rows'] else 0.0:.2f}%), "
                f"en_tokens={en_tokens} ({(en_tokens / stats['total_tokens'] * 100.0) if stats['total_tokens'] else 0.0:.2f}%), "
                f"en_assistant_tokens={en_assistant} ({(en_assistant / stats['total_assistant_tokens'] * 100.0) if stats['total_assistant_tokens'] else 0.0:.2f}%)"
            )

        if args.by_language and "by_language" in breakdown:
            print("  by_language_rows:")
            for lang, count in sorted(breakdown["by_language"]["rows"].items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"    {lang}: {count}")

        if args.by_exercise and "by_exercise" in breakdown:
            print("  by_exercise_rows:")
            for ex, count in sorted(breakdown["by_exercise"]["rows"].items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"    {ex}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
