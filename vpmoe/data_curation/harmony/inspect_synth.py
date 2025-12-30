#!/usr/bin/env python3
"""Inspect SYNTH JSONL samples and print a compact report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_FIELDS = [
    "language",
    "category",
    "exercise_type",
    "topic",
    "source",
]

TEXT_FIELDS = [
    "query",
    "synthetic_reasoning",
    "synthetic_answer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect SYNTH JSONL shards and emit summary stats + samples.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/synth_sample"),
        help="Directory containing JSONL shards from download_synth_sample.py.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=200,
        help="Max rows to scan per JSONL file.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2,
        help="How many example rows to include per file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K category values to print per field.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the report as JSON.",
    )
    return parser.parse_args()


def iter_jsonl_rows(path: Path, limit: int) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def short_text(text: str, max_len: int = 200) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def summarize_text_fields(rows: Iterable[dict]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, List[int]] = {field: [] for field in TEXT_FIELDS}
    for row in rows:
        for field in TEXT_FIELDS:
            val = row.get(field)
            if isinstance(val, str):
                stats[field].append(len(val))
    summary = {}
    for field, lengths in stats.items():
        if not lengths:
            continue
        summary[field] = {
            "count": len(lengths),
            "mean_chars": sum(lengths) / len(lengths),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
        }
    return summary


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    jsonl_files = sorted(input_dir.glob("synth_shard_*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No synth_shard_*.jsonl files found in {input_dir}")

    all_keys: Counter[str] = Counter()
    field_counters: Dict[str, Counter[str]] = defaultdict(Counter)
    samples: Dict[str, List[dict]] = {}
    text_stats_accum: List[Dict[str, Dict[str, float]]] = []
    total_rows = 0

    for path in jsonl_files:
        rows = list(iter_jsonl_rows(path, args.max_rows_per_file))
        total_rows += len(rows)
        for row in rows:
            all_keys.update(row.keys())
            for field in DEFAULT_FIELDS:
                val = row.get(field)
                if isinstance(val, str):
                    field_counters[field][val] += 1
        text_stats_accum.append(summarize_text_fields(rows))
        samples[path.name] = rows[: args.sample_rows]

    # Merge text stats across files.
    merged_text_stats: Dict[str, Dict[str, float]] = {}
    for field in TEXT_FIELDS:
        counts = []
        mins = []
        maxs = []
        means = []
        for stats in text_stats_accum:
            if field not in stats:
                continue
            counts.append(stats[field]["count"])
            mins.append(stats[field]["min_chars"])
            maxs.append(stats[field]["max_chars"])
            means.append(stats[field]["mean_chars"])
        if counts:
            merged_text_stats[field] = {
                "count": sum(int(c) for c in counts),
                "mean_chars": sum(means) / len(means),
                "min_chars": min(mins),
                "max_chars": max(maxs),
            }

    report = {
        "input_dir": str(input_dir),
        "files": [p.name for p in jsonl_files],
        "rows_scanned": total_rows,
        "fields": [k for k, _ in all_keys.most_common()],
        "text_field_stats": merged_text_stats,
        "categories": {
            field: counter.most_common(args.top_k)
            for field, counter in field_counters.items()
            if counter
        },
        "samples": {
            fname: [
                {
                    "query": short_text(r.get("query", "")),
                    "synthetic_reasoning": short_text(r.get("synthetic_reasoning", "")),
                    "synthetic_answer": short_text(r.get("synthetic_answer", "")),
                    **{k: r.get(k) for k in DEFAULT_FIELDS if k in r},
                }
                for r in rows
            ]
            for fname, rows in samples.items()
        },
    }

    print("SYNTH sample inspection")
    print(f"Input dir: {input_dir}")
    print(f"Files: {', '.join(report['files'])}")
    print(f"Rows scanned: {total_rows}")
    print(f"Fields ({len(report['fields'])}): {', '.join(report['fields'])}")
    if merged_text_stats:
        print("Text field stats (chars):")
        for field, stats in merged_text_stats.items():
            print(
                f"  {field}: count={stats['count']} "
                f"mean={stats['mean_chars']:.1f} "
                f"min={stats['min_chars']} max={stats['max_chars']}"
            )
    if report["categories"]:
        print("Top categories:")
        for field, values in report["categories"].items():
            top = ", ".join(f"{k} ({v})" for k, v in values)
            print(f"  {field}: {top}")
    print("Samples:")
    for fname, rows in report["samples"].items():
        print(f"  {fname}:")
        for idx, row in enumerate(rows, start=1):
            print(f"    [{idx}] query: {row.get('query', '')}")
            if row.get("synthetic_reasoning"):
                print(f"        reasoning: {row['synthetic_reasoning']}")
            if row.get("synthetic_answer"):
                print(f"        answer: {row['synthetic_answer']}")
            for field in DEFAULT_FIELDS:
                if field in row:
                    print(f"        {field}: {row[field]}")

    if args.output_json:
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote report JSON to {args.output_json}")


if __name__ == "__main__":
    main()
