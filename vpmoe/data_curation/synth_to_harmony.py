#!/usr/bin/env python3
"""Convert SYNTH rows to Harmony-formatted conversations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from datasets import load_dataset


DEFAULT_METADATA_FIELDS = [
    "synth_id",
    "language",
    "exercise",
    "model",
    "constraints",
    "seed_license",
    "query_seed_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream SYNTH and write Harmony JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synth_harmony"),
        help="Directory to write Harmony JSONL shards and metadata.",
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
        "--streaming",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use streaming mode to avoid full dataset download.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of virtual shards to partition the split into.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index to materialize.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="Max rows to write (set -1 for full shard).",
    )
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "parquet"],
        default="parquet",
        help="Output format. Parquet is recommended for full runs.",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=1000,
        help="Rows per Parquet row-group write (parquet only).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=2000,
        help="Log progress every N written rows (0 disables).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output shard if it already exists.",
    )
    parser.add_argument(
        "--metadata-fields",
        default=",".join(DEFAULT_METADATA_FIELDS),
        help="Comma-separated metadata fields to keep.",
    )
    parser.add_argument(
        "--model-identity",
        default="You are vpMoE, a large language model trained by Veer Pareek.",
        help="System identity string.",
    )
    parser.add_argument(
        "--knowledge-cutoff",
        default="2024-06",
        help="System knowledge cutoff string.",
    )
    parser.add_argument(
        "--conversation-start-date",
        default="2026-01-01",
        help="Value used for the system message `Current date:` line (Harmony field `conversation_start_date`).",
    )
    parser.add_argument(
        "--current-date",
        dest="conversation_start_date",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="high",
        help="Reasoning effort for the system message.",
    )
    parser.add_argument(
        "--developer-instructions",
        default=(
            "Follow the user's request. When you provide an answer, produce both:\n"
            "- reasoning in the analysis channel\n"
            "- the user-facing answer in the final channel"
        ),
        help="Developer instructions string.",
    )
    parser.add_argument(
        "--omit-developer",
        action="store_true",
        help="Omit the developer message.",
    )
    return parser.parse_args()


def make_system_message(
    identity: str,
    knowledge_cutoff: str,
    conversation_start_date: str,
    reasoning_effort: str,
) -> Dict[str, object]:
    return {
        "role": "system",
        "content": [
            {
                "type": "system_content",
                "model_identity": identity,
                "reasoning_effort": reasoning_effort.capitalize(),
                "conversation_start_date": conversation_start_date,
                "knowledge_cutoff": knowledge_cutoff,
                "channel_config": {
                    "valid_channels": ["analysis", "commentary", "final"],
                    "channel_required": True,
                },
            }
        ],
    }


def make_developer_message(instructions: str) -> Dict[str, object]:
    return {
        "role": "developer",
        "content": [
            {
                "type": "developer_content",
                "instructions": instructions,
            }
        ],
    }


def make_text_message(role: str, text: str, channel: Optional[str] = None) -> Dict[str, object]:
    msg: Dict[str, object] = {
        "role": role,
        "content": [{"type": "text", "text": text}],
    }
    if channel is not None:
        msg["channel"] = channel
    return msg


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


def filter_metadata(row: dict, keep_fields: List[str]) -> Dict[str, object]:
    return {field: row.get(field) for field in keep_fields if field in row}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_fields = [f.strip() for f in args.metadata_fields.split(",") if f.strip()]
    if not metadata_fields:
        raise SystemExit("No metadata fields specified.")

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
    )
    if args.num_shards > 1:
        dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index)

    system_msg = make_system_message(
        identity=args.model_identity,
        knowledge_cutoff=args.knowledge_cutoff,
        conversation_start_date=args.conversation_start_date,
        reasoning_effort=args.reasoning_effort,
    )
    developer_msg = None if args.omit_developer else make_developer_message(
        args.developer_instructions
    )

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "streaming": args.streaming,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_rows": args.max_rows,
        "metadata_fields": metadata_fields,
        "system_message": system_msg,
        "developer_message": developer_msg,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    out_basename = f"synth_harmony_shard_{args.shard_index:02d}"
    out_path = output_dir / f"{out_basename}.{args.output_format}"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing output: {out_path} (pass --overwrite)"
        )
    total = 0
    skipped = 0
    started = time.time()

    def maybe_log() -> None:
        if args.log_every <= 0:
            return
        if total == 0 or total % args.log_every != 0:
            return
        elapsed = time.time() - started
        rows_per_sec = total / elapsed if elapsed > 0 else 0.0
        print(f"[progress] wrote={total} skipped={skipped} rows/sec={rows_per_sec:.1f}")

    if args.output_format == "jsonl":
        with out_path.open("w", encoding="utf-8") as handle:
            for row in iter_rows(dataset, args.max_rows):
                query = row.get("query")
                reasoning = row.get("synthetic_reasoning")
                answer = row.get("synthetic_answer")
                if not query or not reasoning or not answer:
                    skipped += 1
                    continue

                messages: List[Dict[str, object]] = [system_msg]
                if developer_msg is not None:
                    messages.append(developer_msg)
                messages.append(make_text_message("user", query))
                messages.append(make_text_message("assistant", reasoning, channel="analysis"))
                messages.append(make_text_message("assistant", answer, channel="final"))

                record = {
                    "messages": messages,
                    "metadata": filter_metadata(row, metadata_fields),
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                total += 1
                maybe_log()
    else:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Parquet output requires `pyarrow`. Either install it or use --output-format jsonl."
            ) from exc

        # Store messages and metadata as JSON strings to keep the schema stable.
        # Also store selected metadata fields as top-level columns for easy slicing.
        schema_fields = [
            pa.field("messages_json", pa.string()),
            pa.field("metadata_json", pa.string()),
        ]
        for field in metadata_fields:
            schema_fields.append(pa.field(field, pa.string()))
        schema = pa.schema(schema_fields)

        writer: Optional[pq.ParquetWriter] = None
        buffer: List[Dict[str, object]] = []

        def flush() -> None:
            nonlocal writer, buffer
            if not buffer:
                return
            table = pa.Table.from_pylist(buffer, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(out_path, schema=schema, compression="zstd")
            writer.write_table(table)
            buffer = []

        for row in iter_rows(dataset, args.max_rows):
            query = row.get("query")
            reasoning = row.get("synthetic_reasoning")
            answer = row.get("synthetic_answer")
            if not query or not reasoning or not answer:
                skipped += 1
                continue

            messages: List[Dict[str, object]] = [system_msg]
            if developer_msg is not None:
                messages.append(developer_msg)
            messages.append(make_text_message("user", query))
            messages.append(make_text_message("assistant", reasoning, channel="analysis"))
            messages.append(make_text_message("assistant", answer, channel="final"))

            md = filter_metadata(row, metadata_fields)
            record: Dict[str, object] = {
                "messages_json": json.dumps({"messages": messages}, ensure_ascii=True),
                "metadata_json": json.dumps(md, ensure_ascii=True),
            }
            for field in metadata_fields:
                value = md.get(field)
                record[field] = None if value is None else str(value)
            buffer.append(record)

            total += 1
            if len(buffer) >= args.parquet_batch_size:
                flush()
            maybe_log()

        flush()
        if writer is not None:
            writer.close()

    elapsed = time.time() - started
    rows_per_sec = total / elapsed if elapsed > 0 else 0.0
    print(f"Wrote {total} rows to {out_path} ({rows_per_sec:.1f} rows/sec)")
    if skipped:
        print(f"Skipped {skipped} rows with missing fields")
    print(f"Metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
