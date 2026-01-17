#!/usr/bin/env python3
"""
Merge multiple DistillKit-ready datasets into one interleaved dataset.

Why this exists:
- Our per-dataset builders currently write different `source` struct schemas.
  You cannot mix parquet files with incompatible nested struct schemas in a
  single directory and expect pyarrow/datasets to load them reliably.

This script rewrites all inputs into a single canonical schema and interleaves
at the *input shard* level (deterministic, low memory).

Canonical output schema:
- id: string
- source: struct{
    dataset: string,
    language: string,
    exercise: string,
    meta_json: string
  }
- input_ids: list<int32>
- labels: list<int32>
- attention_mask: list<int8>
- spans: struct{assistant_token_start:int32, analysis_token_count:int32, final_token_count:int32}
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def _require(name: str):
    try:
        import importlib

        return importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f"Missing required module '{name}': {exc}") from exc


def _sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _hash_to_unit(value: str, seed: int, salt: str) -> float:
    payload = f"{seed}:{salt}:{value}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _list_shards(root: Path, split: str) -> List[Path]:
    split_dir = root / split
    if not split_dir.exists():
        raise SystemExit(f"Missing split dir: {split_dir}")
    shards = sorted(split_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet shards found under: {split_dir}")
    return shards


def _read_manifest(root: Path) -> Optional[Dict[str, Any]]:
    path = root / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _estimate_totals_from_manifest(manifest: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Returns (rows, total_tokens, total_assistant_tokens) if available.
    These are best-effort and used only to compute interleaving weights.
    """

    if not manifest:
        return None, None, None

    rows = None
    total_tokens = None
    total_assistant = None

    output = manifest.get("output") if isinstance(manifest, dict) else None
    if isinstance(output, dict):
        rows = output.get("kept") if isinstance(output.get("kept"), int) else rows
        total_tokens = output.get("total_tokens") if isinstance(output.get("total_tokens"), int) else total_tokens
        total_assistant = (
            output.get("total_assistant_tokens")
            if isinstance(output.get("total_assistant_tokens"), int)
            else total_assistant
        )

    token_stats = manifest.get("token_stats") if isinstance(manifest, dict) else None
    if isinstance(token_stats, dict):
        total = token_stats.get("total")
        if isinstance(total, dict):
            if rows is None and isinstance(total.get("count"), int):
                rows = int(total["count"])
            if total_tokens is None and isinstance(total.get("count"), int) and isinstance(total.get("mean"), (int, float)):
                total_tokens = int(round(float(total["mean"]) * int(total["count"])))
        final = token_stats.get("final")
        analysis = token_stats.get("analysis")
        if total_assistant is None:
            try:
                if isinstance(final, dict) and isinstance(final.get("count"), int) and isinstance(final.get("mean"), (int, float)):
                    n = int(final["count"])
                    a_mean = float(final["mean"])
                    r_mean = 0.0
                    if isinstance(analysis, dict) and isinstance(analysis.get("mean"), (int, float)):
                        r_mean = float(analysis["mean"])
                    total_assistant = int(round((a_mean + r_mean) * n))
            except Exception:
                pass

    return rows, total_tokens, total_assistant


def _smooth_weighted_round_robin(
    queues: Dict[str, List[Path]],
    weights: Dict[str, float],
) -> List[Tuple[str, Path]]:
    """
    Deterministic smooth weighted round robin over *shards*.
    Returns a sequence of (dataset_name, shard_path).
    """

    remaining = {k: list(v) for k, v in queues.items() if v}
    current = {k: 0.0 for k in remaining}
    total_w = sum(float(weights.get(k, 1.0)) for k in remaining) or 1.0

    schedule: List[Tuple[str, Path]] = []
    while True:
        alive = [k for k, q in remaining.items() if q]
        if not alive:
            break

        best = None
        best_score = None
        for k in alive:
            current[k] += float(weights.get(k, 1.0))
            score = current[k]
            if best is None or score > best_score:
                best = k
                best_score = score

        assert best is not None
        shard = remaining[best].pop(0)
        schedule.append((best, shard))
        current[best] -= total_w

    return schedule


class ShardedParquetWriter:
    def __init__(self, out_dir: Path, schema, prefix: str = "part", rows_per_shard: int = 2000):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.prefix = prefix
        self.rows_per_shard = rows_per_shard
        self.buffer: List[Dict[str, Any]] = []
        self.shard_index = 0
        self.total_rows = 0

    def write(self, row: Dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        pa = _require("pyarrow")
        pq = _require("pyarrow.parquet")
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        path = self.out_dir / f"{self.prefix}-{self.shard_index:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        self.total_rows += len(self.buffer)
        self.buffer.clear()
        self.shard_index += 1

    def close(self) -> None:
        self.flush()


@dataclasses.dataclass
class OutputStats:
    input_shards_total: int = 0
    input_rows_total: int = 0
    rows_written: int = 0
    train_rows: int = 0
    validation_rows: int = 0


def _normalize_source(
    dataset_name: str,
    source: Any,
    *,
    default_language: str,
    default_exercise: str,
) -> Dict[str, Any]:
    if not isinstance(source, dict):
        source = {}

    language = str(source.get("language") or default_language or "unknown").strip().lower()
    exercise = str(source.get("exercise") or default_exercise or "unknown").strip().lower()
    meta = dict(source)
    meta.setdefault("dataset", dataset_name)

    return {
        "dataset": dataset_name,
        "language": language,
        "exercise": exercise,
        "meta_json": json.dumps(meta, sort_keys=True, separators=(",", ":")),
    }


def _iter_row_batches_from_shard(shard: Path, *, batch_size: int = 2048) -> Iterator[List[Dict[str, Any]]]:
    pq = _require("pyarrow.parquet")
    pf = pq.ParquetFile(str(shard))
    columns = ["id", "source", "input_ids", "labels", "attention_mask", "spans"]
    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        yield batch.to_pylist()


def _shard_num_rows(shard: Path) -> int:
    pq = _require("pyarrow.parquet")
    try:
        pf = pq.ParquetFile(str(shard))
        md = pf.metadata
        if md is None:
            return 0
        return int(md.num_rows)
    except Exception:
        return 0


def _parse_input_specs(values: List[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for v in values:
        if "=" in v:
            name, path = v.split("=", 1)
            name = name.strip()
            path = path.strip()
        else:
            path = v.strip()
            name = Path(path).name
        if not name:
            raise SystemExit(f"Invalid --input: {v!r}")
        out.append((name, Path(path).expanduser().resolve()))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge/interleave multiple distill datasets into one dataset dir.")
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input dataset root. Use `name=/path/to/dataset` to name it.",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--rows-per-shard", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--weight-by",
        choices=["assistant_tokens", "total_tokens", "rows"],
        default="assistant_tokens",
        help="What to use as interleaving weight (best-effort from each manifest).",
    )
    p.add_argument(
        "--shuffle-within-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle shard order within each input dataset (seeded).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    inputs = _parse_input_specs(args.input)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pa = _require("pyarrow")

    schema = pa.schema(
        [
            ("id", pa.string()),
            (
                "source",
                pa.struct(
                    [
                        ("dataset", pa.string()),
                        ("language", pa.string()),
                        ("exercise", pa.string()),
                        ("meta_json", pa.string()),
                    ]
                ),
            ),
            ("input_ids", pa.list_(pa.int32())),
            ("labels", pa.list_(pa.int32())),
            ("attention_mask", pa.list_(pa.int8())),
            (
                "spans",
                pa.struct(
                    [
                        ("assistant_token_start", pa.int32()),
                        ("analysis_token_count", pa.int32()),
                        ("final_token_count", pa.int32()),
                    ]
                ),
            ),
        ]
    )

    # Collect shard lists and per-dataset weights.
    per_ds: Dict[str, Dict[str, Any]] = {}
    for name, root in inputs:
        if not root.exists():
            raise SystemExit(f"Input not found: {root}")
        manifest = _read_manifest(root)
        rows, total_tokens, total_assistant = _estimate_totals_from_manifest(manifest)

        weights: Dict[str, float] = {"rows": float(rows or 1)}
        weights["total_tokens"] = float(total_tokens or (rows or 1))
        weights["assistant_tokens"] = float(total_assistant or (total_tokens or (rows or 1)))

        per_ds[name] = {
            "root": root,
            "manifest": manifest,
            "weights": weights,
            "train": _list_shards(root, "train"),
            "validation": _list_shards(root, "validation"),
        }

    if args.shuffle_within_dataset:
        for name, meta in per_ds.items():
            seed = args.seed
            rng = random.Random(int(_hash_to_unit(name, seed, "shuffle") * (2**31 - 1)))
            rng.shuffle(meta["train"])
            rng.shuffle(meta["validation"])

    weights_for_mix = {name: float(per_ds[name]["weights"][args.weight_by]) for name in per_ds}

    train_queues = {name: list(meta["train"]) for name, meta in per_ds.items()}
    val_queues = {name: list(meta["validation"]) for name, meta in per_ds.items()}

    train_schedule = _smooth_weighted_round_robin(train_queues, weights_for_mix)
    val_schedule = _smooth_weighted_round_robin(val_queues, weights_for_mix)

    # Default language/exercise per dataset name (best-effort fallback).
    defaults = {}
    for name, meta in per_ds.items():
        root = meta["root"]
        dataset_name = name
        if isinstance(meta.get("manifest"), dict):
            dataset_name = str(meta["manifest"].get("dataset") or name)
        # coarse buckets so we can still do `--by-exercise` after merge.
        exercise = "unknown"
        if "synth" in name.lower():
            exercise = "synth"
        elif "opencode" in name.lower() or "code" in name.lower():
            exercise = "code"
        elif "math" in name.lower():
            exercise = "math"
        elif "oasst" in name.lower():
            exercise = "chat"
        defaults[name] = {
            "dataset_name": dataset_name,
            "default_language": "en",
            "default_exercise": exercise,
        }

    stats = OutputStats(input_shards_total=len(train_schedule) + len(val_schedule))
    stats.input_rows_total = int(sum(_shard_num_rows(p) for _, p in train_schedule) + sum(_shard_num_rows(p) for _, p in val_schedule))

    train_writer = ShardedParquetWriter(output_dir / "train", schema, rows_per_shard=args.rows_per_shard)
    val_writer = ShardedParquetWriter(output_dir / "validation", schema, rows_per_shard=args.rows_per_shard)

    tqdm = _require("tqdm").tqdm

    for split, schedule, writer in (
        ("train", train_schedule, train_writer),
        ("validation", val_schedule, val_writer),
    ):
        split_total_rows = int(sum(_shard_num_rows(p) for _, p in schedule))
        with tqdm(
            total=split_total_rows,
            unit="row",
            dynamic_ncols=True,
            desc=f"merge:{split}",
        ) as pbar:
            for name, shard in schedule:
                pbar.set_postfix_str(f"{name}:{shard.name}")
                dflt = defaults[name]
                dataset_name = dflt["dataset_name"]
                for rows in _iter_row_batches_from_shard(shard):
                    for row in rows:
                        source_norm = _normalize_source(
                            dataset_name,
                            row.get("source"),
                            default_language=dflt["default_language"],
                            default_exercise=dflt["default_exercise"],
                        )
                        out_row = {
                            "id": row.get("id"),
                            "source": source_norm,
                            "input_ids": row.get("input_ids"),
                            "labels": row.get("labels"),
                            "attention_mask": row.get("attention_mask"),
                            "spans": row.get("spans"),
                        }
                        writer.write(out_row)
                        stats.rows_written += 1
                        if split == "train":
                            stats.train_rows += 1
                        else:
                            stats.validation_rows += 1
                    pbar.update(len(rows))

    train_writer.close()
    val_writer.close()

    config = {
        "inputs": [{"name": name, "path": str(root)} for name, root in inputs],
        "rows_per_shard": int(args.rows_per_shard),
        "seed": int(args.seed),
        "weight_by": str(args.weight_by),
        "shuffle_within_dataset": bool(args.shuffle_within_dataset),
    }

    output_manifest = {
        "dataset": "phase1_mix",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config": config,
        "config_hash": _sha256_json(config),
        "inputs": {
            name: {
                "path": str(meta["root"]),
                "dataset": (meta["manifest"] or {}).get("dataset") if isinstance(meta.get("manifest"), dict) else None,
                "estimated": {
                    "rows": _estimate_totals_from_manifest(meta.get("manifest"))[0],
                    "total_tokens": _estimate_totals_from_manifest(meta.get("manifest"))[1],
                    "total_assistant_tokens": _estimate_totals_from_manifest(meta.get("manifest"))[2],
                },
                "weight": float(meta["weights"][args.weight_by]),
                "num_train_shards": len(meta["train"]),
                "num_validation_shards": len(meta["validation"]),
            }
            for name, meta in per_ds.items()
        },
        "output": dataclasses.asdict(stats),
        "interleaving": {
            "train_schedule_shards": [{"dataset": name, "shard": str(path)} for name, path in train_schedule],
            "validation_schedule_shards": [{"dataset": name, "shard": str(path)} for name, path in val_schedule],
        },
    }

    (output_dir / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote merged dataset to {output_dir}", file=os.sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
