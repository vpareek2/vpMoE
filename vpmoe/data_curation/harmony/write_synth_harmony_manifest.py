#!/usr/bin/env python3
"""Write a deterministic manifest for a SYNTH→Harmony dataset directory.

This is intentionally lightweight: it records file inventory, row counts, and the
exact scaffold (system/developer) used during conversion so training can log it
as provenance later.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq


@dataclass(frozen=True)
class ShardInfo:
    path: str
    bytes: int
    rows: int
    row_groups: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a manifest.json for a directory of SYNTH→Harmony parquet shards.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/synth_harmony_full"),
        help="Directory containing synth_harmony_shard_*.parquet and metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output manifest path (default: <dataset-dir>/manifest.json).",
    )
    parser.add_argument(
        "--glob",
        default="synth_harmony_shard_*.parquet",
        help="Glob for shard files relative to --dataset-dir.",
    )
    parser.add_argument(
        "--schema-version",
        default="synth_harmony_v1",
        help="Schema/version identifier for this dataset layout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifest.json if present.",
    )
    return parser.parse_args()


def try_git_sha(root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out if out else None
    except Exception:
        return None


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_shards(dataset_dir: Path, pattern: str) -> List[ShardInfo]:
    shards: List[ShardInfo] = []
    for path in sorted(dataset_dir.glob(pattern)):
        pf = pq.ParquetFile(path)
        stat = path.stat()
        shards.append(
            ShardInfo(
                path=str(path),
                bytes=stat.st_size,
                rows=pf.metadata.num_rows,
                row_groups=pf.num_row_groups,
            )
        )
    return shards


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset dir not found: {dataset_dir}")

    output = args.output or (dataset_dir / "manifest.json")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing manifest: {output} (pass --overwrite)")

    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing conversion metadata.json: {metadata_path}")

    shards = collect_shards(dataset_dir, args.glob)
    if not shards:
        raise SystemExit(f"No shards found in {dataset_dir} matching {args.glob!r}")

    total_rows = sum(s.rows for s in shards)
    total_bytes = sum(s.bytes for s in shards)

    conversion_meta = read_json(metadata_path)

    # Compute a stable schema fingerprint (string form) from the first shard.
    first_pf = pq.ParquetFile(Path(shards[0].path))
    schema_str = str(first_pf.schema_arrow)

    manifest: Dict[str, Any] = {
        "schema_version": args.schema_version,
        "created_unix_s": int(time.time()),
        "git_sha": try_git_sha(Path.cwd()),
        "dataset_dir": str(dataset_dir),
        "conversion": {
            "dataset": conversion_meta.get("dataset"),
            "split": conversion_meta.get("split"),
            "streaming": conversion_meta.get("streaming"),
            "num_shards": conversion_meta.get("num_shards"),
            "metadata_fields": conversion_meta.get("metadata_fields"),
            "system_message": conversion_meta.get("system_message"),
            "developer_message": conversion_meta.get("developer_message"),
        },
        "storage": {
            "format": "parquet",
            "glob": args.glob,
            "schema_arrow": schema_str,
            "total_rows": total_rows,
            "total_bytes": total_bytes,
            "shards": [s.__dict__ for s in shards],
        },
    }

    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {output}")


if __name__ == "__main__":
    main()

