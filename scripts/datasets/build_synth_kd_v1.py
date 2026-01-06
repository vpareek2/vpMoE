#!/usr/bin/env python3
"""Build SYNTH KD v1 Megatron datasets from Harmony parquet shards."""

from __future__ import annotations

import argparse
import os
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vpmoe" / "Megatron-vpmoe"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
from megatron.core.tokenizers.text.libraries.o200k_harmony_tokenizer import (
    O200kHarmonyTokenizer,
    O200K_HARMONY_PADDED_VOCAB_SIZE,
)
from vpmoe.harmony.renderer import render_messages


TOKENS_SUFFIX = "_tokens"
LOSSMASK_SUFFIX = "_lossmask"
SPAN_SUFFIX = "_span"


@dataclass(frozen=True)
class ShardStats:
    name: str
    rows: int
    tokens: int
    paths: Dict[str, str]
    bytes: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Megatron IndexedDatasets for SYNTH KD (Harmony format).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/synth_harmony"),
        help="Directory containing synth_harmony_shard_*.parquet and manifest.json.",
    )
    parser.add_argument(
        "--input-glob",
        default="synth_harmony_shard_*.parquet",
        help="Glob for Harmony parquet shards relative to --input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/megatron/synth_kd_v1"),
        help="Directory to write Megatron shards into.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        required=True,
        help="Path to the local o200k_base.tiktoken file.",
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.001,
        help="Deterministic validation fraction based on synth_id hash.",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=1024,
        help="Rows per parquet batch.",
    )
    parser.add_argument(
        "--max-rows-per-shard",
        type=int,
        default=-1,
        help="Max rows to process per shard (set -1 for full shard).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="Max total rows across all shards (set -1 for full run).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10000,
        help="Log progress every N rows (0 disables).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output manifest and shards.",
    )
    return parser.parse_args()


def try_git_sha(root: Path) -> Optional[str]:
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out if out else None
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_size(path: Path) -> int:
    return path.stat().st_size


def select_split(synth_id: str, valid_threshold: int) -> str:
    digest = hashlib.blake2b(synth_id.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return "valid" if value < valid_threshold else "train"


def parse_shard_name(path: Path) -> str:
    match = re.search(r"shard_(\d+)", path.name)
    if not match:
        raise ValueError(f"Unable to parse shard index from {path.name}")
    return f"shard_{int(match.group(1)):02d}"


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return
    existing = list(output_dir.rglob("*.bin")) + list(output_dir.rglob("*.idx"))
    manifest = output_dir / "manifest.json"
    if existing or manifest.exists():
        raise SystemExit(
            f"Refusing to overwrite existing outputs in {output_dir} (pass --overwrite)."
        )


def make_builders(output_dir: Path, shard: str) -> Dict[str, Dict[str, IndexedDatasetBuilder]]:
    builders: Dict[str, Dict[str, IndexedDatasetBuilder]] = {}
    for split in ("train", "valid"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        builders[split] = {
            "tokens": IndexedDatasetBuilder(
                str(split_dir / f"{shard}{TOKENS_SUFFIX}.bin"), dtype=np.int32
            ),
            "lossmask": IndexedDatasetBuilder(
                str(split_dir / f"{shard}{LOSSMASK_SUFFIX}.bin"), dtype=np.uint8
            ),
            "span": IndexedDatasetBuilder(
                str(split_dir / f"{shard}{SPAN_SUFFIX}.bin"), dtype=np.uint8
            ),
        }
    return builders


def finalize_builders(
    output_dir: Path, shard: str, builders: Dict[str, Dict[str, IndexedDatasetBuilder]]
) -> None:
    for split, split_builders in builders.items():
        split_dir = output_dir / split
        split_builders["tokens"].finalize(
            str(split_dir / f"{shard}{TOKENS_SUFFIX}.idx")
        )
        split_builders["lossmask"].finalize(
            str(split_dir / f"{shard}{LOSSMASK_SUFFIX}.idx")
        )
        split_builders["span"].finalize(
            str(split_dir / f"{shard}{SPAN_SUFFIX}.idx")
        )


def shift_label_alignment(values: List[int]) -> List[int]:
    if not values:
        return values
    return values[1:] + [0]


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    input_manifest = input_dir / "manifest.json"
    if not input_manifest.exists():
        raise SystemExit(f"Missing input manifest: {input_manifest}")

    ensure_output_dir(args.output_dir, args.overwrite)

    shard_paths = sorted(input_dir.glob(args.input_glob))
    if not shard_paths:
        raise SystemExit(f"No shards found in {input_dir} matching {args.input_glob!r}")

    if args.tokenizer_model.name != "o200k_base.tiktoken":
        raise SystemExit(
            "Tokenizer model must be named o200k_base.tiktoken for openai-harmony."
        )
    if not args.tokenizer_model.exists():
        raise SystemExit(f"Tokenizer model not found: {args.tokenizer_model}")
    os.environ.setdefault("TIKTOKEN_ENCODINGS_BASE", str(args.tokenizer_model.parent))

    tokenizer = O200kHarmonyTokenizer(str(args.tokenizer_model))
    if tokenizer.vocab_size != O200K_HARMONY_PADDED_VOCAB_SIZE:
        raise SystemExit(
            "Unexpected tokenizer vocab size. "
            f"got={tokenizer.vocab_size} expected={O200K_HARMONY_PADDED_VOCAB_SIZE}"
        )
    if args.valid_fraction < 0 or args.valid_fraction > 1:
        raise SystemExit("--valid-fraction must be in [0, 1].")

    valid_threshold = int(Decimal(str(args.valid_fraction)) * (1 << 64))
    started = time.time()
    total_rows = 0

    split_stats: Dict[str, Dict[str, object]] = {
        "train": {"rows": 0, "tokens": 0, "shards": []},
        "valid": {"rows": 0, "tokens": 0, "shards": []},
    }

    for shard_path in shard_paths:
        shard_name = parse_shard_name(shard_path)
        builders = make_builders(args.output_dir, shard_name)
        shard_counts = {"train": {"rows": 0, "tokens": 0}, "valid": {"rows": 0, "tokens": 0}}

        pf = pq.ParquetFile(shard_path)
        rows_in_shard = 0
        stop_all = False
        stop_shard = False

        for batch in pf.iter_batches(
            batch_size=args.parquet_batch_size,
            columns=["messages_json", "synth_id"],
        ):
            messages_list = batch.column("messages_json").to_pylist()
            synth_ids = batch.column("synth_id").to_pylist()
            for messages_json, synth_id in zip(messages_list, synth_ids):
                if args.max_rows_per_shard >= 0 and rows_in_shard >= args.max_rows_per_shard:
                    stop_shard = True
                    break
                if args.max_rows >= 0 and total_rows >= args.max_rows:
                    stop_all = True
                    stop_shard = True
                    break
                if synth_id is None or messages_json is None:
                    raise ValueError("Missing synth_id or messages_json in input row.")

                payload = json.loads(messages_json)
                messages = payload.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("messages_json must contain a list field 'messages'.")

                tokens, token_mask, token_span = render_messages(messages)
                if not (len(tokens) == len(token_mask) == len(token_span)):
                    raise ValueError("Token/mask/span lengths do not match.")

                loss_mask = shift_label_alignment(token_mask)
                span_id = shift_label_alignment(token_span)

                split = select_split(str(synth_id), valid_threshold)
                builders[split]["tokens"].add_item(torch.tensor(tokens, dtype=torch.int64))
                builders[split]["lossmask"].add_item(torch.tensor(loss_mask, dtype=torch.uint8))
                builders[split]["span"].add_item(torch.tensor(span_id, dtype=torch.uint8))
                builders[split]["tokens"].end_document()
                builders[split]["lossmask"].end_document()
                builders[split]["span"].end_document()

                shard_counts[split]["rows"] += 1
                shard_counts[split]["tokens"] += len(tokens)

                rows_in_shard += 1
                total_rows += 1
                if args.log_every and total_rows % args.log_every == 0:
                    elapsed = time.time() - started
                    rows_per_sec = total_rows / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[progress] rows={total_rows} rows/sec={rows_per_sec:.1f} shard={shard_name}"
                    )

            if stop_shard:
                break

        finalize_builders(args.output_dir, shard_name, builders)

        for split in ("train", "valid"):
            split_rows = shard_counts[split]["rows"]
            split_tokens = shard_counts[split]["tokens"]
            split_stats[split]["rows"] += split_rows
            split_stats[split]["tokens"] += split_tokens
            split_root = args.output_dir / split
            tokens_prefix = split_root / f"{shard_name}{TOKENS_SUFFIX}"
            lossmask_prefix = split_root / f"{shard_name}{LOSSMASK_SUFFIX}"
            span_prefix = split_root / f"{shard_name}{SPAN_SUFFIX}"
            files = {
                "tokens_bin": tokens_prefix.with_suffix(".bin"),
                "tokens_idx": tokens_prefix.with_suffix(".idx"),
                "lossmask_bin": lossmask_prefix.with_suffix(".bin"),
                "lossmask_idx": lossmask_prefix.with_suffix(".idx"),
                "span_bin": span_prefix.with_suffix(".bin"),
                "span_idx": span_prefix.with_suffix(".idx"),
            }
            if split_rows == 0:
                print(f"[info] {split} {shard_name}: 0 rows, removing empty shard outputs.")
                for path in files.values():
                    if path.exists():
                        path.unlink()
                continue
            split_stats[split]["shards"].append(
                ShardStats(
                    name=shard_name,
                    rows=split_rows,
                    tokens=split_tokens,
                    paths={key: str(path) for key, path in files.items()},
                    bytes={key: file_size(path) for key, path in files.items()},
                ).__dict__
            )

        if stop_all:
            break

    config = {
        "valid_fraction": args.valid_fraction,
        "hash": "blake2b-8",
        "loss_mask_alignment": "labels_shifted_with_pad",
        "eod_token_id": tokenizer.eod,
        "roles_preserved": ["system", "developer", "user", "assistant"],
        "assistant_channels": ["analysis", "final"],
    }

    manifest = {
        "schema_version": "synth_kd_v1",
        "created_unix_s": int(time.time()),
        "git_sha": try_git_sha(REPO_ROOT),
        "config_hash": sha256_json(config),
        "input": {
            "dataset_dir": str(input_dir),
            "manifest_path": str(input_manifest),
            "manifest_sha256": sha256_file(input_manifest),
            "glob": args.input_glob,
        },
        "renderer": {
            "library": "openai-harmony",
            "version": importlib.metadata.version("openai-harmony"),
            "auto_drop_analysis": False,
            "roles_preserved": ["system", "developer", "user", "assistant"],
        },
        "tokenizer": {
            "type": "O200kHarmonyTokenizer",
            "model_path": str(args.tokenizer_model),
            "model_sha256": sha256_file(args.tokenizer_model),
            "vocab_size": O200K_HARMONY_PADDED_VOCAB_SIZE,
        },
        "config": config,
        "splits": split_stats,
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    elapsed = time.time() - started
    rows_per_sec = total_rows / elapsed if elapsed > 0 else 0.0
    print(
        f"Wrote {total_rows} rows in {elapsed:.1f}s ({rows_per_sec:.1f} rows/sec). "
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
