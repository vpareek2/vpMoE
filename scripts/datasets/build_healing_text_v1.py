#!/usr/bin/env python3
"""Build Megatron IndexedDataset for healing_text_v1 from JSONL shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
import torch

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
from megatron.core.tokenizers.text.libraries.o200k_harmony_tokenizer import (
    O200kHarmonyTokenizer,
    O200K_HARMONY_PADDED_VOCAB_SIZE,
)


TOKENS_SUFFIX = "_tokens"


@dataclass(frozen=True)
class SplitStats:
    rows: int
    tokens: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build Megatron IndexedDataset for healing_text_v1."
    )
    ap.add_argument(
        "--input-manifest",
        type=Path,
        required=True,
        help="Manifest produced by materialize_fineweb_edu.py",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/megatron/healing_text_v1"),
    )
    ap.add_argument(
        "--tokenizer-model",
        type=Path,
        default=Path("data/tokenizer/o200k_base.tiktoken"),
    )
    ap.add_argument(
        "--valid-fraction",
        type=float,
        default=0.10,
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=-1,
        help="Optional cap on total tokens processed (<=0 means no cap).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
    )
    return ap.parse_args()


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


def select_split(doc_id: str, valid_threshold: int) -> str:
    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return "valid" if value < valid_threshold else "train"


def make_builders(output_dir: Path) -> Dict[str, IndexedDatasetBuilder]:
    builders: Dict[str, IndexedDatasetBuilder] = {}
    for split in ("train", "valid"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        builders[split] = IndexedDatasetBuilder(
            str(split_dir / f"healing_text_v1{TOKENS_SUFFIX}.bin"),
            dtype=np.int32,
        )
    return builders


def finalize_builders(output_dir: Path, builders: Dict[str, IndexedDatasetBuilder]) -> None:
    for split, builder in builders.items():
        split_dir = output_dir / split
        builder.finalize(str(split_dir / f"healing_text_v1{TOKENS_SUFFIX}.idx"))


def main() -> None:
    args = parse_args()
    if not args.input_manifest.exists():
        raise SystemExit(f"Missing input manifest: {args.input_manifest}")

    ensure_output_dir(args.output_dir, args.overwrite)

    if args.tokenizer_model.name != "o200k_base.tiktoken":
        raise SystemExit("Tokenizer model must be named o200k_base.tiktoken.")
    if not args.tokenizer_model.exists():
        raise SystemExit(f"Tokenizer model not found: {args.tokenizer_model}")
    os.environ.setdefault("TIKTOKEN_ENCODINGS_BASE", str(args.tokenizer_model.parent))

    tokenizer = O200kHarmonyTokenizer(str(args.tokenizer_model))
    if tokenizer.vocab_size != O200K_HARMONY_PADDED_VOCAB_SIZE:
        raise SystemExit(
            "Unexpected tokenizer vocab size. "
            f"got={tokenizer.vocab_size} expected={O200K_HARMONY_PADDED_VOCAB_SIZE}"
        )

    manifest = json.loads(args.input_manifest.read_text())
    shard_paths = [Path(s["path"]) for s in manifest.get("shards", [])]
    if not shard_paths:
        raise SystemExit("Input manifest has no shards.")

    valid_threshold = int(Decimal(str(args.valid_fraction)) * (1 << 64))
    builders = make_builders(args.output_dir)

    split_stats = {
        "train": SplitStats(rows=0, tokens=0),
        "valid": SplitStats(rows=0, tokens=0),
    }
    total_tokens = 0
    started = time.time()

    eod = tokenizer.eod

    for shard_path in shard_paths:
        if not shard_path.exists():
            raise SystemExit(f"Missing shard: {shard_path}")
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if args.max_tokens > 0 and total_tokens >= args.max_tokens:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                doc_id = str(obj.get("doc_id", ""))
                text = str(obj.get("text", ""))
                if not doc_id or not text.strip():
                    continue

                tokens = tokenizer.text_to_ids(text)
                tokens.append(eod)

                split = select_split(doc_id, valid_threshold)
                builders[split].add_item(torch.tensor(tokens, dtype=torch.int32))
                builders[split].end_document()

                split_stats[split] = SplitStats(
                    rows=split_stats[split].rows + 1,
                    tokens=split_stats[split].tokens + len(tokens),
                )
                total_tokens += len(tokens)

            if args.max_tokens > 0 and total_tokens >= args.max_tokens:
                break

    finalize_builders(args.output_dir, builders)

    elapsed = time.time() - started
    output_manifest = {
        "id": "healing_text_v1",
        "input_manifest": str(args.input_manifest),
        "input_manifest_sha256": sha256_file(args.input_manifest),
        "tokenizer_model": str(args.tokenizer_model),
        "tokenizer_sha256": sha256_file(args.tokenizer_model),
        "valid_fraction": args.valid_fraction,
        "git_sha": try_git_sha(REPO_ROOT),
        "elapsed_sec": elapsed,
        "splits": {
            "train": {"rows": split_stats["train"].rows, "tokens": split_stats["train"].tokens},
            "valid": {"rows": split_stats["valid"].rows, "tokens": split_stats["valid"].tokens},
        },
    }
    output_manifest["config_hash"] = sha256_json(output_manifest)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=False)
    )
    print(f"Wrote output manifest: {args.output_dir / 'manifest.json'}")
    print(
        f"train rows={split_stats['train'].rows:,} tokens={split_stats['train'].tokens:,} | "
        f"valid rows={split_stats['valid'].rows:,} tokens={split_stats['valid'].tokens:,}"
    )


if __name__ == "__main__":
    main()
