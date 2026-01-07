#!/usr/bin/env python3
"""Materialize a bounded FineWeb-Edu slice into local JSONL shards + manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vpmoe" / "Megatron-vpmoe"))
sys.path.insert(0, str(REPO_ROOT))

from megatron.core.tokenizers.text.libraries.o200k_harmony_tokenizer import (
    O200kHarmonyTokenizer,
    O200K_HARMONY_PADDED_VOCAB_SIZE,
)


@dataclass(frozen=True)
class ShardMeta:
    path: str
    sha256: str
    bytes: int
    rows: int
    tokens: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Materialize a bounded FineWeb-Edu slice into local JSONL shards."
    )
    ap.add_argument("--repo-id", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", default=None, help="HF commit hash (default: latest).")
    ap.add_argument("--output-root", default="/data/corpora/fineweb_edu")
    ap.add_argument("--max-tokens", type=int, default=200_000_000)
    ap.add_argument("--max-docs", type=int, default=-1)
    ap.add_argument("--max-rows-per-shard", type=int, default=50_000)
    ap.add_argument("--tokenizer-model", default="data/tokenizer/o200k_base.tiktoken")
    ap.add_argument("--english-only", action="store_true")
    ap.add_argument("--text-field", default=None)
    ap.add_argument("--doc-id-field", default=None)
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_text(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    return value


def resolve_revision(repo_id: str, revision: Optional[str]) -> str:
    if revision:
        return revision
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id)
    if not info.sha:
        raise SystemExit(f"Unable to resolve dataset revision for {repo_id}")
    return info.sha


def detect_fields(sample: Dict[str, object]) -> Tuple[str, Optional[str]]:
    text_field = None
    for candidate in ("text", "content"):
        if candidate in sample:
            text_field = candidate
            break
    if text_field is None:
        raise SystemExit(f"Unable to find a text field in sample keys: {sorted(sample)}")

    lang_field = None
    for candidate in ("language", "lang", "language_code"):
        if candidate in sample:
            lang_field = candidate
            break
    return text_field, lang_field


def detect_doc_id_field(sample: Dict[str, object]) -> Optional[str]:
    for candidate in ("doc_id", "document_id", "id", "uuid"):
        if candidate in sample:
            return candidate
    return None


def normalize_text(text: str) -> Tuple[str, bool]:
    if "\x00" in text:
        return text.replace("\x00", ""), True
    return text, False


def main() -> None:
    args = parse_args()

    tokenizer_model = Path(args.tokenizer_model)
    if not tokenizer_model.exists():
        raise SystemExit(f"Tokenizer model not found: {tokenizer_model}")
    os.environ.setdefault("TIKTOKEN_ENCODINGS_BASE", str(tokenizer_model.parent))
    tokenizer = O200kHarmonyTokenizer(str(tokenizer_model))
    if tokenizer.vocab_size != O200K_HARMONY_PADDED_VOCAB_SIZE:
        raise SystemExit(
            "Unexpected tokenizer vocab size. "
            f"got={tokenizer.vocab_size} expected={O200K_HARMONY_PADDED_VOCAB_SIZE}"
        )

    revision = resolve_revision(args.repo_id, args.revision)
    output_root = Path(args.output_root) / revision
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"fineweb_edu_{revision}.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"Manifest already exists: {manifest_path} (use --overwrite)")

    from datasets import load_dataset

    ds = load_dataset(
        args.repo_id,
        split=args.split,
        streaming=True,
        revision=revision,
    )

    iterator = iter(ds)
    first = next(iterator, None)
    if first is None:
        raise SystemExit("Dataset is empty.")

    text_field = args.text_field
    lang_field = None
    if text_field is None:
        text_field, lang_field = detect_fields(first)
    if args.english_only and lang_field is None:
        # Cannot apply an English filter without a language field.
        args.english_only = False

    doc_id_field = args.doc_id_field or detect_doc_id_field(first)

    def row_iter():
        yield first
        for row in iterator:
            yield row

    shard_index = 0
    shard_rows = 0
    shard_tokens = 0
    shard_path = output_root / f"shard_{shard_index:05d}.jsonl"
    shard_handle = shard_path.open("w", encoding="utf-8")

    total_rows = 0
    total_tokens = 0
    total_docs = 0
    nul_replacements = 0
    shards: list[ShardMeta] = []
    started = time.time()

    def finalize_shard(path: Path, rows: int, tokens: int):
        if rows == 0:
            return None
        sha = sha256_file(path)
        return ShardMeta(
            path=str(path),
            sha256=sha,
            bytes=path.stat().st_size,
            rows=rows,
            tokens=tokens,
        )

    for idx, row in enumerate(row_iter()):
        if args.max_docs > 0 and total_docs >= args.max_docs:
            break
        if total_tokens >= args.max_tokens:
            break

        text = safe_text(row.get(text_field))
        if not text.strip():
            continue

        if args.english_only and lang_field:
            lang_value = safe_text(row.get(lang_field)).lower()
            if lang_value not in ("en", "eng", "english"):
                continue

        text, had_nul = normalize_text(text)
        if had_nul:
            nul_replacements += 1

        token_ids = tokenizer.text_to_ids(text)
        token_count = len(token_ids) + 1  # include EOD for budgeting

        if total_tokens + token_count > args.max_tokens and total_tokens > 0:
            break

        if doc_id_field:
            doc_id = safe_text(row.get(doc_id_field))
        else:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            doc_id = f"{idx}_{digest}"

        record = {"doc_id": doc_id, "text": text}
        shard_handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        total_rows += 1
        total_tokens += token_count
        total_docs += 1
        shard_rows += 1
        shard_tokens += token_count

        if shard_rows >= args.max_rows_per_shard:
            shard_handle.close()
            meta = finalize_shard(shard_path, shard_rows, shard_tokens)
            if meta:
                shards.append(meta)
            shard_index += 1
            shard_rows = 0
            shard_tokens = 0
            shard_path = output_root / f"shard_{shard_index:05d}.jsonl"
            shard_handle = shard_path.open("w", encoding="utf-8")

    shard_handle.close()
    meta = finalize_shard(shard_path, shard_rows, shard_tokens)
    if meta:
        shards.append(meta)

    elapsed = time.time() - started
    manifest = {
        "id": f"fineweb_edu_{revision}",
        "repo_id": args.repo_id,
        "split": args.split,
        "revision": revision,
        "output_root": str(output_root),
        "text_field": text_field,
        "doc_id_field": doc_id_field,
        "english_only_requested": args.english_only,
        "english_filter_applied": args.english_only and lang_field is not None,
        "language_field": lang_field,
        "normalization": {"removed_nul": True},
        "tokenizer_model": str(tokenizer_model),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "max_tokens_target": args.max_tokens,
        "max_docs": args.max_docs,
        "rows": total_rows,
        "docs": total_docs,
        "tokens": total_tokens,
        "nul_replacements": nul_replacements,
        "elapsed_sec": elapsed,
        "shards": [s.__dict__ for s in shards],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote shards under: {output_root}")
    print(f"Tokens: {total_tokens:,} Rows: {total_rows:,} Docs: {total_docs:,}")


if __name__ == "__main__":
    main()
