#!/usr/bin/env python3
"""
Print a few human-readable samples from a DistillKit-ready Parquet dataset.

This is intended as a quick sanity check: read prompt / analysis / final text and
spot obvious formatting or quality issues.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


_O200K_BASE_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"


def _require(name: str):
    try:
        import importlib

        return importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f"Missing required module '{name}': {exc}") from exc


def _resolve_split_dir(input_dir: Path, split: str) -> Path:
    if (input_dir / "train").exists() and (input_dir / "validation").exists():
        return input_dir / split
    return input_dir


def _maybe_set_tiktoken_env() -> None:
    # Prefer the container convention first.
    if "TIKTOKEN_ENCODINGS_BASE" not in os.environ:
        for candidate in [
            Path("/data/hf_cache/tiktoken-encodings"),
            Path("data/hf_cache/tiktoken-encodings"),
        ]:
            if candidate.exists():
                os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(candidate.resolve())
                break

    # tiktoken itself uses a separate cache keyed by the sha1 of the URL/path.
    # When network access is restricted, we want o200k_base to be resolvable
    # purely from disk.
    if "TIKTOKEN_CACHE_DIR" not in os.environ:
        for candidate in [
            Path("/data/hf_cache/tiktoken-encodings"),
            Path("data/hf_cache/tiktoken-encodings"),
        ]:
            if candidate.exists():
                os.environ["TIKTOKEN_CACHE_DIR"] = str(candidate.resolve())
                break

    cache_dir = Path(os.environ.get("TIKTOKEN_CACHE_DIR", "")).expanduser()
    encodings_base = Path(os.environ.get("TIKTOKEN_ENCODINGS_BASE", "")).expanduser()
    if not cache_dir or not encodings_base:
        return

    vocab_path = encodings_base / "o200k_base.tiktoken"
    if not vocab_path.exists():
        return

    cache_key = hashlib.sha1(_O200K_BASE_URL.encode("utf-8")).hexdigest()
    cache_path = cache_dir / cache_key
    if cache_path.exists():
        return

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(vocab_path.read_bytes())
    except OSError:
        # Best-effort: if we can't write cache, tiktoken may still work if the
        # file is already cached elsewhere or if network access is available.
        pass


def _clip(text: str, limit: int, *, tail: bool = False) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if tail:
        return "…\n" + text[-limit:]
    return text[:limit] + "\n…"


def _between(text: str, start: str, end: str) -> Optional[str]:
    i = text.find(start)
    if i < 0:
        return None
    i += len(start)
    j = text.find(end, i)
    if j < 0:
        return None
    return text[i:j]


def _strip_harmony_wrappers(text: str, *, channel: Optional[str] = None) -> str:
    if not text:
        return ""
    if channel is None:
        return text
    marker = f"<|channel|>{channel}<|message|>"
    i = text.find(marker)
    if i >= 0:
        text = text[i + len(marker) :]
    for term in ["<|end|>", "<|return|>"]:
        j = text.find(term)
        if j >= 0:
            text = text[:j]
    return text.strip()


def _decode(enc, token_ids: List[int]) -> str:
    if not token_ids:
        return ""

    try:
        return enc.decode(token_ids)
    except KeyError:
        # Harmony tokenization can include special tokens that tiktoken's public
        # decoder does not know about. Best-effort decode by stitching known
        # pieces and rendering unknown IDs as placeholders.
        decoder = getattr(enc, "_decoder", None)
        if not isinstance(decoder, dict):
            return "<decode failed: unknown decoder mapping>"

        chunks: List[bytes] = []
        for tid in token_ids:
            piece = decoder.get(int(tid))
            if piece is None:
                chunks.append(f"<|tok:{tid}|>".encode("utf-8"))
            elif isinstance(piece, bytes):
                chunks.append(piece)
            else:
                chunks.append(str(piece).encode("utf-8"))
        return b"".join(chunks).decode("utf-8", "replace")
    except Exception as exc:
        return f"<decode failed: {type(exc).__name__}: {exc}>"


def _maybe_load_hf_tokenizer(tokenizer_id: Optional[str], trust_remote_code: bool):
    if not tokenizer_id:
        return None
    transformers = _require("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=trust_remote_code
        )
    except Exception as exc:
        raise SystemExit(f"Failed to load tokenizer '{tokenizer_id}': {exc}") from exc


def _sample_row(dataset, rng: random.Random, batch_size: int) -> Optional[Dict[str, Any]]:
    fragments = list(dataset.get_fragments())
    if not fragments:
        return None
    frag = rng.choice(fragments)

    ds = _require("pyarrow.dataset")
    scanner = ds.Scanner.from_fragment(
        frag,
        columns=["id", "source", "input_ids", "spans"],
        batch_size=batch_size,
    )
    for batch in scanner.to_batches():
        d = batch.to_pydict()
        n = len(d.get("input_ids", []))
        if n == 0:
            continue
        i = rng.randrange(n)
        return {k: d[k][i] for k in d.keys()}
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print random samples from a distill dataset.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"])
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default="",
        help="Optional HF tokenizer repo id to decode with (e.g. openai/gpt-oss-20b). "
        "If omitted, decodes with tiktoken o200k_harmony.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-special-tokens", action="store_true")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print decoded Harmony markup as-is (includes <|start|>, <|channel|>, etc.).",
    )
    parser.add_argument(
        "--show-system",
        action="store_true",
        help="When not using --raw, also print the system message content.",
    )
    parser.add_argument("--prompt-tail-chars", type=int, default=2000)
    parser.add_argument("--analysis-chars", type=int, default=1500)
    parser.add_argument("--final-chars", type=int, default=1500)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rng = random.Random(args.seed)

    ds = _require("pyarrow.dataset")
    input_dir = Path(args.input_dir).expanduser().resolve()
    split_dir = _resolve_split_dir(input_dir, args.split)
    if not split_dir.exists():
        raise SystemExit(f"Split directory not found: {split_dir}")

    dataset = ds.dataset(str(split_dir), format="parquet")

    tokenizer = _maybe_load_hf_tokenizer(args.tokenizer_id, args.trust_remote_code)
    enc = None
    if tokenizer is None:
        _maybe_set_tiktoken_env()
        tiktoken = _require("tiktoken")
        # Distill datasets in this repo are tokenized using Harmony, which adds
        # special tokens like <|start|> (200006) and <|channel|> (200005).
        # Those IDs are *not* decodable with plain o200k_base.
        enc = tiktoken.get_encoding("o200k_harmony")

    for _ in range(max(0, args.num_samples)):
        row = _sample_row(dataset, rng, batch_size=args.batch_size)
        if row is None:
            raise SystemExit("No rows found.")

        token_ids = row["input_ids"]
        spans = row["spans"]
        start = int(spans["assistant_token_start"])
        analysis = int(spans["analysis_token_count"])
        final = int(spans["final_token_count"])

        if tokenizer is not None:
            prompt_txt = tokenizer.decode(
                token_ids[:start], skip_special_tokens=args.skip_special_tokens
            )
            analysis_txt = tokenizer.decode(
                token_ids[start : start + analysis], skip_special_tokens=args.skip_special_tokens
            )
            final_txt = tokenizer.decode(
                token_ids[start + analysis : start + analysis + final],
                skip_special_tokens=args.skip_special_tokens,
            )
        else:
            prompt_txt = _decode(enc, token_ids[:start])
            analysis_txt = _decode(enc, token_ids[start : start + analysis])
            final_txt = _decode(enc, token_ids[start + analysis : start + analysis + final])

        print("=" * 120)
        print(row["id"])
        print(row["source"])
        print(f"tokens: total={len(token_ids)} prompt={start} analysis={analysis} final={final}")
        if args.raw:
            print("\nPROMPT (tail):\n", _clip(prompt_txt, args.prompt_tail_chars, tail=True))
            print("\nANALYSIS:\n", _clip(analysis_txt, args.analysis_chars))
            print("\nFINAL:\n", _clip(final_txt, args.final_chars))
            continue

        system_txt = _between(prompt_txt, "<|start|>system<|message|>", "<|end|>") or ""
        user_txt = _between(prompt_txt, "<|start|>user<|message|>", "<|end|>") or prompt_txt

        analysis_clean = _strip_harmony_wrappers(analysis_txt, channel="analysis")
        final_clean = _strip_harmony_wrappers(final_txt, channel="final")

        if args.show_system and system_txt.strip():
            print("\nSYSTEM:\n", _clip(system_txt.strip(), 600))
        print("\nPROMPT (tail):\n", _clip(user_txt.strip(), args.prompt_tail_chars, tail=True))
        print("\nANALYSIS:\n", _clip(analysis_clean, args.analysis_chars))
        print("\nFINAL:\n", _clip(final_clean, args.final_chars))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
