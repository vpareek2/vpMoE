#!/usr/bin/env python3
"""
Build a DistillKit-ready dataset from nvidia/OpenCodeInstruct Parquet shards.

This keeps the dataset minimal and deterministic:
- Prompt = `input`
- Assistant final = `output` (no analysis)
- Drops rows failing tests (average_test_score < 1.0 by default)
- Stops when a target token budget is reached
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import importlib
import json
import os
import random
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


_O200K_BASE_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"


def _require(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise SystemExit(f"Missing required module '{module_name}': {exc}") from exc


def _looks_like_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"}


def _default_hf_cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    if Path("/data").exists():
        return Path("/data/hf_cache")
    return Path("/tmp/hf_cache")


def _ensure_o200k_base_vocab(encodings_dir: Path) -> Path:
    encodings_dir.mkdir(parents=True, exist_ok=True)
    dest = encodings_dir / "o200k_base.tiktoken"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(".tiktoken.tmp")
    try:
        with urllib.request.urlopen(_O200K_BASE_URL, timeout=60) as resp, tmp.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dest


def _hash_to_unit(value: str, seed: int, salt: str) -> float:
    payload = f"{seed}:{salt}:{value}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _sanitize_for_harmony(text: str) -> str:
    if not text:
        return ""
    if "\x00" in text:
        text = text.replace("\x00", "")
    if _SURROGATE_RE.search(text):
        text = _SURROGATE_RE.sub("\uFFFD", text)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = text.encode("utf-8", "replace").decode("utf-8")
    return text


@dataclasses.dataclass
class RunningStats:
    count: int = 0
    total: int = 0
    max_val: int = 0
    min_val: Optional[int] = None

    def update(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.max_val = max(self.max_val, value)
        if self.min_val is None:
            self.min_val = value
        else:
            self.min_val = min(self.min_val, value)

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean(),
            "min": self.min_val,
            "max": self.max_val,
        }


@dataclasses.dataclass
class OutputStats:
    kept: int = 0
    dropped_missing: int = 0
    dropped_failed_tests: int = 0
    dropped_long: int = 0
    dropped_encoding_error: int = 0
    train: int = 0
    validation: int = 0
    total_tokens: int = 0
    total_assistant_tokens: int = 0


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
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        path = self.out_dir / f"{self.prefix}-{self.shard_index:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        self.total_rows += len(self.buffer)
        self.buffer.clear()
        self.shard_index += 1

    def close(self) -> None:
        self.flush()


def _iter_batches(dataset, columns: List[str], batch_size: int) -> Iterator[Dict[str, List[Any]]]:
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pydict()


def _load_encoding() -> Any:
    openai_harmony = _require("openai_harmony")
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _harmony_render_config_keep_analysis() -> Any:
    try:
        from openai_harmony import RenderConversationConfig

        try:
            return RenderConversationConfig(auto_drop_analysis=False)
        except TypeError:
            cfg = RenderConversationConfig()
            setattr(cfg, "auto_drop_analysis", False)
            return cfg
    except Exception:
        return None


def _render_for_completion(encoding, conversation, role, config) -> List[int]:
    fn = getattr(encoding, "render_conversation_for_completion", None)
    if fn is None:
        raise RuntimeError("openai_harmony encoding missing render_conversation_for_completion")
    try:
        return list(fn(conversation, role, config))
    except TypeError:
        return list(fn(conversation, role))


def _render_for_training(encoding, conversation, config) -> List[int]:
    fn = getattr(encoding, "render_conversation_for_training", None)
    if fn is not None:
        try:
            return list(fn(conversation, config))
        except TypeError:
            return list(fn(conversation))
    fn = getattr(encoding, "render_conversation", None)
    if fn is None:
        raise RuntimeError("openai_harmony encoding missing render_conversation")
    try:
        return list(fn(conversation, config))
    except TypeError:
        return list(fn(conversation))


def _make_system_content(level: str):
    from openai_harmony import SystemContent

    if not level:
        return SystemContent.new()
    for kwargs in ({"reasoning_level": level}, {"reasoning": level}):
        try:
            return SystemContent.new(**kwargs)
        except TypeError:
            pass

    system = SystemContent.new()
    for method_name in (
        "with_reasoning_level",
        "with_reasoning",
        "set_reasoning_level",
        "set_reasoning",
    ):
        fn = getattr(system, method_name, None)
        if fn is None:
            continue
        try:
            out = fn(level)
        except TypeError:
            try:
                out = fn(reasoning=level)
            except TypeError:
                out = None
        return out if out is not None else system

    for attr_name in ("reasoning_level", "reasoning"):
        if hasattr(system, attr_name):
            setattr(system, attr_name, level)
            return system

    return (
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n\n"
        f"Reasoning: {level}\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    )


def _render_tokens_for_example(
    encoding,
    *,
    prompt: str,
    answer: str,
    reasoning_level: str,
) -> Tuple[List[int], int, int, int]:
    from openai_harmony import Conversation, Message, Role

    system_message = Message.from_role_and_content(Role.SYSTEM, _make_system_content(reasoning_level))
    user_message = Message.from_role_and_content(Role.USER, prompt)

    base_conv = Conversation.from_messages([system_message, user_message])
    config = _harmony_render_config_keep_analysis()
    prompt_ids = _render_for_completion(encoding, base_conv, Role.ASSISTANT, config)
    assistant_start = len(prompt_ids)

    final_msg = Message.from_role_and_content(Role.ASSISTANT, answer).with_channel("final")
    full_conv = Conversation.from_messages([system_message, user_message, final_msg])
    full_ids = _render_for_training(encoding, full_conv, config)

    analysis_count = 0
    final_count = len(full_ids) - assistant_start
    if final_count < 0:
        raise ValueError("Computed negative final span length.")
    return full_ids, assistant_start, analysis_count, final_count


def _git_sha(repo_root: Path) -> Optional[str]:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return output.decode("utf-8").strip() or None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a DistillKit-ready OpenCodeInstruct dataset.")
    p.add_argument("--input-dir", type=str, required=True, help="Directory of parquet shards.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for parquet splits.")
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--train-frac", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--rows-per-shard", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument(
        "--reasoning-level",
        type=str,
        default="high",
        help="Harmony system reasoning level to embed.",
    )
    p.add_argument(
        "--target-total-tokens",
        type=int,
        default=40_000_000,
        help="Stop after reaching this many total tokens.",
    )
    p.add_argument(
        "--shuffle-shards",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle input shard order deterministically (seeded).",
    )
    p.add_argument(
        "--require-perfect-score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require average_test_score == 1.0 to keep a row.",
    )
    p.add_argument(
        "--tiktoken-encodings-base",
        type=str,
        default=None,
        help="Override TIKTOKEN_ENCODINGS_BASE (must be a local directory).",
    )
    p.add_argument(
        "--tiktoken-cache-dir",
        type=str,
        default=None,
        help="Override the tiktoken cache directory (TIKTOKEN_RS_CACHE_DIR).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    if args.tiktoken_encodings_base:
        if _looks_like_url(args.tiktoken_encodings_base):
            raise SystemExit("--tiktoken-encodings-base must be a local directory path.")
        os.environ["TIKTOKEN_ENCODINGS_BASE"] = args.tiktoken_encodings_base
    if args.tiktoken_cache_dir:
        os.environ["TIKTOKEN_RS_CACHE_DIR"] = args.tiktoken_cache_dir
    if "TIKTOKEN_RS_CACHE_DIR" not in os.environ:
        os.environ["TIKTOKEN_RS_CACHE_DIR"] = str(_default_hf_cache_root() / "tiktoken-rs-cache")
    if "TIKTOKEN_ENCODINGS_BASE" in os.environ and _looks_like_url(
        os.environ["TIKTOKEN_ENCODINGS_BASE"]
    ):
        print(
            "warning: TIKTOKEN_ENCODINGS_BASE looks like a URL; overriding to a local cache directory",
            file=sys.stderr,
        )
        os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(_default_hf_cache_root() / "tiktoken-encodings")
    if "TIKTOKEN_ENCODINGS_BASE" not in os.environ:
        os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(_default_hf_cache_root() / "tiktoken-encodings")

    cache_dir = Path(os.environ["TIKTOKEN_RS_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    encodings_dir = Path(os.environ["TIKTOKEN_ENCODINGS_BASE"])
    _ensure_o200k_base_vocab(encodings_dir)

    tqdm = _require("tqdm").tqdm
    pa = _require("pyarrow")
    ds = _require("pyarrow.dataset")

    parquet_files = sorted(str(p) for p in input_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet shards found under: {input_dir}")

    if args.shuffle_shards:
        rng = random.Random(args.seed)
        rng.shuffle(parquet_files)

    dataset = ds.dataset(parquet_files, format="parquet")
    required_columns = {"id", "input", "output", "domain", "average_test_score"}
    missing = required_columns - set(dataset.schema.names)
    if missing:
        raise SystemExit(f"Input dataset missing required columns: {sorted(missing)}")

    openai_harmony = _require("openai_harmony")
    encoding = _load_encoding()
    harmony_version = getattr(openai_harmony, "__version__", "unknown")

    schema = pa.schema(
        [
            ("id", pa.string()),
            (
                "source",
                pa.struct(
                    [
                        ("dataset", pa.string()),
                        ("opencode_id", pa.string()),
                        ("domain", pa.string()),
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

    train_writer = ShardedParquetWriter(
        output_dir / "train", schema, rows_per_shard=args.rows_per_shard
    )
    val_writer = ShardedParquetWriter(
        output_dir / "validation", schema, rows_per_shard=args.rows_per_shard
    )

    output_stats = OutputStats()
    prompt_stats = RunningStats()
    final_stats = RunningStats()
    total_stats = RunningStats()

    columns = ["id", "input", "output", "domain", "average_test_score"]
    pbar = tqdm(total=None, desc="processed", unit="row")
    token_budget = args.target_total_tokens

    for batch in _iter_batches(dataset, columns, args.batch_size):
        ids = batch.get("id", [])
        inputs = batch.get("input", [])
        outputs = batch.get("output", [])
        domains = batch.get("domain", [])
        scores = batch.get("average_test_score", [])

        for op_id, prompt, answer, domain, score in zip(
            ids, inputs, outputs, domains, scores
        ):
            pbar.update(1)
            if output_stats.total_tokens >= token_budget:
                break

            if op_id is None or prompt is None or answer is None:
                output_stats.dropped_missing += 1
                continue

            if args.require_perfect_score:
                try:
                    if score is None or float(score) < 0.999999:
                        output_stats.dropped_failed_tests += 1
                        continue
                except Exception:
                    output_stats.dropped_failed_tests += 1
                    continue

            prompt_text = _sanitize_for_harmony(str(prompt))
            answer_text = _sanitize_for_harmony(str(answer))
            domain_text = str(domain or "").strip().lower() or "unknown"
            op_id_text = str(op_id)

            try:
                input_ids, assistant_start, analysis_count, final_count = _render_tokens_for_example(
                    encoding,
                    prompt=prompt_text,
                    answer=answer_text,
                    reasoning_level=args.reasoning_level,
                )
            except Exception:
                output_stats.dropped_encoding_error += 1
                continue

            total_len = len(input_ids)
            if total_len > args.max_seq_len:
                output_stats.dropped_long += 1
                continue

            labels = [-100] * total_len
            for i in range(assistant_start, total_len):
                labels[i] = input_ids[i]

            attention_mask = [1] * total_len

            row = {
                "id": f"opencode:{op_id_text}",
                "source": {
                    "dataset": "nvidia/OpenCodeInstruct",
                    "opencode_id": op_id_text,
                    "domain": domain_text,
                },
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "spans": {
                    "assistant_token_start": assistant_start,
                    "analysis_token_count": analysis_count,
                    "final_token_count": final_count,
                },
            }

            split_draw = _hash_to_unit(op_id_text, args.seed, "split")
            if split_draw < (1.0 - args.train_frac):
                val_writer.write(row)
                output_stats.validation += 1
            else:
                train_writer.write(row)
                output_stats.train += 1

            output_stats.kept += 1
            output_stats.total_tokens += total_len
            output_stats.total_assistant_tokens += final_count
            prompt_stats.update(assistant_start)
            final_stats.update(final_count)
            total_stats.update(total_len)

        if output_stats.total_tokens >= token_budget:
            break

    pbar.close()

    train_writer.close()
    val_writer.close()

    config_payload = {
        "max_seq_len": args.max_seq_len,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "rows_per_shard": args.rows_per_shard,
        "reasoning_level": args.reasoning_level,
        "require_perfect_score": args.require_perfect_score,
        "target_total_tokens": args.target_total_tokens,
        "shuffle_shards": args.shuffle_shards,
    }
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode("utf-8")).hexdigest()

    git_sha = _git_sha(Path(__file__).resolve().parents[1])

    manifest = {
        "dataset": "nvidia/OpenCodeInstruct",
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "tokenizer": {
            "name": "o200k_harmony",
            "encoding": "HARMONY_GPT_OSS",
            "openai_harmony_version": harmony_version,
        },
        "config": config_payload,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "output": dataclasses.asdict(output_stats),
        "token_stats": {
            "prompt": prompt_stats.as_dict(),
            "final": final_stats.as_dict(),
            "total": total_stats.as_dict(),
        },
        "input_shards": {
            "input_dir": str(input_dir),
            "count": len(parquet_files),
            "shuffled": args.shuffle_shards,
            "order": [Path(p).name for p in parquet_files],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    shards_path = output_dir / "input_shards_order.txt"
    with shards_path.open("w", encoding="utf-8") as f:
        for name in manifest["input_shards"]["order"]:
            f.write(f"{name}\n")

    print(f"Wrote dataset to {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
