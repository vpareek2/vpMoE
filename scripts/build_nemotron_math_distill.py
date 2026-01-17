#!/usr/bin/env python3
"""
Build a DistillKit-ready dataset from nvidia/Nemotron-Math-v2 JSONL.

Goal for phase-1 (8k context):
- Use the dataset's existing trajectories as teacher-forced targets.
- Emit Harmony-formatted token IDs with an explicit system header that sets
  `Reasoning: high` (or another configured level).
- Mask loss to assistant tokens only (labels == -100 before assistant span).

We intentionally keep this script minimal and deterministic:
- Streaming JSONL reader (no HF datasets dependency required).
- Deterministic sampling via hashing (optional), deterministic train/val split.
- Optional drop of tool-using rows (phase-1 default).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import Counter
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
    dropped_tool: int = 0
    dropped_long: int = 0
    dropped_encoding_error: int = 0
    dropped_filtered: int = 0
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
        self.buffer.clear()
        self.shard_index += 1

    def close(self) -> None:
        self.flush()


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


def _load_encoding() -> Any:
    _require("openai_harmony")
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


def _extract_messages(obj: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extract (prompt, reasoning, final) from Nemotron-Math style rows.

    We accept:
    - First user message content as prompt
    - Last assistant message content as final
    - Optional assistant reasoning_content as reasoning
    """
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("missing messages")

    user_text: Optional[str] = None
    assistant_msg: Optional[Dict[str, Any]] = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user" and user_text is None:
            c = m.get("content")
            if isinstance(c, str):
                user_text = c
        if role == "assistant":
            assistant_msg = m

    if user_text is None or assistant_msg is None:
        raise ValueError("missing user/assistant")

    final = assistant_msg.get("content")
    if not isinstance(final, str):
        raise ValueError("assistant content not string")
    reasoning = assistant_msg.get("reasoning_content", "")
    if not isinstance(reasoning, str):
        reasoning = ""
    return user_text, reasoning, final


def _render_tokens_for_example(
    encoding,
    *,
    prompt: str,
    reasoning: str,
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

    reasoning = reasoning.strip()
    answer = answer.strip()

    base_messages = [system_message, user_message]

    analysis_ids: Optional[List[int]] = None
    analysis_msg = None
    if reasoning:
        analysis_msg = Message.from_role_and_content(Role.ASSISTANT, reasoning).with_channel("analysis")
        analysis_conv = Conversation.from_messages(base_messages + [analysis_msg])
        analysis_ids = _render_for_training(encoding, analysis_conv, config)

    final_msg = Message.from_role_and_content(Role.ASSISTANT, answer).with_channel("final")
    full_messages = base_messages + ([analysis_msg] if analysis_msg else []) + [final_msg]
    full_conv = Conversation.from_messages(full_messages)
    full_ids = _render_for_training(encoding, full_conv, config)

    if analysis_ids is None:
        analysis_count = 0
        final_count = len(full_ids) - assistant_start
    else:
        analysis_count = len(analysis_ids) - assistant_start
        final_count = len(full_ids) - len(analysis_ids)
    if analysis_count < 0 or final_count < 0:
        raise ValueError("Negative span lengths; Harmony render config likely dropped analysis.")

    return full_ids, assistant_start, analysis_count, final_count


def _iter_jsonl_files(paths: List[Path]) -> Iterator[Tuple[Path, int, Dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield path, lineno, obj


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a DistillKit-ready Nemotron-Math-v2 dataset.")
    p.add_argument("--input", action="append", required=True, help="Path to a JSONL file. Repeatable.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for parquet splits.")
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--train-frac", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--rows-per-shard", type=int, default=2000)
    p.add_argument("--reasoning-level", type=str, default="high")
    p.add_argument(
        "--target-total-tokens",
        type=int,
        default=55_000_000,
        help="Stop after reaching this many total tokens.",
    )
    p.add_argument(
        "--drop-tool-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows that specify a tool definition (phase-1 default).",
    )
    p.add_argument(
        "--accept-prob",
        type=float,
        default=1.0,
        help="Deterministic random accept probability based on uuid hashing (1.0 keeps all).",
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

    inputs = [Path(p).expanduser().resolve() for p in args.input]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit(f"Input file(s) not found: {missing}")

    output_dir = Path(args.output_dir).expanduser().resolve()

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
                        ("uuid", pa.string()),
                        ("subset", pa.string()),
                        ("data_source", pa.string()),
                        ("tool", pa.bool_()),
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

    out = OutputStats()
    prompt_stats = RunningStats()
    analysis_stats = RunningStats()
    final_stats = RunningStats()
    total_stats = RunningStats()
    dist_subset = Counter()

    pbar = tqdm(total=None, desc="processed", unit="row")
    token_budget = args.target_total_tokens

    for path, lineno, obj in _iter_jsonl_files(inputs):
        pbar.update(1)
        if out.total_tokens >= token_budget:
            break

        uuid = obj.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            out.dropped_missing += 1
            continue

        if args.accept_prob < 1.0:
            if _hash_to_unit(uuid, args.seed, "accept") >= args.accept_prob:
                out.dropped_filtered += 1
                continue

        tools_field = obj.get("tools")
        has_tool = bool(tools_field)
        if args.drop_tool_rows and has_tool:
            out.dropped_tool += 1
            continue

        try:
            prompt, reasoning, answer = _extract_messages(obj)
        except Exception:
            out.dropped_missing += 1
            continue

        prompt_text = _sanitize_for_harmony(prompt)
        reasoning_text = _sanitize_for_harmony(reasoning)
        answer_text = _sanitize_for_harmony(answer)

        subset = path.name
        data_source = obj.get("data_source")
        data_source_text = str(data_source or "").strip().lower() or "unknown"

        try:
            input_ids, assistant_start, analysis_count, final_count = _render_tokens_for_example(
                encoding,
                prompt=prompt_text,
                reasoning=reasoning_text,
                answer=answer_text,
                reasoning_level=args.reasoning_level,
            )
        except Exception:
            out.dropped_encoding_error += 1
            continue

        total_len = len(input_ids)
        if total_len > args.max_seq_len:
            out.dropped_long += 1
            continue

        labels = [-100] * total_len
        for i in range(assistant_start, total_len):
            labels[i] = input_ids[i]

        attention_mask = [1] * total_len

        row = {
            "id": f"nemotron_math:{uuid}",
            "source": {
                "dataset": "nvidia/Nemotron-Math-v2",
                "uuid": uuid,
                "subset": subset,
                "data_source": data_source_text,
                "tool": has_tool,
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

        split_draw = _hash_to_unit(uuid, args.seed, "split")
        if split_draw < (1.0 - args.train_frac):
            val_writer.write(row)
            out.validation += 1
        else:
            train_writer.write(row)
            out.train += 1

        out.kept += 1
        out.total_tokens += total_len
        out.total_assistant_tokens += (total_len - assistant_start)

        prompt_stats.update(assistant_start)
        analysis_stats.update(analysis_count)
        final_stats.update(final_count)
        total_stats.update(total_len)
        dist_subset[subset] += 1

    pbar.close()
    train_writer.close()
    val_writer.close()

    config_payload = {
        "max_seq_len": args.max_seq_len,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "rows_per_shard": args.rows_per_shard,
        "reasoning_level": args.reasoning_level,
        "target_total_tokens": args.target_total_tokens,
        "drop_tool_rows": args.drop_tool_rows,
        "accept_prob": args.accept_prob,
        "inputs": [str(p) for p in inputs],
    }
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode("utf-8")).hexdigest()

    git_sha = _git_sha(Path(__file__).resolve().parents[1])

    manifest = {
        "dataset": "nvidia/Nemotron-Math-v2",
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "tokenizer": {
            "name": "o200k_harmony",
            "encoding": "HARMONY_GPT_OSS",
            "openai_harmony_version": harmony_version,
        },
        "config": config_payload,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "output": dataclasses.asdict(out),
        "token_stats": {
            "prompt": prompt_stats.as_dict(),
            "analysis": analysis_stats.as_dict(),
            "final": final_stats.as_dict(),
            "total": total_stats.as_dict(),
        },
        "distributions": {
            "subset_rows": dict(dist_subset),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote dataset to {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
