#!/usr/bin/env python3
"""
Build a DistillKit-ready dataset from PleIAs/SYNTH Parquet shards.

This emits per-example Harmony tokenized sequences with assistant-only labels.
Packing (for throughput) is expected to be enabled at training time.
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
import sys
import subprocess
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


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
    """
    Ensure the o200k_base vocab file exists in the directory pointed to by
    TIKTOKEN_ENCODINGS_BASE.

    openai_harmony's Rust loader expects TIKTOKEN_ENCODINGS_BASE to be a *local
    directory* containing `o200k_base.tiktoken` (not an https URL).
    """

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


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _sanitize_for_harmony(text: str) -> str:
    """Make text safe for Harmony/tiktoken (avoid invalid Unicode / NULs)."""

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


_RESULT_RE = re.compile(r"result\s*:\s*([^\s,;]+)", re.IGNORECASE)
_FAIL_RE = re.compile(r"failed after maximum attempts", re.IGNORECASE)


def _extract_result_constraint(constraints: str) -> Optional[str]:
    if not constraints:
        return None
    match = _RESULT_RE.search(constraints)
    if not match:
        return None
    return match.group(1).strip()


def _parse_first_number(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return None
    raw = match.group(0).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.match(r"\s*\**\s*([A-D])\b", text.strip(), re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def _result_mismatch(constraints: str, answer: str) -> bool:
    expected = _extract_result_constraint(constraints)
    if expected is None:
        return False
    if _FAIL_RE.search(expected) or _FAIL_RE.search(constraints):
        return True
    expected_str = expected.strip()
    if re.fullmatch(r"[A-D]", expected_str, re.IGNORECASE):
        expected_letter = expected_str.upper()
        answer_letter = _parse_answer_letter(answer)
        if answer_letter is None:
            return False
        return answer_letter != expected_letter

    expected_num = _parse_first_number(expected_str)
    if expected_num is None:
        return False
    answer_num = _parse_first_number(answer)
    if answer_num is None:
        return False
    tol = 1e-6 * max(1.0, abs(expected_num))
    return abs(expected_num - answer_num) > tol


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


@dataclasses.dataclass
class SamplingPlan:
    memorization_keep: float
    non_en_keep: float


@dataclasses.dataclass
class ScanStats:
    scanned: int = 0
    kept_after_quality: int = 0
    dropped_empty_answer: int = 0
    dropped_failed_result: int = 0
    dropped_result_mismatch: int = 0


@dataclasses.dataclass
class OutputStats:
    kept: int = 0
    dropped_long: int = 0
    dropped_mem_cap: int = 0
    dropped_non_en_cap: int = 0
    dropped_missing_query: int = 0
    dropped_encoding_error: int = 0
    train: int = 0
    validation: int = 0


@dataclasses.dataclass
class DistributionStats:
    language: Counter = dataclasses.field(default_factory=Counter)
    exercise: Counter = dataclasses.field(default_factory=Counter)


def _iter_batches(dataset, columns: List[str], batch_size: int) -> Iterator[Dict[str, List[Any]]]:
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pydict()


def _passes_quality(
    answer: str,
    constraints: str,
    scan_stats: Optional[ScanStats] = None,
    *,
    drop_mismatch: bool = True,
) -> bool:
    answer = answer.strip()
    if not answer:
        if scan_stats is not None:
            scan_stats.dropped_empty_answer += 1
        return False

    if _FAIL_RE.search(constraints or ""):
        if scan_stats is not None:
            scan_stats.dropped_failed_result += 1
        return False

    if drop_mismatch and _result_mismatch(constraints or "", answer):
        if scan_stats is not None:
            scan_stats.dropped_result_mismatch += 1
        return False

    return True


def _scan_counts(dataset, batch_size: int, drop_mismatch: bool) -> Tuple[ScanStats, Dict[str, Counter]]:
    tqdm = _require("tqdm").tqdm
    stats = ScanStats()
    counters: Dict[str, Counter] = {
        "language": Counter(),
        "exercise": Counter(),
        "language_mem": Counter(),
        "language_other": Counter(),
    }
    columns = ["language", "exercise", "synthetic_answer", "constraints"]

    total_rows = None
    try:
        files = getattr(dataset, "files", None)
        if files:
            pq = _require("pyarrow.parquet")
            total_rows = 0
            for f in files:
                md = pq.ParquetFile(str(f)).metadata
                if md is not None:
                    total_rows += int(md.num_rows)
        else:
            total_rows = int(dataset.count_rows())
    except Exception:
        total_rows = None

    pbar = tqdm(total=total_rows, desc="scan", unit="row", dynamic_ncols=True)
    try:
        for batch in _iter_batches(dataset, columns, batch_size):
            languages = batch.get("language", [])
            exercises = batch.get("exercise", [])
            answers = batch.get("synthetic_answer", [])
            constraints_list = batch.get("constraints", [])
            for lang, ex, answer, constraints in zip(languages, exercises, answers, constraints_list):
                stats.scanned += 1
                lang = _clean_text(lang).strip().lower() or "unknown"
                ex = _clean_text(ex).strip().lower() or "unknown"
                answer_text = _clean_text(answer)
                constraints_text = _clean_text(constraints)
                if not _passes_quality(answer_text, constraints_text, stats, drop_mismatch=drop_mismatch):
                    continue
                stats.kept_after_quality += 1
                counters["language"][lang] += 1
                counters["exercise"][ex] += 1
                if ex == "memorization":
                    counters["language_mem"][lang] += 1
                else:
                    counters["language_other"][lang] += 1
            pbar.update(len(languages))
    finally:
        pbar.close()

    return stats, counters


def _compute_sampling_plan(
    counts: Dict[str, Counter],
    *,
    english_frac: float,
    memorization_cap: float,
) -> SamplingPlan:
    exercise_counts = counts["exercise"]
    mem_count = exercise_counts.get("memorization", 0)
    total = sum(exercise_counts.values())
    other_count = total - mem_count

    memorization_keep = 1.0
    if mem_count > 0 and other_count > 0:
        current_mem_frac = mem_count / total
        if current_mem_frac > memorization_cap:
            target_mem = memorization_cap * other_count / (1.0 - memorization_cap)
            memorization_keep = max(0.0, min(1.0, target_mem / mem_count))

    lang_mem = counts["language_mem"]
    lang_other = counts["language_other"]
    non_en_after_mem = 0.0
    for lang, count in lang_mem.items():
        if lang == "en":
            continue
        non_en_after_mem += count * memorization_keep
    for lang, count in lang_other.items():
        if lang == "en":
            continue
        non_en_after_mem += count

    total_after_mem = other_count + mem_count * memorization_keep
    target_non_en = total_after_mem * (1.0 - english_frac)
    non_en_keep = 1.0
    if non_en_after_mem > target_non_en and non_en_after_mem > 0:
        non_en_keep = max(0.0, min(1.0, target_non_en / non_en_after_mem))

    return SamplingPlan(memorization_keep=memorization_keep, non_en_keep=non_en_keep)


def _build_prompt(query: str, seed_text: str) -> str:
    query = query.strip()
    seed_text = seed_text.strip()
    if seed_text:
        return f"{query}\n\nReference:\n{seed_text}"
    return query


def _load_encoding() -> Any:
    openai_harmony = _require("openai_harmony")
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _harmony_render_config_keep_analysis() -> Any:
    """
    Harmony render helpers default to dropping prior analysis content when a
    final answer is present. For distillation, we want the model to learn to
    emit reasoning (analysis) *and* final spans, so we must disable this.
    """

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


def _render_tokens_for_example(
    encoding,
    *,
    query: str,
    seed_text: str,
    reasoning: str,
    answer: str,
    reasoning_level: str,
) -> Tuple[List[int], int, int, int]:
    from openai_harmony import Conversation, Message, Role, SystemContent

    def _make_system_content(level: str):
        if not level:
            return SystemContent.new()
        for kwargs in (
            {"reasoning_level": level},
            {"reasoning": level},
        ):
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

        # Fallback: emit a plain system string with the desired reasoning level.
        # This preserves the conditioning even if openai_harmony lacks setters.
        return (
            "You are ChatGPT, a large language model trained by OpenAI.\n"
            "Knowledge cutoff: 2024-06\n\n"
            f"Reasoning: {level}\n\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message."
        )

    system_message = Message.from_role_and_content(Role.SYSTEM, _make_system_content(reasoning_level))
    user_message = Message.from_role_and_content(Role.USER, _build_prompt(query, seed_text))

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
        analysis_msg = (
            Message.from_role_and_content(Role.ASSISTANT, reasoning).with_channel("analysis")
        )
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
        raise ValueError(
            "Computed negative span lengths "
            f"(analysis={analysis_count}, final={final_count}). "
            "This typically indicates Harmony auto_drop_analysis is enabled."
        )

    return full_ids, assistant_start, analysis_count, final_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DistillKit-ready SYNTH dataset.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of SYNTH parquet shards.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for parquet splits.")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--train-frac", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--english-frac", type=float, default=0.75)
    parser.add_argument("--memorization-cap", type=float, default=0.7)
    parser.add_argument(
        "--reasoning-level",
        type=str,
        default="high",
        help="Harmony system reasoning level to embed (e.g. low/medium/high).",
    )
    parser.add_argument(
        "--global-keep",
        type=float,
        default=1.0,
        help="Optional global downsample rate applied after all other filters (deterministic).",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--rows-per-shard", type=int, default=2000)
    parser.add_argument(
        "--encoding-errors-log",
        type=str,
        default=None,
        help="Optional path to write a JSONL sample of Harmony/tokenization failures (no raw text).",
    )
    parser.add_argument(
        "--max-encoding-errors-log",
        type=int,
        default=1000,
        help="Maximum number of encoding errors to write to --encoding-errors-log.",
    )
    parser.add_argument(
        "--drop-result-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows whose constraints imply a result that mismatches the answer.",
    )
    parser.add_argument(
        "--tiktoken-encodings-base",
        type=str,
        default=None,
        help="Override TIKTOKEN_ENCODINGS_BASE (must be a local directory).",
    )
    parser.add_argument(
        "--tiktoken-cache-dir",
        type=str,
        default=None,
        help="Override the tiktoken cache directory (TIKTOKEN_RS_CACHE_DIR).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not (0.0 < args.global_keep <= 1.0):
        raise SystemExit("--global-keep must be in (0, 1].")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    if args.tiktoken_encodings_base:
        if _looks_like_url(args.tiktoken_encodings_base):
            raise SystemExit(
                "--tiktoken-encodings-base must be a local directory path (not a URL)."
            )
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

    if input_dir.is_file():
        parquet_files = [str(input_dir)]
    else:
        parquet_files = sorted(str(p) for p in input_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet shards found under: {input_dir}")

    missing_files = [p for p in parquet_files if not Path(p).exists()]
    if missing_files:
        example = "\n".join(f"- {p}" for p in missing_files[:10])
        raise SystemExit(
            "Input contains missing/broken parquet paths.\n"
            f"missing_count={len(missing_files)} under: {input_dir}\n"
            "examples:\n"
            f"{example}\n"
            "\n"
            "This usually means your input dir contains broken symlinks.\n"
            "Fix the symlinks or point --input-dir at a directory that contains the real parquet files."
        )

    dataset = ds.dataset(parquet_files, format="parquet")
    required_columns = {
        "synth_id",
        "language",
        "exercise",
        "query",
        "query_seed_text",
        "synthetic_reasoning",
        "synthetic_answer",
        "constraints",
    }
    missing = required_columns - set(dataset.schema.names)
    if missing:
        raise SystemExit(f"Input dataset missing required columns: {sorted(missing)}")

    print("Scanning SYNTH for counts...", file=sys.stderr)
    scan_stats, counters = _scan_counts(dataset, args.batch_size, args.drop_result_mismatch)

    sampling_plan = _compute_sampling_plan(
        counters, english_frac=args.english_frac, memorization_cap=args.memorization_cap
    )

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
                        ("synth_id", pa.string()),
                        ("exercise", pa.string()),
                        ("language", pa.string()),
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
    dist_stats = DistributionStats()
    prompt_stats = RunningStats()
    analysis_stats = RunningStats()
    final_stats = RunningStats()
    total_stats = RunningStats()
    encoding_error_types: Counter = Counter()
    encoding_error_language: Counter = Counter()
    encoding_error_exercise: Counter = Counter()
    encoding_error_log_f = None
    encoding_error_log_remaining = 0
    if args.encoding_errors_log:
        encoding_error_log_path = Path(args.encoding_errors_log).expanduser().resolve()
        encoding_error_log_path.parent.mkdir(parents=True, exist_ok=True)
        encoding_error_log_f = encoding_error_log_path.open("w", encoding="utf-8")
        encoding_error_log_remaining = max(0, int(args.max_encoding_errors_log))

    columns = [
        "synth_id",
        "language",
        "exercise",
        "query",
        "query_seed_text",
        "synthetic_reasoning",
        "synthetic_answer",
        "constraints",
    ]

    total_expected = scan_stats.kept_after_quality
    pbar = tqdm(total=total_expected, desc="processed", unit="row")

    for batch in _iter_batches(dataset, columns, args.batch_size):
        synth_ids = batch.get("synth_id", [])
        languages = batch.get("language", [])
        exercises = batch.get("exercise", [])
        queries = batch.get("query", [])
        seeds = batch.get("query_seed_text", [])
        reasonings = batch.get("synthetic_reasoning", [])
        answers = batch.get("synthetic_answer", [])
        constraints_list = batch.get("constraints", [])

        for synth_id, lang, ex, query, seed, reasoning, answer, constraints in zip(
            synth_ids,
            languages,
            exercises,
            queries,
            seeds,
            reasonings,
            answers,
            constraints_list,
        ):
            lang = _clean_text(lang).strip().lower() or "unknown"
            ex = _clean_text(ex).strip().lower() or "unknown"
            query_text = _clean_text(query)
            seed_text = _clean_text(seed)
            reasoning_text = _clean_text(reasoning)
            answer_text = _clean_text(answer)
            constraints_text = _clean_text(constraints)
            synth_id_text = _clean_text(synth_id)

            if not _passes_quality(
                answer_text, constraints_text, None, drop_mismatch=args.drop_result_mismatch
            ):
                continue
            pbar.update(1)

            if ex == "memorization" and sampling_plan.memorization_keep < 1.0:
                if _hash_to_unit(synth_id_text, args.seed, "mem") > sampling_plan.memorization_keep:
                    output_stats.dropped_mem_cap += 1
                    continue

            if lang != "en" and sampling_plan.non_en_keep < 1.0:
                if _hash_to_unit(synth_id_text, args.seed, "lang") > sampling_plan.non_en_keep:
                    output_stats.dropped_non_en_cap += 1
                    continue

            if args.global_keep < 1.0:
                if _hash_to_unit(synth_id_text, args.seed, "global") > args.global_keep:
                    continue

            if not query_text.strip():
                output_stats.dropped_missing_query += 1
                continue

            query_text = _sanitize_for_harmony(query_text)
            seed_text = _sanitize_for_harmony(seed_text)
            reasoning_text = _sanitize_for_harmony(reasoning_text)
            answer_text = _sanitize_for_harmony(answer_text)

            try:
                input_ids, assistant_start, analysis_count, final_count = _render_tokens_for_example(
                    encoding,
                    query=query_text,
                    seed_text=seed_text,
                    reasoning=reasoning_text,
                    answer=answer_text,
                    reasoning_level=args.reasoning_level,
                )
            except Exception as exc:
                # Skip malformed rows rather than crashing the entire build, but
                # keep enough structured diagnostics to debug systemic failures.
                output_stats.dropped_encoding_error += 1
                encoding_error_types[type(exc).__name__] += 1
                encoding_error_language[lang] += 1
                encoding_error_exercise[ex] += 1
                if encoding_error_log_f is not None and encoding_error_log_remaining > 0:
                    payload = {
                        "synth_id": synth_id_text,
                        "language": lang,
                        "exercise": ex,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "query_chars": len(query_text),
                        "seed_chars": len(seed_text),
                        "reasoning_chars": len(reasoning_text),
                        "answer_chars": len(answer_text),
                    }
                    encoding_error_log_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    encoding_error_log_remaining -= 1
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
                "id": f"pleias_synth:{synth_id_text}",
                "source": {
                    "dataset": "PleIAs/SYNTH",
                    "synth_id": synth_id_text,
                    "exercise": ex,
                    "language": lang,
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

            split_draw = _hash_to_unit(synth_id_text, args.seed, "split")
            if split_draw < (1.0 - args.train_frac):
                val_writer.write(row)
                output_stats.validation += 1
            else:
                train_writer.write(row)
                output_stats.train += 1

            output_stats.kept += 1
            dist_stats.language[lang] += 1
            dist_stats.exercise[ex] += 1
            prompt_stats.update(assistant_start)
            analysis_stats.update(analysis_count)
            final_stats.update(final_count)
            total_stats.update(total_len)

    pbar.close()

    train_writer.close()
    val_writer.close()
    if encoding_error_log_f is not None:
        encoding_error_log_f.close()

    config_payload = {
        "max_seq_len": args.max_seq_len,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "english_frac": args.english_frac,
        "memorization_cap": args.memorization_cap,
        "global_keep": args.global_keep,
        "rows_per_shard": args.rows_per_shard,
        "drop_result_mismatch": args.drop_result_mismatch,
        "reasoning_level": args.reasoning_level,
    }
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode("utf-8")).hexdigest()

    git_sha = _git_sha(Path(__file__).resolve().parents[1])

    manifest = {
        "dataset": "PleIAs/SYNTH",
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "tokenizer": {
            "name": "o200k_harmony",
            "encoding": "HARMONY_GPT_OSS",
            "openai_harmony_version": harmony_version,
        },
        "config": config_payload,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "sampling": dataclasses.asdict(sampling_plan),
        "scan": dataclasses.asdict(scan_stats),
        "output": dataclasses.asdict(output_stats),
        "distributions": {
            "language": dict(dist_stats.language),
            "exercise": dict(dist_stats.exercise),
        },
        "token_stats": {
            "prompt": prompt_stats.as_dict(),
            "analysis": analysis_stats.as_dict(),
            "final": final_stats.as_dict(),
            "total": total_stats.as_dict(),
        },
        "encoding_errors": {
            "types": dict(encoding_error_types),
            "language": dict(encoding_error_language),
            "exercise": dict(encoding_error_exercise),
            "log_path": str(Path(args.encoding_errors_log).expanduser().resolve())
            if args.encoding_errors_log
            else None,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote dataset to {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
