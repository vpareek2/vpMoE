#!/usr/bin/env python3
"""
Run gpt-oss teacher inference on a small prompt pack and append stats to JSONL.

This uses the Harmony renderer/parser from openai_harmony for correct formatting.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import multiprocessing as mp
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _require(module_name: str):
    try:
        return __import__(module_name)
    except Exception as exc:  # pragma: no cover - CLI guard
        raise SystemExit(f"Missing required module '{module_name}': {exc}") from exc


def _load_prompts(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
            records.append(record)
    return records


def _load_existing_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("run_key")
            if key:
                keys.add(key)
    return keys


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_key(record: Dict[str, Any], index: int) -> str:
    dataset = record.get("dataset") or record.get("name") or "unknown"
    prompt_id = record.get("id") or f"index_{index}"
    return f"{dataset}::{prompt_id}"


def _get_max_context(config: Any) -> Optional[int]:
    for attr in (
        "max_position_embeddings",
        "max_sequence_length",
        "max_seq_len",
        "max_model_len",
        "seq_length",
    ):
        val = getattr(config, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    return None


def _extract_text_from_entry(entry_dict: Dict[str, Any]) -> str:
    content = entry_dict.get("content") or []
    chunks: List[str] = []
    for item in content:
        text = item.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


_O200K_BASE_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"


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
        with urllib.request.urlopen(_O200K_BASE_URL, timeout=60) as resp, tmp.open(
            "wb"
        ) as f:
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


def _build_generation_record(
    *,
    prompt_record: Dict[str, Any],
    prompt_index: int,
    model_id: str,
    reasoning_effort: str,
    conversation_date: str,
    generation_cfg: Dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int,
    analysis_tokens: int,
    final_tokens: int,
    stop_reason: str,
    truncated: bool,
    latency_s: float,
    tokens_per_s: float,
    raw_text: str,
    parsed_entries: List[Dict[str, Any]],
    parse_error: Optional[str],
) -> Dict[str, Any]:
    run_key = f"{_prompt_key(prompt_record, prompt_index)}::{model_id}::{reasoning_effort}"
    prompt_text = prompt_record.get("prompt") or prompt_record.get("text") or ""
    prompt_sha = _sha256_text(prompt_text)
    analysis_text = ""
    final_text = ""
    for entry in parsed_entries:
        channel = entry.get("channel")
        if channel == "analysis":
            analysis_text += _extract_text_from_entry(entry)
        elif channel == "final":
            final_text += _extract_text_from_entry(entry)

    return {
        "run_key": run_key,
        "prompt_key": _prompt_key(prompt_record, prompt_index),
        "prompt_sha256": prompt_sha,
        "prompt_record": prompt_record,
        "model_id": model_id,
        "reasoning_effort": reasoning_effort,
        "conversation_date": conversation_date,
        "generation_cfg": generation_cfg,
        "counts": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "analysis_tokens": analysis_tokens,
            "final_tokens": final_tokens,
        },
        "stop_reason": stop_reason,
        "truncated": truncated,
        "timing": {
            "latency_s": latency_s,
            "tokens_per_s": tokens_per_s,
        },
        "raw_text": raw_text,
        "analysis_text": analysis_text,
        "final_text": final_text,
        "parsed_entries": parsed_entries,
        "parse_error": parse_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe gpt-oss teacher outputs and record stats.")
    parser.add_argument("--prompts", type=Path, default=Path("src/data/prompts.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("/data/teacher_runs.jsonl"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai/gpt-oss-20b"],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Inference backend to use. vllm requires the vLLM Python package in the container image.",
    )
    parser.add_argument(
        "--reasoning",
        nargs="+",
        default=["low", "medium", "high"],
        choices=["low", "medium", "high"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=65536)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of independent prompts to run per generate() call (reduces overhead, increases throughput).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument(
        "--allow-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow CPU-only inference (very slow, will dequantize MXFP4 weights).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Required for gpt-oss models.",
    )
    parser.add_argument(
        "--disable-system-site-packages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove system dist-packages from sys.path to avoid broken torchvision imports.",
    )
    parser.add_argument(
        "--tiktoken-encodings-base",
        type=str,
        default=None,
        help=(
            "Override TIKTOKEN_ENCODINGS_BASE (must be a local directory; the script "
            "will download o200k_base.tiktoken into it if missing)."
        ),
    )
    parser.add_argument(
        "--tiktoken-cache-dir",
        type=str,
        default=None,
        help="Override the tiktoken cache directory (TIKTOKEN_RS_CACHE_DIR).",
    )
    parser.add_argument("--conversation-date", type=str, default=None)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    if args.backend == "vllm":
        # vLLM starts a CUDA worker subprocess. If CUDA is initialized in the parent
        # process and the worker uses fork, torch will crash with:
        #   "Cannot re-initialize CUDA in forked subprocess"
        #
        # Force spawn early to avoid fork+CUDA issues.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    if args.disable_system_site_packages:
        venv_prefix = sys.prefix
        pruned = []
        for entry in sys.path:
            if "dist-packages" in entry and not entry.startswith(venv_prefix):
                continue
            pruned.append(entry)
        sys.path[:] = pruned

    if args.tiktoken_encodings_base:
        if _looks_like_url(args.tiktoken_encodings_base):
            raise SystemExit(
                "--tiktoken-encodings-base must be a local directory path (not a URL). "
                "openai_harmony expects a directory containing o200k_base.tiktoken."
            )
        os.environ["TIKTOKEN_ENCODINGS_BASE"] = args.tiktoken_encodings_base
    if args.tiktoken_cache_dir:
        os.environ["TIKTOKEN_RS_CACHE_DIR"] = args.tiktoken_cache_dir
    if "TIKTOKEN_RS_CACHE_DIR" not in os.environ:
        os.environ["TIKTOKEN_RS_CACHE_DIR"] = str(_default_hf_cache_root() / "tiktoken-rs-cache")

    if "TIKTOKEN_ENCODINGS_BASE" in os.environ and _looks_like_url(os.environ["TIKTOKEN_ENCODINGS_BASE"]):
        # A common misconfiguration is setting this to the OpenAI blob URL. The Rust
        # loader treats it as a filesystem path and fails with ENOENT. Override to a
        # local directory so we can self-bootstrap the vocab file.
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
    vocab_path = _ensure_o200k_base_vocab(encodings_dir)

    torch = _require("torch")
    transformers = _require("transformers")
    openai_harmony = _require("openai_harmony")
    tqdm = _require("tqdm").tqdm

    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Message,
        ReasoningEffort,
        Role,
        StreamableParser,
        SystemContent,
        load_harmony_encoding,
    )

    from transformers import AutoConfig, AutoModelForCausalLM

    if not args.prompts.exists():
        raise SystemExit(f"Prompts file not found: {args.prompts}")

    prompts = _load_prompts(args.prompts)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts}")

    existing_keys = _load_existing_keys(args.out)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.backend == "transformers" and not torch.cuda.is_available() and not getattr(args, "allow_cpu", False):
        raise SystemExit(
            "CUDA is not available. The gpt-oss MXFP4 weights require a GPU. "
            "Install a CUDA-enabled torch build in this container or rerun with "
            "--allow-cpu to force CPU (very slow)."
        )

    try:
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    except Exception as exc:
        hint = (
            "Failed to load Harmony encoding. If this is a vocab download error, "
            "ensure TIKTOKEN_ENCODINGS_BASE points to a local directory containing "
            "o200k_base.tiktoken. This script tries to bootstrap it automatically, "
            f"but failed (encodings_dir={encodings_dir}, vocab_exists={vocab_path.exists()})."
        )
        raise SystemExit(f"{hint}\\nRoot error: {exc}") from exc
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    reasoning_map = {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH,
    }

    conversation_date = args.conversation_date or _dt.datetime.now().strftime("%Y-%m-%d")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("a", encoding="utf-8") as out_f:
        pending_runs = 0
        for model_id in args.models:
            for prompt_index, prompt_record in enumerate(prompts):
                prompt_text = prompt_record.get("prompt") or prompt_record.get("text") or ""
                if not isinstance(prompt_text, str) or not prompt_text.strip():
                    continue
                for reasoning_effort in args.reasoning:
                    run_key = f"{_prompt_key(prompt_record, prompt_index)}::{model_id}::{reasoning_effort}"
                    if run_key in existing_keys:
                        continue
                    pending_runs += 1

        pbar = tqdm(total=pending_runs, desc="teacher runs", unit="run")

        for model_id in args.models:
            if args.backend == "transformers":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=args.torch_dtype,
                    device_map=args.device_map,
                    trust_remote_code=args.trust_remote_code,
                )
                model.eval()
                max_context = _get_max_context(model.config)
                vllm_engine = None
            else:
                try:
                    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
                    from vllm.inputs import TokensPrompt  # type: ignore[import-not-found]
                except Exception as exc:
                    raise SystemExit(
                        "Backend 'vllm' requested but vLLM is not available in this container. "
                        "Install vLLM in the image or use a vLLM-enabled container build."
                    ) from exc

                cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
                max_context = _get_max_context(cfg)

                if args.device_map not in ("auto", "cuda"):
                    print(
                        f"warning: --device-map={args.device_map!r} ignored for backend=vllm",
                        file=sys.stderr,
                    )
                if args.torch_dtype not in ("auto", "bf16", "fp16", "float16", "bfloat16"):
                    print(
                        f"warning: --torch-dtype={args.torch_dtype!r} may be ignored for backend=vllm",
                        file=sys.stderr,
                    )

                vllm_engine = LLM(
                    model=model_id,
                    trust_remote_code=args.trust_remote_code,
                    seed=args.seed,
                )
                model = None
            stop_token_set = set(stop_token_ids)

            def flush_batch(
                batch_items: List[Dict[str, Any]],
                *,
                max_input_len: int,
                effective_max_new_tokens: int,
            ) -> None:
                if not batch_items:
                    return

                pad_id = stop_token_ids[0]
                outputs: List[Dict[str, Any]] = []
                total_completion_tokens = 0

                if args.backend == "transformers":
                    assert model is not None
                    input_ids = torch.full(
                        (len(batch_items), max_input_len),
                        pad_id,
                        device=model.device,
                        dtype=torch.long,
                    )
                    attention_mask = torch.zeros_like(input_ids)
                    for row, item in enumerate(batch_items):
                        ids = item["prefill_ids"]
                        seq_len = len(ids)
                        # Decoder-only generation expects left-padding so that the last token of each
                        # sequence is a real prompt token (not padding).
                        start_col = max_input_len - seq_len
                        input_ids[row, start_col:max_input_len] = torch.tensor(
                            ids, device=model.device, dtype=torch.long
                        )
                        attention_mask[row, start_col:max_input_len] = 1

                    start = time.perf_counter()
                    with torch.inference_mode():
                        output = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=effective_max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            eos_token_id=stop_token_ids,
                            pad_token_id=pad_id,
                            use_cache=True,
                        )
                    latency_s = time.perf_counter() - start

                    for row in range(len(batch_items)):
                        output_ids = output[row].tolist()
                        completion_raw = output_ids[
                            max_input_len : max_input_len + effective_max_new_tokens
                        ]

                        stop_index: Optional[int] = None
                        for idx, token in enumerate(completion_raw):
                            if token in stop_token_set:
                                stop_index = idx
                                break

                        if stop_index is not None:
                            completion_ids = completion_raw[: stop_index + 1]
                            stop_reason = "harmony_stop"
                            truncated = False
                        else:
                            completion_ids = completion_raw
                            stop_reason = "max_new_tokens"
                            truncated = True

                        completion_tokens = len(completion_ids)
                        total_completion_tokens += completion_tokens
                        outputs.append(
                            {
                                "completion_ids": completion_ids,
                                "completion_tokens": completion_tokens,
                                "stop_reason": stop_reason,
                                "truncated": truncated,
                            }
                        )
                else:
                    assert vllm_engine is not None
                    sampling_params = SamplingParams(
                        max_tokens=effective_max_new_tokens,
                        temperature=args.temperature if args.do_sample else 0.0,
                        top_p=args.top_p,
                        stop_token_ids=stop_token_ids,
                        seed=args.seed,
                    )
                    prompts = [TokensPrompt(prompt_token_ids=item["prefill_ids"]) for item in batch_items]

                    start = time.perf_counter()
                    request_outputs = vllm_engine.generate(prompts, sampling_params)
                    latency_s = time.perf_counter() - start

                    for request_output in request_outputs:
                        out0 = request_output.outputs[0]
                        completion_raw = list(out0.token_ids)[:effective_max_new_tokens]

                        stop_index: Optional[int] = None
                        for idx, token in enumerate(completion_raw):
                            if token in stop_token_set:
                                stop_index = idx
                                break

                        if stop_index is not None:
                            completion_ids = completion_raw[: stop_index + 1]
                            stop_reason = "harmony_stop"
                            truncated = False
                        else:
                            finish = getattr(out0, "finish_reason", None)
                            if finish == "stop":
                                completion_ids = completion_raw
                                stop_reason = "harmony_stop"
                                truncated = False
                            else:
                                completion_ids = completion_raw
                                stop_reason = "max_new_tokens"
                                truncated = True

                        completion_tokens = len(completion_ids)
                        total_completion_tokens += completion_tokens
                        outputs.append(
                            {
                                "completion_ids": completion_ids,
                                "completion_tokens": completion_tokens,
                                "stop_reason": stop_reason,
                                "truncated": truncated,
                            }
                        )

                batch_tokens_per_s = 0.0
                if latency_s > 0:
                    batch_tokens_per_s = total_completion_tokens / latency_s

                for item, out in zip(batch_items, outputs):
                    completion_ids = out["completion_ids"]
                    completion_tokens = out["completion_tokens"]
                    stop_reason = out["stop_reason"]
                    truncated = out["truncated"]

                    parser = StreamableParser(encoding, role=Role.ASSISTANT)
                    analysis_tokens = 0
                    final_tokens = 0
                    for token in completion_ids:
                        parser.process(token)
                        channel = parser.current_channel
                        if channel == "analysis":
                            analysis_tokens += 1
                        elif channel == "final":
                            final_tokens += 1
                    # Some backends may omit explicit stop tokens; ensure the parser finalizes.
                    parser.process_eos()

                    parsed_entries: List[Dict[str, Any]] = []
                    parse_error: Optional[str] = None
                    try:
                        for message in parser.messages:
                            parsed_entries.append(message.to_dict())
                    except Exception as exc:
                        parse_error = str(exc)

                    try:
                        raw_text = encoding.decode(completion_ids)
                    except Exception:
                        raw_text = ""

                    tokens_per_s = 0.0
                    if latency_s > 0:
                        tokens_per_s = completion_tokens / latency_s

                    record = _build_generation_record(
                        prompt_record=item["prompt_record"],
                        prompt_index=item["prompt_index"],
                        model_id=model_id,
                        reasoning_effort=item["reasoning_effort"],
                        conversation_date=conversation_date,
                        generation_cfg={
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "do_sample": args.do_sample,
                            "max_new_tokens": args.max_new_tokens,
                            "effective_max_new_tokens": effective_max_new_tokens,
                            "stop_token_ids": stop_token_ids,
                            "batch_size": len(batch_items),
                        },
                        prompt_tokens=item["prompt_tokens"],
                        completion_tokens=completion_tokens,
                        analysis_tokens=analysis_tokens,
                        final_tokens=final_tokens,
                        stop_reason=stop_reason,
                        truncated=truncated,
                        latency_s=latency_s,
                        tokens_per_s=tokens_per_s,
                        raw_text=raw_text,
                        parsed_entries=parsed_entries,
                        parse_error=parse_error,
                    )
                    record["timing"]["batch"] = {
                        "size": len(batch_items),
                        "total_completion_tokens": total_completion_tokens,
                        "tokens_per_s": batch_tokens_per_s,
                    }

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    existing_keys.add(item["run_key"])
                    pbar.update(1)

            batch_items: List[Dict[str, Any]] = []
            batch_max_input_len = 0
            batch_effective_max_new_tokens: Optional[int] = None

            for prompt_index, prompt_record in enumerate(prompts):
                prompt_text = prompt_record.get("prompt") or prompt_record.get("text") or ""
                if not isinstance(prompt_text, str) or not prompt_text.strip():
                    continue

                for reasoning_effort in args.reasoning:
                    run_key = f"{_prompt_key(prompt_record, prompt_index)}::{model_id}::{reasoning_effort}"
                    if run_key in existing_keys:
                        continue

                    pbar.set_postfix_str(f"{model_id} | {reasoning_effort}")
                    system_content = (
                        SystemContent.new()
                        .with_reasoning_effort(reasoning_map[reasoning_effort])
                        .with_conversation_start_date(conversation_date)
                    )
                    system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
                    user_message = Message.from_role_and_content(Role.USER, prompt_text)
                    conversation = Conversation.from_messages([system_message, user_message])

                    prefill_ids = encoding.render_conversation_for_completion(
                        conversation, Role.ASSISTANT
                    )
                    prompt_tokens = len(prefill_ids)

                    effective_max_new_tokens = args.max_new_tokens
                    if max_context is not None:
                        remaining = max_context - prompt_tokens
                        if remaining <= 0:
                            flush_batch(
                                batch_items,
                                max_input_len=batch_max_input_len,
                                effective_max_new_tokens=batch_effective_max_new_tokens or 0,
                            )
                            batch_items = []
                            batch_max_input_len = 0
                            batch_effective_max_new_tokens = None

                            record = _build_generation_record(
                                prompt_record=prompt_record,
                                prompt_index=prompt_index,
                                model_id=model_id,
                                reasoning_effort=reasoning_effort,
                                conversation_date=conversation_date,
                                generation_cfg={
                                    "temperature": args.temperature,
                                    "top_p": args.top_p,
                                    "do_sample": args.do_sample,
                                    "max_new_tokens": args.max_new_tokens,
                                    "effective_max_new_tokens": 0,
                                    "stop_token_ids": stop_token_ids,
                                },
                                prompt_tokens=prompt_tokens,
                                completion_tokens=0,
                                analysis_tokens=0,
                                final_tokens=0,
                                stop_reason="context_overflow",
                                truncated=True,
                                latency_s=0.0,
                                tokens_per_s=0.0,
                                raw_text="",
                                parsed_entries=[],
                                parse_error="Prompt exceeds model context window.",
                            )
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            out_f.flush()
                            existing_keys.add(run_key)
                            continue
                        effective_max_new_tokens = min(effective_max_new_tokens, remaining)

                    if batch_effective_max_new_tokens is None:
                        batch_effective_max_new_tokens = effective_max_new_tokens
                    if effective_max_new_tokens != batch_effective_max_new_tokens:
                        flush_batch(
                            batch_items,
                            max_input_len=batch_max_input_len,
                            effective_max_new_tokens=batch_effective_max_new_tokens,
                        )
                        batch_items = []
                        batch_max_input_len = 0
                        batch_effective_max_new_tokens = effective_max_new_tokens

                    batch_items.append(
                        {
                            "run_key": run_key,
                            "prompt_record": prompt_record,
                            "prompt_index": prompt_index,
                            "reasoning_effort": reasoning_effort,
                            "prefill_ids": prefill_ids,
                            "prompt_tokens": prompt_tokens,
                        }
                    )
                    batch_max_input_len = max(batch_max_input_len, len(prefill_ids))

                    if len(batch_items) >= args.batch_size:
                        flush_batch(
                            batch_items,
                            max_input_len=batch_max_input_len,
                            effective_max_new_tokens=batch_effective_max_new_tokens,
                        )
                        batch_items = []
                        batch_max_input_len = 0
                        batch_effective_max_new_tokens = None

            flush_batch(
                batch_items,
                max_input_len=batch_max_input_len,
                effective_max_new_tokens=batch_effective_max_new_tokens or 0,
            )

            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
