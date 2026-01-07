#!/usr/bin/env python3
"""Run CORE-8 evaluation (DCLM CORE subset) against an HF or Megatron model."""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]

# Megatron imports are optional; only used in megatron mode.
MEGATRON_ROOT = REPO_ROOT / "vpmoe" / "Megatron-vpmoe"


@dataclass(frozen=True)
class TaskSpec:
    label: str
    dataset_uri: str
    num_fewshot: int
    icl_task_type: str
    continuation_delimiter: str


class ModelAdapter:
    def __init__(self, device: torch.device):
        self.device = device

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def bos_id(self) -> int:
        raise NotImplementedError

    def max_seq_len(self) -> Optional[int]:
        return None

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError


class HFAdapter(ModelAdapter):
    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype):
        super().__init__(device)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.bos_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.bos_token_id = 0
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self._max_seq_len = getattr(self.model.config, "max_position_embeddings", None)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def bos_id(self) -> int:
        return int(self.tokenizer.bos_token_id)

    def max_seq_len(self) -> Optional[int]:
        return self._max_seq_len

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del attention_mask, position_ids
        with torch.no_grad():
            out = self.model(input_ids=input_ids)
            return out.logits


class MegatronAdapter(ModelAdapter):
    def __init__(
        self,
        load_path: str,
        model_config: Path,
        tokenizer_model: Path,
        device: torch.device,
        dtype: torch.dtype,
        compat_mode: bool,
        rope_base: Optional[int],
        dist_ckpt_strictness: Optional[str],
    ):
        super().__init__(device)

        if not MEGATRON_ROOT.exists():
            raise RuntimeError(f"Megatron root not found: {MEGATRON_ROOT}")
        sys.path.insert(0, str(MEGATRON_ROOT))
        sys.path.insert(0, str(REPO_ROOT))

        from megatron.training.arguments import parse_args
        from megatron.training.initialize import initialize_megatron
        from megatron.training.training import get_model
        from megatron.training.checkpointing import load_checkpoint
        from gpt_builders import gpt_builder
        from model_provider import model_provider

        config = tomllib.loads(model_config.read_text(encoding="utf-8"))
        overrides = infer_checkpoint_overrides(load_path)
        args_list = build_megatron_args(
            config=config,
            load_path=load_path,
            tokenizer_model=tokenizer_model,
            dtype=dtype,
            compat_mode=compat_mode,
            rope_base=rope_base,
            hetereogenous_dist_checkpoint=overrides.get("hetereogenous_dist_checkpoint", False),
            dist_ckpt_strictness=dist_ckpt_strictness,
        )

        argv_backup = sys.argv
        try:
            sys.argv = [argv_backup[0]] + args_list
            parsed = parse_args()
        finally:
            sys.argv = argv_backup

        initialize_megatron(parsed_args=parsed)

        model = get_model(functools.partial(model_provider, gpt_builder), wrap_with_ddp=False)
        if len(model) != 1:
            raise RuntimeError("CORE-8 eval expects a single-stage model (pp=1, vp=1).")
        self.model = model[0]
        self.model.eval()

        # load_checkpoint expects a list of model stages.
        load_checkpoint(model, optimizer=None, opt_param_scheduler=None)

        from megatron.core.tokenizers.text.libraries.o200k_harmony_tokenizer import (
            O200kHarmonyTokenizer,
        )

        self.tokenizer = O200kHarmonyTokenizer(str(tokenizer_model))
        self._max_seq_len = config.get("attention", {}).get("max_sequence_length")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.text_to_ids(text)

    def bos_id(self) -> int:
        return int(self.tokenizer.bos_id)

    def max_seq_len(self) -> Optional[int]:
        return self._max_seq_len

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.model(
                input_ids,
                position_ids,
                attention_mask,
                labels=None,
                runtime_gather_output=True,
            )


def build_megatron_args(
    *,
    config: Dict[str, object],
    load_path: str,
    tokenizer_model: Path,
    dtype: torch.dtype,
    compat_mode: bool,
    rope_base: Optional[int],
    hetereogenous_dist_checkpoint: bool,
    dist_ckpt_strictness: Optional[str],
) -> List[str]:
    model_cfg = config.get("model", {})
    attn_cfg = config.get("attention", {})
    tpa_cfg = config.get("tpa", {})
    grape_cfg = config.get("grape", {})
    moe_cfg = config.get("moe", {})

    num_layers = int(model_cfg["num_layers"])
    max_pos = int(attn_cfg.get("max_sequence_length", 4096))

    args = [
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", "1",
        "--micro-batch-size", "1",
        "--global-batch-size", "1",
        "--num-layers", str(num_layers),
        "--hidden-size", str(model_cfg["hidden_size"]),
        "--num-attention-heads", str(model_cfg["num_attention_heads"]),
        "--max-position-embeddings", str(max_pos),
        "--seq-length", str(max_pos),
        "--tokenizer-type", "O200kHarmonyTokenizer",
        "--tokenizer-model", str(tokenizer_model),
        "--load", load_path,
        "--ckpt-format", "torch_dist",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--exit-on-missing-checkpoint",
    ]

    if dist_ckpt_strictness:
        args += ["--dist-ckpt-strictness", dist_ckpt_strictness]

    if compat_mode:
        # Compat checkpoints from surgery may not carry iteration metadata.
        args.append("--finetune")

    if dtype == torch.bfloat16:
        args.append("--bf16")
    elif dtype == torch.float16:
        args.append("--fp16")

    if model_cfg.get("num_query_groups", 1) != 1:
        args.append("--group-query-attention")
        args += ["--num-query-groups", str(model_cfg["num_query_groups"])]

    if model_cfg.get("ffn_hidden_size") is not None:
        args += ["--ffn-hidden-size", str(model_cfg["ffn_hidden_size"])]

    if model_cfg.get("normalization"):
        args += ["--normalization", str(model_cfg["normalization"])]

    if model_cfg.get("qk_layernorm"):
        args.append("--qk-layernorm")

    if model_cfg.get("softmax_type"):
        args += ["--softmax-type", str(model_cfg["softmax_type"])]

    if model_cfg.get("activation") == "squaredrelu":
        args.append("--squared-relu")
        # Megatron only supports bias activation fusion with gelu/swiglu/quick_geglu.
        args += ["--no-bias-gelu-fusion", "--no-bias-swiglu-fusion"]

    if model_cfg.get("untied_embeddings"):
        args.append("--untie-embeddings-and-output-weights")

    if not model_cfg.get("add_bias_linear", True):
        args.append("--disable-bias-linear")

    if compat_mode:
        args += ["--position-embedding-type", "rope"]
        if rope_base is not None:
            args += ["--rotary-base", str(rope_base)]
    else:
        pos_type = attn_cfg.get("position_embedding_type")
        if pos_type:
            args += ["--position-embedding-type", str(pos_type)]

        if attn_cfg.get("window_size"):
            args += ["--window-size", str(attn_cfg["window_size"])]
        if attn_cfg.get("window_attn_skip_freq"):
            args += ["--window-attn-skip-freq", str(attn_cfg["window_attn_skip_freq"])]

        if tpa_cfg.get("use_tpa"):
            args.append("--use-tpa")
            if tpa_cfg.get("tpa_rank") is not None:
                args += ["--tpa-rank", str(tpa_cfg["tpa_rank"])]
            if tpa_cfg.get("tpa_q_rank") is not None:
                args += ["--tpa-q-rank", str(tpa_cfg["tpa_q_rank"])]

        if grape_cfg.get("grape_a"):
            args.append("--grape-a")
            # Skip GRAPE-M on global layers when GRAPE-A is enabled.
            if attn_cfg.get("window_attn_skip_freq"):
                args += ["--no-rope-freq", str(attn_cfg["window_attn_skip_freq"])]
        if grape_cfg.get("grapem_learnable_freq") is False:
            args.append("--no-grapem-learnable-freq")
        if grape_cfg.get("grapem_share_across_heads") is False:
            args.append("--grapem-per-head")
        if grape_cfg.get("grapem_log_freq_scale") is not None:
            args += ["--grapem-log-freq-scale", str(grape_cfg["grapem_log_freq_scale"])]

    if model_cfg.get("head_dim") is not None:
        args += ["--kv-channels", str(model_cfg["head_dim"])]

    if model_cfg.get("value_residual"):
        args.append("--value-residual")

    if model_cfg.get("value_residual_init") is not None:
        args += ["--value-residual-init", str(model_cfg["value_residual_init"])]

    if model_cfg.get("hetereogenous_dist_checkpoint") or hetereogenous_dist_checkpoint:
        args.append("--hetereogenous-dist-checkpoint")

    # MoE configuration.
    if moe_cfg.get("num_experts"):
        args += ["--num-experts", str(moe_cfg["num_experts"])]
        if moe_cfg.get("topk") is not None:
            args += ["--moe-router-topk", str(moe_cfg["topk"])]
        if moe_cfg.get("shared_expert") and moe_cfg.get("shared_expert_size") is not None:
            args += ["--moe-shared-expert-intermediate-size", str(moe_cfg["shared_expert_size"])]
        if moe_cfg.get("first_layer_dense"):
            pattern = f"([0]+[1]*{num_layers - 1})"
            args += ["--moe-layer-freq", pattern]
        moe_ffn = moe_cfg.get("moe_ffn_hidden_size", model_cfg.get("ffn_hidden_size"))
        if moe_ffn is not None:
            args += ["--moe-ffn-hidden-size", str(moe_ffn)]

    return args


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run CORE-8 eval.")
    ap.add_argument("--config", type=Path, default=Path("configs/eval/core8.toml"))
    ap.add_argument("--bundle-dir", type=Path, default=None)
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--max-per-task", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--report-dir", type=Path, default=Path("reports/eval/core8"))
    ap.add_argument("--checkpoint-id", type=str, default=None)
    ap.add_argument("--compare-to", type=Path, default=None)

    ap.add_argument("--model-format", choices=["hf", "megatron"], required=True)
    ap.add_argument("--hf-path", type=str, default=None)
    ap.add_argument("--megatron-load", type=str, default=None)
    ap.add_argument("--megatron-config", type=Path, default=None)
    ap.add_argument("--tokenizer-model", type=Path, default=Path("data/tokenizer/o200k_base.tiktoken"))
    ap.add_argument("--compat", action="store_true")
    ap.add_argument("--rope-base", type=int, default=None)
    ap.add_argument("--dist-ckpt-strictness", type=str, default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return ap.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_baselines(csv_path: Path) -> Dict[str, float]:
    baselines: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row.get("Eval Task")
            baseline = row.get("Random baseline")
            if not label or baseline is None:
                continue
            baselines[label] = float(baseline)
    return baselines


def load_jsonl(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def infer_checkpoint_overrides(load_path: str) -> Dict[str, bool]:
    path = Path(load_path)
    candidates = [path, path.parent]
    for base in candidates:
        prov = base / "provenance.json"
        if not prov.exists():
            continue
        try:
            payload = json.loads(prov.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        dist_cfg = payload.get("dist_checkpoint", {})
        if dist_cfg.get("non_homogeneous_layers"):
            return {"hetereogenous_dist_checkpoint": True}
    return {}


def shuffle_and_slice(data: List[dict], seed: int, max_per_task: int) -> List[dict]:
    rng = random.Random(seed)
    rng.shuffle(data)
    if max_per_task > 0:
        return data[:max_per_task]
    return data


def render_prompts_mc(item, delimiter: str, fewshot_examples: List[dict]) -> List[str]:
    chunks: List[str] = []
    for ex in fewshot_examples:
        chunks.append(f"{ex['query']}{delimiter}{ex['choices'][ex['gold']]}\n\n")
    prefix = "".join(chunks)
    return [f"{prefix}{item['query']}{delimiter}{choice}" for choice in item["choices"]]


def render_prompts_schema(item, delimiter: str, fewshot_examples: List[dict]) -> List[str]:
    chunks: List[str] = []
    for ex in fewshot_examples:
        chunks.append(f"{ex['context_options'][ex['gold']]}{delimiter}{ex['continuation']}\n\n")
    prefix = "".join(chunks)
    return [f"{prefix}{context}{delimiter}{item['continuation']}" for context in item["context_options"]]


def render_prompts_lm(item, delimiter: str, fewshot_examples: List[dict]) -> List[str]:
    chunks: List[str] = []
    for ex in fewshot_examples:
        context = str(ex["context"]).strip()
        chunks.append(f"{context}{delimiter}{ex['continuation']}\n\n")
    prefix = "".join(chunks)
    context = str(item["context"]).strip()
    prompt_without = f"{prefix}{context}{delimiter}".strip()
    prompt_with = f"{prompt_without}{item['continuation']}"
    return [prompt_without, prompt_with]


def find_common_length(seqs: List[List[int]], direction: str) -> int:
    min_len = min(len(s) for s in seqs)
    if direction == "left":
        indices = range(min_len)
    elif direction == "right":
        indices = range(-1, -min_len - 1, -1)
    else:
        raise ValueError("direction must be 'left' or 'right'")

    for i, idx in enumerate(indices):
        token = seqs[0][idx]
        if not all(seq[idx] == token for seq in seqs):
            return i
    return min_len


def batch_sequences_mc(tokenizer: ModelAdapter, prompts: List[str]) -> Tuple[List[List[int]], List[int], List[int]]:
    tokens = [([tokenizer.bos_id()] + tokenizer.encode(p)) for p in prompts]
    start = find_common_length(tokens, direction="left")
    start_idxs = [start] * len(tokens)
    end_idxs = [len(t) for t in tokens]
    return tokens, start_idxs, end_idxs


def batch_sequences_schema(tokenizer: ModelAdapter, prompts: List[str]) -> Tuple[List[List[int]], List[int], List[int]]:
    tokens = [([tokenizer.bos_id()] + tokenizer.encode(p)) for p in prompts]
    suffix = find_common_length(tokens, direction="right")
    end_idxs = [len(t) for t in tokens]
    start_idxs = [end - suffix for end in end_idxs]
    return tokens, start_idxs, end_idxs


def batch_sequences_lm(tokenizer: ModelAdapter, prompts: List[str]) -> Tuple[List[List[int]], List[int], List[int]]:
    tokens = [([tokenizer.bos_id()] + tokenizer.encode(p)) for p in prompts]
    tokens_without, tokens_with = tokens
    end = len(tokens_with)
    if tokens_without == tokens_with[: len(tokens_without)]:
        start = len(tokens_without)
    else:
        # Tokenization is not always prefix-stable across the boundary; fall back to the
        # longest shared prefix (includes any boundary-spanning token in the continuation).
        start = find_common_length([tokens_without, tokens_with], direction="left")
    if start == 0 or start >= end:
        raise ValueError("LM prompt alignment failed: empty continuation after tokenization.")
    return [tokens_with], [start], [end]


def crop_sequences(
    tokens: List[List[int]],
    start_idxs: List[int],
    end_idxs: List[int],
    max_len: Optional[int],
) -> Tuple[List[List[int]], List[int], List[int]]:
    if max_len is None:
        return tokens, start_idxs, end_idxs
    new_tokens = []
    new_starts = []
    new_ends = []
    for t, s, e in zip(tokens, start_idxs, end_idxs):
        if len(t) <= max_len:
            new_tokens.append(t)
            new_starts.append(s)
            new_ends.append(e)
            continue
        crop = len(t) - max_len
        new_tokens.append(t[-max_len:])
        new_starts.append(s - crop)
        new_ends.append(e - crop)
    return new_tokens, new_starts, new_ends


def stack_sequences(tokens: List[List[int]], pad_id: int, device: torch.device) -> torch.Tensor:
    bsz = len(tokens)
    seq_len = max(len(t) for t in tokens)
    input_ids = torch.full((bsz, seq_len), pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(tokens):
        input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    return input_ids


def build_attention_mask(seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    mask = (mask < 0.5).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
    return mask.expand(batch_size, -1, -1, -1)


def forward_losses_and_preds(
    adapter: ModelAdapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = adapter.forward_logits(input_ids, attention_mask, position_ids)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(shift_labels.size(0), shift_labels.size(1))
    loss_full = torch.full((input_ids.size(0), input_ids.size(1)), float("nan"), device=input_ids.device)
    loss_full[:, :-1] = loss
    preds = shift_logits.argmax(dim=-1)
    return loss_full, preds


def evaluate_example(idx: int, adapter: ModelAdapter, data: List[dict], task: TaskSpec) -> bool:
    item = data[idx]
    fewshot = []
    if task.num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available, task.num_fewshot)
        fewshot = [data[i] for i in fewshot_indices]

    if task.icl_task_type == "multiple_choice":
        prompts = render_prompts_mc(item, task.continuation_delimiter, fewshot)
        tokens, starts, ends = batch_sequences_mc(adapter, prompts)
    elif task.icl_task_type == "schema":
        prompts = render_prompts_schema(item, task.continuation_delimiter, fewshot)
        tokens, starts, ends = batch_sequences_schema(adapter, prompts)
    elif task.icl_task_type == "language_modeling":
        prompts = render_prompts_lm(item, task.continuation_delimiter, fewshot)
        tokens, starts, ends = batch_sequences_lm(adapter, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task.icl_task_type}")

    tokens, starts, ends = crop_sequences(tokens, starts, ends, adapter.max_seq_len())

    input_ids = stack_sequences(tokens, adapter.bos_id(), adapter.device)
    position_ids = torch.arange(input_ids.size(1), device=adapter.device).unsqueeze(0).expand_as(input_ids)
    attention_mask = build_attention_mask(input_ids.size(1), input_ids.size(0), adapter.device)

    loss_full, preds = forward_losses_and_preds(adapter, input_ids, attention_mask, position_ids)

    if task.icl_task_type == "language_modeling":
        si = starts[0]
        ei = ends[0]
        predicted = preds[0, si - 1 : ei - 1]
        actual = input_ids[0, si:ei]
        return torch.all(predicted == actual).item()

    mean_losses = []
    for i, (si, ei) in enumerate(zip(starts, ends)):
        segment = loss_full[i, si - 1 : ei - 1]
        mean_losses.append(torch.nanmean(segment).item())
    pred_idx = mean_losses.index(min(mean_losses))
    return pred_idx == item["gold"]


def evaluate_task(adapter: ModelAdapter, data: List[dict], task: TaskSpec) -> float:
    rank = 0
    world_size = 1
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    correct = torch.zeros(len(data), dtype=torch.float32, device=adapter.device)
    for idx in range(rank, len(data), world_size):
        correct[idx] = float(evaluate_example(idx, adapter, data, task))

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
    return correct.mean().item()


def build_report(
    *,
    checkpoint_id: str,
    results: Dict[str, float],
    centered: Dict[str, float],
    core8_centered: float,
    config: Dict[str, object],
    compare_to: Optional[Path],
) -> Dict[str, object]:
    report = {
        "checkpoint_id": checkpoint_id,
        "git_sha": git_sha(),
        "config": config,
        "results": results,
        "centered_results": centered,
        "core8_centered_mean": core8_centered,
    }
    if compare_to:
        prev = json.loads(compare_to.read_text(encoding="utf-8"))
        report["delta"] = {
            "core8_centered_mean": core8_centered - prev.get("core8_centered_mean", 0.0),
            "tasks": {
                k: centered.get(k, 0.0) - prev.get("centered_results", {}).get(k, 0.0)
                for k in centered
            },
        }
    return report


def git_sha() -> Optional[str]:
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO_ROOT}", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    core_cfg = config["core8"]

    bundle_dir = args.bundle_dir or Path(core_cfg["eval_bundle_dir"])
    if not bundle_dir.exists():
        raise SystemExit(
            f"Eval bundle not found at {bundle_dir}. Run scripts/eval/fetch_core_bundle.py first."
        )

    eval_meta = bundle_dir / "eval_meta_data.csv"
    if not eval_meta.exists():
        raise SystemExit(f"Missing eval_meta_data.csv at {eval_meta}")

    baselines = load_baselines(eval_meta)

    mode_max = core_cfg["max_per_task_smoke"] if args.mode == "smoke" else core_cfg["max_per_task_full"]
    max_per_task = args.max_per_task if args.max_per_task is not None else int(mode_max)
    seed = args.seed if args.seed is not None else int(core_cfg["seed"])

    tasks = [
        TaskSpec(
            label=t["label"],
            dataset_uri=t["dataset_uri"],
            num_fewshot=int(t["num_fewshot"]),
            icl_task_type=t["icl_task_type"],
            continuation_delimiter=t.get("continuation_delimiter", " "),
        )
        for t in config.get("tasks", [])
    ]
    if not tasks:
        raise SystemExit("No tasks configured in core8.toml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.model_format == "hf":
        if not args.hf_path:
            raise SystemExit("--hf-path is required for model-format=hf")
        adapter: ModelAdapter = HFAdapter(args.hf_path, device, dtype)
    else:
        if not args.megatron_load or not args.megatron_config:
            raise SystemExit("--megatron-load and --megatron-config are required for model-format=megatron")
        adapter = MegatronAdapter(
            load_path=args.megatron_load,
            model_config=args.megatron_config,
            tokenizer_model=args.tokenizer_model,
            device=device,
            dtype=dtype,
            compat_mode=args.compat,
            rope_base=args.rope_base,
            dist_ckpt_strictness=args.dist_ckpt_strictness,
        )

    results: Dict[str, float] = {}
    centered: Dict[str, float] = {}

    data_root = bundle_dir / "eval_data"
    for task in tasks:
        data_path = data_root / task.dataset_uri
        if not data_path.exists():
            raise SystemExit(f"Missing dataset: {data_path}")
        data = load_jsonl(data_path)
        data = shuffle_and_slice(data, seed=seed, max_per_task=max_per_task)

        acc = evaluate_task(adapter, data, task)
        results[task.label] = acc

        baseline = baselines.get(task.label, 0.0)
        centered_score = (acc - 0.01 * baseline) / (1.0 - 0.01 * baseline)
        centered[task.label] = centered_score
        print(f"{task.label}: acc={acc:.4f} centered={centered_score:.4f}")

    core8_centered = sum(centered.values()) / len(centered)
    print(f"CORE-8 centered mean: {core8_centered:.6f}")

    checkpoint_id = args.checkpoint_id or f"core8_{int(time.time())}"
    report = build_report(
        checkpoint_id=checkpoint_id,
        results=results,
        centered=centered,
        core8_centered=core8_centered,
        config={
            "config": str(args.config),
            "bundle_url": core_cfg.get("eval_bundle_url"),
            "bundle_sha256": core_cfg.get("eval_bundle_sha256"),
            "mode": args.mode,
            "max_per_task": max_per_task,
            "seed": seed,
            "model_format": args.model_format,
        },
        compare_to=args.compare_to,
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"{checkpoint_id}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
