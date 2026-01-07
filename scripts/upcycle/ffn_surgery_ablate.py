#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class CalibBatch:
    input_ids: "torch.Tensor"
    attention_mask: "torch.Tensor"


def _pick_device(torch, device: Optional[str]) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _load_prompts(path: Optional[str], max_samples: int) -> List[str]:
    if path is None:
        return [
            "Explain why 1/3 is 0.333... and not 0.33.",
            "Compute 17 * 29 and show your steps.",
            "Write a short Python function to check if a number is prime.",
            "Repeat exactly: 2025-08-05",
            "Solve: If x + 3 = 10, what is x?",
            "Give a concise summary of the key points of the U.S. Constitution.",
            "Translate 'Good morning' into Spanish and Japanese.",
        ][:max_samples]

    prompts: List[str] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"calibration file not found: {path}")
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    if "text" in obj:
                        prompts.append(str(obj["text"]))
                        continue
                    if "prompt" in obj:
                        prompts.append(str(obj["prompt"]))
                        continue
            except Exception:
                pass
        prompts.append(line)
        if len(prompts) >= max_samples:
            break
    return prompts[:max_samples]


def _tokenize_prompts(
    tokenizer,
    prompts: List[str],
    *,
    max_length: int,
    batch_size: int,
) -> Iterable[CalibBatch]:
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        yield CalibBatch(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])


def _get_mlp_weights(mlp):
    # Qwen-style (gate/up/down)
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        return mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    # LLaMA-style naming (w1 gate, w3 up, w2 down)
    if hasattr(mlp, "w1") and hasattr(mlp, "w2") and hasattr(mlp, "w3"):
        return mlp.w1.weight, mlp.w3.weight, mlp.w2.weight
    raise ValueError("Unsupported MLP module naming; expected gate/up/down or w1/w2/w3.")


def _select_topk_by_weight_norm(w_down, k: int):
    import torch
    # w_down: [hidden, intermediate]
    scores = w_down.float().pow(2).sum(dim=0).sqrt()
    topk = torch.topk(scores, k, largest=True).indices
    return topk


def _select_topk_by_activation(stats, k: int):
    import torch
    # stats: [intermediate] mean abs activation
    topk = torch.topk(stats, k, largest=True).indices
    return topk


def _relu2(x):
    import torch
    return torch.relu(x) ** 2


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ablate FFN surgery mappings (SwiGLU→ReLU²) using Qwen weights."
    )
    ap.add_argument("--model-dir", default="weights/upcycle/qwen3-0_6B-o200k")
    ap.add_argument("--method", choices=["A", "D", "A+D"], default="A")
    ap.add_argument("--intermediate-size", type=int, default=512)
    ap.add_argument("--calib-path", default=None, help="Text file (or JSONL with 'text') for calibration.")
    ap.add_argument("--max-samples", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--no-eval", action="store_true", help="Skip MSE evaluation pass.")
    ap.add_argument("--out-json", default="artifacts/ffn_surgery_ablate.json")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _pick_device(torch, args.device)
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_fast=True,
        fix_mistral_regex=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()

    prompts = _load_prompts(args.calib_path, args.max_samples)
    batches = list(_tokenize_prompts(tokenizer, prompts, max_length=args.max_length, batch_size=args.batch_size))

    num_layers = len(model.model.layers)
    k = args.intermediate_size

    want_a = args.method in ("A", "A+D")
    want_d = args.method in ("D", "A+D")

    # Pass 1: activation stats for method D (mean abs activation per intermediate dim).
    act_stats = [None] * num_layers
    act_counts = [0] * num_layers

    hook_handles = []
    mode = "idle"

    def _register_hooks():
        for layer_idx, layer in enumerate(model.model.layers):
            gate_w, _, _ = _get_mlp_weights(layer.mlp)
            if act_stats[layer_idx] is None:
                act_stats[layer_idx] = torch.zeros(
                    gate_w.shape[0], device=device, dtype=torch.float32
                )

            def _make_hook(idx):
                def _hook(module, inputs, output):
                    if mode == "idle":
                        return
                    x = inputs[0]
                    gate_w, up_w, down_w = _get_mlp_weights(module)
                    gate = F.linear(x, gate_w)
                    up = F.linear(x, up_w)
                    inter = F.silu(gate) * up

                    if mode == "stats":
                        act_stats[idx] += inter.abs().sum(dim=(0, 1))
                        act_counts[idx] += inter.shape[0] * inter.shape[1]
                        return

                    if mode != "eval":
                        return

                    y_donor = F.linear(inter, down_w)
                    if want_a:
                        sel = torch.tensor(sel_a[idx], device=device)
                        w1 = up_w.index_select(0, sel)
                        w2 = down_w.index_select(1, sel)
                        y_a = F.linear(_relu2(F.linear(x, w1)), w2)
                        mse_a[idx] += F.mse_loss(y_a, y_donor, reduction="sum")
                    if want_d:
                        sel = torch.tensor(sel_d[idx], device=device)
                        w1 = up_w.index_select(0, sel)
                        w2 = down_w.index_select(1, sel)
                        y_d = F.linear(_relu2(F.linear(x, w1)), w2)
                        mse_d[idx] += F.mse_loss(y_d, y_donor, reduction="sum")
                    counts[idx] += y_donor.numel()

                return _hook

            hook_handles.append(layer.mlp.register_forward_hook(_make_hook(layer_idx)))

    _register_hooks()

    if want_d:
        mode = "stats"
        with torch.no_grad():
            for batch in batches:
                input_ids = batch.input_ids.to(device)
                attention_mask = batch.attention_mask.to(device)
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        act_stats = [s / max(1, c) for s, c in zip(act_stats, act_counts)]

    # Build per-layer selections.
    sel_a = [None] * num_layers
    sel_d = [None] * num_layers
    for i, layer in enumerate(model.model.layers):
        gate_w, up_w, down_w = _get_mlp_weights(layer.mlp)
        if want_a:
            sel_a[i] = _select_topk_by_weight_norm(down_w, k).tolist()
        if want_d:
            sel_d[i] = _select_topk_by_activation(act_stats[i], k).tolist()

    results = {
        "model_dir": args.model_dir,
        "intermediate_size": k,
        "method": args.method,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": device,
        "selections": {},
        "mse": {},
    }
    if want_a:
        results["selections"]["A_weight_norm"] = sel_a
    if want_d:
        results["selections"]["D_activation"] = sel_d

    if not args.no_eval:
        mse_a = torch.zeros(num_layers, device=device, dtype=torch.float32)
        mse_d = torch.zeros(num_layers, device=device, dtype=torch.float32)
        counts = torch.zeros(num_layers, device=device, dtype=torch.float32)

        mode = "eval"
        with torch.no_grad():
            for batch in batches:
                input_ids = batch.input_ids.to(device)
                attention_mask = batch.attention_mask.to(device)
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

        if want_a:
            results["mse"]["A_weight_norm"] = (mse_a / counts).tolist()
        if want_d:
            results["mse"]["D_activation"] = (mse_d / counts).tolist()

    mode = "idle"
    for h in hook_handles:
        h.remove()

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
