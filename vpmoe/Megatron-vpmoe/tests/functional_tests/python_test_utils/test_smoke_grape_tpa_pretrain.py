# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _find_o200k_vocab_path(megatron_root: Path) -> Path | None:
    env_path = os.environ.get("O200K_HARMONY_VOCAB_PATH")
    if env_path:
        return Path(env_path)

    candidates = [Path("/workspace/vpmoe/data/tokenizer/o200k_base.tiktoken")]
    for base in [megatron_root, *megatron_root.parents]:
        candidates.append(base / "data/tokenizer/o200k_base.tiktoken")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@pytest.mark.smoke
def test_smoke_pretrain_grape_tpa_mock_data():
    """Opt-in smoke test for GRAPE-M + GRAPE-A + TPA wiring.

    Enable with `VPMOE_RUN_SMOKE=1`.
    """
    if os.environ.get("VPMOE_RUN_SMOKE") != "1":
        pytest.skip("Set VPMOE_RUN_SMOKE=1 to enable this smoke test.")

    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch is not available: {exc}")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available; this smoke test expects a GPU environment.")

    megatron_root = Path(__file__).resolve().parents[3]
    vocab_path = _find_o200k_vocab_path(megatron_root)
    if vocab_path is None:
        pytest.skip("O200k vocab not found; set O200K_HARMONY_VOCAB_PATH.")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "pretrain_gpt.py",
        "--mock-data",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--num-layers",
        "4",
        "--hidden-size",
        "128",
        "--num-attention-heads",
        "4",
        "--group-query-attention",
        "--num-query-groups",
        "4",
        "--seq-length",
        "128",
        "--max-position-embeddings",
        "128",
        "--tokenizer-type",
        "O200kHarmonyTokenizer",
        "--tokenizer-model",
        str(vocab_path),
        "--position-embedding-type",
        "grapem",
        "--grape-a",
        "--window-size",
        "(127,0)",
        "--window-attn-skip-freq",
        "4",
        "--use-tpa",
        "--tpa-rank",
        "8",
        "--tpa-q-rank",
        "8",
        "--disable-bias-linear",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "1",
        "--train-iters",
        "2",
        "--lr",
        "1e-4",
        "--min-lr",
        "1e-4",
        "--lr-decay-style",
        "constant",
        "--log-interval",
        "1",
        "--eval-interval",
        "1000000000",
        "--eval-iters",
        "0",
        "--save-interval",
        "0",
    ]

    completed = subprocess.run(
        cmd,
        cwd=str(megatron_root),
        check=False,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60 * 30,
    )

    if completed.returncode != 0:
        raise AssertionError(completed.stdout)
