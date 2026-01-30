#!/usr/bin/env python3
"""
Download the canonical Phase-1 mix dataset snapshot from Hugging Face.

This pulls the already-built dataset so we do not re-mix raw sources locally.
The output directory should be used directly with datasets.load_from_disk.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download veerpareek/vpmoe-phase1-mix-4k-665m (dataset snapshot)."
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/data/distillation_1/phase1_mix_4k_665m",
        help="Destination directory for the dataset snapshot (load_from_disk path).",
    )
    p.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help="Optional HF_HOME override (cache root).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_home:
        os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser().resolve())

    from huggingface_hub import snapshot_download

    print("Downloading dataset repo: veerpareek/vpmoe-phase1-mix-4k-665m")
    print(f"  -> out_dir: {out_dir}")
    if "HF_HOME" in os.environ:
        print(f"  -> HF_HOME: {os.environ['HF_HOME']}")

    snapshot_download(
        repo_id="veerpareek/vpmoe-phase1-mix-4k-665m",
        repo_type="dataset",
        local_dir=str(out_dir),
    )

    print("\nDownloaded OK.")
    print("Use directly in DistillKit config:")
    print(f"  dataset.train_dataset.disk_path: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
