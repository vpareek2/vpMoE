#!/usr/bin/env python3
"""
Download OpenAssistant/oasst2 as a local snapshot (parquet) for use via the /datasets mount.

This is intentionally small and explicit: one dataset, one output directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download OpenAssistant/oasst2 (dataset snapshot).")
    p.add_argument(
        "--out-dir",
        type=str,
        default="/data/raw_0/OpenAssistant__oasst2",
        help="Destination directory for the dataset snapshot.",
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
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if out_dir.as_posix().startswith("/datasets/"):
            raise SystemExit(
                f"Cannot write to {out_dir} ({exc}).\n"
                "In this container, `/datasets` is typically a read-only host mount.\n"
                "Use a writable path like `/data/raw_0/OpenAssistant__oasst2`, or download on the host and mount it into `/datasets`."
            ) from exc
        raise

    if args.hf_home:
        os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser().resolve())

    from huggingface_hub import snapshot_download

    print("Downloading dataset repo: OpenAssistant/oasst2")
    print(f"  -> out_dir: {out_dir}")
    if "HF_HOME" in os.environ:
        print(f"  -> HF_HOME: {os.environ['HF_HOME']}")

    snapshot_download(
        repo_id="OpenAssistant/oasst2",
        repo_type="dataset",
        local_dir=str(out_dir),
        # Dataset is small; fetch everything (includes parquet + metadata).
    )
    print(f"Done. Local snapshot: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
