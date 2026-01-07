#!/usr/bin/env python3
"""Download and unpack the DCLM CORE eval bundle (nanochat) into /data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fetch and unpack the DCLM CORE eval bundle (nanochat).",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/core8.toml"),
        help="Eval suite config file.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Target directory for the eval bundle (overrides config).",
    )
    ap.add_argument(
        "--url",
        type=str,
        default=None,
        help="Bundle URL (overrides config).",
    )
    ap.add_argument(
        "--sha256",
        type=str,
        default=None,
        help="Expected sha256 (overrides config).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing bundle if present.",
    )
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def load_defaults(config_path: Path) -> tuple[str, str, Path]:
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    core_cfg = payload.get("core8", {})
    url = core_cfg.get("eval_bundle_url")
    sha256 = core_cfg.get("eval_bundle_sha256")
    out_dir = core_cfg.get("eval_bundle_dir")
    if not url or not sha256 or not out_dir:
        raise SystemExit("core8.toml missing eval_bundle_url/sha256/dir")
    return url, sha256, Path(out_dir)


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        raise SystemExit(f"Missing config: {args.config}")

    url, sha256, out_dir = load_defaults(args.config)
    if args.url:
        url = args.url
    if args.sha256:
        sha256 = args.sha256
    if args.output_dir:
        out_dir = args.output_dir

    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output dir already exists: {out_dir} (use --overwrite)")
        shutil.rmtree(out_dir)

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpzip = Path(tmpdir) / "eval_bundle.zip"
        print(f"[download] {url} -> {tmpzip}")
        download(url, tmpzip)

        digest = sha256_file(tmpzip)
        if digest != sha256:
            raise SystemExit(f"SHA256 mismatch: expected {sha256} got {digest}")

        extract_root = Path(tmpdir) / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmpzip, "r") as zf:
            zf.extractall(extract_root)

        extracted = extract_root / "eval_bundle"
        if not extracted.exists():
            raise SystemExit("Expected eval_bundle/ in zip payload")

        shutil.move(str(extracted), out_dir)

    manifest = {
        "url": url,
        "sha256": sha256,
        "fetched_unix_s": int(time.time()),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] bundle ready at {out_dir}")


if __name__ == "__main__":
    main()
