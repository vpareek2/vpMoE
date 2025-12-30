import hashlib
import os
from argparse import Namespace
from pathlib import Path

import pytest
import requests

from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer as build_core_tokenizer
from megatron.core.tokenizers.text.libraries.o200k_harmony_tokenizer import (
    O200kHarmonyTokenizer,
)
from megatron.training import tokenizer as training_tokenizer

O200K_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
O200K_HASH = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
O200K_PADDED_VOCAB = 201088


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_o200k(path: Path) -> None:
    response = requests.get(O200K_URL, stream=True, timeout=60)
    response.raise_for_status()
    with path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


@pytest.fixture(scope="session")
def o200k_vocab_path(tmp_path_factory) -> Path:
    env_path = os.getenv("O200K_HARMONY_VOCAB_PATH")
    if env_path:
        path = Path(env_path)
        if not path.exists():
            pytest.skip(f"O200K_HARMONY_VOCAB_PATH not found: {path}")
        return path

    path = tmp_path_factory.mktemp("o200k_harmony") / "o200k_base.tiktoken"
    if not path.exists():
        try:
            _download_o200k(path)
        except Exception as exc:
            pytest.skip(f"Failed to download o200k_base.tiktoken: {exc}")

    file_hash = _sha256(path)
    if file_hash != O200K_HASH:
        raise AssertionError(f"o200k_base.tiktoken hash mismatch: {file_hash}")

    return path


def test_o200k_harmony_core_tokenizer(o200k_vocab_path: Path):
    args = Namespace(
        tokenizer_type="O200kHarmonyTokenizer",
        tokenizer_model=str(o200k_vocab_path),
        tokenizer_metadata=None,
        vocab_file=None,
        merges_file=None,
        tokenizer_special_tokens=None,
        tiktoken_pattern=None,
        tiktoken_num_special_tokens=None,
        vocab_size=None,
        tokenizer_hf_use_fast=False,
        trust_remote_code=False,
        tokenizer_hf_include_special_tokens=False,
    )
    tok = build_core_tokenizer(args)

    assert tok.vocab_size == O200K_PADDED_VOCAB
    assert tok.eod == 199999
    assert tok.vocab["<|start|>"] == 200006
    assert tok.vocab["<|message|>"] == 200008

    text = "hello world"
    assert tok.detokenize(tok.tokenize(text)) == text


def test_o200k_harmony_training_builder_uses_core(o200k_vocab_path: Path):
    args = Namespace(
        rank=0,
        tokenizer_type="O200kHarmonyTokenizer",
        tokenizer_model=str(o200k_vocab_path),
        tokenizer_metadata=None,
        tokenizer_special_tokens=None,
        tokenizer_sentencepiece_legacy=False,
        tokenizer_hf_use_fast=False,
        tokenizer_hf_include_special_tokens=False,
        trust_remote_code=False,
        vocab_file=None,
        merges_file=None,
        tiktoken_pattern=None,
        tiktoken_num_special_tokens=None,
        vocab_size=None,
        padded_vocab_size=None,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=8,
        legacy_tokenizer=False,
    )
    tok = training_tokenizer.build_tokenizer(args)

    assert tok.vocab_size == O200K_PADDED_VOCAB
    assert isinstance(tok._tokenizer, O200kHarmonyTokenizer)
    assert tok.vocab["<|end|>"] == 200007
    assert tok.tokenize("<|start|>") == [200006]
