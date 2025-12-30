# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
except ImportError:
    tiktoken = None
    load_tiktoken_bpe = None

from megatron.core.tokenizers.text.libraries.abstract_tokenizer import MegatronTokenizerTextAbstract
from megatron.core.tokenizers.text.libraries.chat_template import MegatronTokenizerChatTemplate

O200K_HARMONY_PADDED_VOCAB_SIZE = 201088
O200K_HARMONY_EXPECTED_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
O200K_HARMONY_PAT_STR = "|".join(
    [
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

REQUIRED_HARMONY_SPECIAL_TOKENS = {
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|message|>": 200008,
    "<|channel|>": 200005,
    "<|constrain|>": 200003,
    "<|return|>": 200002,
    "<|call|>": 200012,
}


def build_o200k_harmony_special_tokens() -> Dict[str, int]:
    special_tokens = {
        "<|startoftext|>": 199998,
        "<|endoftext|>": 199999,
        "<|endofprompt|>": 200018,
        "<|reserved_200000|>": 200000,
        "<|reserved_200001|>": 200001,
        "<|return|>": 200002,
        "<|constrain|>": 200003,
        "<|reserved_200004|>": 200004,
        "<|channel|>": 200005,
        "<|start|>": 200006,
        "<|end|>": 200007,
        "<|message|>": 200008,
        "<|reserved_200009|>": 200009,
        "<|reserved_200010|>": 200010,
        "<|reserved_200011|>": 200011,
        "<|call|>": 200012,
    }
    special_tokens.update(
        {f"<|reserved_{i}|>": i for i in range(200013, O200K_HARMONY_PADDED_VOCAB_SIZE)}
    )
    return special_tokens


def _validate_local_path(path: str) -> None:
    if "://" in path:
        raise ValueError(
            "O200k Harmony tokenizer requires a local o200k_base.tiktoken file. "
            "Download it and pass the local path via --tokenizer-model."
        )
    if not os.path.exists(path):
        raise ValueError(
            f"Tokenizer file not found: {path}. "
            "Download o200k_base.tiktoken and pass its local path via --tokenizer-model."
        )


def build_o200k_harmony_encoding(
    tokenizer_path: str,
) -> Tuple["tiktoken.Encoding", Dict[bytes, int], Dict[str, int]]:
    if tiktoken is None or load_tiktoken_bpe is None:
        raise ImportError("tiktoken is required for O200k Harmony tokenizer")

    _validate_local_path(tokenizer_path)

    mergeable_ranks = load_tiktoken_bpe(
        tokenizer_path, expected_hash=O200K_HARMONY_EXPECTED_SHA256
    )
    special_tokens = build_o200k_harmony_special_tokens()

    encoding = tiktoken.Encoding(
        name=Path(tokenizer_path).parent.name or "o200k_harmony",
        pat_str=O200K_HARMONY_PAT_STR,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    max_rank = max(mergeable_ranks.values()) if mergeable_ranks else -1
    max_special = max(special_tokens.values()) if special_tokens else -1
    max_id = max(max_rank, max_special)
    if max_id + 1 != O200K_HARMONY_PADDED_VOCAB_SIZE:
        raise ValueError(
            "Unexpected o200k_harmony vocab size. "
            f"max_id={max_id} expected_vocab={O200K_HARMONY_PADDED_VOCAB_SIZE}"
        )

    for token, token_id in REQUIRED_HARMONY_SPECIAL_TOKENS.items():
        if special_tokens.get(token) != token_id:
            raise ValueError(
                f"Missing or incorrect special token mapping: {token}"
            )

    return encoding, mergeable_ranks, special_tokens


def build_o200k_harmony_vocab(
    mergeable_ranks: Dict[bytes, int], special_tokens: Dict[str, int]
) -> Tuple[Dict[str, int], Dict[int, str]]:
    token_to_id: Dict[str, int] = dict(special_tokens)
    id_to_token: Dict[int, str] = {v: k for k, v in special_tokens.items()}

    for token_bytes, token_id in mergeable_ranks.items():
        if token_id in id_to_token:
            continue
        token_str = token_bytes.decode("utf-8", errors="replace")
        token_to_id.setdefault(token_str, token_id)
        id_to_token.setdefault(token_id, token_str)

    return token_to_id, id_to_token


class O200kHarmonyTokenizer(MegatronTokenizerTextAbstract, MegatronTokenizerChatTemplate):
    """O200k Harmony tokenizer backed by a local o200k_base.tiktoken file."""

    def __init__(self, tokenizer_path: str, **_kwargs):
        if not tokenizer_path:
            raise ValueError("tokenizer_path must be provided for O200k Harmony tokenizer")

        encoding, mergeable_ranks, special_tokens = build_o200k_harmony_encoding(tokenizer_path)

        self._encoding = encoding
        self._special_tokens = special_tokens
        self._vocab_size = O200K_HARMONY_PADDED_VOCAB_SIZE
        self._bos_id = special_tokens["<|startoftext|>"]
        self._eos_id = special_tokens["<|endoftext|>"]
        self._eod_id = special_tokens["<|endoftext|>"]
        self._pad_id = -1

        self._vocab, self._inv_vocab = build_o200k_harmony_vocab(
            mergeable_ranks, special_tokens
        )

    def text_to_tokens(self, text: str) -> List[str]:
        token_ids = self._encoding.encode(text)
        return [self._encoding.decode_single_token_bytes(token) for token in token_ids]

    def tokens_to_text(self, tokens: List[str]) -> str:
        token_ids = self.tokens_to_ids(tokens)
        return self._encoding.decode(token_ids)

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        return [self.id_to_token(token_id) for token_id in token_ids]

    def text_to_ids(self, text: str) -> List[int]:
        return self._encoding.encode(text, allowed_special="all")

    def ids_to_text(self, tokens: List[int]) -> str:
        return self._encoding.decode(tokens)

    def add_special_tokens(self):
        raise NotImplementedError("O200k Harmony tokenizer does not support adding tokens")

    def token_to_id(self, token: str) -> int:
        if token in self._special_tokens:
            return self._special_tokens[token]
        return self._encoding.encode_single_token(token)

    def id_to_token(self, token_id: int) -> str:
        token = self._inv_vocab.get(token_id)
        if token is not None:
            return token
        token_bytes = self._encoding.decode_single_token_bytes(token_id)
        return token_bytes.decode("utf-8", errors="replace")

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @property
    def inv_vocab(self) -> Dict[int, str]:
        return self._inv_vocab

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def eod(self) -> int:
        return self._eod_id

    @property
    def pad_id(self) -> int:
        return self._pad_id
