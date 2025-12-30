# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from megatron.core.tokenizers.text.utils.build_tokenizer import (
    build_tokenizer as _build_core_tokenizer,
)
from megatron.training.tokenizer.tokenizer import (
    build_tokenizer as _build_legacy_tokenizer,
)


def build_tokenizer(args, **kwargs):
    if getattr(args, "legacy_tokenizer", False):
        return _build_legacy_tokenizer(args, **kwargs)
    return _build_core_tokenizer(args)
