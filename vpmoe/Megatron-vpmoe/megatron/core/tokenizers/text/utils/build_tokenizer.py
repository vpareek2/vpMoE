# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math

from megatron.core.tokenizers import MegatronTokenizer

MEGATRON_TOKENIZERS = ['BertWordPieceLowerCase', 'BertWordPieceCase', 'GPT2BPETokenizer']

SP_TOKENIZERS = ['SentencePieceTokenizer', 'GPTSentencePieceTokenizer', 'Llama2Tokenizer']


def build_tokenizer(args):
    """ """
    kwargs = {}
    tokenizer_library = None
    tokenizer_path = None
    if args.tokenizer_type in MEGATRON_TOKENIZERS:
        tokenizer_library = 'megatron'
        tokenizer_path = args.tokenizer_type
        kwargs['additional_special_tokens'] = (
            args.tokenizer_special_tokens if args.tokenizer_special_tokens else []
        )
        if tokenizer_path == 'BertWordPieceCase':
            special_tokens = {}
            special_tokens['additional_special_tokens'] = [f'<extra_id_{i}>' for i in range(100)]
            kwargs = special_tokens
        kwargs['vocab_file'] = args.vocab_file
        kwargs['merges_file'] = args.merge_file
        kwargs['use_fast'] = args.tokenizer_hf_use_fast
        kwargs['trust_remote_code'] = args.trust_remote_code
        kwargs['include_special_tokens'] = args.tokenizer_hf_include_special_tokens
    elif args.tokenizer_type in SP_TOKENIZERS:
        tokenizer_library = 'sentencepiece'
        tokenizer_path = args.tokenizer_model
        kwargs['legacy'] = args.tokenizer_sentencepiece_legacy
        kwargs['special_tokens'] = args.tokenizer_special_tokens
    elif args.tokenizer_type == 'TikTokenizer':
        tokenizer_library = 'tiktoken'
        tokenizer_path = args.tokenizer_model
        if args.tiktoken_pattern:
            kwargs['pattern'] = args.tiktoken_pattern
        if args.vocab_size:
            kwargs['vocab_size'] = args.vocab_size
        kwargs['num_special_tokens'] = args.tiktoken_num_special_tokens
        kwargs['special_tokens'] = args.tokenizer_special_tokens
    elif args.tokenizer_type == 'O200kHarmonyTokenizer':
        tokenizer_library = 'o200k-harmony'
        tokenizer_path = args.tokenizer_model
        assert tokenizer_path is not None, "O200k Harmony tokenizer requires --tokenizer-model"
    elif args.tokenizer_type == 'HuggingFaceTokenizer':
        tokenizer_library = 'huggingface'
        tokenizer_path = args.tokenizer_model
        kwargs['vocab_file'] = args.vocab_file
        kwargs['merges_file'] = args.merge_file
        kwargs['additional_special_tokens'] = (
            args.tokenizer_special_tokens if args.tokenizer_special_tokens else []
        )
        kwargs['use_fast'] = args.tokenizer_hf_use_fast
        kwargs['trust_remote_code'] = args.trust_remote_code
        kwargs['include_special_tokens'] = args.tokenizer_hf_include_special_tokens
    elif args.tokenizer_type == 'NullTokenizer':
        tokenizer_library = 'null'
        metadata = {'library': tokenizer_library}
        if args.vocab_size:
            kwargs['vocab_size'] = args.vocab_size
        tokenizer = MegatronTokenizer.from_pretrained(metadata_path=metadata, **kwargs)

        return tokenizer

    if args.tokenizer_metadata:
        metadata = args.tokenizer_metadata
    else:
        metadata = {'library': tokenizer_library}
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=tokenizer_path, metadata_path=metadata, **kwargs
    )

    # Training code expects these to be populated for embedding construction.
    # Keep behavior aligned with the legacy tokenizer builder.
    if args.tokenizer_type == 'O200kHarmonyTokenizer':
        from megatron.core.tokenizers.text.libraries.o200k_harmony_tokenizer import (
            O200K_HARMONY_PADDED_VOCAB_SIZE,
        )

        if getattr(args, "vocab_size", None) not in (None, O200K_HARMONY_PADDED_VOCAB_SIZE):
            raise ValueError(
                "O200k Harmony tokenizer requires "
                f"vocab_size={O200K_HARMONY_PADDED_VOCAB_SIZE}"
            )
        if getattr(args, "padded_vocab_size", None) not in (None, O200K_HARMONY_PADDED_VOCAB_SIZE):
            raise ValueError(
                "O200k Harmony tokenizer requires "
                f"padded_vocab_size={O200K_HARMONY_PADDED_VOCAB_SIZE}"
            )
        args.vocab_size = O200K_HARMONY_PADDED_VOCAB_SIZE
        args.padded_vocab_size = O200K_HARMONY_PADDED_VOCAB_SIZE
    else:
        if getattr(args, "vocab_size", None) is None:
            args.vocab_size = getattr(tokenizer, "vocab_size", None)
        if getattr(args, "padded_vocab_size", None) is None:
            make_divisible_by = getattr(args, "make_vocab_size_divisible_by", None)
            tp_size = getattr(args, "tensor_model_parallel_size", None)
            vocab_size = getattr(args, "vocab_size", None)
            if make_divisible_by is not None and tp_size is not None and vocab_size is not None:
                multiple = int(make_divisible_by) * int(tp_size)
                args.padded_vocab_size = int(math.ceil(int(vocab_size) / multiple) * multiple)

    return tokenizer
