# Vocabulary Pruning Strategy

## Overview

We train with the full GPT-OSS 200k vocabulary to enable token-level distillation from GPT-OSS-120B, then prune the vocabulary post-training for efficient deployment. This allows us to release two checkpoints:

1. **Full Vocab (201k)** - For users needing multilingual, continued distillation, or fine-tuning
2. **Pruned Vocab (~156k)** - English + code + math focused, multilingual tokens removed

## Motivation

With `h=2048` and 201k vocab, the output head alone is **~411M parameters** - consuming ~33% of active parameter budget for a small model. Since we don't care about multilingual, we can prune unused tokens post-training.

### Parameter Savings (h=2048)

Based on Flab-Pruner's ~22% vocab pruning, we target **~156k tokens** (201k × 0.78).

| Vocab Size | Embedding | LM Head | Total Vocab Params | Savings |
|------------|-----------|---------|-------------------|---------|
| 201k (full) | 411.6M | 411.6M | 823.3M | - |
| 156k (pruned) | 319.5M | 319.5M | 639.0M | ~184M (22%) |

## Research Backing

### Flab-Pruner (arXiv:2412.15921, Dec 2024)

"Less is More: Towards Green Code Large Language Models via Unified Structural Pruning"

- Pruned **22% of parameters** while retaining **97% of performance** on code generation
- Vocabulary pruning is a major component
- Full recovery after light post-training

**Algorithm**:
```
V' = {v ∈ V | U(v) > τ}  # Keep tokens with usage > threshold

# Prune both:
# 1. Embedding matrix: W_e ∈ ℝ^{|V|×d} → W'_e ∈ ℝ^{|V'|×d}
# 2. Output head: W_o ∈ ℝ^{d×|V|} → W'_o ∈ ℝ^{d×|V'|}
```

### VocabTrim (arXiv:2506.22694, ICML 2025 Workshop)

Training-free vocabulary pruning for speculative decoding. Achieves **16% speedup** on Llama-3.2-3B-Instruct.

## Implementation

### Pruning Process

```python
def prune_vocabulary(model, tokenizer, corpus, keep_special=True):
    """
    Prune vocabulary based on token usage in corpus.
    
    Args:
        model: Trained model with full vocab
        tokenizer: Original tokenizer (200k vocab)
        corpus: Target corpus (e.g., English code + text)
        keep_special: Always keep special tokens
    
    Returns:
        pruned_model, pruned_tokenizer
    """
    # Step 1: Collect used tokens
    used_tokens = set()
    for text in corpus:
        tokens = tokenizer.encode(text)
        used_tokens.update(tokens)
    
    # Step 2: Add special tokens
    if keep_special:
        used_tokens.update(tokenizer.all_special_ids)
    
    # Step 3: Create mapping old_id -> new_id
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(used_tokens))}
    
    # Step 4: Prune embedding matrix
    new_embed = model.embed_tokens.weight[list(sorted(used_tokens))]
    
    # Step 5: Prune LM head
    new_lm_head = model.lm_head.weight[list(sorted(used_tokens))]
    
    # Step 6: Update tokenizer vocab + merges (BPE-specific)
    # Need to update merge rules to only include valid token pairs
    
    return pruned_model, pruned_tokenizer
```

### Corpus Selection for Pruning

To determine which tokens to keep, use a representative corpus:

1. **Code**: The Stack v2, StarCoder training data
2. **English text**: FineWeb, RedPajama English subset
3. **Math**: GSM8K, MATH, OpenWebMath

**Important**: Include math/numbers corpus to preserve math capability.

### Tokens to Always Keep

- All digit tokens (0-9 and multi-digit patterns)
- Math operators (+, -, *, /, =, <, >, etc.)
- Common programming symbols
- All special tokens (BOS, EOS, PAD, etc.)

## Workflow

```
Training Phase:
├── Train with full GPT-OSS 201k vocab
├── Token-level distillation from GPT-OSS-120B
└── No compromises during training

Release Phase:
├── Checkpoint 1: Full Vocab (201k)
│   ├── For multilingual users
│   ├── For continued distillation/fine-tuning
│   └── ~412M params in output head
│
└── Checkpoint 2: Pruned Vocab (~156k)
    ├── English + code + math focused
    ├── ~320M params in output head (~22% reduction)
    ├── Faster inference, smaller memory footprint
    └── Optional: Light fine-tuning to recover any degradation
```

## Math Capability Preservation

Flab-Pruner specifically targets code tasks and shows 97% retention. For math:

- Use **corpus-driven pruning** - if training/eval corpus includes math, those tokens are preserved
- GPT-OSS has extensive number tokenization - keep all digit-related tokens
- The pruned model maintains math capability as long as relevant tokens are kept

## Related Tools

- **MergeKit TokenSurgeon**: For transplanting tokenizers between models (different use case)
- **Flab-Pruner**: Reference implementation for unified structural pruning

## References

1. Flab-Pruner: https://arxiv.org/abs/2412.15921
2. VocabTrim: https://arxiv.org/abs/2506.22694
3. MergeKit TokenSurgeon: https://github.com/arcee-ai/mergekit

