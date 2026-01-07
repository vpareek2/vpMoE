import torch

from megatron.core.transformer.transformer_config import TransformerConfig


def _build_test_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=4,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        add_bias_linear=False,
        add_qkv_bias=False,
    )


def test_grapem_embedding_has_grad_and_no_cache():
    from megatron.core.models.common.embeddings.rotary_pos_embedding import (
        GrapeMRotaryEmbedding,
    )

    emb = GrapeMRotaryEmbedding(
        kv_channels=8,
        rotary_percent=1.0,
        rotary_interleaved=False,
        rotary_base=10000,
        learnable_freq=True,
        share_across_heads=True,
        log_freq_scale=16.0,
        use_cpu_initialization=True,
    )

    out = emb(4)
    assert out.shape == (4, 1, 1, 8)

    out.sum().backward()
    assert emb.log_freq.grad is not None

    with torch.no_grad():
        before = emb(4).clone()
        emb.log_freq.add_(0.1)
        after = emb(4)

    assert not torch.allclose(before, after)


def test_alibi_bias_values():
    from megatron.core.models.common.embeddings.alibi import AlibiBias, build_alibi_slopes

    slopes = build_alibi_slopes(2)
    bias_module = AlibiBias(slopes=torch.tensor(slopes, dtype=torch.float32))
    bias = bias_module.get_bias(seq_len=4, device=torch.device("cpu"), dtype=torch.float32)

    assert bias.shape == (1, 2, 4, 4)
    assert bias[0, 0, 0, 0].item() == 0.0
    assert bias[0, 1, 3, 1].item() == -2.0 * slopes[1]
    assert bias[0, 0, 0, 3].item() == 0.0


def test_tpa_linear_qkv_matches_reference():
    from megatron.core.transformer.tpa import TPALinearQKV

    config = _build_test_config()
    config.tpa_rank = 2
    config.tpa_q_rank = 4

    tpa = TPALinearQKV(
        config.hidden_size,
        config.kv_channels * config.num_attention_heads
        + 2 * config.kv_channels * config.num_query_groups,
        config=config,
        init_method=config.init_method,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
        tp_comm_buffer_name="qkv",
        tp_group=None,
    )

    seq_len = 3
    batch_size = 2
    hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)

    mixed, _ = tpa(hidden_states)

    a_q, _ = tpa.linear_a_q(hidden_states)
    a_k, _ = tpa.linear_a_k(hidden_states)
    a_v, _ = tpa.linear_a_v(hidden_states)
    b_q, _ = tpa.linear_b_q(hidden_states)
    b_k, _ = tpa.linear_b_k(hidden_states)
    b_v, _ = tpa.linear_b_v(hidden_states)

    h = tpa.num_attention_heads_per_partition
    g = tpa.num_query_groups_per_partition
    d = tpa.head_dim
    q_rank = tpa.tpa_q_rank
    rank = tpa.tpa_rank

    a_q = a_q.view(seq_len, batch_size, h, q_rank)
    a_k = a_k.view(seq_len, batch_size, g, rank)
    a_v = a_v.view(seq_len, batch_size, g, rank)
    b_q = b_q.view(seq_len, batch_size, q_rank, d)
    b_k = b_k.view(seq_len, batch_size, rank, d)
    b_v = b_v.view(seq_len, batch_size, rank, d)

    q = torch.matmul(a_q.view(seq_len * batch_size, h, q_rank), b_q.view(seq_len * batch_size, q_rank, d))
    q = q.div_(q_rank).view(seq_len, batch_size, h, d)
    k = torch.matmul(a_k.view(seq_len * batch_size, g, rank), b_k.view(seq_len * batch_size, rank, d))
    k = k.div_(rank).view(seq_len, batch_size, g, d)
    v = torch.matmul(a_v.view(seq_len * batch_size, g, rank), b_v.view(seq_len * batch_size, rank, d))
    v = v.div_(rank).view(seq_len, batch_size, g, d)

    heads_per_group = h // g
    q = q.view(seq_len, batch_size, g, heads_per_group, d)
    q = q.reshape(seq_len, batch_size, g, heads_per_group * d)
    k = k.reshape(seq_len, batch_size, g, d)
    v = v.reshape(seq_len, batch_size, g, d)
    mixed_ref = torch.cat([q, k, v], dim=3).reshape(seq_len, batch_size, -1)

    assert mixed.shape == mixed_ref.shape
    assert torch.allclose(mixed, mixed_ref)


def test_tpa_linear_qkv_bias_adds_offset():
    from megatron.core.transformer.tpa import TPALinearQKV

    config = _build_test_config()
    config.tpa_rank = 2
    config.tpa_q_rank = 4

    tpa = TPALinearQKV(
        config.hidden_size,
        config.kv_channels * config.num_attention_heads
        + 2 * config.kv_channels * config.num_query_groups,
        config=config,
        init_method=config.init_method,
        gather_output=False,
        bias=True,
        skip_bias_add=False,
        is_expert=False,
        tp_comm_buffer_name="qkv",
        tp_group=None,
    )

    assert tpa.bias is not None
    with torch.no_grad():
        tpa.bias.copy_(torch.linspace(0, 1, tpa.bias.numel()))

    seq_len = 3
    batch_size = 2
    hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)

    mixed, _ = tpa(hidden_states)

    a_q, _ = tpa.linear_a_q(hidden_states)
    a_k, _ = tpa.linear_a_k(hidden_states)
    a_v, _ = tpa.linear_a_v(hidden_states)
    b_q, _ = tpa.linear_b_q(hidden_states)
    b_k, _ = tpa.linear_b_k(hidden_states)
    b_v, _ = tpa.linear_b_v(hidden_states)

    h = tpa.num_attention_heads_per_partition
    g = tpa.num_query_groups_per_partition
    d = tpa.head_dim
    q_rank = tpa.tpa_q_rank
    rank = tpa.tpa_rank

    a_q = a_q.view(seq_len, batch_size, h, q_rank)
    a_k = a_k.view(seq_len, batch_size, g, rank)
    a_v = a_v.view(seq_len, batch_size, g, rank)
    b_q = b_q.view(seq_len, batch_size, q_rank, d)
    b_k = b_k.view(seq_len, batch_size, rank, d)
    b_v = b_v.view(seq_len, batch_size, rank, d)

    q = torch.matmul(a_q.view(seq_len * batch_size, h, q_rank), b_q.view(seq_len * batch_size, q_rank, d))
    q = q.div_(q_rank).view(seq_len, batch_size, h, d)
    k = torch.matmul(a_k.view(seq_len * batch_size, g, rank), b_k.view(seq_len * batch_size, rank, d))
    k = k.div_(rank).view(seq_len, batch_size, g, d)
    v = torch.matmul(a_v.view(seq_len * batch_size, g, rank), b_v.view(seq_len * batch_size, rank, d))
    v = v.div_(rank).view(seq_len, batch_size, g, d)

    heads_per_group = h // g
    q = q.view(seq_len, batch_size, g, heads_per_group, d)
    q = q.reshape(seq_len, batch_size, g, heads_per_group * d)
    k = k.reshape(seq_len, batch_size, g, d)
    v = v.reshape(seq_len, batch_size, g, d)
    mixed_ref = torch.cat([q, k, v], dim=3).reshape(seq_len, batch_size, -1)
    mixed_ref = mixed_ref + tpa.bias

    assert mixed.shape == mixed_ref.shape
    assert torch.allclose(mixed, mixed_ref)
