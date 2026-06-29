"""Tests for the KV cache managers (no LiteRT stack required)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from hermes.kv_cache import (  # noqa: E402
    PagedKVCache,
    SlidingWindowKVCache,
    StaticKVCache,
)


def _kv(num_kv_heads, t, head_dim, batch=1):
    return (
        torch.randn(batch, num_kv_heads, t, head_dim),
        torch.randn(batch, num_kv_heads, t, head_dim),
    )


def test_static_cache_update_get():
    cache = StaticKVCache(num_layers=2, num_kv_heads=2, max_seq_len=16, head_dim=4)
    k, v = _kv(2, 5, 4)
    out_k, out_v = cache.update(0, k, v, position=0)
    assert out_k.shape == (1, 2, 5, 4)
    assert cache.current_len == 5
    got_k, got_v = cache.get(0)
    assert torch.allclose(got_k, k)
    assert torch.allclose(got_v, v)


def test_static_cache_overflow_raises():
    cache = StaticKVCache(num_layers=1, num_kv_heads=2, max_seq_len=8, head_dim=4)
    k, v = _kv(2, 6, 4)
    with pytest.raises(ValueError):
        cache.update(0, k, v, position=4)  # 4 + 6 = 10 > 8


def test_sliding_window_evicts():
    window = 8
    cache = SlidingWindowKVCache(
        num_layers=1, num_kv_heads=1, max_seq_len=64, head_dim=2, window_size=window
    )
    # Insert window_size + 4 tokens one at a time with distinct values.
    total = window + 4
    for i in range(total):
        k = torch.full((1, 1, 1, 2), float(i))
        v = torch.full((1, 1, 1, 2), float(i))
        cache.update(0, k, v, position=i)
    got_k, _ = cache.get(0)
    assert got_k.shape[2] == window
    # Oldest 4 tokens (values 0..3) should be gone; newest value present.
    seen = {int(x) for x in got_k[0, 0, :, 0].tolist()}
    assert 0 not in seen and 3 not in seen
    assert (total - 1) in seen


def test_paged_cache_alloc_free():
    cache = PagedKVCache(num_layers=2, num_kv_heads=1, head_dim=4, num_blocks=8, block_size=16)
    b0 = cache.allocate_block(seq_id=1)
    b1 = cache.allocate_block(seq_id=1)
    b2 = cache.allocate_block(seq_id=2)
    assert cache.num_used_blocks == 3
    assert {b0, b1} == set(cache.get_page_table()[1])
    freed = cache.free_sequence(1)
    assert set(freed) == {b0, b1}
    assert cache.num_used_blocks == 1
    assert cache.get_page_table()[2] == [b2]


def test_static_cache_serialization_roundtrip():
    cache = StaticKVCache(num_layers=1, num_kv_heads=2, max_seq_len=8, head_dim=4)
    k, v = _kv(2, 3, 4)
    cache.update(0, k, v, position=0)
    state = cache.state_dict()
    restored = StaticKVCache(num_layers=1, num_kv_heads=2, max_seq_len=8, head_dim=4)
    restored.load_state_dict(state)
    assert restored.current_len == 3
    assert torch.allclose(restored.get(0)[0], cache.get(0)[0])
