import pytest
import torch
from torch.autograd import gradcheck
from torch.nn import functional as F

from drills.week03 import StaticKVCache, gqa


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("masked", [False, True])
def test_attention_reference_forward_backward(kv_heads, masked):
    q = torch.randn(1, 4, 3, 2, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, kv_heads, 3, 2, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    allowed = torch.ones(3, 3, dtype=torch.bool).tril() if masked else None
    actual = gqa(q, k, v, allowed)
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(4 // kv_heads, dim=1),
        v.repeat_interleave(4 // kv_heads, dim=1), attn_mask=allowed)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    probe = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (q, k, v), probe)
    ge = torch.autograd.grad(expected, (q, k, v), probe)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-9)
    assert gradcheck(lambda a, b, c: gqa(a, b, c, allowed), (q, k, v))


def test_fully_masked_rows_are_zero_with_finite_gradients():
    q = torch.randn(1, 2, 2, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    mask = torch.tensor([[False, False, False], [True, False, True]])
    out = gqa(q, k, v, mask)
    torch.testing.assert_close(out[:, :, 0], torch.zeros_like(out[:, :, 0]))
    grads = torch.autograd.grad(out.sum(), (q, k, v))
    assert all(torch.isfinite(g).all() for g in grads)
    torch.testing.assert_close(grads[0][:, :, 0], torch.zeros_like(grads[0][:, :, 0]))


def test_causal_future_isolation_and_offset_cached_chunks():
    q = torch.randn(1, 4, 5, 4, dtype=torch.float64)
    k = torch.randn(1, 2, 5, 4, dtype=torch.float64)
    v = torch.randn_like(k)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    full = gqa(q, k, v, mask)
    changed = v.clone()
    changed[:, :, 3:] += 100
    torch.testing.assert_close(gqa(q, k, changed, mask)[:, :, :3], full[:, :, :3])
    cache = StaticKVCache(1, 2, 5, 4, dtype=torch.float64)
    outputs = []
    for start, stop in [(0, 2), (2, 4), (4, 5)]:
        keys, values = cache.append(k[:, :, start:stop], v[:, :, start:stop])
        allowed = torch.arange(stop)[None, :] <= torch.arange(start, stop)[:, None]
        outputs.append(gqa(q[:, :, start:stop], keys, values, allowed))
    torch.testing.assert_close(torch.cat(outputs, dim=2), full, rtol=1e-9, atol=1e-10)


def test_static_cache_storage_bytes_reset_and_atomic_overflow():
    cache = StaticKVCache(2, 2, 4, 3, dtype=torch.float64)
    assert cache.length == 0
    pointers = (cache.k.untyped_storage().data_ptr(), cache.v.untyped_storage().data_ptr())
    assert pointers[0] != pointers[1]
    assert cache.k.untyped_storage().nbytes() + cache.v.untyped_storage().nbytes() == 768
    key = torch.randn(2, 2, 3, 3, dtype=torch.float64, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    keys, values = cache.append(key, value)
    assert cache.length == 3 and keys.shape == (2, 2, 3, 3)
    assert not keys.requires_grad and not values.requires_grad
    torch.testing.assert_close(keys, key.detach())
    torch.testing.assert_close(values, value.detach())
    assert keys.untyped_storage().data_ptr() == pointers[0]
    assert values.untyped_storage().data_ptr() == pointers[1]
    before = (cache.k.clone(), cache.v.clone())
    with pytest.raises(ValueError):
        cache.append(key, value)
    assert cache.length == 3
    torch.testing.assert_close(cache.k, before[0], equal_nan=True)
    torch.testing.assert_close(cache.v, before[1], equal_nan=True)
    with pytest.raises(ValueError):
        cache.append(torch.zeros(1, 2, 1, 3), torch.zeros(1, 2, 1, 3))
    assert cache.length == 3
    cache.reset()
    assert cache.length == 0 and cache.prefix()[0].shape[2] == 0
    assert (cache.k.untyped_storage().data_ptr(), cache.v.untyped_storage().data_ptr()) == pointers


def test_attention_rejects_nondivisible_heads():
    with pytest.raises(ValueError):
        gqa(torch.zeros(1, 3, 2, 4), torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocation telemetry only")
def test_cuda_append_peak_and_storage_identity(record_property):
    cache = StaticKVCache(1, 2, 128, 16, device="cuda")
    k = torch.randn(1, 2, 8, 16, device="cuda")
    v = torch.randn_like(k)
    pointers = (cache.k.data_ptr(), cache.v.data_ptr())
    torch.cuda.synchronize()
    live_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    cache.append(k, v)
    torch.cuda.synchronize()
    record_property("logical_cache_bytes", 2 * 1 * 2 * 128 * 16 * 4)
    record_property("live_before", live_before)
    record_property("live_after", torch.cuda.memory_allocated())
    record_property("synchronized_peak_allocated", torch.cuda.max_memory_allocated())
    record_property("reserved_after", torch.cuda.memory_reserved())
    assert (cache.k.data_ptr(), cache.v.data_ptr()) == pointers
    assert torch.cuda.max_memory_allocated() >= live_before
