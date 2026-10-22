from __future__ import annotations

import torch

from d3trt.model.cache import StateCache, StateEntry, tensor_bytes


def test_tensor_bytes_handles_nested():
    payload = [torch.zeros(2, 3, dtype=torch.float32), (torch.zeros(4, dtype=torch.float16),)]
    value = tensor_bytes(payload)
    assert value == 2 * 3 * 4 + 4 * 2


def test_put_get_and_lru_eviction():
    cache = StateCache(64)
    a = torch.zeros(8, dtype=torch.float32)  # 32 bytes
    b = torch.zeros(8, dtype=torch.float32)
    c = torch.zeros(8, dtype=torch.float32)
    assert cache.put("a", [a], 1) is not None
    assert cache.put("b", [b], 1) is not None
    assert cache.bytes == 64
    assert cache.get("a") is not None
    # c forces one eviction; a was just used, so b is the LRU entry.
    assert cache.put("c", [c], 1) is not None
    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache
    assert cache.evictions == 1


def test_refcount_pins_an_entry():
    cache = StateCache(64)
    a = torch.zeros(8, dtype=torch.float32)
    b = torch.zeros(8, dtype=torch.float32)
    c = torch.zeros(8, dtype=torch.float32)
    cache.put("a", [a], 1)
    cache.acquire("a")
    cache.put("b", [b], 1)
    cache.put("c", [c], 1)  # cannot evict a; b/c may be evicted instead
    assert "a" in cache
    cache.release("a")
    cache.put("d", [torch.zeros(8)], 1)
    assert "a" not in cache or cache.bytes <= cache.capacity_bytes


def test_get_or_compute_reports_hit():
    cache = StateCache(1024)
    calls = []

    def factory():
        calls.append(1)
        return [torch.ones(2, 2)], 3

    entry, hit = cache.get_or_compute("k", factory)
    assert not hit and entry.state_len == 3 and len(calls) == 1
    entry2, hit2 = cache.get_or_compute("k", factory)
    assert hit2 and len(calls) == 1
    assert entry2 is entry


def test_budget_too_small_returns_uncached_entry():
    cache = StateCache(1)
    entry, hit = cache.get_or_compute("k", lambda: ([torch.ones(2, 2)], 2))
    assert not hit
    assert entry.nbytes == 16
    assert cache.count == 0
