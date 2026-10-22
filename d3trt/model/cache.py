"""Cross-request state cache.

One state (a prompt prefix) can be reused by every question that refers to it,
and by later requests with the same state. The cache stores the per-layer
snapshot produced by the state pass: K/V for the 16 full-attention layers and
the conv tail + recurrent state for the 48 GDN layers.

Entries are content-addressed by the state token ids (and, for media, a hash of
the decoded pixels), reference-counted while a question batch is using them, and
evicted LRU-first when the byte budget would be exceeded.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from ..errors import StateCacheError


def tensor_bytes(obj: Any) -> int:
    """Bytes held by a tensor, a (nested) list/tuple of tensors, or a dataclass."""
    if hasattr(obj, "numel") and hasattr(obj, "element_size"):
        return int(obj.numel()) * int(obj.element_size())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(x) for x in obj)
    if hasattr(obj, "__dict__"):
        return sum(tensor_bytes(v) for v in vars(obj).values())
    if hasattr(obj, "__slots__"):
        return sum(tensor_bytes(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
    return 0


@dataclass
class StateEntry:
    key: str
    snapshot: Any
    state_len: int
    nbytes: int
    media_hash: str = ""
    created: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    refcount: int = 0
    hits: int = 0


class StateCache:
    """A byte-budgeted LRU cache of state snapshots."""

    def __init__(self, capacity_bytes: int):
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be >= 0")
        self.capacity_bytes = int(capacity_bytes)
        self._entries: OrderedDict[str, StateEntry] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.rejected = 0

    # ------------------------------------------------------------------
    @property
    def bytes(self) -> int:
        return self._bytes

    @property
    def count(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "capacity_bytes": self.capacity_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "evictions": self.evictions,
                "rejected": self.rejected,
            }

    # ------------------------------------------------------------------
    def get(self, key: str) -> StateEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            entry.last_used = time.monotonic()
            entry.hits += 1
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key: str, snapshot: Any, state_len: int, media_hash: str = "") -> StateEntry | None:
        nbytes = tensor_bytes(snapshot)
        if self.capacity_bytes == 0 or nbytes > self.capacity_bytes:
            self.rejected += 1
            return None
        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._bytes -= existing.nbytes
            self._entries[key] = StateEntry(
                key=key,
                snapshot=snapshot,
                state_len=int(state_len),
                nbytes=nbytes,
                media_hash=media_hash,
            )
            self._bytes += nbytes
            self._trim()
            return self._entries.get(key)

    def acquire(self, key: str) -> StateEntry:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise StateCacheError(f"state {key} is not in the cache")
            entry.refcount += 1
            entry.last_used = time.monotonic()
            self._entries.move_to_end(key)
            return entry

    def release(self, key: str) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.refcount > 0:
                entry.refcount -= 1

    def drop(self, key: str) -> None:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._bytes -= entry.nbytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def _trim(self) -> None:
        while self._bytes > self.capacity_bytes and self._entries:
            key, entry = next(iter(self._entries.items()))
            if entry.refcount > 0:
                # Do not evict an entry a running batch is using. If every entry
                # is pinned, let the caller overshoot until the batch releases.
                break
            self._entries.pop(key, None)
            self._bytes -= entry.nbytes
            self.evictions += 1

    # ------------------------------------------------------------------
    def get_or_compute(
        self,
        key: str,
        factory: Callable[[], tuple[Any, int]],
        *,
        media_hash: str = "",
    ) -> tuple[StateEntry, bool]:
        """Return ``(entry, hit)``; compute and store on a miss.

        The new entry is returned without a reference; callers that need to keep
        it alive across an asynchronous batch should call ``acquire``.
        """
        entry = self.get(key)
        if entry is not None:
            return entry, True
        snapshot, state_len = factory()
        entry = self.put(key, snapshot, state_len, media_hash)
        if entry is None:
            # Budget too small for this state: return an uncached entry.
            entry = StateEntry(key=key, snapshot=snapshot, state_len=state_len, nbytes=tensor_bytes(snapshot))
        return entry, False
