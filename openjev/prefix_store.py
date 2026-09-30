"""Prefilled prefixes kept across requests, so a repeated or growing state is not re-read.

A request looks up the stored prefix sharing the most leading tokens with its own,
prefills only the tokens past it, and stores the result. Clients are not identified:
two requests share an entry exactly when their prompts share leading tokens. Growing
states (a dropper appending to its history) therefore chain naturally, and the entry
a request extended is replaced by the extended one instead of both being kept.

Eviction must not let cheap requests push out expensive ones:

- prefixes shorter than ``min_tokens`` are never stored; re-reading them costs well
  under a second, so a stream of short requests (command gating, model routing) never
  touches the store;
- entries idle for more than ``ttl_s`` go first;
- past that, when over ``max_bytes``, entries are evicted by GreedyDual-Size: each
  entry's priority is the clock value at its last use plus its re-prefill time per byte,
  and evicting an entry advances the clock to its priority. Long prefixes cost more
  seconds per byte to rebuild (attention grows with length), so a large dropper state
  outlives a burst of medium file reads, but still ages out if it stops being used.

Stored caches are never written after insertion. Readers take ``trim`` copies (or read
them through packed.PackedCache), which is what lets entries be shared.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from .scorer import WINDOW_SLACK

DEFAULT_MAX_BYTES = 4 << 30
DEFAULT_TTL_S = 900.0
DEFAULT_MIN_TOKENS = 1024


def cache_nbytes(cache: list) -> int:
    return sum(c.keys.nbytes + c.values.nbytes for c in cache if c.keys is not None)


def common_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def trimmable(cache: list, n: int) -> bool:
    """Whether ``trim(cache, n)`` is exact: every sliding-window layer still holds the
    window the next token needs."""
    for c in cache:
        if isinstance(c, RotatingKVCache):
            if c.keys.shape[2] - (c.offset - n) < min(n, c.max_size - 1):
                return False
        elif not isinstance(c, KVCache):
            # Recurrent state cannot be rewound.
            return False
    return True


def trim(cache: list, n: int) -> list:
    """A copy of ``cache`` covering its first ``n`` tokens (see ``trimmable``).

    Slicing to exactly the kept length also leaves no spare capacity, so the next
    write reallocates rather than touching the source buffers.
    """
    out = []
    for c in cache:
        if isinstance(c, RotatingKVCache):
            keys, values = c._temporal_order(c.keys), c._temporal_order(c.values)
            keep = keys.shape[2] - (c.offset - n)
            e = RotatingKVCache(max_size=c.max_size, keep=c.keep)
            e.keys, e.values = keys[..., :keep, :], values[..., :keep, :]
            e._idx = keep
        else:
            e = KVCache()
            e.keys, e.values = c.keys[..., :n, :], c.values[..., :n, :]
        e.offset = n
        out.append(e)
    return out


@dataclass
class Entry:
    ids: list[int]
    cache: list
    nbytes: int
    cost_s: float  # estimated seconds to prefill ids from scratch
    last_used: float = field(default_factory=time.monotonic)
    priority: float = 0.0
    hits: int = 0


class PrefixStore:
    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_BYTES,
        ttl_s: float = DEFAULT_TTL_S,
        min_tokens: int = DEFAULT_MIN_TOKENS,
    ) -> None:
        self.max_bytes = max_bytes
        self.ttl_s = ttl_s
        self.min_tokens = min_tokens
        self.entries: list[Entry] = []
        self.clock = 0.0
        self.stats = {"hits": 0, "misses": 0, "reused_tokens": 0, "evictions": 0}

    @property
    def nbytes(self) -> int:
        return sum(e.nbytes for e in self.entries)

    def lookup(self, ids: list[int]) -> tuple[Entry | None, int]:
        """The entry reusable for the most leading tokens of ``ids``, and that count."""
        self._expire()
        best, best_n = None, 0
        for e in self.entries:
            n = common_len(e.ids, ids)
            if n > best_n and n >= len(e.ids) - WINDOW_SLACK and trimmable(e.cache, n):
                best, best_n = e, n
        if best is None:
            self.stats["misses"] += 1
            return None, 0
        self.stats["hits"] += 1
        self.stats["reused_tokens"] += best_n
        best.hits += 1
        self._touch(best)
        return best, best_n

    def put(self, ids: list[int], cache: list, cost_s: float, parent: Entry | None = None) -> Entry | None:
        """Store ``cache`` for ``ids``; the caller must not modify it afterwards."""
        if len(ids) < self.min_tokens:
            return None
        if parent is not None and parent in self.entries:
            self.entries.remove(parent)
        # An identical or shorter prefix of the same lineage is now redundant.
        self.entries = [e for e in self.entries if not (len(e.ids) <= len(ids) and ids[: len(e.ids)] == e.ids)]
        e = Entry(ids=list(ids), cache=cache, nbytes=cache_nbytes(cache), cost_s=cost_s)
        if e.nbytes > self.max_bytes:
            return None
        self._touch(e)
        self.entries.append(e)
        while self.nbytes > self.max_bytes:
            victim = min(self.entries, key=lambda x: x.priority)
            self.clock = victim.priority
            self.entries.remove(victim)
            self.stats["evictions"] += 1
        return e if e in self.entries else None

    def _touch(self, e: Entry) -> None:
        e.last_used = time.monotonic()
        e.priority = self.clock + e.cost_s / max(1, e.nbytes) * (1 << 30)

    def _expire(self) -> None:
        now = time.monotonic()
        keep = [e for e in self.entries if now - e.last_used <= self.ttl_s]
        self.stats["evictions"] += len(self.entries) - len(keep)
        self.entries = keep

    def info(self) -> dict:
        return {
            **self.stats,
            "entries": len(self.entries),
            "bytes": self.nbytes,
            "tokens": [len(e.ids) for e in self.entries],
        }


def prefill(scorer, store: PrefixStore | None, ids: list[int]) -> tuple[list, dict]:
    """A cache holding ``ids``, reusing and updating ``store``.

    The returned cache may be the stored one, so callers must only read it (through
    ``scorer.last_logits``). Returns reuse and timing details alongside.
    """
    t = time.perf_counter()
    entry, n = store.lookup(ids) if store is not None else (None, 0)
    if entry is not None and n == len(ids) == len(entry.ids):
        return entry.cache, {"prefill_s": time.perf_counter() - t, "reused_tokens": n, "new_tokens": 0}
    cache = trim(entry.cache, n) if entry is not None else None
    if len(ids) > n:
        cache, _ = scorer._prefill(ids[n:], cache=cache)
    dt = time.perf_counter() - t
    if store is not None:
        # Whole-prefix rebuild cost: the reused part's recorded cost plus this delta.
        base = entry.cost_s * n / max(1, len(entry.ids)) if entry is not None else 0.0
        store.put(ids, cache, cost_s=base + dt, parent=entry)
    return cache, {"prefill_s": dt, "reused_tokens": n, "new_tokens": len(ids) - n}
