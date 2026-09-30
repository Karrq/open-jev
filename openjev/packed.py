"""Score many question suffixes against one prefilled prefix in a single forward pass.

Scoring each question on its own clone of the prefix cache costs one small forward pass
per question, and batching them the usual way (``mx.repeat`` of the cache across the
batch) copies the whole prefix once per question. At a 20k-token state that copy is
~0.6 GB per question with Gemma 4, and a dropper-sized request has hundreds of questions.

Here the questions stay in a (B, L) batch, so RoPE gives every question the same
positions it would have alone (they all start at the prefix offset). Inside attention
the batch is folded into the query axis and every query attends to one shared copy of
the prefix plus its own question's keys, through an explicit mask. The prefix is read,
never written, so the stored cache stays reusable.

Only attention caches (``KVCache``, ``RotatingKVCache``) can be shared this way. Models
with recurrent layers (Qwen3.5's ``ArraysCache``) are rejected by ``supported``.
"""
from __future__ import annotations

import sys
from typing import Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache


class PackedCache:
    """One layer's view of a shared prefix plus B packed question suffixes.

    Presents the ``update_and_fetch``/``make_mask``/``offset`` surface the model's
    attention uses. Keys come back as (1, kv_heads, P + B*L, D): the P retained prefix
    tokens followed by every question's tokens, batch-major.
    """

    def __init__(self, layer_cache, batch: int) -> None:
        if isinstance(layer_cache, RotatingKVCache):
            keys = layer_cache._temporal_order(layer_cache.keys)
            values = layer_cache._temporal_order(layer_cache.values)
            self.window: Optional[int] = layer_cache.max_size
        else:
            keys, values = layer_cache.keys, layer_cache.values
            self.window = None
        n = layer_cache.offset
        # A KVCache buffer can have spare capacity past the offset.
        used = min(n, keys.shape[2])
        self.keys = keys[..., keys.shape[2] - used :, :] if self.window else keys[..., :used, :]
        self.values = values[..., values.shape[2] - used :, :] if self.window else values[..., :used, :]
        self.offset = n
        self.retained = used
        self.batch = batch

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, L, D = keys.shape
        flat_k = keys.transpose(1, 0, 2, 3).reshape(1, H, B * L, D)
        flat_v = values.transpose(1, 0, 2, 3).reshape(1, H, B * L, values.shape[-1])
        return (
            mx.concatenate([self.keys, flat_k], axis=2),
            mx.concatenate([self.values, flat_v], axis=2),
        )

    def make_mask(self, N: int, window_size: Optional[int] = None, return_array: bool = False):
        """Boolean (B*N, P + B*N) mask: the shared prefix within the window, then causal
        attention inside the query's own question only."""
        B, P = self.batch, self.retained
        window = window_size or self.window
        j = mx.arange(N)
        # Prefix key i sits at absolute position offset - P + i; query j at offset + j.
        pre = mx.broadcast_to(mx.arange(P)[None, :] >= 0, (N, P))
        if window is not None:
            pre = mx.arange(P)[None, :] > (P + j[:, None] - window)
        own = j[:, None] >= j[None, :]
        if window is not None:
            own = own & (j[:, None] - j[None, :] < window)
        eye = mx.eye(B, dtype=mx.bool_)
        # (B, N, B, N): query (b, j) sees key (b', j') only when b == b'.
        suf = eye[:, None, :, None] & own[None, :, None, :]
        pre = mx.broadcast_to(pre[None], (B, N, P))
        return mx.concatenate([pre.reshape(B * N, P), suf.reshape(B * N, B * N)], axis=1)


def _packed_sdpa(original):
    def sdpa(queries, keys, values, cache, scale, mask, sinks=None, **kw):
        if not isinstance(cache, PackedCache):
            return original(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks, **kw)
        B, H, L, D = queries.shape
        q = queries.transpose(1, 0, 2, 3).reshape(1, H, B * L, D)
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale, mask=mask, sinks=sinks)
        return out.reshape(H, B, L, -1).transpose(1, 0, 2, 3)

    sdpa.__openjev_packed__ = True
    return sdpa


def install(model) -> None:
    """Route attention through the packed path whenever a PackedCache is passed.

    Model files import ``scaled_dot_product_attention`` into their own namespace, so the
    wrapper replaces that name in every module the model's classes come from. Calls with
    ordinary caches go straight to the original function.
    """
    for mod in {type(m).__module__ for _, m in model.named_modules()}:
        module = sys.modules.get(mod)
        fn = getattr(module, "scaled_dot_product_attention", None)
        if fn is not None and not getattr(fn, "__openjev_packed__", False):
            module.scaled_dot_product_attention = _packed_sdpa(fn)


def supported(cache: list) -> bool:
    return all(type(c) in (KVCache, RotatingKVCache) for c in cache)


def text_model(model):
    """The (text model, head) pair used to read logits at chosen positions only.

    Returns None when the model does not have the usual mlx-lm layout, in which case the
    caller computes full logits instead.
    """
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", None)
    if inner is None or not hasattr(inner, "embed_tokens"):
        return None
    args = getattr(lm, "args", None)
    tied = getattr(args, "tie_word_embeddings", not hasattr(lm, "lm_head"))
    cap = getattr(lm, "final_logit_softcapping", None)

    def head(h: mx.array) -> mx.array:
        out = inner.embed_tokens.as_linear(h) if tied else lm.lm_head(h)
        if cap is not None:
            out = mx.tanh(out / cap) * cap
        return out

    return inner, head
