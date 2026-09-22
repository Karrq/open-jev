"""Zero-shot option scoring with Gemma 3 via mlx-lm (Route B in docs/design/one-pass-option-scoring.md).

The context is prefilled once. Its KV cache is then expanded across the batch
dimension so every option is scored in one padded forward pass that shares the
context prefix. No decoding happens anywhere; the score is the log-probability the
language model assigns to the option tokens given the context.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache

DEFAULT_MODEL = "models/gemma-3-4b-it"
NORMS = ("mean", "sum", "pmi")


@dataclass(frozen=True)
class PrefixCache:
    """A prefilled KV cache for a token prefix shared by several prompts.

    ``ids`` is deliberately one token shorter than the prefix text tokenises to:
    the final token can merge with whatever follows, so dropping it guarantees
    ``ids`` is a true token prefix of every full prompt built on that text.
    """

    ids: list[int]
    cache: list[KVCache]


@dataclass(frozen=True)
class OptionScore:
    option: str
    n_tokens: int
    logprob_sum: float
    logprob_mean: float
    logprob_uncond: float | None
    score: float
    probability: float

    def to_dict(self) -> dict:
        return asdict(self)


def _softmax(xs: Sequence[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


class OptionScorer:
    """Scores pre-written options as continuations of a context with a causal LM.

    Args:
        model_path: local directory or HF repo id understood by ``mlx_lm.load``.
        batch_size: maximum number of options scored in one forward pass.
        chat: wrap the context in Gemma's chat template (user turn, generation
            prompt appended) so options are scored as the start of the reply.
        sep: literal string placed between context and each option (ignored
            when ``chat`` is set, since the template already ends the turn).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        batch_size: int = 8,
        chat: bool = False,
        sep: str = "",
    ) -> None:
        self.model, self.tok = load(model_path)
        self.batch_size = max(1, batch_size)
        self.chat = chat
        self.sep = sep
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.bos_id = self.tok.bos_token_id
        self.last_timing: dict[str, float] = {}

    # ------------------------------------------------------------------ text
    def context_ids(self, context: str, chat: bool | None = None, sep: str | None = None) -> list[int]:
        chat = self.chat if chat is None else chat
        sep = self.sep if sep is None else sep
        if chat:
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": context}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = self.tok.encode(text, add_special_tokens=False)
            if self.bos_id is not None and ids[:1] != [self.bos_id]:
                ids = [self.bos_id] + ids
            return ids
        ids = self.tok.encode(context + sep)  # adds BOS
        if self.bos_id is not None and ids[:1] != [self.bos_id]:
            ids = [self.bos_id] + ids
        return ids

    def option_ids(self, option: str) -> list[int]:
        ids = self.tok.encode(option, add_special_tokens=False)
        if not ids:
            raise ValueError(f"option tokenises to nothing: {option!r}")
        return ids

    # --------------------------------------------------------------- prefill
    def _prefill(self, ids: list[int]) -> tuple[list[KVCache], mx.array]:
        """Prefill with the model's own cache layout.

        Gemma 3 interleaves 29 sliding-window layers with 5 global ones. Giving
        every layer a full KVCache both wastes memory and lets local layers
        attend past their window, which disagrees with a single full-sequence
        forward pass once the context exceeds the window.
        """
        cache = make_prompt_cache(self.model)
        logits = self.model(mx.array(ids)[None], cache=cache)
        last = logits[0, -1]
        mx.eval(last, *[c.keys for c in cache], *[c.values for c in cache])
        self._compact(cache)
        return cache, last

    @staticmethod
    def _compact(cache: Iterable) -> None:
        """Drop entries a sliding-window layer can never attend to again.

        A bulk prefill stores every token, because RotatingKVCache only trims on
        its next update. Trimming to the window now makes the retained prefix
        several times smaller without changing any later result: the next update
        would have trimmed to exactly this same tail anyway.
        """
        for c in cache:
            if not isinstance(c, RotatingKVCache) or c.keys is None:
                continue
            if c.keys.shape[2] <= c.max_size:
                continue
            keys, values = c._temporal_order(c.keys), c._temporal_order(c.values)
            tail = c.max_size - c.keep
            if c.keep:
                keys = mx.concatenate([keys[..., : c.keep, :], keys[..., -tail:, :]], axis=2)
                values = mx.concatenate([values[..., : c.keep, :], values[..., -tail:, :]], axis=2)
            else:
                keys, values = keys[..., -tail:, :], values[..., -tail:, :]
            c.keys, c.values = keys, values
            c._idx = keys.shape[2]
            mx.eval(c.keys, c.values)

    @staticmethod
    def _clone(cache: list[KVCache]) -> list[KVCache]:
        """Independent caches over the same prefix.

        ``KVCache.update_and_fetch`` writes into spare buffer capacity in place,
        so sharing the arrays would let one scoring call corrupt the retained
        prefix. Slicing to ``offset`` both copies and removes the spare capacity,
        which forces the next write to reallocate instead.
        """
        out = []
        for c in cache:
            if isinstance(c, RotatingKVCache):
                e = RotatingKVCache(max_size=c.max_size, keep=c.keep)
                e._idx = c._idx
                # The buffer is in rotated order, so keep it whole.
                used = c.keys.shape[2]
            else:
                e = KVCache()
                used = c.offset
            e.offset = c.offset
            e.keys = c.keys[..., :used, :]
            e.values = c.values[..., :used, :]
            out.append(e)
        return out

    @staticmethod
    def _expand(cache: list, n: int) -> list:
        out = []
        for c in cache:
            if isinstance(c, RotatingKVCache):
                e = RotatingKVCache(max_size=c.max_size, keep=c.keep)
                e._idx = c._idx
            else:
                e = KVCache()
            e.offset = c.offset
            e.keys = mx.repeat(c.keys, n, axis=0)
            e.values = mx.repeat(c.values, n, axis=0)
            out.append(e)
        return out

    def _score_with_prefix(
        self, cache: list[KVCache], last: mx.array, opts: list[list[int]]
    ) -> list[float]:
        """Sum of log p(option tokens | prefix) for each option, sharing the prefix cache."""
        sums: list[float] = []
        for start in range(0, len(opts), self.batch_size):
            chunk = opts[start : start + self.batch_size]
            n, L = len(chunk), max(len(x) for x in chunk)
            arr = mx.array([x + [self.pad_id] * (L - len(x)) for x in chunk])
            mask = mx.array([[1.0] * len(x) + [0.0] * (L - len(x)) for x in chunk])
            logits = self.model(arr, cache=self._expand(cache, n))  # (n, L, V)
            first = mx.broadcast_to(last[None, None, :], (n, 1, logits.shape[-1]))
            pred = mx.concatenate([first, logits[:, :-1]], axis=1).astype(mx.float32)
            lse = mx.logsumexp(pred, axis=-1)  # (n, L)
            tgt = mx.take_along_axis(pred, arr[..., None], axis=-1)[..., 0]
            s = ((tgt - lse) * mask).sum(-1)
            mx.eval(s)
            sums.extend(s.tolist())
        return sums

    # ---------------------------------------------------------------- public
    def score(
        self,
        context: str,
        options: Sequence[str],
        norm: str = "mean",
        chat: bool | None = None,
        sep: str | None = None,
    ) -> list[OptionScore]:
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        if len(options) < 2:
            raise ValueError("need at least two options")
        t0 = time.perf_counter()
        opts = [self.option_ids(o) for o in options]
        ctx_ids = self.context_ids(context, chat=chat, sep=sep)
        cache, last = self._prefill(ctx_ids)
        t1 = time.perf_counter()
        sums = self._score_with_prefix(cache, last, opts)
        t2 = time.perf_counter()

        uncond: list[float] | None = None
        if norm == "pmi":
            base_cache, base_last = self._prefill([self.bos_id])
            uncond = self._score_with_prefix(base_cache, base_last, opts)
        t3 = time.perf_counter()
        self.last_timing = {
            "prefill_s": t1 - t0,
            "options_s": t2 - t1,
            "uncond_s": t3 - t2,
            "total_s": t3 - t0,
            "context_tokens": len(ctx_ids),
            "option_tokens": sum(len(o) for o in opts),
        }

        return self._finalize(options, opts, sums, uncond, norm)

    @staticmethod
    def _finalize(
        options: Sequence[str],
        opts: list[list[int]],
        sums: list[float],
        uncond: list[float] | None,
        norm: str,
    ) -> list[OptionScore]:
        means = [s / len(o) for s, o in zip(sums, opts)]
        if norm == "sum":
            scores = sums
        elif norm == "mean":
            scores = means
        else:
            scores = [s - u for s, u in zip(sums, uncond)]
        probs = _softmax(scores)
        return [
            OptionScore(
                option=o,
                n_tokens=len(ids),
                logprob_sum=s,
                logprob_mean=m,
                logprob_uncond=(uncond[i] if uncond is not None else None),
                score=sc,
                probability=p,
            )
            for i, (o, ids, s, m, sc, p) in enumerate(
                zip(options, opts, sums, means, scores, probs)
            )
        ]

    # ------------------------------------------------------- shared prefixes
    def prefill_prefix(self, text: str) -> PrefixCache | None:
        """Prefill a context prefix once so many prompts can reuse it.

        Returns None when the text is too short to be worth sharing.
        """
        ids = self.context_ids(text, chat=False, sep="")[:-1]
        if len(ids) < 2:
            return None
        t0 = time.perf_counter()
        cache, _ = self._prefill(ids)
        self.last_timing = {"prefix_prefill_s": time.perf_counter() - t0, "prefix_tokens": len(ids)}
        return PrefixCache(ids=ids, cache=cache)

    def score_with_prefix(
        self,
        prefix: PrefixCache,
        context: str,
        options: Sequence[str],
        norm: str = "mean",
    ) -> list[OptionScore]:
        """Like ``score``, but only the tokens past ``prefix`` are encoded.

        ``context`` is still the full prompt; it must begin with the text the
        prefix was built from.
        """
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        if norm == "pmi":
            raise ValueError("pmi normalisation needs an unconditional pass and cannot reuse a prefix")
        if len(options) < 2:
            raise ValueError("need at least two options")

        t0 = time.perf_counter()
        opts = [self.option_ids(o) for o in options]
        ctx_ids = self.context_ids(context, chat=False, sep="")
        n = len(prefix.ids)
        if len(ctx_ids) <= n or ctx_ids[:n] != prefix.ids:
            raise ValueError("context does not start with the prefilled prefix")

        suffix = ctx_ids[n:]
        cache = self._clone(prefix.cache)
        last = self.model(mx.array(suffix)[None], cache=cache)[0, -1]
        mx.eval(last)
        t1 = time.perf_counter()
        sums = self._score_with_prefix(cache, last, opts)
        t2 = time.perf_counter()

        self.last_timing = {
            "prefill_s": t1 - t0,
            "options_s": t2 - t1,
            "uncond_s": 0.0,
            "total_s": t2 - t0,
            "context_tokens": len(ctx_ids),
            "suffix_tokens": len(suffix),
            "option_tokens": sum(len(o) for o in opts),
        }
        return self._finalize(options, opts, sums, None, norm)

    def score_naive(self, context: str, options: Sequence[str]) -> list[float]:
        """Reference: re-encode context + option from scratch per option, no cache.

        Returns the sum of option-token log-probs. Used to verify the prefix-shared
        batched path and to benchmark against it.
        """
        ctx = self.context_ids(context)
        out = []
        for o in options:
            oid = self.option_ids(o)
            ids = mx.array(ctx + oid)[None]
            logits = self.model(ids)[0].astype(mx.float32)
            pred = logits[len(ctx) - 1 : len(ctx) - 1 + len(oid)]
            lp = pred - mx.logsumexp(pred, axis=-1, keepdims=True)
            s = mx.take_along_axis(lp, mx.array(oid)[:, None], axis=-1).sum()
            mx.eval(s)
            out.append(s.item())
        return out


def iter_jsonl(path: str) -> Iterable[dict]:
    import json

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
