"""HTTP server: one loaded model, many scoring requests.

    openjev serve --host 0.0.0.0 --port 8000
    curl -s localhost:8000/score -H 'content-type: application/json' \\
      -d '{"context": "The capital of France is", "options": [" Paris", " Berlin"]}'
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import packed
from .prefix_store import DEFAULT_MAX_BYTES, DEFAULT_MIN_TOKENS, DEFAULT_TTL_S, PrefixStore
from .scheduler import PriorityScheduler, QueueFull
from .scorer import DEFAULT_MODEL, OptionScorer
from .systemone import SystemOneRequest, SystemOneResponse, state_cache_key, system_one

logger = logging.getLogger("openjev.server")


def _log_request(kind: str, request: Request, fields: dict, timing: dict) -> None:
    """One line per served request: identity, request shape, then timing and memory."""
    client = request.client.host if request.client else "-"
    values = {
        **fields,
        **{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in timing.items()},
    }
    logger.info("%s client=%s %s", kind, client, " ".join(f"{k}={v}" for k, v in values.items()))


def _mlx():
    """The ``mlx.core`` module on the MLX backend, else ``None``.

    Imported lazily so the package and the torch backend load on machines
    without MLX.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return None
    return mx


class ScoreRequest(BaseModel):
    context: str
    options: list[str] = Field(min_length=2)
    norm: Literal["mean", "sum", "pmi"] = "mean"
    chat: bool = False
    sep: str = ""


class OptionOut(BaseModel):
    option: str
    n_tokens: int
    logprob_sum: float
    logprob_mean: float
    logprob_uncond: float | None
    score: float
    probability: float


class ScoreResponse(BaseModel):
    best: str
    best_index: int
    options: list[OptionOut]
    timing: dict[str, float]


# Cost estimates for the scheduler. They only need to rank requests against each
# other, so a cheap character count stands in for tokenization, which would need
# the model and its lock at admission time. Four characters per token is the
# usual English ratio; non-English text shifts the scale, not the ordering.
# systemone additionally subtracts the state a prefix-store entry already holds,
# matched by state content prefix rather than by token ids.
_CHARS_PER_TOKEN = 4.0


def _text_weight(value) -> int:
    """Approximate rendered length of a JSON-ish value without dumping it."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(len(str(k)) + _text_weight(v) for k, v in value.items()) + len(value)
    if isinstance(value, (list, tuple)):
        return sum(_text_weight(v) for v in value) + len(value)
    if value is None:
        return 4
    return len(str(value))


def _estimate_tokens(value) -> float:
    return max(1.0, _text_weight(value) / _CHARS_PER_TOKEN)


def _score_cost(req: ScoreRequest) -> float:
    cost = _estimate_tokens(req.context) + sum(_estimate_tokens(o) for o in req.options)
    return cost * 2 if req.norm == "pmi" else cost


def _systemone_cost(req: SystemOneRequest, store: PrefixStore | None = None, cache_key: str | None = None) -> float:
    # The state is prefilled once and every question scores against it, so the
    # state dominates and the summed questions are the rest. A state the prefix
    # store already holds costs only its uncached remainder.
    state_tokens = _estimate_tokens(req.state)
    if store is not None and cache_key is not None:
        cached = store.lookup_key(cache_key)
        if cached:
            state_tokens = max(0.0, state_tokens - cached)
    return state_tokens + sum(_estimate_tokens(q.model_dump()) for q in req.questions.values())


def create_app(model_path: str | None = None, batch_size: int = 8, backend: str = "auto", device: str = "auto") -> FastAPI:
    model_path = model_path or os.environ.get("OPENJEV_MODEL", DEFAULT_MODEL)
    adapter_path = os.environ.get("OPENJEV_ADAPTER")  # optional LoRA adapter dir
    app = FastAPI(title="openjev", version="0.1.0")
    state: dict = {}
    api_key = os.environ.get("OPENJEV_API_KEY")  # if set, /v1/systemone requires "Authorization: Bearer <key>"
    model_name = os.path.basename(model_path.rstrip("/"))
    readout = os.environ.get("OPENJEV_READOUT", "letters")
    # Waiting requests run cheapest first so short ones are not stuck behind long
    # ones. aging_s is how long a request must wait before its size stops
    # mattering (0 = pure shortest-first); max_pending=0 leaves the queue unbounded.
    scheduler = PriorityScheduler(
        aging_s=float(os.environ.get("OPENJEV_SCHED_AGING_S", "30")),
        max_pending=int(os.environ.get("OPENJEV_SCHED_MAX_PENDING", "0")),
    )
    # Prefilled states kept across /v1/systemone requests; OPENJEV_PREFIX_CACHE_GB=0 disables.
    store_bytes = int(float(os.environ.get("OPENJEV_PREFIX_CACHE_GB", DEFAULT_MAX_BYTES / 2**30)) * 2**30)
    store = PrefixStore(
        max_bytes=store_bytes,
        ttl_s=float(os.environ.get("OPENJEV_PREFIX_TTL_S", DEFAULT_TTL_S)),
        min_tokens=int(os.environ.get("OPENJEV_PREFIX_MIN_TOKENS", DEFAULT_MIN_TOKENS)),
    ) if store_bytes > 0 else None
    # MLX is not thread-safe and one model serves every request, so jobs take
    # turns on it: the scheduler runs them one at a time, here under this lock.
    # Reentrant so a running job can load the model on demand.
    lock = threading.RLock()
    # 0 keeps the weights loaded forever; otherwise they are freed after this many
    # idle seconds and loaded again on the next request.
    idle_unload_s = float(os.environ.get("OPENJEV_IDLE_UNLOAD_S", "0"))
    activity = {"at": time.monotonic(), "unloads": 0}

    # MLX's peak counter is cumulative unless reset, so each request resets it
    # and records its own peak; state["peak_mb"] keeps the high-water mark across
    # requests. Reset at request start, read at request end, under the model lock.
    def _mem() -> dict:
        mx = _mlx()
        if mx is None:
            return {}
        return {
            "active_mb": round(mx.get_active_memory() / 2**20, 1),
            "cache_mb": round(mx.get_cache_memory() / 2**20, 1),
        }

    def _measure_start() -> None:
        mx = _mlx()
        if mx is not None:
            mx.reset_peak_memory()

    def _measure_end(kind: str) -> dict:
        mx = _mlx()
        if mx is None:
            return {}
        peak = round(mx.get_peak_memory() / 2**20, 1)
        state["peak_mb"] = max(state.get("peak_mb", 0.0), peak)
        info = {**_mem(), "request_peak_mb": peak, "peak_mb": round(state["peak_mb"], 1)}
        state["last_request"] = {"kind": kind, **info}
        return info

    def _load_scorer() -> OptionScorer:
        t = time.perf_counter()
        scorer = OptionScorer(model_path, batch_size=batch_size, adapter_path=adapter_path, backend=backend, device=device)
        scorer.score("warm up", ["a", "b"])  # compile kernels before the first request
        state["scorer"] = scorer
        state["load_s"] = time.perf_counter() - t
        state["loaded_at"] = time.monotonic()
        logger.info("loaded %s in %.2fs", model_path, state["load_s"])
        return scorer

    def _unload_scorer() -> None:
        idle_s = time.monotonic() - activity["at"]
        state["scorer"] = None
        if store is not None:
            store.clear()  # caches belong to the weights just freed
        gc.collect()
        mx = _mlx()
        if mx is not None:
            mx.clear_cache()
        activity["unloads"] += 1
        logger.info(
            "unloaded model after %.0fs idle (unload #%d), active_mb=%.1f",
            idle_s,
            activity["unloads"],
            mx.get_active_memory() / 2**20,
        )

    def _scorer() -> OptionScorer:
        """The loaded scorer, loading it under the lock if an idle period unloaded it."""
        scorer = state.get("scorer")
        if scorer is None:
            with lock:
                scorer = state.get("scorer") or _load_scorer()
        activity["at"] = time.monotonic()
        return scorer

    def _idle_watch() -> None:
        while True:
            time.sleep(min(30.0, max(1.0, idle_unload_s / 4)))
            if idle_unload_s <= 0 or state.get("scorer") is None:
                continue
            with lock:
                if time.monotonic() - activity["at"] >= idle_unload_s:
                    _unload_scorer()

    if idle_unload_s > 0:
        threading.Thread(target=_idle_watch, daemon=True).start()

    def _auth(authorization: str | None = Header(default=None)) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid or missing API key")

    # Model work runs on the scheduler thread, one job at a time, under the lock.
    def _run_score(req: ScoreRequest):
        with lock:
            scorer = _scorer()
            _measure_start()
            res = scorer.score(req.context, req.options, norm=req.norm, chat=req.chat, sep=req.sep)
            timing = dict(scorer.last_timing)
            timing.update(_measure_end("score"))
            activity["at"] = time.monotonic()
        return res, timing

    def _run_systemone(req: SystemOneRequest, cache_key: str | None):
        with lock:
            scorer = _scorer()
            _measure_start()
            resp = system_one(
                scorer,
                req,
                model_name=req.model or model_name,
                readout=readout,
                store=store,
                cache_key=cache_key,
            )
            timing = dict(scorer.last_timing)
            timing.update(_measure_end("systemone"))
            activity["at"] = time.monotonic()
        return resp, timing

    @app.on_event("startup")
    def _load() -> None:
        _load_scorer()
        scheduler.start()

    @app.on_event("shutdown")
    def _stop() -> None:
        scheduler.stop()

    @app.get("/health")
    def health() -> dict:
        fused = packed.fused_status()
        return {
            "ok": state.get("scorer") is not None,
            "loaded": state.get("scorer") is not None,
            "model": model_path,
            "load_s": round(state.get("load_s", 0), 2),
            "idle_for_s": round(time.monotonic() - activity["at"], 1),
            "unloads": activity["unloads"],
            "backend": getattr(state.get("scorer"), "backend", None),
            "device": getattr(state.get("scorer"), "device", None),
            "memory": {
                **_mem(),
                "peak_mb": round(state.get("peak_mb", 0.0), 1),
                "last_request": state.get("last_request"),
            },
            "force_fused": fused["enabled"],
            "fused_fallback": fused["fallback"],
            "queue": scheduler.info(),
            "prefix_cache": store.info() if store is not None else None,
        }

    @app.post("/score", response_model=ScoreResponse)
    async def score(req: ScoreRequest, request: Request) -> ScoreResponse:
        cost = _score_cost(req)
        try:
            job = scheduler.submit(cost, lambda: _run_score(req))
        except QueueFull as e:
            raise HTTPException(503, str(e))
        try:
            res, timing = await job
        except ValueError as e:
            raise HTTPException(400, str(e))
        best = max(range(len(res)), key=lambda i: res[i].score)
        _log_request(
            "score",
            request,
            {"options": len(req.options), "best": res[best].option, "cost": round(cost)},
            timing,
        )
        return ScoreResponse(
            best=res[best].option,
            best_index=best,
            options=[OptionOut(**r.to_dict()) for r in res],
            timing=timing,
        )

    @app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(_auth)])
    async def systemone(req: SystemOneRequest, request: Request) -> SystemOneResponse:
        """TypeSafe System One contract: state + typed questions -> typed answers (docs.typesafe.ai)."""
        cache_key = state_cache_key(req.state) if store is not None and readout == "letters" else None
        cost = _systemone_cost(req, store, cache_key)
        try:
            job = scheduler.submit(cost, lambda: _run_systemone(req, cache_key))
        except QueueFull as e:
            raise HTTPException(503, str(e))
        try:
            resp, timing = await job
        except ValueError as e:
            raise HTTPException(400, str(e))
        _log_request(
            "systemone",
            request,
            {
                "questions": len(req.questions),
                "readout": readout,
                "cost": round(cost),
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
            timing,
        )
        return resp

    return app


def serve(host: str, port: int, model_path: str | None, batch_size: int, backend: str = "auto", device: str = "auto") -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    uvicorn.run(create_app(model_path, batch_size, backend=backend, device=device), host=host, port=port, workers=1)
