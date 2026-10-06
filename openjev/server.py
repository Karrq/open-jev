"""HTTP server: one loaded model, many scoring requests.

    openjev serve --host 0.0.0.0 --port 8000
    curl -s localhost:8000/score -H 'content-type: application/json' \\
      -d '{"context": "The capital of France is", "options": [" Paris", " Berlin"]}'
"""
from __future__ import annotations

import gc
import os
import threading
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import packed
from .prefix_store import DEFAULT_MAX_BYTES, DEFAULT_MIN_TOKENS, DEFAULT_TTL_S, PrefixStore
from .scorer import DEFAULT_MODEL, OptionScorer
from .systemone import SystemOneRequest, SystemOneResponse, system_one


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


def create_app(model_path: str | None = None, batch_size: int = 8, backend: str = "auto", device: str = "auto") -> FastAPI:
    model_path = model_path or os.environ.get("OPENJEV_MODEL", DEFAULT_MODEL)
    adapter_path = os.environ.get("OPENJEV_ADAPTER")  # optional LoRA adapter dir
    app = FastAPI(title="openjev", version="0.1.0")
    state: dict = {}
    api_key = os.environ.get("OPENJEV_API_KEY")  # if set, /v1/systemone requires "Authorization: Bearer <key>"
    model_name = os.path.basename(model_path.rstrip("/"))
    readout = os.environ.get("OPENJEV_READOUT", "letters")
    # Prefilled states kept across /v1/systemone requests; OPENJEV_PREFIX_CACHE_GB=0 disables.
    store_bytes = int(float(os.environ.get("OPENJEV_PREFIX_CACHE_GB", DEFAULT_MAX_BYTES / 2**30)) * 2**30)
    store = PrefixStore(
        max_bytes=store_bytes,
        ttl_s=float(os.environ.get("OPENJEV_PREFIX_TTL_S", DEFAULT_TTL_S)),
        min_tokens=int(os.environ.get("OPENJEV_PREFIX_MIN_TOKENS", DEFAULT_MIN_TOKENS)),
    ) if store_bytes > 0 else None
    # MLX is not thread-safe and FastAPI runs sync handlers on a thread pool, so
    # requests take turns on the model.
    # Reentrant so a handler that already holds it can load the model on demand.
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
        return scorer

    def _unload_scorer() -> None:
        state["scorer"] = None
        if store is not None:
            store.entries.clear()  # caches belong to the weights just freed
        gc.collect()
        mx = _mlx()
        if mx is not None:
            mx.clear_cache()
        activity["unloads"] += 1

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

    @app.on_event("startup")
    def _load() -> None:
        _load_scorer()

    @app.get("/health")
    def health() -> dict:
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
            "force_fused": packed._FORCE_FUSED,
            "prefix_cache": store.info() if store is not None else None,
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        try:
            with lock:
                scorer = _scorer()
                _measure_start()
                res = scorer.score(req.context, req.options, norm=req.norm, chat=req.chat, sep=req.sep)
                timing = dict(scorer.last_timing)
                timing.update(_measure_end("score"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        best = max(range(len(res)), key=lambda i: res[i].score)
        return ScoreResponse(
            best=res[best].option,
            best_index=best,
            options=[OptionOut(**r.to_dict()) for r in res],
            timing=timing,
        )

    @app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(_auth)])
    def systemone(req: SystemOneRequest) -> SystemOneResponse:
        """TypeSafe System One contract: state + typed questions -> typed answers (docs.typesafe.ai)."""
        try:
            with lock:
                _measure_start()
                resp = system_one(_scorer(), req, model_name=req.model or model_name, readout=readout, store=store)
                _measure_end("systemone")
                return resp
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


def serve(host: str, port: int, model_path: str | None, batch_size: int, backend: str = "auto", device: str = "auto") -> None:
    import uvicorn

    uvicorn.run(create_app(model_path, batch_size, backend=backend, device=device), host=host, port=port, workers=1)
