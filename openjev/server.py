"""HTTP server: one loaded model, many scoring requests.

    openjev serve --host 0.0.0.0 --port 8000
    curl -s localhost:8000/score -H 'content-type: application/json' \\
      -d '{"context": "The capital of France is", "options": [" Paris", " Berlin"]}'
"""
from __future__ import annotations

import os
import threading
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .prefix_store import DEFAULT_MAX_BYTES, DEFAULT_MIN_TOKENS, DEFAULT_TTL_S, PrefixStore
from .scorer import DEFAULT_MODEL, OptionScorer
from .systemone import SystemOneRequest, SystemOneResponse, system_one


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


def create_app(model_path: str | None = None, batch_size: int = 8) -> FastAPI:
    model_path = model_path or os.environ.get("OPENJEV_MODEL", DEFAULT_MODEL)
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
    lock = threading.Lock()

    def _auth(authorization: str | None = Header(default=None)) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid or missing API key")

    @app.on_event("startup")
    def _load() -> None:
        t = time.perf_counter()
        scorer = OptionScorer(model_path, batch_size=batch_size)
        scorer.score("warm up", ["a", "b"])  # compile kernels before the first request
        state["scorer"] = scorer
        state["load_s"] = time.perf_counter() - t

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": "scorer" in state,
            "model": model_path,
            "load_s": state.get("load_s"),
            "prefix_cache": store.info() if store is not None else None,
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        scorer: OptionScorer = state.get("scorer")
        if scorer is None:
            raise HTTPException(503, "model still loading")
        try:
            with lock:
                res = scorer.score(req.context, req.options, norm=req.norm, chat=req.chat, sep=req.sep)
                timing = dict(scorer.last_timing)
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
        scorer: OptionScorer = state.get("scorer")
        if scorer is None:
            raise HTTPException(503, "model still loading")
        try:
            with lock:
                return system_one(scorer, req, model_name=req.model or model_name, readout=readout, store=store)
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


def serve(host: str, port: int, model_path: str | None, batch_size: int) -> None:
    import uvicorn

    uvicorn.run(create_app(model_path, batch_size), host=host, port=port, workers=1)
