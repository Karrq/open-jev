"""Shortest-job-first scheduling for the single-model server.

The model serves one request at a time, so the only scheduling decision is which
of the waiting requests runs next. A single worker thread pulls jobs cheapest
first, which keeps a burst of large requests from queueing small ones behind them.

Cost is an estimate in tokens, a proxy for latency: prefill and option scoring are
roughly linear in token count, and a systemone state dominates every question
against it. Only the ordering between queued jobs matters, so the proxy's units
are irrelevant.

A purely shortest-first queue starves large jobs whenever small ones keep
arriving. Aging prevents that: a job's effective cost decays linearly to zero over
``aging_s`` seconds of waiting, so after that long its size no longer matters and
it runs. ``aging_s=0`` disables aging and leaves pure shortest-first. If the queue
is bounded, ``submit`` raises :class:`QueueFull` instead of growing without limit.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable

__all__ = ["Job", "PriorityScheduler", "QueueFull"]


class QueueFull(Exception):
    """Raised by :meth:`PriorityScheduler.submit` when the queue is at capacity."""


def _running_loop():
    """The running asyncio loop, or None when submit is called off-loop."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class Job:
    """A submitted unit of work and the handle used to collect its result.

    Await it from an asyncio handler, or call :meth:`wait` from a thread.
    """

    __slots__ = (
        "cost",
        "seq",
        "fn",
        "cancelled",
        "_enqueued_at",
        "_started_at",
        "_finished_at",
        "_event",
        "_loop",
        "_future",
        "_value",
        "_error",
    )

    def __init__(self, cost: float, seq: int, fn: Callable[[], Any], loop) -> None:
        self.cost = float(cost)
        self.seq = seq
        self.fn = fn
        self.cancelled = False
        self._enqueued_at = time.monotonic()
        self._started_at = 0.0
        self._finished_at = 0.0
        self._event = threading.Event()
        self._loop = loop
        self._future = loop.create_future() if loop is not None else None
        self._value = None
        self._error = None
        if self._future is not None:
            self._future.add_done_callback(self._on_future_done)

    # ------------------------------------------------------------ scheduling
    @property
    def waited_s(self) -> float:
        """Seconds from submit to start, or to now while still queued."""
        end = self._started_at or time.monotonic()
        return max(0.0, end - self._enqueued_at)

    @property
    def run_s(self) -> float:
        """Seconds spent executing, or 0 if it has not finished."""
        if not self._started_at or not self._finished_at:
            return 0.0
        return max(0.0, self._finished_at - self._started_at)

    def priority_key(self, now: float, aging_s: float) -> tuple[float, int]:
        """Sort key: smallest runs first, ties in arrival order.

        Aging scales the cost down as the job waits; at ``aging_s`` the cost is
        zero and only arrival order remains.
        """
        if aging_s <= 0:
            return self.cost, self.seq
        decay = 1.0 - (now - self._enqueued_at) / aging_s
        return (self.cost * decay if decay > 0 else 0.0), self.seq

    # ---------------------------------------------------------------- result
    def _complete(self, value: Any) -> None:
        self._value = value
        self._event.set()
        self._schedule(self._set_result, value)

    def _fail(self, exc: BaseException) -> None:
        self._error = exc
        self._event.set()
        self._schedule(self._set_exception, exc)

    def _schedule(self, fn: Callable[[Any], None], arg: Any) -> None:
        fut = self._future
        if fut is not None and not fut.done():
            try:
                self._loop.call_soon_threadsafe(fn, arg)
            except RuntimeError:
                pass  # loop closed during shutdown

    def _set_result(self, value: Any) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_result(value)

    def _set_exception(self, exc: BaseException) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_exception(exc)

    def _on_future_done(self, fut) -> None:
        if fut.cancelled():
            # The waiting handler went away; drop the job if it has not started.
            self.cancelled = True
            self._event.set()

    def wait(self, timeout: float | None = None) -> Any:
        """Blocking result. Raises the job's exception, or TimeoutError."""
        if not self._event.wait(timeout):
            raise TimeoutError(f"job {self.seq} did not finish within {timeout}s")
        if self._error is not None:
            raise self._error
        return self._value

    def __await__(self):
        if self._future is None:
            raise RuntimeError("job has no event loop; use wait()")
        return self._future.__await__()


class PriorityScheduler:
    """One worker thread that runs queued jobs cheapest first, with aging."""

    def __init__(self, *, aging_s: float = 30.0, max_pending: int = 0, name: str = "openjev") -> None:
        self.aging_s = max(0.0, float(aging_s))
        self.max_pending = max(0, int(max_pending))
        self.name = name
        self._pending: list[Job] = []
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._seq = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._current: Job | None = None
        self.stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "rejected": 0,
            "cancelled": 0,
            "max_pending": 0,
            "busy_s": 0.0,
        }

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-scheduler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop accepting dispatch. A running job is allowed to finish."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    def submit(self, cost: float, fn: Callable[[], Any]) -> Job:
        """Queue ``fn`` and return a :class:`Job` that yields its result."""
        loop = _running_loop()
        with self._lock:
            live = sum(1 for h in self._pending if not h.cancelled)
            if self.max_pending and live >= self.max_pending:
                self.stats["rejected"] += 1
                raise QueueFull(f"queue holds {live} pending requests (max {self.max_pending})")
            self._seq += 1
            job = Job(cost, self._seq, fn, loop)
            self._pending.append(job)
            self.stats["submitted"] += 1
            self.stats["max_pending"] = max(self.stats["max_pending"], live + 1)
            self._wake.notify()
        return job

    def info(self) -> dict:
        with self._lock:
            now = time.monotonic()
            pending = [h for h in self._pending if not h.cancelled]
            current = self._current
            return {
                **self.stats,
                "busy_s": round(self.stats["busy_s"], 1),
                "pending": len(pending),
                "running_cost": round(current.cost, 1) if current is not None else None,
                "running_for_s": round(now - current._started_at, 1) if current is not None else None,
                "aging_s": self.aging_s,
                "max_pending_limit": self.max_pending,
                "pending_costs": [round(h.cost) for h in pending],
                "pending_wait_s": [round(h.waited_s, 1) for h in pending],
            }

    # --------------------------------------------------------------- worker
    def _loop(self) -> None:
        while True:
            with self._lock:
                while self._running and all(h.cancelled for h in self._pending):
                    self._wake.wait()
                if not self._running:
                    return
                if any(h.cancelled for h in self._pending):
                    dropped = sum(1 for h in self._pending if h.cancelled)
                    self._pending[:] = [h for h in self._pending if not h.cancelled]
                    self.stats["cancelled"] += dropped
                now = time.monotonic()
                job = min(self._pending, key=lambda h: h.priority_key(now, self.aging_s))
                self._pending.remove(job)
                job._started_at = now
                self._current = job
            self._execute(job)
            with self._lock:
                self._current = None

    def _execute(self, job: Job) -> None:
        try:
            value = job.fn()
        except Exception as exc:  # keep the worker alive for later jobs
            job._finished_at = time.monotonic()
            self.stats["failed"] += 1
            self.stats["busy_s"] += job.run_s
            job._fail(exc)
        else:
            job._finished_at = time.monotonic()
            self.stats["completed"] += 1
            self.stats["busy_s"] += job.run_s
            job._complete(value)
