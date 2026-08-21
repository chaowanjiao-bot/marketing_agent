from __future__ import annotations

import time
from collections import Counter
from threading import BoundedSemaphore, Lock
from typing import Callable, TypeVar

from .schemas import Observation
from .tools import AgentTool


T = TypeVar("T")


class GpuScheduler:
    def __init__(self, max_concurrent: int = 1) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self.semaphore = BoundedSemaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.lock = Lock()
        self.active = 0
        self.peak_active = 0
        self.completed: Counter[str] = Counter()

    def run(self, workload: str, callback: Callable[[], T]) -> tuple[T, float]:
        queued_at = time.monotonic()
        self.semaphore.acquire()
        wait = time.monotonic() - queued_at
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            return callback(), wait
        finally:
            with self.lock:
                self.active -= 1
                self.completed[workload] += 1
            self.semaphore.release()

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "max_concurrent": self.max_concurrent, "active": self.active,
                "peak_active": self.peak_active, "completed": dict(self.completed),
            }


class ScheduledTool(AgentTool):
    def __init__(self, tool: AgentTool, scheduler: GpuScheduler) -> None:
        self.tool = tool
        self.scheduler = scheduler
        self.name = tool.name

    def execute(self, arguments: dict[str, object]) -> Observation:
        observation, wait = self.scheduler.run(
            self.name, lambda: self.tool.execute(arguments)
        )
        observation.metrics["gpu_queue_seconds"] = round(wait, 6)
        return observation

    def close(self) -> None:
        close = getattr(self.tool, "close", None)
        if callable(close):
            close()
