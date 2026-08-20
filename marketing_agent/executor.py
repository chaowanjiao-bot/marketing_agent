from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from .graph import run_task
from .schemas import TaskRequest
from .task_store import TaskStore
from .tools import ToolRegistry


def classify_error(exc: Exception) -> tuple[str, bool]:
    if isinstance(exc, FileNotFoundError):
        return "dependency_unavailable", False
    if isinstance(exc, ValueError):
        return "invalid_input", False
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
        return "gpu_out_of_memory", True
    if isinstance(exc, TimeoutError):
        return "execution_timeout", True
    return "execution_error", True


class TaskExecutor:
    def __init__(self, store: TaskStore, registry: ToolRegistry, workers: int = 1) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self.store = store
        self.registry = registry
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent")
        self.futures: dict[str, Future[None]] = {}
        self.lock = Lock()

    def submit(self, task_id: str, request: TaskRequest) -> None:
        self.store.set_status(task_id, "queued", phase="waiting_for_worker")
        future = self.pool.submit(self._execute, task_id, request)
        with self.lock:
            self.futures[task_id] = future

    def cancel(self, task_id: str) -> bool:
        self.store.path(task_id)
        with self.lock:
            future = self.futures.get(task_id)
        if future is None or not future.cancel():
            return False
        self.store.set_status(task_id, "cancelled", phase="cancelled_before_start")
        return True

    def _execute(self, task_id: str, request: TaskRequest) -> None:
        self.store.set_status(task_id, "running", phase="agent_execution")
        try:
            result = run_task(request, registry=self.registry)
            result = self.store.materialize_outputs(task_id, result)
            self.store.save_result(task_id, result)
        except Exception as exc:
            code, retryable = classify_error(exc)
            self.store.set_status(
                task_id,
                "failed",
                phase="failed",
                error_type=type(exc).__name__,
                error_code=code,
                retryable=retryable,
            )

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
