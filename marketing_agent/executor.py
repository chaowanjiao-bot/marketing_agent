from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from .case_memory import CaseMemory, format_retrieval_context
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
    def __init__(
        self, store: TaskStore, registry: ToolRegistry, workers: int = 1,
        memory: CaseMemory | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self.store = store
        self.registry = registry
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent")
        self.futures: dict[str, Future[None]] = {}
        self.lock = Lock()
        self.memory = memory

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
            retrieved: list[dict[str, object]] = []
            decision_reason = "memory_disabled"
            if self.memory is not None:
                retrieval_decision, retrieved = self.memory.retrieve_if_needed(
                    request.prompt, limit=request.memory_top_k, override=request.use_memory
                )
                decision_reason = retrieval_decision.reason
                request = request.model_copy(
                    update={"memory_context": format_retrieval_context(retrieved)}
                )
            result = run_task(request, registry=self.registry)
            retrieved_ids = [str(case["case_id"]) for case in retrieved]
            result = result.model_copy(update={
                "memory_used": bool(retrieved),
                "retrieved_case_ids": retrieved_ids,
                "trace": [{
                    "event": "memory_retrieval",
                    "used": bool(retrieved),
                    "reason": decision_reason,
                    "case_ids": retrieved_ids,
                }] + result.trace,
            })
            result = self.store.materialize_outputs(task_id, result)
            if self.memory is not None and result.assets:
                best = next(
                    (asset for asset in result.assets if asset.asset_id == result.best_asset_id),
                    result.assets[-1],
                )
                compliant = result.best_compliant_asset_id == best.asset_id
                saved_case_id = self.memory.add(
                    prompt=request.prompt,
                    enhanced_prompt=best.prompt,
                    asset_path=best.file_path,
                    score=result.best_score,
                    compliant=compliant,
                    source="generated",
                    metadata={"task_id": task_id, "terminal_reason": result.terminal_reason},
                )
                result = result.model_copy(update={"saved_case_id": saved_case_id})
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
