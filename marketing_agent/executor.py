from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from .case_memory import CaseMemory, format_retrieval_context
from .candidates import run_campaign_batch
from .experience import ExperienceMemory
from .schemas import (
    FinalResult, ReviewDecision, ReviewRecord, ReviewStatus, TaskRequest,
)
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
        experience: ExperienceMemory | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self.store = store
        self.registry = registry
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent")
        self.futures: dict[str, Future[None]] = {}
        self.lock = Lock()
        self.memory = memory
        self.experience = experience

    def submit(
        self, task_id: str, request: TaskRequest,
        review_history: list[ReviewRecord] | None = None,
    ) -> None:
        self.store.set_status(task_id, "queued", phase="waiting_for_worker")
        future = self.pool.submit(self._execute, task_id, request, review_history or [])
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

    def _execute(
        self, task_id: str, request: TaskRequest, review_history: list[ReviewRecord]
    ) -> None:
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
            if self.experience is not None:
                request = request.model_copy(update={
                    "experience_strategies": self.experience.strategies()
                })
            result = run_campaign_batch(request, registry=self.registry)
            retrieved_ids = [str(case["case_id"]) for case in retrieved]
            result = result.model_copy(update={
                "memory_used": bool(retrieved),
                "retrieved_case_ids": retrieved_ids,
                "experience_used": bool(request.experience_strategies),
                "trace": [{
                    "event": "memory_retrieval",
                    "used": bool(retrieved),
                    "reason": decision_reason,
                    "case_ids": retrieved_ids,
                }] + result.trace,
            })
            result = self.store.materialize_outputs(task_id, result)
            result = result.model_copy(update={
                "review_round": request.review_round,
                "review_history": review_history,
            })
            if request.review_required and result.assets:
                result = result.model_copy(update={
                    "status": "waiting_for_review",
                    "terminal_reason": "human_review_required",
                    "review_status": ReviewStatus.WAITING,
                    "trace": result.trace + [{
                        "event": "human_review_requested",
                        "round": request.review_round,
                        "best_asset_id": result.best_asset_id,
                    }],
                })
            else:
                result = self._write_memory(task_id, request, result)
                result = self._write_experience(task_id, request, result)
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

    def _write_memory(
        self, task_id: str, request: TaskRequest, result: FinalResult
    ) -> FinalResult:
        if self.memory is None or not result.assets or result.saved_case_id:
            return result
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
        return result.model_copy(update={"saved_case_id": saved_case_id})

    def _write_experience(
        self, task_id: str, request: TaskRequest, result: FinalResult
    ) -> FinalResult:
        if self.experience is None or result.learned_experience_count:
            return result
        learned = self.experience.learn_from_trace(result.trace, metadata={
            "task_id": task_id, "brand_id": result.brand_id,
            "review_round": request.review_round,
        })
        return result.model_copy(update={"learned_experience_count": len(learned)})

    def review(
        self, task_id: str, decision: ReviewDecision, *, feedback: str = "",
        reviewer: str = "human",
    ) -> dict[str, object]:
        with self.lock:
            status = self.store.status(task_id)
            if status.get("status") != "waiting_for_review":
                raise ValueError("task is not waiting for review")
            self.store.set_status(task_id, "review_processing", phase="human_review")
        result = self.store.result_model(task_id)
        if result is None:
            raise RuntimeError("review result is missing")
        request = self.store.request(task_id)
        normalized_feedback = " ".join(feedback.split())
        record = ReviewRecord(
            round=request.review_round, decision=decision,
            feedback=normalized_feedback, reviewer=reviewer,
        )
        history = result.review_history + [record]
        if decision == ReviewDecision.APPROVE:
            approved = result.model_copy(update={
                "status": "completed",
                "terminal_reason": "human_review_approved",
                "review_status": ReviewStatus.APPROVED,
                "review_history": history,
                "trace": result.trace + [{
                    "event": "human_review_approved", "round": request.review_round,
                    "reviewer": reviewer,
                }],
            })
            approved = self._write_memory(task_id, request, approved)
            approved = self._write_experience(task_id, request, approved)
            self.store.save_result(task_id, approved)
            return {"task_id": task_id, "status": "completed", "review_round": request.review_round}
        if not normalized_feedback:
            self.store.set_status(task_id, "waiting_for_review", phase="human_review")
            raise ValueError("revision feedback is required")
        if request.review_round >= request.max_review_rounds:
            self.store.set_status(task_id, "waiting_for_review", phase="human_review")
            raise ValueError("maximum review rounds reached")
        self.store.archive_result(task_id, request.review_round)
        revised_request = request.model_copy(update={
            "review_round": request.review_round + 1,
            "review_feedback": normalized_feedback,
            "seed": request.seed + 97_409,
        })
        self.store.update_request(task_id, revised_request)
        self.submit(task_id, revised_request, history)
        return {
            "task_id": task_id, "status": "queued",
            "review_round": revised_request.review_round,
        }

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
