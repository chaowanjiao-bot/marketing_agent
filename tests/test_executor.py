import time
from pathlib import Path

from marketing_agent.executor import TaskExecutor
from marketing_agent.case_memory import CaseMemory
from marketing_agent.schemas import TaskRequest
from marketing_agent.schemas import ReviewDecision
from marketing_agent.task_store import TaskStore
from marketing_agent.tools import (
    MockEditTool,
    MockEvaluateTool,
    MockGenerateTool,
    ToolRegistry,
    build_default_registry,
)


def test_executor_runs_task_and_persists_terminal_state(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    request = TaskRequest(prompt="Create a premium perfume poster", max_iterations=8)
    task_id = store.create(request)
    executor = TaskExecutor(store, build_default_registry(), workers=1)
    executor.submit(task_id, request)
    for _ in range(100):
        if store.status(task_id)["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert store.status(task_id)["status"] == "completed"
    executor.shutdown()


class SlowGenerateTool(MockGenerateTool):
    def execute(self, arguments):
        time.sleep(0.2)
        return super().execute(arguments)


def test_executor_cancels_a_queued_task(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(SlowGenerateTool())
    registry.register(MockEditTool())
    registry.register(MockEvaluateTool())
    store = TaskStore(tmp_path)
    first = TaskRequest(prompt="Create poster one", max_iterations=8)
    second = TaskRequest(prompt="Create poster two", max_iterations=8)
    first_id = store.create(first)
    second_id = store.create(second)
    executor = TaskExecutor(store, registry, workers=1)

    executor.submit(first_id, first)
    executor.submit(second_id, second)

    assert executor.cancel(second_id)
    assert store.status(second_id)["status"] == "cancelled"
    assert store.status(second_id)["phase"] == "cancelled_before_start"
    executor.shutdown()


def test_executor_retrieves_and_writes_back_case(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")
    memory = CaseMemory(tmp_path / "memory.sqlite3")
    seed_id = memory.add(prompt="高端香水广告海报", enhanced_prompt="黑金棚拍构图", score=0.9)
    request = TaskRequest(prompt="生成高端香水营销海报", max_iterations=8)
    task_id = store.create(request)
    executor = TaskExecutor(store, build_default_registry(), memory=memory)
    executor.submit(task_id, request)
    for _ in range(100):
        if store.status(task_id)["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    result = store.result(task_id)
    assert result is not None
    assert result["memory_used"] is True
    assert seed_id in result["retrieved_case_ids"]
    assert result["saved_case_id"].startswith("case_")
    assert memory.count() == 2
    executor.shutdown()


def wait_for(store: TaskStore, task_id: str, expected: set[str]) -> str:
    status = ""
    for _ in range(200):
        status = str(store.status(task_id)["status"])
        if status in expected:
            return status
        time.sleep(0.01)
    return status


def test_review_approval_delays_memory_writeback(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")
    memory = CaseMemory(tmp_path / "memory.sqlite3")
    request = TaskRequest(
        prompt="生成高端香水活动海报", max_iterations=8, review_required=True,
    )
    task_id = store.create(request)
    executor = TaskExecutor(store, build_default_registry(), memory=memory)
    executor.submit(task_id, request)
    assert wait_for(store, task_id, {"waiting_for_review", "failed"}) == "waiting_for_review"
    preview = store.result(task_id)
    assert preview["review_status"] == "waiting_for_review"
    assert memory.count() == 0

    response = executor.review(task_id, ReviewDecision.APPROVE, reviewer="creative_lead")
    assert response["status"] == "completed"
    approved = store.result(task_id)
    assert approved["review_status"] == "approved"
    assert approved["review_history"][0]["reviewer"] == "creative_lead"
    assert memory.count() == 1
    executor.shutdown()


def test_review_revision_resumes_with_feedback_and_archives_preview(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")
    request = TaskRequest(
        prompt="生成高端精华活动海报", max_iterations=8, review_required=True,
    )
    task_id = store.create(request)
    executor = TaskExecutor(store, build_default_registry())
    executor.submit(task_id, request)
    assert wait_for(store, task_id, {"waiting_for_review", "failed"}) == "waiting_for_review"

    response = executor.review(
        task_id, ReviewDecision.REVISE,
        feedback="产品放大到画面高度 55%，标题上移",
    )
    assert response["review_round"] == 1
    assert wait_for(store, task_id, {"waiting_for_review", "failed"}) == "waiting_for_review"
    revised = store.result(task_id)
    assert revised["review_round"] == 1
    assert revised["review_history"][0]["decision"] == "revise"
    assert "人工审核修订要求" in revised["assets"][0]["prompt"]
    assert "产品放大到画面高度 55%" in revised["assets"][0]["prompt"]
    assert (store.path(task_id) / "result.review_0.json").is_file()
    executor.shutdown()
