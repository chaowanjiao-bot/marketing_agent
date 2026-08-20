import time
from pathlib import Path

from marketing_agent.executor import TaskExecutor
from marketing_agent.schemas import TaskRequest
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
