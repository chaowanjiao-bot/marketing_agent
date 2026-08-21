import time
from pathlib import Path

import pytest

from marketing_agent.experience import (
    ExperienceMemory, ExperienceRecord, InMemoryExperienceStore, JsonlExperienceStore,
)
from marketing_agent.executor import TaskExecutor
from marketing_agent.schemas import TaskRequest
from marketing_agent.task_store import TaskStore
from marketing_agent.tools import build_default_registry


def evaluation(score: float, issue: str, action: str) -> dict:
    return {
        "event": "observation", "tool": "evaluate_image",
        "metrics": {"marketing_alignment": score}, "issues": [issue],
        "repair_actions": [action], "output_format": "1:1",
    }


def test_experience_learns_positive_transition_and_ranks_strategy() -> None:
    memory = ExperienceMemory()
    learned = memory.learn_from_trace([
        evaluation(0.61, "composition", "产品居中并扩大安全边距"),
        evaluation(0.79, "composition", "unused next action"),
    ], metadata={"task_id": "task_1"})
    assert len(learned) == 1
    assert learned[0].delta == pytest.approx(0.18)
    assert learned[0].success is True
    assert memory.strategies()["composition"] == ["产品居中并扩大安全边距"]


def test_failed_experience_is_recorded_but_not_recommended() -> None:
    store = InMemoryExperienceStore()
    memory = ExperienceMemory(store)
    memory.learn_from_trace([
        evaluation(0.70, "lighting", "增加强烈顶光"),
        evaluation(0.60, "lighting", "next"),
    ])
    assert memory.count() == 1
    assert memory.strategies() == {}


def test_jsonl_store_is_lazy_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "memory" / "experience.jsonl"
    store = JsonlExperienceStore(path)
    assert not path.exists()
    store.add(ExperienceRecord(
        issue="focus", action="放大主体", score_before=0.5,
        score_after=0.8, delta=0.3, success=True,
    ))
    assert path.is_file()
    restored = JsonlExperienceStore(path)
    assert restored.records()[0].action == "放大主体"


def wait_for(store: TaskStore, task_id: str) -> str:
    status = ""
    for _ in range(200):
        status = str(store.status(task_id)["status"])
        if status in {"completed", "failed"}:
            return status
        time.sleep(0.01)
    return status


def test_executor_learns_then_reuses_strategy(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")
    experience = ExperienceMemory()
    executor = TaskExecutor(store, build_default_registry(), experience=experience)
    first = TaskRequest(prompt="生成高端香水活动海报", max_iterations=8)
    first_id = store.create(first)
    executor.submit(first_id, first)
    assert wait_for(store, first_id) == "completed"
    assert experience.count() == 1

    second = TaskRequest(prompt="生成高端精华活动海报", max_iterations=8)
    second_id = store.create(second)
    executor.submit(second_id, second)
    assert wait_for(store, second_id) == "completed"
    result = store.result(second_id)
    assert result["experience_used"] is True
    assert "根据评估意见重新生成或编辑" in result["assets"][1]["prompt"]
    executor.shutdown()
