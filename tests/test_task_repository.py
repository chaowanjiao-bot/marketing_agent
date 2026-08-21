from pathlib import Path

from marketing_agent.schemas import TaskRequest
from marketing_agent.task_store import TaskStore


def test_task_metadata_and_events_survive_store_restart(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    store = TaskStore(root)
    task_id = store.create(TaskRequest(prompt="Create a premium skincare poster"))
    store.set_status(task_id, "queued", phase="waiting_for_worker")
    store.set_status(task_id, "running", phase="agent_execution")

    restarted = TaskStore(root)
    tasks = restarted.list_tasks()
    events = restarted.events(task_id)

    assert tasks[0]["task_id"] == task_id
    assert tasks[0]["status"] == "running"
    assert [event["status"] for event in events] == ["created", "queued", "running"]


def test_task_metadata_supports_status_filter(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")
    first = store.create(TaskRequest(prompt="First campaign"))
    store.create(TaskRequest(prompt="Second campaign"))
    store.set_status(first, "completed", phase="finished")

    completed = store.list_tasks(status="completed")

    assert [task["task_id"] for task in completed] == [first]
