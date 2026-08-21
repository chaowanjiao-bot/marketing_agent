from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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


def test_concurrent_repository_startup_is_migration_safe(tmp_path: Path) -> None:
    path = tmp_path / "metadata.sqlite3"
    # Create the pre-auth schema that requires both migration columns.
    import sqlite3
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE tasks (task_id TEXT PRIMARY KEY, status TEXT NOT NULL,
            phase TEXT, prompt TEXT NOT NULL, request_json TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL)"""
        )
    with ThreadPoolExecutor(max_workers=2) as pool:
        stores = list(pool.map(lambda _: TaskStore(tmp_path / "tasks", path), range(2)))
    assert all(store.metadata.path == path for store in stores)
