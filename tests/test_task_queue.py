from pathlib import Path

from marketing_agent.task_queue import DurableTaskQueue


def test_queue_claim_is_durable_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    queue = DurableTaskQueue(path)
    queue.enqueue("task_one", {"request": {"prompt": "one"}})

    claimed = DurableTaskQueue(path).claim("worker-a")

    assert claimed is not None and claimed["task_id"] == "task_one"
    assert queue.claim("worker-b") is None
    assert queue.stats()["running"] == 1


def test_queue_cancel_and_recover(tmp_path: Path) -> None:
    queue = DurableTaskQueue(tmp_path / "queue.sqlite3")
    queue.enqueue("task_cancel", {})
    assert queue.cancel("task_cancel") is True
    queue.enqueue("task_recover", {})
    assert queue.claim("dead-worker") is not None

    assert queue.recover_running(older_than_seconds=0) == 1
    assert queue.claim("new-worker")["task_id"] == "task_recover"
