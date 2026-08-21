from __future__ import annotations

import os
import signal
import socket
import time
from pathlib import Path

from .case_memory import CaseMemory
from .executor import TaskExecutor
from .experience import ExperienceMemory, JsonlExperienceStore
from .provenance import build_provenance_service
from .schemas import ReviewRecord, TaskRequest
from .task_queue import DurableTaskQueue
from .task_store import TaskStore
from .tools import build_default_registry


def _enabled(name: str) -> bool:
    return os.environ.get(name, "false").lower() in {"1", "true", "yes", "on"}


def build_executor(root: Path) -> tuple[TaskExecutor, object]:
    task_root = Path(os.environ.get("TASK_ROOT", root / "runtime/tasks"))
    store = TaskStore(task_root, metadata_path=Path(os.environ.get(
        "TASK_DATABASE_PATH", task_root.parent / "task_metadata.sqlite3"
    )))
    if os.environ.get("AGENT_TOOLSET", "mock") == "production":
        from .real_tools import build_production_registry
        tools = build_production_registry()
    else:
        tools = build_default_registry()
    memory = CaseMemory(Path(os.environ.get(
        "RAG_DATABASE_PATH", task_root.parent / "memory/cases.sqlite3"
    ))) if _enabled("RAG_ENABLED") else None
    experience = ExperienceMemory(JsonlExperienceStore(Path(os.environ.get(
        "EXPERIENCE_MEMORY_PATH", task_root.parent / "memory/experience.jsonl"
    )))) if _enabled("EXPERIENCE_MEMORY_ENABLED") else None
    executor = TaskExecutor(
        store, tools, memory=memory, experience=experience,
        provenance=build_provenance_service(task_root),
    )
    return executor, tools


def run() -> int:
    root = Path(os.environ.get("MARKETING_AGENT_ROOT", Path.cwd()))
    task_root = Path(os.environ.get("TASK_ROOT", root / "runtime/tasks"))
    queue = DurableTaskQueue(Path(os.environ.get(
        "TASK_QUEUE_PATH", task_root.parent / "task_queue.sqlite3"
    )))
    worker_id = os.environ.get("WORKER_ID", f"{socket.gethostname()}-{os.getpid()}")
    executor, tools = build_executor(root)
    stopping = False

    def stop(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    queue.recover_running(int(os.environ.get("WORKER_STALE_SECONDS", "3600")))
    poll_seconds = float(os.environ.get("WORKER_POLL_SECONDS", "1"))
    try:
        while not stopping:
            job = queue.claim(worker_id)
            if job is None:
                time.sleep(poll_seconds)
                continue
            task_id = str(job["task_id"])
            try:
                payload = dict(job["payload"])
                request = TaskRequest.model_validate(payload["request"])
                history = [ReviewRecord.model_validate(item) for item in payload.get("review_history", [])]
                executor.execute_now(task_id, request, history)
                state = str(executor.store.status(task_id).get("status"))
                queue.finish(task_id, "failed" if state == "failed" else "completed")
            except Exception:
                queue.finish(task_id, "failed")
        return 0
    finally:
        executor.shutdown()
        tools.close()


if __name__ == "__main__":
    raise SystemExit(run())
