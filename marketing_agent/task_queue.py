from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DurableTaskQueue:
    """SQLite-backed queue with atomic claiming for one or more GPU workers."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS task_jobs (
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    worker_id TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    claimed_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_task_jobs_state_id
                    ON task_jobs(state, job_id);
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def enqueue(self, task_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO task_jobs
                   (task_id, payload_json, state, created_at)
                   VALUES (?, ?, 'queued', ?)
                   ON CONFLICT(task_id) DO UPDATE SET
                     payload_json=excluded.payload_json, state='queued', worker_id=NULL,
                     claimed_at=NULL, finished_at=NULL""",
                (task_id, json.dumps(payload, ensure_ascii=False), _now()),
            )

    def claim(self, worker_id: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT job_id, task_id, payload_json FROM task_jobs "
                "WHERE state='queued' ORDER BY job_id LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                """UPDATE task_jobs SET state='running', worker_id=?,
                   attempts=attempts+1, claimed_at=? WHERE job_id=?""",
                (worker_id, _now(), row["job_id"]),
            )
            connection.commit()
            return {
                "job_id": row["job_id"], "task_id": row["task_id"],
                "payload": json.loads(row["payload_json"]),
            }
        finally:
            connection.close()

    def finish(self, task_id: str, state: str = "completed") -> None:
        if state not in {"completed", "failed"}:
            raise ValueError("invalid terminal queue state")
        with self._connect() as connection:
            connection.execute(
                "UPDATE task_jobs SET state=?, finished_at=? WHERE task_id=?",
                (state, _now(), task_id),
            )

    def cancel(self, task_id: str) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                """UPDATE task_jobs SET state='cancelled', finished_at=?
                   WHERE task_id=? AND state='queued'""",
                (_now(), task_id),
            )
        return result.rowcount == 1

    def recover_running(self, older_than_seconds: int = 3600) -> int:
        """Requeue jobs left running after a worker process died."""
        cutoff = (datetime.now(timezone.utc) - timedelta(
            seconds=max(0, older_than_seconds)
        )).isoformat()
        with self._connect() as connection:
            result = connection.execute(
                """UPDATE task_jobs SET state='queued', worker_id=NULL, claimed_at=NULL
                   WHERE state='running' AND claimed_at <= ?""",
                (cutoff,),
            )
        return result.rowcount

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM task_jobs GROUP BY state"
            ).fetchall()
        values = {row["state"]: row["count"] for row in rows}
        return {name: values.get(name, 0) for name in (
            "queued", "running", "completed", "failed", "cancelled"
        )}
