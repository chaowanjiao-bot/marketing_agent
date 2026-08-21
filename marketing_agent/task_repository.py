from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteTaskRepository:
    """Durable, queryable task metadata; large artifacts remain in TaskStore."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    phase TEXT,
                    prompt TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_updated_at
                    ON tasks(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tasks_status_updated_at
                    ON tasks(status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_task_events_task_id
                    ON task_events(task_id, event_id);
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)")}
            if "owner_id" not in columns:
                try:
                    connection.execute(
                        "ALTER TABLE tasks ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'anonymous'"
                    )
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise
            if "project_id" not in columns:
                try:
                    connection.execute("ALTER TABLE tasks ADD COLUMN project_id TEXT")
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise

    def create(
        self, task_id: str, request: dict[str, Any], *, owner_id: str = "anonymous",
        project_id: str | None = None,
    ) -> None:
        timestamp = _now()
        request_json = json.dumps(request, ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO tasks
                   (task_id, status, phase, prompt, request_json, created_at, updated_at,
                    owner_id, project_id)
                   VALUES (?, 'created', 'created', ?, ?, ?, ?, ?, ?)""",
                (task_id, str(request.get("prompt", "")), request_json, timestamp, timestamp,
                 owner_id, project_id),
            )
            connection.execute(
                """INSERT INTO task_events
                   (task_id, status, phase, details_json, created_at)
                   VALUES (?, 'created', 'created', '{}', ?)""",
                (task_id, timestamp),
            )

    def update_status(
        self, task_id: str, status: str, phase: str | None, details: dict[str, Any]
    ) -> None:
        timestamp = _now()
        details_json = json.dumps(details, ensure_ascii=False)
        with self._connect() as connection:
            updated = connection.execute(
                """UPDATE tasks SET status=?, phase=?, details_json=?, updated_at=?
                   WHERE task_id=?""",
                (status, phase, details_json, timestamp, task_id),
            )
            if updated.rowcount != 1:
                raise KeyError(task_id)
            connection.execute(
                """INSERT INTO task_events
                   (task_id, status, phase, details_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (task_id, status, phase, details_json, timestamp),
            )

    def update_request(self, task_id: str, request: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET prompt=?, request_json=?, updated_at=? WHERE task_id=?",
                (str(request.get("prompt", "")), json.dumps(request, ensure_ascii=False), _now(), task_id),
            )

    def list(
        self, *, status: str | None = None, limit: int = 50,
        owner_id: str | None = None, project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM tasks"
        parameters: list[Any] = []
        clauses = []
        if status:
            clauses.append("status=?")
            parameters.append(status)
        if owner_id is not None:
            clauses.append("owner_id=?")
            parameters.append(owner_id)
        if project_id is not None:
            clauses.append("project_id=?")
            parameters.append(project_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._task_row(row) for row in rows]

    def owner(self, task_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT owner_id FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return str(row["owner_id"])

    def events(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(task_id)
            rows = connection.execute(
                """SELECT event_id, status, phase, details_json, created_at
                   FROM task_events WHERE task_id=? ORDER BY event_id""",
                (task_id,),
            ).fetchall()
        return [{
            "event_id": row["event_id"], "status": row["status"],
            "phase": row["phase"], "details": json.loads(row["details_json"]),
            "created_at": row["created_at"],
        } for row in rows]

    @staticmethod
    def _task_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["task_id"], "status": row["status"],
            "phase": row["phase"], "prompt": row["prompt"],
            "details": json.loads(row["details_json"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "project_id": row["project_id"],
        }
