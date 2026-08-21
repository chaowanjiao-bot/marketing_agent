from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuthService:
    def __init__(self, path: Path, session_hours: int = 168) -> None:
        self.path = path
        self.session_hours = session_hours
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL, password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL, created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    name TEXT NOT NULL, created_at TEXT NOT NULL,
                    FOREIGN KEY(owner_id) REFERENCES users(user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id);
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @staticmethod
    def _password(password: str, salt: bytes) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000).hex()

    def register(self, email: str, password: str, display_name: str) -> dict[str, str]:
        email = email.strip().casefold()
        display_name = " ".join(display_name.split())
        if "@" not in email or len(email) > 254:
            raise ValueError("invalid email")
        if len(password) < 10 or len(password) > 200:
            raise ValueError("password must be between 10 and 200 characters")
        if not display_name or len(display_name) > 80:
            raise ValueError("invalid display name")
        user_id, project_id = f"user_{uuid4().hex[:12]}", f"project_{uuid4().hex[:12]}"
        salt = os.urandom(16)
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, email, display_name, self._password(password, salt), salt.hex(), _now()),
                )
                connection.execute(
                    "INSERT INTO projects VALUES (?, ?, ?, ?)",
                    (project_id, user_id, "我的第一个项目", _now()),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("email is already registered") from exc
        return {"user_id": user_id, "email": email, "display_name": display_name,
                "default_project_id": project_id}

    def login(self, email: str, password: str) -> tuple[str, dict[str, str]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email=?", (email.strip().casefold(),)
            ).fetchone()
        if row is None or not hmac.compare_digest(
            self._password(password, bytes.fromhex(row["password_salt"])), row["password_hash"]
        ):
            raise ValueError("invalid email or password")
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(hours=self.session_hours)).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?)",
                (hashlib.sha256(token.encode()).hexdigest(), row["user_id"], expires, _now()),
            )
        return token, {"user_id": row["user_id"], "email": row["email"],
                       "display_name": row["display_name"]}

    def identity(self, token: str | None) -> dict[str, str] | None:
        if not token:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """SELECT u.user_id,u.email,u.display_name FROM sessions s JOIN users u
                   ON u.user_id=s.user_id WHERE s.token_hash=? AND s.expires_at>?""",
                (hashlib.sha256(token.encode()).hexdigest(), _now()),
            ).fetchone()
        return dict(row) if row else None

    def logout(self, token: str | None) -> None:
        if token:
            with self._connect() as connection:
                connection.execute("DELETE FROM sessions WHERE token_hash=?",
                                   (hashlib.sha256(token.encode()).hexdigest(),))

    def create_project(self, owner_id: str, name: str) -> dict[str, str]:
        name = " ".join(name.split())
        if not name or len(name) > 100:
            raise ValueError("invalid project name")
        project_id = f"project_{uuid4().hex[:12]}"
        with self._connect() as connection:
            connection.execute("INSERT INTO projects VALUES (?, ?, ?, ?)",
                               (project_id, owner_id, name, _now()))
        return {"project_id": project_id, "name": name}

    def projects(self, owner_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT project_id,name,created_at FROM projects WHERE owner_id=? ORDER BY created_at",
                (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def owns_project(self, owner_id: str, project_id: str) -> bool:
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM projects WHERE project_id=? AND owner_id=?",
                (project_id, owner_id),
            ).fetchone() is not None
