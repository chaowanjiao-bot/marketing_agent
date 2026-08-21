import time
from pathlib import Path

from fastapi.testclient import TestClient

from marketing_agent.api import create_app


def test_authenticated_users_cannot_read_each_others_tasks_or_uploads(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    app = create_app(task_root=tmp_path / "tasks")
    with TestClient(app) as alice, TestClient(app) as bob:
        alice_register = alice.post("/auth/register", json={
            "email": "alice@example.com", "password": "correct-horse-123",
            "display_name": "Alice",
        })
        bob_register = bob.post("/auth/register", json={
            "email": "bob@example.com", "password": "battery-staple-456",
            "display_name": "Bob",
        })
        assert alice_register.status_code == bob_register.status_code == 201
        project_id = alice_register.json()["default_project_id"]
        uploaded = alice.post("/assets", files={
            "file": ("product.png", b"\x89PNG\r\n\x1a\nmock", "image/png")
        }).json()
        created = alice.post("/tasks", json={
            "prompt": "Create an isolated campaign", "project_id": project_id,
            "input_asset_id": uploaded["asset_id"], "max_iterations": 8,
        })
        assert created.status_code == 202
        task_id = created.json()["task_id"]
        for _ in range(100):
            if alice.get(f"/tasks/{task_id}").json()["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)

        assert bob.get(f"/tasks/{task_id}").status_code == 404
        assert bob.get(f"/tasks/{task_id}/result").status_code == 404
        assert bob.post("/tasks", json={
            "prompt": "Try another user's upload",
            "project_id": bob_register.json()["default_project_id"],
            "input_asset_id": uploaded["asset_id"],
        }).status_code == 404
        assert alice.get("/tasks").json()["count"] == 1
        assert bob.get("/tasks").json()["count"] == 0


def test_authentication_required_when_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        assert client.get("/tasks").status_code == 401
        assert client.post("/tasks", json={"prompt": "No session"}).status_code == 401
